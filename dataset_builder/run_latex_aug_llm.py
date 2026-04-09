import argparse
import gc
import json
import logging
import random
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore")

DATASET_DIR = Path("/kaggle/input/latex-ocr-raw")
OUT_DIR     = Path("/kaggle/working/heavy_text")
SEED        = 42

logging.basicConfig(
    level   = logging.DEBUG,
    format  = "%(asctime)s | %(levelname)-8s | %(message)s",
    handlers= [logging.StreamHandler()],
)
log = logging.getLogger("latex_aug")

for _noisy in ("httpx", "httpcore", "huggingface_hub", "huggingface_hub.file_download",
               "filelock", "transformers", "accelerate"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_SYSTEM_PROMPT = (
    "You are a LaTeX rewriting assistant. "
    "You will receive a LaTeX math expression and a rewrite instruction. "
    "You MUST output a DIFFERENT expression — never return it unchanged. "
    "Examples of valid rewrites: a+b → b+a, \\frac{1}{2} → \\frac{2}{4}, x^2 → x\\cdot x. "
    "Output ONLY the rewritten LaTeX, no explanation, no markdown, no $...$ wrapper."
)

_TRANSFORM_TEMPLATES = {
    "commute": (
        "Reorder the terms or factors in this expression (e.g. a+b → b+a, xy → yx). "
        "Pick any subexpression and swap its order.\n{latex}"
    ),
    "neutral": (
        "Add a neutral element to this expression without changing its value "
        "(e.g. add +0, add -0, multiply by 1, add x-x for any variable x present).\n{latex}"
    ),
    "expand": (
        "Expand or distribute any product, power, or fraction in this expression "
        "(e.g. (a+b)^2 → a^2+2ab+b^2, 2(x+y) → 2x+2y). "
        "If nothing to expand, rewrite a fraction as a sum.\n{latex}"
    ),
    "factor": (
        "Factor or group terms in this expression "
        "(e.g. a^2-b^2 → (a+b)(a-b), 2x+2y → 2(x+y)). "
        "If nothing to factor, group any repeated subexpression.\n{latex}"
    ),
    "constant": (
        "Replace any symbolic constant or variable with a plausible numeric value, "
        "or replace a number with an equivalent expression "
        "(e.g. \\pi → 3.14159, e → 2.71828, 1 → \\frac{2}{2}, 0 → 1-1).\n{latex}"
    ),
}

_TRANSFORMS = list(_TRANSFORM_TEMPLATES.keys())

_AUGMENTABLE_RE = re.compile(
    r'\\frac|\\sum|\\int|\\prod|\\sqrt|\\cdot|\^|_|\+|-|='
)


def is_augmentable(latex: str) -> bool:
    if len(latex.strip()) < 15:
        return False
    return bool(_AUGMENTABLE_RE.search(latex))


class Checkpoint:
    def __init__(self, path: Path):
        self.path         = path
        self.done: set    = set()
        self.failed: dict = {}
        self._load()

    def _load(self):
        if self.path.exists():
            d           = json.loads(self.path.read_text(encoding="utf-8"))
            self.done   = set(d.get("done", []))
            self.failed = d.get("failed", {})
            log.info(f"Checkpoint loaded — done: {len(self.done):,}  failed: {len(self.failed):,}")

    def save(self):
        self.path.write_text(
            json.dumps({"done": sorted(self.done), "failed": self.failed},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def mark_done(self, idx: str):
        self.done.add(idx)
        self.failed.pop(idx, None)

    def mark_failed(self, idx: str, err: str):
        self.failed[idx] = err

    def is_done(self, idx: str) -> bool:
        return idx in self.done


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int = 5_000):
        self.out_dir         = out_dir
        self.shard_size      = shard_size
        self.buffer: list    = []
        self.shard_idx       = 0
        self.tmp_paths: list = []
        out_dir.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict):
        self.buffer.append(record)
        if len(self.buffer) >= self.shard_size:
            self._flush()

    def _flush(self):
        if not self.buffer:
            return
        tmp = self.out_dir / f"_tmp_llm_aug_{self.shard_idx:05d}.parquet"
        pq.write_table(
            pa.table({
                "idx":    pa.array([r["idx"]    for r in self.buffer], type=pa.int64()),
                "latex":  pa.array([r["latex"]  for r in self.buffer], type=pa.string()),
                "source": pa.array([r["source"] for r in self.buffer], type=pa.string()),
            }),
            str(tmp),
            compression="snappy",
        )
        self.tmp_paths.append(tmp)
        log.info(f"Flushed shard {self.shard_idx} → {tmp.name}  ({len(self.buffer)} rows)")
        self.buffer.clear()
        self.shard_idx += 1

    def finalize(self):
        self._flush()
        n = len(self.tmp_paths)
        if n == 0:
            log.warning("No shards to finalize.")
            return []
        final_paths = []
        for i, tmp in enumerate(self.tmp_paths):
            final = self.out_dir / f"heavy_text_train-{i:05d}-of-{n:05d}.parquet"
            tmp.rename(final)
            final_paths.append(final)
        log.info(f"Finalized {n} shard(s) → {self.out_dir}")
        return final_paths


def load_latex_corpus(raw_dir: Path, n_samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    reservoir: list[dict] = []
    total_seen = 0
    for pfile in sorted(raw_dir.glob("*.parquet")):
        tbl = pq.read_table(str(pfile), columns=["idx", "latex", "source"])
        for idx, lat, src in zip(
            tbl["idx"].to_pylist(),
            tbl["latex"].to_pylist(),
            tbl["source"].to_pylist(),
        ):
            if not lat or not is_augmentable(lat):
                continue
            total_seen += 1
            item = {"idx": idx, "latex": lat, "source": src}
            if len(reservoir) < n_samples:
                reservoir.append(item)
            else:
                j = rng.randint(0, total_seen - 1)
                if j < n_samples:
                    reservoir[j] = item

    rng.shuffle(reservoir)
    return reservoir


def clean_output(raw: str) -> str:
    out = raw.strip()
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
    out = re.sub(r"^```[^\n]*\n", "", out)
    out = re.sub(r"\n?```$", "", out)
    out = re.sub(r"^\$\$?(.+?)\$\$?$",                                 r"\1", out, flags=re.DOTALL)
    out = re.sub(r"^\\\[(.+?)\\\]$",                                    r"\1", out, flags=re.DOTALL)
    out = re.sub(r"^\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}$", r"\1", out, flags=re.DOTALL)
    return out.strip()


def is_valid_output(original: str, output: str) -> bool:
    if not isinstance(output, str) or not output or len(output.strip()) < 2:
        return False
    if output.strip() == original.strip():
        return False
    if len(output) > max(len(original) * 4, 512):
        return False
    return True


def log_gpu_memory():
    for i in range(torch.cuda.device_count()):
        used  = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        log.info(f"  GPU {i}: {used:.2f} / {total:.2f} GB")


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.float16,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_storage    = torch.uint8,
    )


def load_model(model_name: str):
    log.info(f"Loading {model_name}  |  4-bit NF4 + double-quant")

    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_gpu = torch.cuda.device_count()
    if n_gpu >= 2:
        total_0 = torch.cuda.get_device_properties(0).total_memory / 1e9
        total_1 = torch.cuda.get_device_properties(1).total_memory / 1e9
        max_mem = {
            0: f"{max(1, int(total_0) - 3)}GiB",
            1: f"{max(1, int(total_1) - 1)}GiB",
            "cpu": "24GiB",
        }
        dmap = "auto"
    else:
        total_0 = torch.cuda.get_device_properties(0).total_memory / 1e9
        max_mem = {0: f"{max(1, int(total_0) - 2)}GiB", "cpu": "24GiB"}
        dmap = "auto"

    log.info(f"  device_map={dmap}  max_memory={max_mem}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = build_bnb_config(),
        device_map          = dmap,
        max_memory          = max_mem,
    )
    model.eval()
    log.info("Model ready ✓")
    log_gpu_memory()
    return model, tokenizer


def build_messages(latex: str, transform: str) -> list[dict]:
    user_content = _TRANSFORM_TEMPLATES[transform].replace("{latex}", latex)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def get_first_device(model) -> torch.device:
    return next(model.parameters()).device


def build_prompt(messages: list[dict], tokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def tokenize_batch(tokenizer, batch: list[dict]) -> dict:
    texts = [build_prompt(item["messages"], tokenizer) for item in batch]
    return tokenizer(
        texts,
        return_tensors = "pt",
        padding        = True,
        truncation     = True,
        max_length     = 512,
    )


def batch_generate(model, tokenizer, batch: list[dict], max_new_tokens: int,
                   prefetched_inputs: dict | None = None) -> list[str]:
    first_device = get_first_device(model)
    cpu_inputs = prefetched_inputs if prefetched_inputs is not None \
                 else tokenize_batch(tokenizer, batch)
    inputs = {k: v.to(first_device) for k, v in cpu_inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            input_ids          = inputs["input_ids"],
            attention_mask     = inputs["attention_mask"],
            max_new_tokens     = max_new_tokens,
            do_sample          = True,
            temperature        = 0.7,
            top_p              = 0.8,
            top_k              = 20,
            min_p              = 0.0,
            repetition_penalty = 1.5,
            use_cache          = True,
            pad_token_id       = tokenizer.pad_token_id,
            eos_token_id       = tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)
    results = [clean_output(t) for t in decoded]

    del inputs, output_ids
    return results


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",          type=str,   default="Qwen/Qwen3-8B")
    ap.add_argument("--raw_dir",        type=str,   default=str(DATASET_DIR))
    ap.add_argument("--out_dir",        type=str,   default=str(OUT_DIR))
    ap.add_argument("--n_samples",      type=int,   default=50_000)
    ap.add_argument("--batch_size",     type=int,   default=96)
    ap.add_argument("--max_new_tokens", type=int,   default=128)
    ap.add_argument("--shard_size",     type=int,   default=5_000)
    ap.add_argument("--ckpt_every",     type=int,   default=50)
    ap.add_argument("--gc_every",       type=int,   default=200)
    ap.add_argument("--max_retries",    type=int,   default=3)
    ap.add_argument("--retry_delay",    type=float, default=2.0)
    ap.add_argument("--seed",           type=int,   default=SEED)
    return ap.parse_args()


def main():
    args    = parse_args()
    rng     = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "llm_aug.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    log.addHandler(fh)

    log.info("=" * 60)
    log.info("  LaTeX Augmentation  |  Qwen3-8B  |  2x T4 15GB")
    log.info("=" * 60)

    ckpt   = Checkpoint(out_dir / "llm_aug_checkpoint.json")
    writer = ShardWriter(out_dir, shard_size=args.shard_size)

    log.info(f"Loading up to {args.n_samples:,} samples from {args.raw_dir}...")
    records = load_latex_corpus(Path(args.raw_dir), args.n_samples, args.seed)
    log.info(f"  loaded: {len(records):,}")

    for r in records:
        r["transform"] = rng.choice(_TRANSFORMS)
        r["messages"]  = build_messages(r["latex"], r["transform"])

    pending      = [r for r in records if not ckpt.is_done(str(r["idx"]))]
    already_done = len(records) - len(pending)
    log.info(f"Done: {already_done:,}  |  Remaining: {len(pending):,}")

    if not pending:
        log.info("All samples already processed. Nothing to do.")
        return

    pending.sort(key=lambda r: len(r["latex"]))
    log.info("Samples sorted by latex length to minimize padding ✓")

    model, tokenizer = load_model(args.model)

    stats = {"success": 0, "failed": 0}
    bs    = args.batch_size

    def _tokenize(batch):
        return tokenize_batch(tokenizer, batch)

    batches   = [pending[s : s + bs] for s in range(0, len(pending), bs)]
    n_batches = len(batches)

    executor     = ThreadPoolExecutor(max_workers=1)
    prefetch_fut = executor.submit(_tokenize, batches[0]) if batches else None

    pbar = tqdm(range(n_batches), desc="Augmenting", ncols=90)
    for batch_no in pbar:
        batch = batches[batch_no]

        prefetched = prefetch_fut.result() if prefetch_fut is not None else None

        if batch_no + 1 < n_batches:
            prefetch_fut = executor.submit(_tokenize, batches[batch_no + 1])
        else:
            prefetch_fut = None

        outputs = None
        for attempt in range(1, args.max_retries + 1):
            try:
                outputs = batch_generate(model, tokenizer, batch, args.max_new_tokens,
                                         prefetched_inputs=prefetched)
                prefetched = None
                break
            except torch.cuda.OutOfMemoryError:
                log.warning(f"OOM batch {batch_no} attempt {attempt}/{args.max_retries} — clearing cache")
                gc.collect()
                torch.cuda.empty_cache()
                prefetched = None
                if attempt < args.max_retries:
                    time.sleep(args.retry_delay)
            except Exception as e:
                log.warning(f"Batch {batch_no} attempt {attempt}/{args.max_retries}: {type(e).__name__}: {e}")
                prefetched = None
                if attempt < args.max_retries:
                    time.sleep(args.retry_delay * attempt)

        if outputs is None:
            log.error(f"All {args.max_retries} retries failed for batch {batch_no}, skipping.")
            outputs = [""] * len(batch)

        for r, out in zip(batch, outputs):
            idx = str(r["idx"])
            if is_valid_output(r["latex"], out):
                writer.write({
                    "idx":    r["idx"],
                    "latex":  out,
                    "source": r["source"],
                })
                ckpt.mark_done(idx)
                stats["success"] += 1
            else:
                reason = "empty" if not out or len(out.strip()) < 2 else \
                         "unchanged" if out.strip() == r["latex"].strip() else "too_long"
                if batch_no == 0:
                    log.debug(f"  FAIL [{reason}] orig={r['latex'][:60]!r} → out={out[:60]!r}")
                ckpt.mark_failed(idx, reason)
                stats["failed"] += 1

        pbar.set_postfix(ok=stats["success"], fail=stats["failed"])

        if (batch_no + 1) % args.ckpt_every == 0:
            ckpt.save()

        if (batch_no + 1) % args.gc_every == 0:
            gc.collect()
            torch.cuda.empty_cache()
            log_gpu_memory()

    ckpt.save()
    executor.shutdown(wait=False)

    final_paths = writer.finalize()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    log.info("Model unloaded, VRAM freed ✓")

    log.info("─" * 45)
    log.info(f"Success : {stats['success']:,}")
    log.info(f"Failed  : {stats['failed']:,}")
    log.info(f"Output  : {out_dir}")
    for p in final_paths:
        log.info(f"  {p.name}")
    if stats["failed"]:
        log.warning("Re-run để auto-retry failed samples (resume tự động).")
    log.info("Done ✓")


if __name__ == "__main__":
    main()