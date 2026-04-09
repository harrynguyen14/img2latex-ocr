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
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def iter_parquet_files(raw_dir: Path):
    for pfile in sorted(raw_dir.glob("*.parquet")):
        tbl = pq.read_table(str(pfile), columns=["idx", "latex", "source"])
        rows = list(zip(
            tbl["idx"].to_pylist(),
            tbl["latex"].to_pylist(),
            tbl["source"].to_pylist(),
        ))
        del tbl
        for idx, lat, src in rows:
            if lat and is_augmentable(lat):
                yield {"idx": idx, "latex": lat, "source": src}
        gc.collect()


def clean_output(raw: str) -> str:
    out = raw.strip()
    out = re.sub(r"^```[^\n]*\n", "", out)
    out = re.sub(r"\n?```$", "", out)
    out = re.sub(r"^\$\$?(.+?)\$\$?$",                                 r"\1", out, flags=re.DOTALL)
    out = re.sub(r"^\\\[(.+?)\\\]$",                                    r"\1", out, flags=re.DOTALL)
    out = re.sub(r"^\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}$", r"\1", out, flags=re.DOTALL)
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    if lines:
        out = lines[0]
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


def load_model(model_name: str, gpu_id: int | None = None, load_in_4bit: bool = False):
    quant_str = "int4" if load_in_4bit else "fp16"
    log.info(f"Loading {model_name}  |  {quant_str}")

    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = f"cuda:{gpu_id}" if gpu_id is not None else "cuda:0"

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_compute_dtype    = torch.float16,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type       = "nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = bnb_cfg,
            device_map          = device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype      = torch.float16,
            device_map = device_map,
        )
    model.eval()
    log.info(f"Model ready ✓  ({quant_str}, device={device_map})")
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
            repetition_penalty = 1.1,
            use_cache          = True,
            pad_token_id       = tokenizer.pad_token_id,
            eos_token_id       = tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)
    results = [clean_output(t) for t in decoded]

    del inputs, output_ids
    return results


def process_batch(batch, model, tokenizer, args, ckpt, writer, stats,
                  executor, prefetch_fut, next_batch):
    prefetched = prefetch_fut.result() if prefetch_fut is not None else None

    new_fut = executor.submit(tokenize_batch, tokenizer, next_batch) \
              if next_batch is not None else None

    outputs = None
    for attempt in range(1, args.max_retries + 1):
        try:
            outputs = batch_generate(model, tokenizer, batch, args.max_new_tokens,
                                     prefetched_inputs=prefetched)
            prefetched = None
            break
        except torch.cuda.OutOfMemoryError:
            log.warning(f"OOM attempt {attempt}/{args.max_retries} — clearing cache")
            gc.collect()
            torch.cuda.empty_cache()
            prefetched = None
            if attempt < args.max_retries:
                time.sleep(args.retry_delay)
        except Exception as e:
            log.warning(f"Attempt {attempt}/{args.max_retries}: {type(e).__name__}: {e}")
            prefetched = None
            if attempt < args.max_retries:
                time.sleep(args.retry_delay * attempt)

    if outputs is None:
        log.error(f"All retries failed, skipping batch.")
        outputs = [""] * len(batch)

    for r, out in zip(batch, outputs):
        idx = str(r["idx"])
        if is_valid_output(r["latex"], out):
            writer.write({"idx": r["idx"], "latex": out, "source": r["source"]})
            ckpt.mark_done(idx)
            stats["success"] += 1
        else:
            reason = "empty"     if not out or len(out.strip()) < 2 else \
                     "unchanged" if out.strip() == r["latex"].strip() else "too_long"
            ckpt.mark_failed(idx, reason)
            stats["failed"] += 1

    return new_fut


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",          type=str,   default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--raw_dir",        type=str,   default=str(DATASET_DIR))
    ap.add_argument("--out_dir",        type=str,   default=str(OUT_DIR))
    ap.add_argument("--n_samples",      type=int,   default=1_400_000)
    ap.add_argument("--chunk_size",     type=int,   default=20_000)
    ap.add_argument("--batch_size",     type=int,   default=128)
    ap.add_argument("--max_new_tokens", type=int,   default=96)
    ap.add_argument("--shard_size",     type=int,   default=5_000)
    ap.add_argument("--ckpt_every",     type=int,   default=50)
    ap.add_argument("--gc_every",       type=int,   default=200)
    ap.add_argument("--max_retries",    type=int,   default=3)
    ap.add_argument("--retry_delay",    type=float, default=2.0)
    ap.add_argument("--seed",           type=int,   default=SEED)
    ap.add_argument("--load_in_4bit",   action="store_true")
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
    log.info(f"  LaTeX Augmentation  |  Qwen2.5-Math-1.5B  |  2x T4 15GB")
    log.info("=" * 60)

    ckpt   = Checkpoint(out_dir / "llm_aug_checkpoint.json")
    writer = ShardWriter(out_dir, shard_size=args.shard_size)
    model, tokenizer = load_model(args.model, load_in_4bit=args.load_in_4bit)

    stats      = {"success": 0, "failed": 0, "skipped": 0}
    total_seen = 0
    batch_no   = 0
    executor   = ThreadPoolExecutor(max_workers=1)
    chunk: list[dict] = []

    pbar = tqdm(desc="Augmenting", ncols=90, unit="sample", total=args.n_samples)

    def flush_chunk(chunk):
        nonlocal batch_no
        chunk.sort(key=lambda r: len(r["latex"]))
        batches      = [chunk[s: s + args.batch_size] for s in range(0, len(chunk), args.batch_size)]
        n            = len(batches)
        prefetch_fut = executor.submit(tokenize_batch, tokenizer, batches[0])

        for i, batch in enumerate(batches):
            next_b = batches[i + 1] if i + 1 < n else None
            prefetch_fut = process_batch(
                batch, model, tokenizer, args, ckpt, writer, stats,
                executor, prefetch_fut, next_b,
            )
            batch_no += 1
            pbar.update(len(batch))
            pbar.set_postfix(ok=stats["success"], fail=stats["failed"], seen=total_seen)

            if batch_no % args.ckpt_every == 0:
                ckpt.save()
            if batch_no % args.gc_every == 0:
                gc.collect()
                torch.cuda.empty_cache()
                log_gpu_memory()

    for record in iter_parquet_files(Path(args.raw_dir)):
        if total_seen >= args.n_samples:
            break

        idx_str = str(record["idx"])
        if ckpt.is_done(idx_str):
            stats["skipped"] += 1
            continue

        total_seen += 1
        record["transform"] = rng.choice(_TRANSFORMS)
        record["messages"]  = build_messages(record["latex"], record["transform"])
        chunk.append(record)

        if len(chunk) >= args.chunk_size:
            flush_chunk(chunk)
            chunk.clear()
            gc.collect()

    if chunk:
        flush_chunk(chunk)
        chunk.clear()

    pbar.close()
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
    log.info(f"Skipped : {stats['skipped']:,}")
    log.info(f"Output  : {out_dir}")
    for p in final_paths:
        log.info(f"  {p.name}")
    if stats["failed"]:
        log.warning("Re-run để auto-retry failed samples (resume tự động).")
    log.info("Done ✓")


if __name__ == "__main__":
    main()