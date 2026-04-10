import argparse
import gc
import json
import logging
import random
import re
import warnings
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# CuDNN optimizations for RTX 3090 Ti (sm86, Ampere)
torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True  # auto-tune kernels for fixed input shapes
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for faster matmul on Ampere+
torch.backends.cudnn.allow_tf32       = True

HF_DATASET = "harryrobert/latex-raw"
OUT_DIR    = Path("/workspace/output")
SEED       = 42

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(message)s",
    handlers= [logging.StreamHandler()],
)
log = logging.getLogger("latex_aug")

for _noisy in ("httpx", "httpcore", "huggingface_hub", "filelock", "transformers", "accelerate"):
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


def iter_hf_dataset(hf_dataset: str, hf_token: str | None = None):
    log.info(f"Loading dataset from HuggingFace: {hf_dataset}")
    ds = load_dataset(hf_dataset, split="raw_train", token=hf_token)
    for row in ds:
        lat = row.get("latex")
        if lat and is_augmentable(lat):
            yield {"idx": row["idx"], "latex": lat, "source": row.get("source", "")}


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


def build_messages(latex: str, transform: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _TRANSFORM_TEMPLATES[transform].replace("{latex}", latex)},
    ]


def load_model(model_name: str):
    log.info(f"Loading {model_name}  |  transformers  |  bfloat16 + FA2")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype                = torch.bfloat16,
        device_map           = "cuda",
        attn_implementation  = "sdpa",
        trust_remote_code    = True,
    )
    model.eval()
    log.info("Model ready ✓")
    log_gpu_memory()
    return model, tokenizer


def flush_batch(batch: list[dict], model, tokenizer, args, writer, ckpt, stats, pbar, batch_no_ref):
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(r["latex"], r["transform"]),
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in batch
    ]

    # Sort by prompt length — minimizes padding within batch (dynamic padding)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    batch_sorted   = [batch[i]   for i in order]
    prompts_sorted = [prompts[i] for i in order]

    inputs = tokenizer(
        prompts_sorted,
        return_tensors = "pt",
        padding        = True,
        truncation     = True,
        max_length     = 512,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens     = args.max_new_tokens,
            do_sample          = True,
            temperature        = 0.7,
            top_p              = 0.8,
            top_k              = 20,
            repetition_penalty = 1.1,
            pad_token_id       = tokenizer.pad_token_id,
        )

    # Each sequence has different input length due to left-padding
    input_lens = inputs["attention_mask"].sum(dim=1).tolist()
    for r, out_ids, inp_len in zip(batch_sorted, outputs, input_lens):
        text = tokenizer.decode(out_ids[int(inp_len):], skip_special_tokens=True)
        text = clean_output(text)
        idx  = str(r["idx"])
        if is_valid_output(r["latex"], text):
            writer.write({"idx": r["idx"], "latex": text, "source": r["source"]})
            ckpt.mark_done(idx)
            stats["success"] += 1
        else:
            reason = "empty"     if not text or len(text.strip()) < 2 else \
                     "unchanged" if text.strip() == r["latex"].strip() else "too_long"
            ckpt.mark_failed(idx, reason)
            stats["failed"] += 1

    pbar.update(len(batch))
    pbar.set_postfix(ok=stats["success"], fail=stats["failed"])
    batch_no_ref[0] += 1
    if batch_no_ref[0] % args.ckpt_every == 0:
        ckpt.save()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",          type=str,   default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--hf_dataset",     type=str,   default=HF_DATASET)
    ap.add_argument("--hf_token",       type=str,   default=None)
    ap.add_argument("--out_dir",        type=str,   default=str(OUT_DIR))
    ap.add_argument("--n_samples",      type=int,   default=1_400_000)
    ap.add_argument("--batch_size",     type=int,   default=256)  # RTX 3090 Ti 24GB, model ~3GB
    ap.add_argument("--max_new_tokens", type=int,   default=128)
    ap.add_argument("--shard_size",     type=int,   default=5_000)
    ap.add_argument("--ckpt_every",     type=int,   default=10)
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
    log.info(f"  LaTeX Augmentation  |  transformers  |  {args.model}")
    log.info("=" * 60)

    ckpt   = Checkpoint(out_dir / "llm_aug_checkpoint.json")
    writer = ShardWriter(out_dir, shard_size=args.shard_size)
    model, tokenizer = load_model(args.model)

    stats      = {"success": 0, "failed": 0, "skipped": 0}
    total_seen = 0
    batch_no   = [0]
    batch: list[dict] = []
    stop = [False]

    import signal
    def _handler(*_):
        log.warning("Interrupt received — finishing current batch then saving...")
        stop[0] = True
    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)

    pbar = tqdm(desc="Augmenting", ncols=90, unit="sample", total=args.n_samples)

    for record in iter_hf_dataset(args.hf_dataset, args.hf_token):
        if total_seen >= args.n_samples or stop[0]:
            break

        idx_str = str(record["idx"])
        if ckpt.is_done(idx_str):
            stats["skipped"] += 1
            continue

        total_seen += 1
        record["transform"] = rng.choice(_TRANSFORMS)
        batch.append(record)

        if len(batch) >= args.batch_size:
            flush_batch(batch, model, tokenizer, args, writer, ckpt, stats, pbar, batch_no)
            batch.clear()
            gc.collect()
            torch.cuda.empty_cache()

    if batch:
        flush_batch(batch, model, tokenizer, args, writer, ckpt, stats, pbar, batch_no)
        batch.clear()

    pbar.close()
    ckpt.save()
    final_paths = writer.finalize()

    log.info("─" * 45)
    log.info(f"Success : {stats['success']:,}")
    log.info(f"Failed  : {stats['failed']:,}")
    log.info(f"Skipped : {stats['skipped']:,}")
    log.info(f"Output  : {out_dir}")
    for p in final_paths:
        log.info(f"  {p.name}")
    log.info("Done ✓")


if __name__ == "__main__":
    main()
