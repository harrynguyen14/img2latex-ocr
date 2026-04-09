"""
run_latex_aug_llm.py
--------------------
Group 3: Logic-based LaTeX augmentation dùng Qwen2.5-Math-7B-Instruct (4-bit NF4).

Các phép biến đổi:
  - expand:   (a+b)^2 ↔ a^2 + 2ab + b^2
  - factor:   a^2-b^2 → (a+b)(a-b)
  - commute:  a + b → b + a
  - neutral:  thêm +x-x, nhân \\frac{y}{y}
  - constant: \\pi ↔ 3.14159..., e ↔ 2.71828...

Features:
  - 4-bit NF4 + double-quant (BitsAndBytes) — fit 2x T4 15GB
  - max_memory cân bằng tải đều 2 GPU
  - Checkpoint/resume crash-safe (save mỗi N batch)
  - ShardWriter — flush ra disk định kỳ, không OOM RAM
  - Batch inference với attention_mask đúng

Chạy:
    python run_latex_aug_llm.py
    python run_latex_aug_llm.py --model Qwen/Qwen2.5-Math-7B-Instruct --n_samples 50000
"""

import argparse
import gc
import json
import logging
import random
import re
import time
import warnings
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore")

# ── Defaults ──────────────────────────────────────────────────────────────────

DATASET_DIR = Path("D:/dataset-ocr-builder/latex-ocr-dataset")
OUT_DIR     = DATASET_DIR / "train" / "heavy_text"
SEED        = 42

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(message)s",
    handlers= [logging.StreamHandler()],
)
log = logging.getLogger("latex_aug")

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a LaTeX math expert. "
    "When given a LaTeX math expression, apply ONE of the following transformations "
    "and return ONLY the transformed LaTeX — no explanation, no markdown, no $...$ wrapper:\n"
    "1. Expand: expand algebraic expressions (e.g. (a+b)^2 → a^2+2ab+b^2)\n"
    "2. Factor: factor expressions (e.g. a^2-b^2 → (a+b)(a-b))\n"
    "3. Commute: reorder commutative terms (e.g. a+b+c → c+a+b)\n"
    "4. Neutral: add neutral elements (e.g. x → x + y - y)\n"
    "5. Constant: replace symbolic constant with numeric (e.g. \\pi → 3.14159)\n"
    "Return the LaTeX expression ONLY. If transformation is not applicable, "
    "return the original expression unchanged."
)

_TRANSFORMS    = ["expand", "factor", "commute", "neutral", "constant"]
_USER_TEMPLATE = "Transform this LaTeX expression using '{transform}':\n{latex}"


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# ShardWriter
# ══════════════════════════════════════════════════════════════════════════════

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
                "image":  pa.array([r["image"]  for r in self.buffer], type=pa.binary()),
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


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_latex_corpus(raw_dir: Path, n_samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for pfile in sorted(raw_dir.glob("*.parquet")):
        tbl = pq.read_table(str(pfile), columns=["idx", "image", "latex", "source"])
        for idx, img, lat, src in zip(
            tbl["idx"].to_pylist(),
            tbl["image"].to_pylist(),
            tbl["latex"].to_pylist(),
            tbl["source"].to_pylist(),
        ):
            if lat and lat.strip():
                records.append({"idx": idx, "image": img, "latex": lat, "source": src})
    rng.shuffle(records)
    return records[:n_samples]


def clean_output(raw: str) -> str:
    out = raw.strip()
    out = re.sub(r"^```.*?\n", "", out, flags=re.DOTALL)
    out = re.sub(r"```$",      "", out)
    out = re.sub(r"^\$\$?(.*?)\$\$?$",                                  r"\1", out, flags=re.DOTALL)
    out = re.sub(r"^\\\[(.*?)\\\]$",                                     r"\1", out, flags=re.DOTALL)
    out = re.sub(r"^\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}$",   r"\1", out, flags=re.DOTALL)
    return out.strip()


def is_valid_output(original: str, output: str) -> bool:
    if not output or len(output.strip()) < 2:
        return False
    if output.strip() == original.strip():
        return False
    if len(output) > len(original) * 5:
        return False
    return True


def log_gpu_memory():
    for i in range(torch.cuda.device_count()):
        used  = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        log.info(f"  GPU {i}: {used:.2f} / {total:.2f} GB")


# ══════════════════════════════════════════════════════════════════════════════
# Model loading — tối ưu 2x T4 15GB
# ══════════════════════════════════════════════════════════════════════════════

def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.float16,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_storage    = torch.uint8,
    )


def build_max_memory() -> dict:
    """Cân bằng tải đều 2x T4 — dành 1GB buffer cho CUDA overhead."""
    n_gpu = torch.cuda.device_count()
    mem   = {}
    for i in range(n_gpu):
        total_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        mem[i]   = f"{int(total_gb) - 1}GiB"
    mem["cpu"] = "24GiB"   # spill sang CPU RAM nếu cần
    return mem


def load_model(model_name: str):
    log.info(f"Loading {model_name}  |  4-bit NF4 + double-quant")
    max_mem = build_max_memory()
    log.info(f"  max_memory: {max_mem}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = build_bnb_config(),
        device_map          = "auto",
        max_memory          = max_mem,
    )
    model.eval()

    log.info("Model ready ✓")
    log_gpu_memory()
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def build_messages(latex: str, transform: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _USER_TEMPLATE.format(transform=transform, latex=latex)},
    ]


def get_first_device(model) -> torch.device:
    """Lấy device của embedding layer — dùng để đưa inputs lên đúng GPU đầu tiên."""
    return next(model.parameters()).device


def batch_generate(model, tokenizer, batch: list[dict], max_new_tokens: int) -> list[str]:
    texts = [
        tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in batch
    ]

    inputs = tokenizer(
        texts,
        return_tensors = "pt",
        padding        = True,
        truncation     = True,
        max_length     = 512,
    )
    # Đưa input lên GPU đầu tiên — accelerate/device_map tự route các layer sau
    first_device = get_first_device(model)
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            input_ids      = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            max_new_tokens = max_new_tokens,
            do_sample      = False,
            pad_token_id   = tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    results = [
        clean_output(tokenizer.decode(ids[input_len:], skip_special_tokens=True))
        for ids in output_ids
    ]

    del inputs, output_ids
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    ap.add_argument("--raw_dir",    type=str, default=str(DATASET_DIR / "train" / "raw"))
    ap.add_argument("--out_dir",    type=str, default=str(OUT_DIR))
    ap.add_argument("--n_samples",      type=int,   default=50_000)
    ap.add_argument("--batch_size",     type=int,   default=8)
    ap.add_argument("--max_new_tokens", type=int,   default=128)
    ap.add_argument("--shard_size",     type=int,   default=5_000)
    ap.add_argument("--ckpt_every",     type=int,   default=50,
                    help="Save checkpoint mỗi N batch")
    ap.add_argument("--gc_every",       type=int,   default=200,
                    help="gc.collect() mỗi N batch")
    ap.add_argument("--max_retries",    type=int,   default=3)
    ap.add_argument("--retry_delay",    type=float, default=2.0)
    ap.add_argument("--seed",           type=int,   default=SEED)
    return ap.parse_args()


def main():
    args    = parse_args()
    rng     = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # File log
    fh = logging.FileHandler(out_dir / "llm_aug.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    log.addHandler(fh)

    log.info("=" * 60)
    log.info("  LaTeX Augmentation  |  Qwen2.5-Math-7B  |  2x T4 15GB")
    log.info("=" * 60)

    ckpt   = Checkpoint(out_dir / "llm_aug_checkpoint.json")
    writer = ShardWriter(out_dir, shard_size=args.shard_size)

    # 1. Load corpus
    log.info(f"Loading up to {args.n_samples:,} samples from {args.raw_dir}...")
    records = load_latex_corpus(Path(args.raw_dir), args.n_samples, args.seed)
    log.info(f"  loaded: {len(records):,}")

    # 2. Assign transform + stable idx dựa trên seed
    for i, r in enumerate(records):
        r["idx"]       = i
        r["transform"] = rng.choice(_TRANSFORMS)
        r["messages"]  = build_messages(r["latex"], r["transform"])

    pending      = [r for r in records if not ckpt.is_done(str(r["idx"]))]
    already_done = len(records) - len(pending)
    log.info(f"Done: {already_done:,}  |  Remaining: {len(pending):,}")

    if not pending:
        log.info("All samples already processed. Nothing to do.")
        return

    # 3. Load model
    model, tokenizer = load_model(args.model)

    # 4. Batch inference
    stats = {"success": 0, "failed": 0}
    bs    = args.batch_size

    pbar = tqdm(range(0, len(pending), bs), desc="Augmenting", ncols=90)
    for batch_no, start in enumerate(pbar):
        batch = pending[start : start + bs]

        # Retry loop per batch
        outputs = None
        for attempt in range(1, args.max_retries + 1):
            try:
                outputs = batch_generate(model, tokenizer, batch, args.max_new_tokens)
                break
            except torch.cuda.OutOfMemoryError:
                log.warning(f"OOM batch {batch_no} attempt {attempt} — clearing cache")
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(args.retry_delay)
            except Exception as e:
                log.warning(f"Batch {batch_no} attempt {attempt}/{args.max_retries}: {e}")
                if attempt < args.max_retries:
                    time.sleep(args.retry_delay * attempt)

        if outputs is None:
            outputs = [""] * len(batch)

        # Record results
        for r, out in zip(batch, outputs):
            idx = str(r["idx"])
            if is_valid_output(r["latex"], out):
                writer.write({
                    "idx":    r["idx"],
                    "image":  r["image"],
                    "latex":  out,
                    "source": r["source"],
                })
                ckpt.mark_done(idx)
                stats["success"] += 1
            else:
                ckpt.mark_failed(idx, "invalid_output")
                stats["failed"] += 1

        pbar.set_postfix(ok=stats["success"], fail=stats["failed"])

        # Checkpoint mỗi ckpt_every batch (không phải mỗi batch — giảm I/O)
        if (batch_no + 1) % args.ckpt_every == 0:
            ckpt.save()

        # GC định kỳ
        if (batch_no + 1) % args.gc_every == 0:
            gc.collect()
            torch.cuda.empty_cache()
            log_gpu_memory()

    # Save checkpoint lần cuối
    ckpt.save()

    # 5. Finalize shards
    final_paths = writer.finalize()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    log.info("Model unloaded, VRAM freed ✓")

    # 6. Stats
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
