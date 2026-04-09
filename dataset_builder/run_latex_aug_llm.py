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

Chạy (1 process, device_map tự trải model qua 2 GPU):
    python run_latex_aug_llm.py
    python run_latex_aug_llm.py --raw_dir /path/to/raw --n_samples 50000

KHÔNG dùng torchrun — torchrun spawn 2 process riêng, mỗi process load full model → OOM.
"""

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

# ── Defaults ──────────────────────────────────────────────────────────────────

DATASET_DIR = Path("/kaggle/input/latex-ocr-raw")
OUT_DIR     = Path("/kaggle/working/heavy_text")
SEED        = 42

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.DEBUG,
    format  = "%(asctime)s | %(levelname)-8s | %(message)s",
    handlers= [logging.StreamHandler()],
)
log = logging.getLogger("latex_aug")

# Tắt verbose logs từ các thư viện bên ngoài
for _noisy in ("httpx", "httpcore", "huggingface_hub", "huggingface_hub.file_download",
               "filelock", "transformers", "accelerate"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a LaTeX rewriting assistant. "
    "You will receive a LaTeX math expression and a rewrite instruction. "
    "You MUST rewrite the expression — never return it unchanged. "
    "Output ONLY the rewritten LaTeX, no explanation, no markdown, no $...$ wrapper."
)

# Template riêng cho từng transform — cụ thể hơn để tránh unchanged
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
            if lat and len(lat.strip()) >= 15:  # bỏ latex quá ngắn — < 15 chars thường không transform được
                records.append({"idx": idx, "image": img, "latex": lat, "source": src})
    rng.shuffle(records)
    return records[:n_samples]


def clean_output(raw: str) -> str:
    out = raw.strip()
    out = re.sub(r"^```[^\n]*\n", "", out)        # strip opening ```lang
    out = re.sub(r"\n?```$",       "", out)        # strip closing ```
    out = re.sub(r"^\$\$?(.+?)\$\$?$",                                  r"\1", out, flags=re.DOTALL)
    out = re.sub(r"^\\\[(.+?)\\\]$",                                     r"\1", out, flags=re.DOTALL)
    out = re.sub(r"^\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}$",   r"\1", out, flags=re.DOTALL)
    return out.strip()


def is_valid_output(original: str, output: str) -> bool:
    if not isinstance(output, str) or not output or len(output.strip()) < 2:
        return False
    if output.strip() == original.strip():
        return False
    # p99 input = 303 chars; expand tối đa ~3x → output hợp lệ < ~900 chars
    # Dùng max(len*4, 512) để không reject input ngắn bị expand nhiều
    if len(output) > max(len(original) * 4, 512):
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
        bnb_4bit_compute_dtype    = torch.float16,  # T4 không có bfloat16 native
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_storage    = torch.uint8,
    )


def build_max_memory() -> dict:
    """Cân bằng tải đều 2x T4 — dành 1.5GB buffer cho CUDA overhead + KV cache."""
    n_gpu = torch.cuda.device_count()
    mem   = {}
    for i in range(n_gpu):
        total_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        mem[i]   = f"{max(1, int(total_gb) - 2)}GiB"  # buffer rộng hơn để tránh OOM
    mem["cpu"] = "24GiB"
    return mem


def load_model(model_name: str):
    log.info(f"Loading {model_name}  |  4-bit NF4 + double-quant")
    max_mem = build_max_memory()
    log.info(f"  max_memory: {max_mem}")

    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = build_bnb_config(),
        device_map          = "auto",  # để accelerate tự quyết — balanced không hiệu quả với BnB quantized model
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
    user_content = _TRANSFORM_TEMPLATES[transform].replace("{latex}", latex)
    return [
        {"role": "user", "content": _SYSTEM_PROMPT + "\n\n" + user_content},
    ]


def get_first_device(model) -> torch.device:
    """Lấy device của embedding layer — dùng để đưa inputs lên đúng GPU đầu tiên."""
    return next(model.parameters()).device


def build_prompt(messages: list[dict]) -> str:
    """Build Qwen3 chat prompt thủ công — nhanh hơn Jinja2 apply_chat_template ~10x."""
    # Format: <|im_start|>role\ncontent<|im_end|>\n
    # Với enable_thinking=False: assistant prefix là <|im_start|>assistant\n<think>\n\n</think>\n
    text = ""
    for msg in messages:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n<think>\n\n</think>\n"
    return text


def tokenize_batch(tokenizer, batch: list[dict]) -> dict:
    """Tokenize batch trên CPU — an toàn chạy trên background thread (không .to(device))."""
    texts = [build_prompt(item["messages"]) for item in batch]
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
    # .to(device) luôn chạy trên main thread để đảm bảo thread-safety với CUDA
    cpu_inputs = prefetched_inputs if prefetched_inputs is not None \
                 else tokenize_batch(tokenizer, batch)
    inputs = {k: v.to(first_device) for k, v in cpu_inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            input_ids      = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            max_new_tokens = max_new_tokens,
            do_sample      = True,
            temperature    = 0.3,   # đủ để thử transform, không quá random làm hỏng LaTeX
            top_p          = 0.9,
            use_cache      = True,
            pad_token_id   = tokenizer.pad_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)
    results = [clean_output(t) for t in decoded]

    del inputs, output_ids
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--raw_dir",    type=str, default=str(DATASET_DIR))
    ap.add_argument("--out_dir",    type=str, default=str(OUT_DIR))
    ap.add_argument("--n_samples",      type=int,   default=50_000)
    ap.add_argument("--batch_size",     type=int,   default=64)   # GPU dùng ~6.5/15GB → còn ~8GB, tăng batch để tận dụng
    ap.add_argument("--max_new_tokens", type=int,   default=96)   # thực tế transform output < 96 tok; eos_token_id sẽ stop sớm hơn
    ap.add_argument("--shard_size",     type=int,   default=5_000)
    ap.add_argument("--ckpt_every",     type=int,   default=50,
                    help="Save checkpoint mỗi N batch")
    ap.add_argument("--gc_every",       type=int,   default=200,
                    help="gc.collect() mỗi N batch")
    ap.add_argument("--max_retries",    type=int,   default=3)
    ap.add_argument("--retry_delay",    type=float, default=2.0)
    ap.add_argument("--seed",           type=int,   default=SEED)
    # Multi-GPU: chạy 2 process độc lập, mỗi process 1 GPU xử lý nửa dataset
    ap.add_argument("--gpu_id",   type=int, default=None,
                    help="GPU index để dùng (0 hoặc 1). None = dùng tất cả GPU (device_map=auto)")
    ap.add_argument("--worker_id",   type=int, default=0,
                    help="Worker index (0 hoặc 1) — quyết định nửa dataset nào sẽ xử lý")
    ap.add_argument("--num_workers", type=int, default=1,
                    help="Tổng số workers đang chạy song song")
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
    log.info("  LaTeX Augmentation  |  Qwen3-8B  |  2x T4 15GB")
    log.info("=" * 60)

    ckpt   = Checkpoint(out_dir / "llm_aug_checkpoint.json")
    writer = ShardWriter(out_dir, shard_size=args.shard_size)

    # 1. Load corpus
    log.info(f"Loading up to {args.n_samples:,} samples from {args.raw_dir}...")
    records = load_latex_corpus(Path(args.raw_dir), args.n_samples, args.seed)
    log.info(f"  loaded: {len(records):,}")

    # 2. Assign transform — dùng idx gốc từ dataset (không overwrite) để checkpoint resume hoạt động đúng
    for r in records:
        r["transform"] = rng.choice(_TRANSFORMS)
        r["messages"]  = build_messages(r["latex"], r["transform"])

    pending      = [r for r in records if not ckpt.is_done(str(r["idx"]))]
    already_done = len(records) - len(pending)
    log.info(f"Done: {already_done:,}  |  Remaining: {len(pending):,}")

    if not pending:
        log.info("All samples already processed. Nothing to do.")
        return

    # Sort by latex length để giảm padding waste trong mỗi batch
    pending.sort(key=lambda r: len(r["latex"]))
    log.info("Samples sorted by latex length to minimize padding ✓")

    # 3. Load model
    model, tokenizer = load_model(args.model)

    # 4. Batch inference với prefetch tokenization
    stats = {"success": 0, "failed": 0}
    bs    = args.batch_size

    def _tokenize(batch):
        return tokenize_batch(tokenizer, batch)

    batches   = [pending[s : s + bs] for s in range(0, len(pending), bs)]
    n_batches = len(batches)

    # Prefetch batch đầu tiên
    executor      = ThreadPoolExecutor(max_workers=1)
    prefetch_fut  = executor.submit(_tokenize, batches[0]) if batches else None

    pbar = tqdm(range(n_batches), desc="Augmenting", ncols=90)
    for batch_no in pbar:
        batch = batches[batch_no]

        # Lấy tokenized inputs từ prefetch
        prefetched = prefetch_fut.result() if prefetch_fut is not None else None

        # Prefetch batch tiếp theo ngay lúc GPU đang chạy inference
        if batch_no + 1 < n_batches:
            prefetch_fut = executor.submit(_tokenize, batches[batch_no + 1])
        else:
            prefetch_fut = None

        # Retry loop per batch
        outputs = None
        for attempt in range(1, args.max_retries + 1):
            try:
                outputs = batch_generate(model, tokenizer, batch, args.max_new_tokens,
                                         prefetched_inputs=prefetched)
                prefetched = None  # đã dùng, tránh double-use khi retry
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
                reason = "empty" if not out or len(out.strip()) < 2 else \
                         "unchanged" if out.strip() == r["latex"].strip() else "too_long"
                if batch_no == 0:  # debug: chỉ log batch đầu tiên
                    log.debug(f"  FAIL [{reason}] orig={r['latex'][:60]!r} → out={out[:60]!r}")
                ckpt.mark_failed(idx, reason)
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
    executor.shutdown(wait=False)

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
