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
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

DATASET_DIR = Path("D:/dataset-ocr-builder/latex-ocr-dataset/train/raw")
OUT_DIR     = Path("D:/dataset-ocr-builder/latex-ocr-dataset/train/heavy_text")
SEED        = 42

logging.basicConfig(
    level   = logging.DEBUG,
    format  = "%(asctime)s | %(levelname)-8s | %(message)s",
    handlers= [logging.StreamHandler()],
)
log = logging.getLogger("latex_aug")

for _noisy in ("httpx", "httpcore", "huggingface_hub", "huggingface_hub.file_download",
               "filelock", "transformers", "accelerate", "vllm"):
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


def build_prompt(latex: str, transform: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _TRANSFORM_TEMPLATES[transform].replace("{latex}", latex)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model(model_name: str, tensor_parallel_size: int, gpu_memory_utilization: float,
               quantization: str | None):
    log.info(f"Loading {model_name}  |  vLLM  |  tp={tensor_parallel_size}  quant={quantization or 'fp16'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model                  = model_name,
        dtype                  = "float16",
        tensor_parallel_size   = tensor_parallel_size,
        gpu_memory_utilization = gpu_memory_utilization,
        quantization           = quantization,
        trust_remote_code      = True,
        max_model_len          = 640,   # 512 input + 128 output
        disable_log_stats      = True,
    )
    log.info("Model ready ✓")
    log_gpu_memory()
    return llm, tokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",                  type=str,   default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--raw_dir",                type=str,   default=str(DATASET_DIR))
    ap.add_argument("--out_dir",                type=str,   default=str(OUT_DIR))
    ap.add_argument("--n_samples",              type=int,   default=1_400_000)
    ap.add_argument("--batch_size",             type=int,   default=32)
    ap.add_argument("--max_new_tokens",         type=int,   default=128)
    ap.add_argument("--shard_size",             type=int,   default=5_000)
    ap.add_argument("--ckpt_every",             type=int,   default=10)
    ap.add_argument("--max_retries",            type=int,   default=3)
    ap.add_argument("--seed",                   type=int,   default=SEED)
    ap.add_argument("--tensor_parallel_size",   type=int,   default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--quantization",           type=str,   default=None,
                    help="vLLM quantization: awq, gptq, or None for fp16")
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
    log.info(f"  LaTeX Augmentation  |  vLLM  |  {args.model}")
    log.info("=" * 60)

    ckpt   = Checkpoint(out_dir / "llm_aug_checkpoint.json")
    writer = ShardWriter(out_dir, shard_size=args.shard_size)

    llm, tokenizer = load_model(
        args.model,
        tensor_parallel_size   = args.tensor_parallel_size,
        gpu_memory_utilization = args.gpu_memory_utilization,
        quantization           = args.quantization,
    )

    sampling_params = SamplingParams(
        temperature        = 0.7,
        top_p              = 0.8,
        top_k              = 20,
        repetition_penalty = 1.1,
        max_tokens         = args.max_new_tokens,
    )

    stats      = {"success": 0, "failed": 0, "skipped": 0}
    total_seen = 0
    batch_no   = 0
    batch: list[dict] = []

    pbar = tqdm(desc="Augmenting", ncols=90, unit="sample", total=args.n_samples)

    def flush_batch(batch: list[dict]):
        nonlocal batch_no
        prompts = [build_prompt(r["latex"], r["transform"], tokenizer) for r in batch]

        outputs = llm.generate(prompts, sampling_params)

        for r, out in zip(batch, outputs):
            text = clean_output(out.outputs[0].text)
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
        pbar.set_postfix(ok=stats["success"], fail=stats["failed"], seen=total_seen)
        batch_no += 1
        if batch_no % args.ckpt_every == 0:
            ckpt.save()

    for record in iter_parquet_files(Path(args.raw_dir)):
        if total_seen >= args.n_samples:
            break

        idx_str = str(record["idx"])
        if ckpt.is_done(idx_str):
            stats["skipped"] += 1
            continue

        total_seen += 1
        record["transform"] = rng.choice(_TRANSFORMS)
        batch.append(record)

        if len(batch) >= args.batch_size:
            flush_batch(batch)
            batch.clear()
            gc.collect()

    if batch:
        flush_batch(batch)
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
    if stats["failed"]:
        log.warning("Re-run để auto-retry failed samples (resume tự động).")
    log.info("Done ✓")


if __name__ == "__main__":
    main()
