import argparse
import gc
import json
import logging
import random
import re
import signal
import warnings
from pathlib import Path

try:
    from pylatexenc.latexwalker import LatexWalker, LatexWalkerError
    _PYLATEXENC_AVAILABLE = True
except ImportError:
    _PYLATEXENC_AVAILABLE = False

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

warnings.filterwarnings("ignore")

HF_DATASET = "harryrobert/latex-raw"
RAW_SPLIT  = "raw_train"
OUT_DIR    = Path("D:/dataset-ocr-builder/latex-ocr-dataset/train/heavy_text_v2")
SEED       = 42

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(message)s",
    handlers= [logging.StreamHandler()],
)
log = logging.getLogger("latex_aug")

for _noisy in ("httpx", "httpcore", "huggingface_hub", "filelock", "transformers", "accelerate", "vllm"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_TRANSFORM_TEMPLATES = {
    "commute_safe": (
        "Rewrite the following LaTeX math expression into a mathematically equivalent but clearly different form. "
        "Only reorder terms or factors when the change is locally safe and preserves meaning. "
        "Do not explain anything. Output only the rewritten LaTeX on one line.\n\nLaTeX:\n{latex}"
    ),
    "neutral_insert": (
        "Rewrite the following LaTeX math expression into a mathematically equivalent but clearly different form. "
        "You may insert a neutral term such as +0, -0, multiplying by 1, or equivalent parentheses around one local subexpression. "
        "Do not explain anything. Output only the rewritten LaTeX on one line.\n\nLaTeX:\n{latex}"
    ),
    "fraction_rewrite": (
        "Rewrite the following LaTeX math expression into a mathematically equivalent but clearly different form. "
        "Prefer rewriting a fraction, reciprocal, or power locally into another equivalent algebraic form. "
        "Examples: \\frac{a}{b} -> a b^{-1}, x^2 -> x\\cdot x when safe. "
        "Do not explain anything. Output only the rewritten LaTeX on one line.\n\nLaTeX:\n{latex}"
    ),
    "group_factor": (
        "Rewrite the following LaTeX math expression into a mathematically equivalent but clearly different form. "
        "Prefer factoring or regrouping one local subexpression if it is clearly valid. "
        "Do not change constants, variable names, indices, dimensions, or the mathematical meaning. "
        "Output only the rewritten LaTeX on one line.\n\nLaTeX:\n{latex}"
    ),
    "expand_local": (
        "Rewrite the following LaTeX math expression into a mathematically equivalent but clearly different form. "
        "Prefer a safe local expansion or distribution of exactly one subexpression only. "
        "Do not rewrite the entire expression aggressively. Output only the rewritten LaTeX on one line.\n\nLaTeX:\n{latex}"
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
                "transform": pa.array([r["transform"] for r in self.buffer], type=pa.string()),
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
    ds = load_dataset(hf_dataset, split=RAW_SPLIT, token=hf_token).shuffle(seed=SEED)
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


# Phrases that indicate the model produced an English explanation instead of LaTeX.
_PROSE_PREFIXES = (
    "to solve", "to add", "to factor", "to expand", "to find",
    "sure,", "let's", "the given", "we need", "we can",
    "first,", "step ", "note that", "therefore", "thus,",
    "this is", "this equation", "this expression", "in this case",
    "we rewrite", "we can rewrite", "rewriting", "since ",
    "because ", "by ", "using ",
)
_BAD_EXACT = {"system", "user", "assistant"}
_ROLE_PREFIX_RE = re.compile(r"^(system|user|assistant)\s*:?", re.IGNORECASE)

_LATEX_SIGNAL_RE = re.compile(
    r"\\[a-zA-Z]|[_^{}]|\d+\s*[+\-*/=<>]\s*\d|\\frac|\\sum|\\int"
)

# Plain sqrt/frac without backslash — programming style, not LaTeX
_PLAIN_SQRT_RE  = re.compile(r"(?<!\\)\bsqrt\s*\(")
_PLAIN_FRAC_RE  = re.compile(r"(?<!\\)\bfrac\s*\{")
# Markdown artifacts
_MARKDOWN_RE    = re.compile(r"\\\_|&amp;|&lt;|&gt;")
# English word density: if >3 consecutive lowercase words → prose
_WORD_RUN_RE    = re.compile(r"[a-z]{3,}(?:\s+[a-z]{3,}){3,}")


def _pylatexenc_ok(text: str) -> bool:
    """Return False if pylatexenc finds a hard parse error."""
    if not _PYLATEXENC_AVAILABLE:
        return True
    try:
        w = LatexWalker(text, tolerant_parsing=False)
        w.get_latex_nodes()[0]
        return True
    except LatexWalkerError:
        return False


def is_valid_output(original: str, output: str) -> bool:
    if not isinstance(output, str) or not output or len(output.strip()) < 2:
        return False
    out = output.strip()
    if out == original.strip():
        return False
    if out.lower() in _BAD_EXACT or _ROLE_PREFIX_RE.match(out):
        return False
    if len(out) > max(len(original) * 4, 512):
        return False
    lower = out.lower()
    # Reject verbose English prose
    if any(lower.startswith(p) for p in _PROSE_PREFIXES):
        return False
    # Reject if contains a run of ≥4 plain English words (prose hallucination)
    if _WORD_RUN_RE.search(out):
        return False
    # Output must contain at least one LaTeX signal
    if _LATEX_SIGNAL_RE.search(original) and not _LATEX_SIGNAL_RE.search(out):
        return False
    # Reject programming-style sqrt/frac without backslash
    if _PLAIN_SQRT_RE.search(out) or _PLAIN_FRAC_RE.search(out):
        return False
    # Reject markdown/HTML artifacts
    if _MARKDOWN_RE.search(out):
        return False
    # pylatexenc structural validation
    if not _pylatexenc_ok(out):
        return False
    return True


def build_prompt(latex: str, transform: str, tokenizer) -> str:
    user_prompt = _TRANSFORM_TEMPLATES[transform].replace("{latex}", latex)
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(model_name: str, gpu_memory_utilization: float, max_model_len: int):
    log.info(f"Loading {model_name}  |  vLLM  |  gpu_util={gpu_memory_utilization}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model                  = model_name,
        dtype                  = "bfloat16",
        gpu_memory_utilization = gpu_memory_utilization,
        trust_remote_code      = True,
        max_model_len          = max_model_len,
        tensor_parallel_size   = 1,
        disable_log_stats      = True,
        enable_prefix_caching  = True,
    )
    log.info("Model ready ✓")
    return llm, tokenizer


def flush_batch(batch: list[dict], llm, tokenizer, sampling_params, writer, ckpt, stats, pbar, batch_no_ref, ckpt_every):
    prompts = [build_prompt(r["latex"], r["transform"], tokenizer) for r in batch]
    outputs = llm.generate(prompts, sampling_params)

    for r, out in zip(batch, outputs):
        text = clean_output(out.outputs[0].text if out.outputs else "")
        idx  = str(r["idx"])
        if is_valid_output(r["latex"], text):
            writer.write({"idx": r["idx"], "latex": text, "source": r["source"], "transform": r["transform"]})
            ckpt.mark_done(idx)
            stats["success"] += 1
            pbar.update(1)
        else:
            lower = text.strip().lower()
            reason = "empty" if not text or len(text.strip()) < 2 else \
                     "unchanged" if text.strip() == r["latex"].strip() else \
                     "role_token" if lower in _BAD_EXACT or _ROLE_PREFIX_RE.match(text.strip()) else \
                     "too_long" if len(text.strip()) > max(len(r["latex"]) * 4, 512) else \
                     "prose" if any(lower.startswith(p) for p in _PROSE_PREFIXES) else \
                     "not_latex"
            ckpt.mark_failed(idx, reason)
            stats["failed"] += 1

    pbar.set_postfix(ok=stats["success"], fail=stats["failed"], seen=stats["seen"])
    batch_no_ref[0] += 1
    if batch_no_ref[0] % ckpt_every == 0:
        ckpt.save()


def flush_pending_bucket(
    pending: list[dict],
    args,
    llm,
    tokenizer,
    sampling_params,
    writer,
    ckpt,
    stats,
    pbar,
    batch_no,
):
    if not pending:
        return
    pending.sort(key=lambda row: len(row["latex"]))
    while pending and stats["success"] < args.n_samples:
        batch = pending[:args.batch_size]
        del pending[:args.batch_size]
        flush_batch(batch, llm, tokenizer, sampling_params, writer, ckpt, stats, pbar, batch_no, args.ckpt_every)
        gc.collect()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",                  type=str,   default="deepseek-ai/deepseek-math-7b-instruct")
    ap.add_argument("--hf_dataset",             type=str,   default=HF_DATASET)
    ap.add_argument("--hf_token",               type=str,   default=None)
    ap.add_argument("--out_dir",                type=str,   default=str(OUT_DIR))
    ap.add_argument("--n_samples",              type=int,   default=0,
                    help="Target number of successful heavy samples (0 = raw_n // 2)")
    ap.add_argument("--batch_size",             type=int,   default=4096,
                    help="Large batch — vLLM handles internal scheduling")
    ap.add_argument("--bucket_size",            type=int,   default=4096,
                    help="Sort by latex length within each bucket before batching")
    ap.add_argument("--max_new_tokens",         type=int,   default=128)
    ap.add_argument("--max_model_len",          type=int,   default=768)
    ap.add_argument("--shard_size",             type=int,   default=5_000)
    ap.add_argument("--ckpt_every",             type=int,   default=10)
    ap.add_argument("--seed",                   type=int,   default=SEED)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    return ap.parse_args()


def main():
    args    = parse_args()
    rng     = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "llm_aug.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    log.addHandler(fh)

    ds = load_dataset(args.hf_dataset, split=RAW_SPLIT, token=args.hf_token)
    raw_n = len(ds)
    target_success = args.n_samples if args.n_samples > 0 else raw_n // 2
    ds = ds.shuffle(seed=args.seed)

    log.info("=" * 60)
    log.info(f"  LaTeX Augmentation  |  vLLM  |  {args.model}")
    log.info(f"  raw_n={raw_n:,}  target_success={target_success:,}")
    log.info("=" * 60)

    ckpt   = Checkpoint(out_dir / "llm_aug_checkpoint.json")
    writer = ShardWriter(out_dir, shard_size=args.shard_size)
    llm, tokenizer = load_model(args.model, args.gpu_memory_utilization, args.max_model_len)

    sampling_params = SamplingParams(
        temperature        = 0.6,
        top_p              = 0.85,
        top_k              = 20,
        repetition_penalty = 1.08,
        max_tokens         = args.max_new_tokens,
        stop               = ["<｜end▁of▁sentence｜>", "\nUser:", "\nAssistant:", "User:", "Assistant:"],
    )

    stats      = {"success": 0, "failed": 0, "skipped": 0, "seen": 0, "filtered": 0}
    batch_no   = [0]
    batch: list[dict] = []
    stop = [False]

    def _handler(*_):
        log.warning("Interrupt - finishing current batch then saving...")
        stop[0] = True
    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)

    pbar = tqdm(desc="Heavy augment", ncols=100, unit="ok", total=target_success)

    for row in ds:
        if stats["success"] >= target_success or stop[0]:
            break

        idx_str = str(row["idx"])
        if ckpt.is_done(idx_str):
            stats["skipped"] += 1
            continue

        lat = row.get("latex")
        if not lat or not is_augmentable(lat):
            stats["filtered"] += 1
            continue

        stats["seen"] += 1
        batch.append({
            "idx": row["idx"],
            "latex": lat,
            "source": row.get("source", ""),
            "transform": rng.choice(_TRANSFORMS),
        })

        if len(batch) >= min(args.batch_size, target_success * 2):
            flush_batch(batch, llm, tokenizer, sampling_params, writer, ckpt, stats, pbar, batch_no, args.ckpt_every)
            batch.clear()
            gc.collect()

    if batch and stats["success"] < target_success:
        flush_batch(batch, llm, tokenizer, sampling_params, writer, ckpt, stats, pbar, batch_no, args.ckpt_every)
        batch.clear()

    pbar.close()
    ckpt.save()
    final_paths = writer.finalize()

    log.info("-" * 45)
    log.info(f"Success : {stats['success']:,}")
    log.info(f"Failed  : {stats['failed']:,}")
    log.info(f"Skipped : {stats['skipped']:,}")
    log.info(f"Filtered: {stats['filtered']:,}")
    log.info(f"Seen    : {stats['seen']:,}")
    log.info(f"Output  : {out_dir}")
    for p in final_paths:
        log.info(f"  {p.name}")
    log.info("Done")


if __name__ == "__main__":
    main()
