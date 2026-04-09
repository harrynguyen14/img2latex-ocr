"""
tokenizer.py
------------
Train BPE tokenizer trên toàn bộ LaTeX corpus từ cleaned/*.parquet
Output: latex-tokenizer/ (compatible với HuggingFace tokenizers)

Usage:
    python tokenizer.py
    python tokenizer.py --vocab_size 12000 --out_dir D:/my-tokenizer
"""

import argparse
import re
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers, normalizers, decoders
from tokenizers.processors import TemplateProcessing

CLEANED_DIR = Path("D:/dataset-ocr-builder/cleaned")
OUT_DIR     = Path("D:/dataset-ocr-builder/latex-tokenizer")

VOCAB_SIZE  = 8192
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

# Token IDs cho special tokens
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


# ── LaTeX pre-tokenizer ───────────────────────────────────────────────────────
# Tách LaTeX thành các unit có nghĩa trước khi BPE merge:
#   \command   → 1 unit  (e.g. \frac, \alpha)
#   {, }, ^, _ → 1 unit mỗi ký tự
#   chữ số     → riêng lẻ từng digit
#   chữ cái    → riêng lẻ
#   khoảng trắng → bỏ qua

LATEX_TOKEN_RE = re.compile(
    r"\\[a-zA-Z]+|"   # \command
    r"\\[^a-zA-Z]|"   # \, \{ \} v.v.
    r"\d|"             # digit riêng lẻ
    r"[a-zA-Z]|"       # chữ cái riêng lẻ
    r"[^\s]"           # ký tự còn lại (không phải space)
)


def latex_pre_tokenize(text: str) -> list[str]:
    return LATEX_TOKEN_RE.findall(text)


def corpus_iterator(cleaned_dir: Path):
    """Yield từng LaTeX string từ tất cả parquet trong cleaned/."""
    for pfile in sorted(cleaned_dir.glob("*.parquet")):
        table   = pq.read_table(str(pfile), columns=["latex"])
        latexes = table["latex"].to_pylist()
        for latex in latexes:
            if latex and latex.strip():
                yield latex


def corpus_to_pretokenized(cleaned_dir: Path):
    """
    HuggingFace BpeTrainer nhận iterator of list[str] khi dùng
    pre-tokenized input. Yield từng sample đã tách thành tokens.
    """
    for latex in corpus_iterator(cleaned_dir):
        tokens = latex_pre_tokenize(latex)
        if tokens:
            yield tokens


# ── Train ─────────────────────────────────────────────────────────────────────

def train_tokenizer(cleaned_dir: Path, vocab_size: int, out_dir: Path):
    print(f"Training BPE tokenizer")
    print(f"  cleaned_dir : {cleaned_dir}")
    print(f"  vocab_size  : {vocab_size}")
    print(f"  out_dir     : {out_dir}")

    # Đếm corpus size
    total = 0
    for pfile in sorted(cleaned_dir.glob("*.parquet")):
        table  = pq.read_table(str(pfile), columns=["latex"])
        total += len(table)
    print(f"  corpus size : {total:,} samples\n")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Normalizer: strip whitespace đầu cuối
    tokenizer.normalizer = normalizers.Strip()

    # Pre-tokenizer: dùng custom regex split
    # HuggingFace hỗ trợ Split pre-tokenizer với regex pattern
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=pre_tokenizers.SplitPattern(
            r"\\[a-zA-Z]+|\\[^a-zA-Z]|\d|[a-zA-Z]|[^\s]"
        ),
        behavior="isolated",
        invert=False,
    )

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    # Train trên corpus
    tokenizer.train_from_iterator(
        corpus_iterator(cleaned_dir),
        trainer=trainer,
        length=total,
    )

    # Post-processor: tự động thêm <bos> và <eos>
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    # Decoder: nối tokens lại thành string
    tokenizer.decoder = decoders.BPEDecoder()

    return tokenizer


# ── Validate ──────────────────────────────────────────────────────────────────

def validate(tokenizer: Tokenizer, cleaned_dir: Path, n_samples: int = 1000):
    print("\nValidating tokenizer...")
    lengths = []
    unknowns = 0
    total_tokens = 0

    for i, latex in enumerate(corpus_iterator(cleaned_dir)):
        if i >= n_samples:
            break
        enc = tokenizer.encode(latex)
        lengths.append(len(enc.ids))
        total_tokens += len(enc.ids)
        unknowns += enc.ids.count(UNK_ID)

    avg_len = total_tokens / max(len(lengths), 1)
    unk_rate = unknowns / max(total_tokens, 1) * 100

    print(f"  samples checked : {len(lengths):,}")
    print(f"  avg token len   : {avg_len:.1f}")
    print(f"  max token len   : {max(lengths)}")
    print(f"  min token len   : {min(lengths)}")
    print(f"  <unk> rate      : {unk_rate:.4f}%")

    # Sample encode/decode
    print("\n  Sample encode/decode:")
    examples = [
        r"\frac{dy}{dt}",
        r"\sum_{i=0}^{n} x_i^2",
        r"\int_0^\infty e^{-x} dx",
        r"E = mc^2",
    ]
    for ex in examples:
        enc  = tokenizer.encode(ex)
        dec  = tokenizer.decode(enc.ids, skip_special_tokens=True)
        print(f"  input : {ex}")
        print(f"  tokens: {enc.tokens}")
        print(f"  decode: {dec}")
        print()


# ── Save ──────────────────────────────────────────────────────────────────────

def save_tokenizer(tokenizer: Tokenizer, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lưu tokenizer.json (HuggingFace format)
    tokenizer.save(str(out_dir / "tokenizer.json"))

    # Lưu tokenizer_config.json để load bằng AutoTokenizer
    import json
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "model_max_length": 512,
        "padding_side": "right",
    }
    with open(out_dir / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Lưu special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    with open(out_dir / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens_map, f, indent=2)

    print(f"\nTokenizer saved to {out_dir}")
    print(f"  vocab size: {tokenizer.get_vocab_size():,}")
    print(f"  files     : tokenizer.json, tokenizer_config.json, special_tokens_map.json")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned_dir", type=str, default=str(CLEANED_DIR))
    ap.add_argument("--vocab_size",  type=int, default=VOCAB_SIZE)
    ap.add_argument("--out_dir",     type=str, default=str(OUT_DIR))
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cleaned = Path(args.cleaned_dir)
    out     = Path(args.out_dir)

    tokenizer = train_tokenizer(cleaned, args.vocab_size, out)
    validate(tokenizer, cleaned)
    save_tokenizer(tokenizer, out)
