import argparse
import json
import random
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from tokenizers import Tokenizer, normalizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing

from vocab import (
    VOCAB_SIZE, SPECIAL_TOKENS,
    PAD_ID, UNK_ID, BOS_ID, EOS_ID,
    LATEX_TOKEN_RE, TOKENIZER_CONFIG,
    TOP_LATEX_COMMANDS, pretokenize,
)

DATASET_CONFIGS = {
    "raw": {
        "path":  Path("D:/dataset-ocr-builder/latex-ocr-dataset/train/raw"),
        "ratio": 1.0,
        "col":   "latex",
    },
    "light_text": {
        "path":  Path("D:/dataset-ocr-builder/latex-ocr-dataset/train/light_text"),
        "ratio": 1.0,
        "col":   "latex",
    },
    "heavy_text": {
        "path":  Path("D:/dataset-ocr-builder/latex-ocr-dataset/train/heavy_text"),
        "ratio": 1.0,
        "col":   "latex",
    },
}

DEFAULT_OUT_DIR = Path("D:/img2latex/pretrain_decoder/tokenizer")


def corpus_iterator(dataset_configs: dict, seed: int = 42, verbose: bool = True) -> Iterator[str]:
    rng = random.Random(seed)
    for name, cfg in dataset_configs.items():
        files = sorted(cfg["path"].glob("*.parquet"))
        n_keep = max(1, round(len(files) * cfg["ratio"]))
        sampled = rng.sample(files, n_keep)
        if verbose:
            print(f"  {name}: {n_keep}/{len(files)} files ({cfg['ratio']*100:.0f}%)", flush=True)
        for pfile in sampled:
            table = pq.read_table(str(pfile), columns=[cfg["col"]])
            for val in table[cfg["col"]].to_pylist():
                if val and isinstance(val, str) and val.strip():
                    yield val.strip()


def pretokenized_iterator(dataset_configs: dict, seed: int = 42, verbose: bool = True) -> Iterator[list[str]]:
    for text in corpus_iterator(dataset_configs, seed=seed, verbose=verbose):
        tokens = pretokenize(text)
        if tokens:
            yield tokens


def count_corpus(dataset_configs: dict) -> int:
    total = 0
    for name, cfg in dataset_configs.items():
        files = sorted(cfg["path"].glob("*.parquet"))
        n_keep = max(1, round(len(files) * cfg["ratio"]))
        for pfile in files[:n_keep]:
            table = pq.read_table(str(pfile), columns=[cfg["col"]])
            total += len(table)
    return total


def _build_initial_alphabet() -> list[str]:
    chars    = [chr(i) for i in range(32, 127)]
    commands = list(TOP_LATEX_COMMANDS)
    extra    = ["\u00b0", "\u2032", "\u2033", "\u2013", "\u2014", "\u2212"]
    return chars + commands + extra


def build_tokenizer(dataset_configs: dict, vocab_size: int = VOCAB_SIZE, seed: int = 42, verbose: bool = True) -> Tokenizer:
    if verbose:
        print(f"\nBuilding BPE tokenizer  vocab_size={vocab_size}  seed={seed}")
        print(f"\nSampling corpus:")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Strip()
    tokenizer.pre_tokenizer = Split(pattern=LATEX_TOKEN_RE.pattern, behavior="isolated")

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        initial_alphabet=_build_initial_alphabet(),
        show_progress=verbose,
        end_of_word_suffix="",
    )

    if verbose:
        print(f"\nTraining BPE ...")

    tokenizer.train_from_iterator(
        pretokenized_iterator(dataset_configs, seed=seed, verbose=False),
        trainer=trainer,
        length=count_corpus(dataset_configs),
    )

    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    tokenizer.decoder = decoders.Sequence([
        decoders.BPEDecoder(suffix=""),
        decoders.Strip(content=" ", left=0, right=0),
    ])

    if verbose:
        print(f"  Final vocab size: {tokenizer.get_vocab_size():,}")

    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, out_dir: Path, verbose: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_dir / "tokenizer.json"))

    cfg = dict(TOKENIZER_CONFIG)
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    with open(out_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    stmap = {
        "pad_token": {"content": "<pad>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
        "unk_token": {"content": "<unk>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
        "bos_token": {"content": "<bos>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
        "eos_token": {"content": "<eos>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
    }
    with open(out_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(stmap, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\nTokenizer saved to: {out_dir.resolve()}")
        print(f"  vocab_size : {tokenizer.get_vocab_size():,}")
        print(f"  files      : tokenizer.json | tokenizer_config.json | special_tokens_map.json")


def load_tokenizer(out_dir: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(Path(out_dir) / "tokenizer.json"))


def load_fast_tokenizer(out_dir: str | Path):
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(str(out_dir))


def validate(tokenizer: Tokenizer, dataset_configs: dict, n_samples: int = 2000):
    print(f"\nValidating tokenizer ({n_samples} samples) ...")
    lengths, unknowns, total_tokens = [], 0, 0

    for i, text in enumerate(corpus_iterator(dataset_configs, verbose=False)):
        if i >= n_samples:
            break
        enc = tokenizer.encode(text)
        lengths.append(len(enc.ids))
        total_tokens += len(enc.ids)
        unknowns += enc.ids.count(UNK_ID)

    unk_rate = unknowns / max(total_tokens, 1) * 100
    print(f"  samples       : {len(lengths):,}")
    print(f"  token len min : {min(lengths)}")
    print(f"  token len p50 : {sorted(lengths)[len(lengths)//2]}")
    print(f"  token len p95 : {sorted(lengths)[int(len(lengths)*0.95)]}")
    print(f"  token len p99 : {sorted(lengths)[int(len(lengths)*0.99)]}")
    print(f"  token len max : {max(lengths)}")
    print(f"  avg tokens    : {total_tokens/len(lengths):.1f}")
    print(f"  <unk> rate    : {unk_rate:.4f}%  ({'OK' if unk_rate < 0.01 else 'HIGH - check corpus'})")

    assert tokenizer.token_to_id("<pad>") == PAD_ID, "PAD_ID mismatch"
    assert tokenizer.token_to_id("<unk>") == UNK_ID, "UNK_ID mismatch"
    assert tokenizer.token_to_id("<bos>") == BOS_ID, "BOS_ID mismatch"
    assert tokenizer.token_to_id("<eos>") == EOS_ID, "EOS_ID mismatch"
    print(f"  special token IDs: <pad>={PAD_ID} <unk>={UNK_ID} <bos>={BOS_ID} <eos>={EOS_ID}  OK")

    examples = [
        r"\frac{dy}{dt} = f(y, t)",
        r"\sum_{i=0}^{n} x_i^2 = \int_0^\infty e^{-x} dx",
        r"E = mc^2",
        r"\mathbb{R}^n \to \mathbb{R}",
        r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",
        r"\alpha + \beta = \gamma \delta",
        r"\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u",
    ]
    print(f"\n  Encode/decode examples:")
    for ex in examples:
        enc = tokenizer.encode(ex)
        dec = tokenizer.decode(enc.ids, skip_special_tokens=True)
        tokens_str = " | ".join(enc.tokens[:12])
        if len(enc.tokens) > 12:
            tokens_str += f" ... (+{len(enc.tokens)-12})"
        print(f"\n  input : {ex}")
        print(f"  tokens: [{tokens_str}]  (n={len(enc.ids)})")
        print(f"  decode: {dec}")


def parse_args():
    ap = argparse.ArgumentParser(description="Train BPE tokenizer for LaTeX decoder pretrain")
    ap.add_argument("--train",      action="store_true")
    ap.add_argument("--validate",   action="store_true")
    ap.add_argument("--out-dir",    type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--raw-dir",    type=str, default=None)
    ap.add_argument("--light-dir",  type=str, default=None)
    ap.add_argument("--heavy-dir",  type=str, default=None)
    ap.add_argument("--n-validate", type=int, default=2000)
    return ap.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    configs = dict(DATASET_CONFIGS)

    if args.raw_dir:   configs["raw"]["path"]        = Path(args.raw_dir)
    if args.light_dir: configs["light_text"]["path"] = Path(args.light_dir)
    if args.heavy_dir: configs["heavy_text"]["path"] = Path(args.heavy_dir)

    if args.train:
        tok = build_tokenizer(configs, vocab_size=args.vocab_size, seed=args.seed)
        save_tokenizer(tok, out_dir)
        validate(tok, configs, n_samples=args.n_validate)

    elif args.validate:
        if not (out_dir / "tokenizer.json").exists():
            print(f"ERROR: tokenizer.json not found in {out_dir}")
            print("Run with --train first.")
            return
        tok = load_tokenizer(out_dir)
        validate(tok, configs, n_samples=args.n_validate)

    else:
        print("Specify --train or --validate")
        print("Example: python tokenizer.py --train")


if __name__ == "__main__":
    main()
