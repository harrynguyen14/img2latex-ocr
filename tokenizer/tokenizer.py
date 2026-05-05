"""
LaTeX Tokenizer v2 — Frozen vocab + BPE chỉ cho subword zones.

Kiến trúc:
  1. Lexer (lexer.py) chặt LaTeX thành token stream với type tags.
  2. Frozen tokens (COMMAND, SYMBOL, DIGIT, LETTER, ...) → ID cố định.
  3. BPE chỉ train trên các "BPE zones" (chuỗi LETTER/DIGIT liên tiếp).
     → merge "d"+"x"→"dx", "i"+"n"→"in", nhung KHONG cross qua cmd hay {}.
  4. Encode/decode hoàn toàn deterministic.

Usage:
  python tokenizer_v2.py --train
  python tokenizer_v2.py --validate
  python tokenizer_v2.py --demo
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Iterator

import sys
_here = str(Path(__file__).parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from lexer import tokenize, tokenize_to_strings, split_bpe_zones, TT, FROZEN_TYPES, BPE_TYPES
from tokenizer.vocab import (
    ALL_FROZEN_TOKENS, N_FROZEN, VOCAB_SIZE,
    PAD_ID, UNK_ID, BOS_ID, EOS_ID,
    SPECIAL_TOKENS, TOKENIZER_CONFIG, DATASET_CONFIGS,
)

DEFAULT_OUT_DIR = Path(__file__).parent / "saved"


# ══════════════════════════════════════════════════════════════════════════════
# BPE trainer (minimal, chỉ train trên BPE zones)
# ══════════════════════════════════════════════════════════════════════════════

def _get_pair_stats(vocab: dict[tuple[str, ...], int]) -> Counter:
    pairs: Counter = Counter()
    for word, freq in vocab.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += freq
    return pairs


def _merge_vocab(vocab: dict[tuple[str, ...], int],
                 pair: tuple[str, str]) -> dict[tuple[str, ...], int]:
    merged = pair[0] + pair[1]
    new_vocab: dict[tuple[str, ...], int] = {}
    for word, freq in vocab.items():
        new_word: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab


def train_bpe(
    zone_counter: Counter,          # Counter({('d','x'): 1234, ...})
    n_merges: int,
    min_frequency: int = 2,
    verbose: bool = True,
) -> list[tuple[str, str]]:
    """Train BPE trên zone_counter, trả về list merge rules."""
    vocab = {word: freq for word, freq in zone_counter.items() if freq >= min_frequency}
    merges: list[tuple[str, str]] = []

    for step in range(n_merges):
        pairs = _get_pair_stats(vocab)
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        if pairs[best] < min_frequency:
            break
        vocab = _merge_vocab(vocab, best)
        merges.append(best)
        if verbose and (step + 1) % 500 == 0:
            print(f"  BPE merge {step+1}/{n_merges}: {best[0]!r}+{best[1]!r} "
                  f"(freq={pairs[best]})", flush=True)

    if verbose:
        print(f"  Total merges learned: {len(merges)}")
    return merges


# ══════════════════════════════════════════════════════════════════════════════
# LaTeXTokenizerV2
# ══════════════════════════════════════════════════════════════════════════════

class LaTeXTokenizerV2:
    """
    Tokenizer hoàn chỉnh với frozen vocab + BPE subword.

    Attributes:
        token2id  : dict[str, int]
        id2token  : dict[int, str]
        merges    : list[tuple[str,str]]  — BPE merge rules theo thứ tự
    """

    def __init__(
        self,
        token2id: dict[str, int],
        id2token: dict[int, str],
        merges: list[tuple[str, str]],
    ):
        self.token2id = token2id
        self.id2token = id2token
        self.merges = merges
        # Build merge rank lookup: (a,b) → rank (thấp = ưu tiên cao)
        self._merge_rank: dict[tuple[str, str], int] = {
            pair: i for i, pair in enumerate(merges)
        }

    # ── Encode ────────────────────────────────────────────────────────────────

    def _apply_bpe(self, chars: list[str]) -> list[str]:
        """Apply BPE merges lên một list ký tự."""
        word = list(chars)
        while len(word) > 1:
            # Tìm cặp có rank thấp nhất (ưu tiên nhất)
            best_rank = len(self.merges)  # sentinel
            best_idx = -1
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self._merge_rank.get(pair, len(self.merges))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            merged = word[best_idx] + word[best_idx + 1]
            word = word[:best_idx] + [merged] + word[best_idx + 2:]
        return word

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: int | None = None,
    ) -> list[int]:
        """LaTeX string → list[int] token IDs."""
        lex_tokens = tokenize(text)
        ids: list[int] = []

        if add_special_tokens:
            ids.append(BOS_ID)

        bpe_buf: list[str] = []

        def flush_bpe():
            if not bpe_buf:
                return
            subwords = self._apply_bpe(bpe_buf)
            for sw in subwords:
                ids.append(self.token2id.get(sw, UNK_ID))
            bpe_buf.clear()

        for tok in lex_tokens:
            if tok.is_frozen:
                flush_bpe()
                if tok.ttype == TT.NUMBER and tok.text not in self.token2id:
                    # Fallback: split number thành từng digit/dot
                    for ch in tok.text:
                        ids.append(self.token2id.get(ch, UNK_ID))
                else:
                    ids.append(self.token2id.get(tok.text, UNK_ID))
            else:
                bpe_buf.append(tok.text)

        flush_bpe()

        if add_special_tokens:
            ids.append(EOS_ID)

        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        skip = {PAD_ID, EOS_ID}
        if skip_special_tokens:
            skip.add(BOS_ID)
        parts = [self.id2token[i] for i in ids
                 if i in self.id2token and i not in skip]
        return "".join(parts)

    def token_to_id(self, token: str) -> int:
        return self.token2id.get(token, UNK_ID)

    def id_to_token(self, id_: int) -> str:
        return self.id2token.get(id_, "<unk>")

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, out_dir: str | Path, verbose: bool = True) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # tokenizer_v2.json — vocab + merges
        data = {
            "token2id": self.token2id,
            "id2token": {int(k): v for k, v in self.id2token.items()},
            "merges":   self.merges,
            "config":   TOKENIZER_CONFIG,
        }
        with open(out_dir / "tokenizer_v2.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # tokenizer_config.json — HuggingFace compatible
        cfg = dict(TOKENIZER_CONFIG)
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg["vocab_size"] = self.vocab_size
        with open(out_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        # special_tokens_map.json
        stmap = {
            "pad_token": {"content": "<pad>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
            "unk_token": {"content": "<unk>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
            "bos_token": {"content": "<bos>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
            "eos_token": {"content": "<eos>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
        }
        with open(out_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
            json.dump(stmap, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"Tokenizer saved → {out_dir}")
            print(f"  tokenizer_v2.json | tokenizer_config.json | special_tokens_map.json")
            print(f"  vocab_size : {self.vocab_size}")
            print(f"  n_frozen   : {N_FROZEN}")
            print(f"  bpe_merges : {len(self.merges)}")

    @classmethod
    def load(cls, out_dir: str | Path) -> "LaTeXTokenizerV2":
        out_dir = Path(out_dir)
        path = out_dir / "tokenizer_v2.json" if (out_dir / "tokenizer_v2.json").exists() else out_dir / "tokenizer.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            token2id=data["token2id"],
            id2token={int(k): v for k, v in data["id2token"].items()},
            merges=   [tuple(m) for m in data["merges"]],
        )


# ══════════════════════════════════════════════════════════════════════════════
# Build pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _corpus_iterator(configs: dict, seed: int = 42, verbose: bool = True) -> Iterator[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pip install pyarrow")

    rng = random.Random(seed)
    for name, cfg in configs.items():
        path = Path(cfg["path"])
        if not path.exists():
            if verbose:
                print(f"  [SKIP] {name}: path not found ({path})")
            continue
        files = sorted(path.glob("*.parquet"))
        if not files:
            if verbose:
                print(f"  [SKIP] {name}: no parquet files in {path}")
            continue
        n_keep = max(1, round(len(files) * cfg.get("ratio", 1.0)))
        sampled = rng.sample(files, n_keep)
        if verbose:
            print(f"  {name}: {n_keep}/{len(files)} files", flush=True)
        col = cfg["col"]
        for pfile in sampled:
            try:
                table = pq.read_table(str(pfile), columns=[col])
                for val in table[col].to_pylist():
                    if val and isinstance(val, str) and val.strip():
                        yield val.strip()
            except Exception as e:
                if verbose:
                    print(f"  [WARN] {pfile.name}: {e}")


def _collect_corpus_stats(
    configs: dict,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[Counter, Counter]:
    """
    Scan corpus một lần, trả về:
      bpe_zone_counter : Counter({('d','x'): freq, ...})  — BPE training
      number_counter   : Counter({'3.14': freq, ...})      — frozen numbers
    """
    bpe_zones: Counter = Counter()
    numbers: Counter = Counter()
    n_docs = 0

    for text in _corpus_iterator(configs, seed=seed, verbose=verbose):
        lex_tokens = tokenize(text)
        # BPE zones (chỉ LETTER/GREEK/OTHER)
        for zone in split_bpe_zones(lex_tokens):
            bpe_zones[tuple(zone)] += 1
        # NUMBER tokens
        for tok in lex_tokens:
            if tok.ttype == TT.NUMBER:
                numbers[tok.text] += 1
        n_docs += 1
        if verbose and n_docs % 100_000 == 0:
            print(f"  processed {n_docs:,} docs | "
                  f"zones={len(bpe_zones):,} numbers={len(numbers):,}", flush=True)

    if verbose:
        print(f"  Total docs: {n_docs:,} | "
              f"BPE zones: {len(bpe_zones):,} | Unique numbers: {len(numbers):,}")
    return bpe_zones, numbers


def build_tokenizer(
    configs: dict | None = None,
    n_merges: int | None = None,
    min_frequency: int = 2,
    seed: int = 42,
    verbose: bool = True,
) -> LaTeXTokenizerV2:
    if configs is None:
        configs = DATASET_CONFIGS
    if n_merges is None:
        n_merges = VOCAB_SIZE - N_FROZEN  # dùng hết slot còn lại

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Building LaTeXTokenizerV2")
        print(f"  vocab_size  = {VOCAB_SIZE}")
        print(f"  n_frozen    = {N_FROZEN}")
        print(f"  n_bpe_slots = {n_merges}")
        print(f"{'='*60}")

    # ── 1. Build frozen token2id ──────────────────────────────────────────────
    token2id: dict[str, int] = {}
    id2token: dict[int, str] = {}
    for idx, tok in enumerate(ALL_FROZEN_TOKENS):
        if tok in token2id:
            raise ValueError(f"Duplicate frozen token: {tok!r}")
        token2id[tok] = idx
        id2token[idx] = tok

    if verbose:
        print(f"\nFrozen vocab built: {len(token2id)} tokens")

    # ── 2. Collect corpus stats (một lần duy nhất) ───────────────────────────
    if verbose:
        print(f"\nCollecting corpus statistics:")
    bpe_zone_counter, number_counter = _collect_corpus_stats(
        configs, seed=seed, verbose=verbose
    )

    # ── 3. Add NUMBER tokens vào frozen vocab (sorted by freq) ───────────────
    # Chỉ thêm số thực sự xuất hiện trong dataset, min_freq=2
    # Giới hạn slots để không chiếm quá nhiều BPE budget
    MAX_NUMBER_SLOTS = min(512, n_merges // 4)
    next_id = len(token2id)
    added_numbers = 0
    for num_str, freq in number_counter.most_common():
        if freq < min_frequency:
            break
        if added_numbers >= MAX_NUMBER_SLOTS:
            break
        if num_str not in token2id:
            token2id[num_str] = next_id
            id2token[next_id] = num_str
            next_id += 1
            added_numbers += 1

    if verbose:
        print(f"  Number tokens added: {added_numbers} (cap={MAX_NUMBER_SLOTS})")

    # ── 4. Train BPE ──────────────────────────────────────────────────────────
    remaining_slots = VOCAB_SIZE - len(token2id)
    actual_merges = min(n_merges, remaining_slots)
    if verbose:
        print(f"\nTraining BPE ({actual_merges} merges, min_freq={min_frequency}):")
    merges = train_bpe(bpe_zone_counter, n_merges=actual_merges,
                       min_frequency=min_frequency, verbose=verbose)

    # ── 5. Add BPE subword tokens vào vocab ───────────────────────────────────
    for a, b in merges:
        merged = a + b
        if merged not in token2id:
            token2id[merged] = next_id
            id2token[next_id] = merged
            next_id += 1

    if verbose:
        n_total = len(token2id)
        print(f"\nVocab summary:")
        print(f"  frozen (commands+symbols+chars) : {N_FROZEN}")
        print(f"  frozen numbers (corpus)         : {added_numbers}")
        print(f"  BPE learned subwords            : {len(merges)}")
        print(f"  total                           : {n_total}")

    return LaTeXTokenizerV2(token2id=token2id, id2token=id2token, merges=merges)


# ══════════════════════════════════════════════════════════════════════════════
# Validate
# ══════════════════════════════════════════════════════════════════════════════

def validate(tok: LaTeXTokenizerV2, configs: dict, n_samples: int = 2000,
             verbose: bool = True) -> None:
    print(f"\nValidating ({n_samples} samples)...")
    lengths, unknowns, total_tokens = [], 0, 0

    for i, text in enumerate(_corpus_iterator(configs, verbose=False)):
        if i >= n_samples:
            break
        ids = tok.encode(text)
        lengths.append(len(ids))
        total_tokens += len(ids)
        unknowns += ids.count(UNK_ID)

    if not lengths:
        print("  No samples found — check dataset paths.")
        return

    lengths_s = sorted(lengths)
    n = len(lengths_s)
    unk_rate = unknowns / max(total_tokens, 1) * 100

    print(f"  samples       : {n:,}")
    print(f"  token len min : {lengths_s[0]}")
    print(f"  token len p50 : {lengths_s[n//2]}")
    print(f"  token len p95 : {lengths_s[int(n*0.95)]}")
    print(f"  token len p99 : {lengths_s[int(n*0.99)]}")
    print(f"  token len max : {lengths_s[-1]}")
    print(f"  avg tokens    : {total_tokens/n:.1f}")
    print(f"  <unk> rate    : {unk_rate:.4f}%  "
          f"({'OK' if unk_rate < 0.01 else 'HIGH'})")

    # Kiểm tra frozen IDs không đổi
    assert tok.token_to_id("<pad>") == PAD_ID
    assert tok.token_to_id("<unk>") == UNK_ID
    assert tok.token_to_id("<bos>") == BOS_ID
    assert tok.token_to_id("<eos>") == EOS_ID
    print(f"  special IDs   : pad={PAD_ID} unk={UNK_ID} bos={BOS_ID} eos={EOS_ID}  OK")

    examples = [
        r"\frac{dy}{dt} = f(y, t)",
        r"\int_0^\infty e^{-x^2}\,dx",
        r"E = mc^2",
        r"\alpha + \beta = \gamma",
        r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}",
        r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",
        r"\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u",
        r"\lim_{x \to 0} \frac{\sin x}{x} = 1",
    ]
    print(f"\n  Encode / decode examples:")
    all_ok = True
    for ex in examples:
        ids = tok.encode(ex)
        decoded = tok.decode(ids)
        ok = decoded.strip() == ex.strip()
        if not ok:
            all_ok = False
        status = "OK" if ok else "MISMATCH"
        toks_preview = [tok.id_to_token(i) for i in ids[:12]]
        if len(ids) > 12:
            toks_preview.append(f"...+{len(ids)-12}")
        print(f"\n  [{status}] {ex}")
        print(f"   tokens : {toks_preview}")
        print(f"   decode : {decoded}")

    print(f"\n  Roundtrip: {'ALL OK' if all_ok else 'SOME MISMATCHES — check lexer'}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(description="LaTeX Tokenizer v2")
    ap.add_argument("--train",       action="store_true")
    ap.add_argument("--validate",    action="store_true")
    ap.add_argument("--demo",        action="store_true")
    ap.add_argument("--out-dir",     type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--vocab-size",  type=int, default=VOCAB_SIZE)
    ap.add_argument("--min-freq",    type=int, default=2)
    ap.add_argument("--n-validate",  type=int, default=2000)
    ap.add_argument("--seed",        type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    if args.train:
        tok = build_tokenizer(
            configs=DATASET_CONFIGS,
            min_frequency=args.min_freq,
            seed=args.seed,
            verbose=True,
        )
        tok.save(out_dir)
        validate(tok, DATASET_CONFIGS, n_samples=args.n_validate)

    elif args.validate:
        if not (out_dir / "tokenizer_v2.json").exists():
            print(f"ERROR: tokenizer_v2.json not found in {out_dir}")
            print("Run --train first.")
            return
        tok = LaTeXTokenizerV2.load(out_dir)
        print(f"Loaded tokenizer: {tok.vocab_size} tokens, {len(tok.merges)} merges")
        validate(tok, DATASET_CONFIGS, n_samples=args.n_validate)

    elif args.demo:
        if not (out_dir / "tokenizer_v2.json").exists():
            print(f"ERROR: tokenizer_v2.json not found in {out_dir}")
            return
        tok = LaTeXTokenizerV2.load(out_dir)
        from lexer import print_tokens
        examples = [
            r"\frac{dy}{dt}",
            r"\int_0^\infty e^{-x^2}\,dx",
            r"\alpha + \beta = \gamma",
            r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}",
            r"E = mc^2",
        ]
        for ex in examples:
            print("\n" + "─"*50)
            print_tokens(ex)
            ids = tok.encode(ex)
            tokens = [tok.id_to_token(i) for i in ids]
            dec = tok.decode(ids)
            print(f"IDs    : {ids}")
            print(f"Tokens : {tokens}")
            print(f"Decode : {dec}")
    else:
        print("Specify --train, --validate, or --demo")
        print("  python tokenizer_v2.py --train")
        print("  python tokenizer_v2.py --validate")
        print("  python tokenizer_v2.py --demo")


if __name__ == "__main__":
    main()
