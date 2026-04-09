"""Shared utilities for analyze_text_*.py scripts."""

import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

OUT_DIR = Path("D:/dataset-ocr-builder/latex-ocr-dataset")
SEP = "=" * 65


def stat_dict(arr) -> dict:
    arr = np.array(arr, dtype=float)
    if len(arr) == 0:
        return {}
    return {
        "min":    round(float(arr.min()), 2),
        "max":    round(float(arr.max()), 2),
        "mean":   round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "p90":    round(float(np.percentile(arr, 90)), 2),
        "p99":    round(float(np.percentile(arr, 99)), 2),
    }


def load_split(split_dir: Path, max_shards: int = 0) -> tuple[list[str], list[str]]:
    files = sorted(split_dir.glob("*.parquet"))
    if max_shards:
        files = files[:max_shards]
    if not files:
        raise FileNotFoundError(f"No parquet files in {split_dir}")
    latexes, sources = [], []
    for pfile in files:
        table = pq.read_table(str(pfile), columns=["latex", "source"])
        latexes.extend(table["latex"].to_pylist())
        sources.extend(table["source"].to_pylist())
        del table
    return latexes, sources


def analyze_latex(latexes: list[str], label: str) -> dict:
    char_lens  = [len(t) for t in latexes]
    token_lens = [len(t.split()) for t in latexes]
    empty      = sum(1 for t in latexes if not t or not t.strip())

    print(f"\n  [{label}]  n={len(latexes):,}  empty={empty}")
    print(f"    char_len  : min={min(char_lens)}  max={max(char_lens)}"
          f"  mean={np.mean(char_lens):.1f}"
          f"  p90={np.percentile(char_lens, 90):.0f}"
          f"  p99={np.percentile(char_lens, 99):.0f}")
    print(f"    token_cnt : min={min(token_lens)}  max={max(token_lens)}"
          f"  mean={np.mean(token_lens):.1f}"
          f"  p90={np.percentile(token_lens, 90):.0f}")

    return {
        "n":           len(latexes),
        "empty":       empty,
        "char_len":    stat_dict(char_lens),
        "token_count": stat_dict(token_lens),
    }


def compare_pairs(raw: list[str], aug: list[str], label: str, n_print: int = 10) -> dict:
    assert len(raw) == len(aug), f"Length mismatch: {len(raw)} vs {len(aug)}"

    changed = sum(1 for r, a in zip(raw, aug) if r != a)
    pct     = changed / len(raw) * 100
    deltas  = [len(a) - len(r) for r, a in zip(raw, aug)]
    grew    = sum(1 for d in deltas if d > 0)
    shrunk  = sum(1 for d in deltas if d < 0)

    print(f"\n  [{label} vs raw]")
    print(f"    changed    : {changed:,}/{len(raw):,} ({pct:.1f}%)")
    print(f"    char delta : mean={np.mean(deltas):.2f}  grew={grew:,}  shrunk={shrunk:,}")

    changed_idx = [i for i, (r, a) in enumerate(zip(raw, aug)) if r != a]
    sample_idx  = random.sample(changed_idx, min(n_print, len(changed_idx)))

    print(f"\n    Sample changed pairs ({len(sample_idx)}):")
    samples = []
    for i in sample_idx:
        print(f"      RAW: {raw[i][:80]}")
        print(f"      AUG: {aug[i][:80]}")
        print()
        samples.append({"raw": raw[i], "aug": aug[i]})

    return {
        "changed":     changed,
        "pct_changed": round(pct, 2),
        "char_delta":  stat_dict(deltas),
        "grew":        grew,
        "shrunk":      shrunk,
        "samples":     samples,
    }


def per_source_change_rate(raw: list[str], aug: list[str], sources: list[str]) -> dict:
    by_src: dict[str, list] = {}
    for r, a, src in zip(raw, aug, sources):
        by_src.setdefault(src, []).append(r != a)

    result = {}
    print(f"\n    Change rate per source:")
    for src, flags in sorted(by_src.items()):
        rate = sum(flags) / len(flags) * 100
        print(f"      {src:<35} {sum(flags):>8,}/{len(flags):>8,}  ({rate:.1f}%)")
        result[src] = {"changed": sum(flags), "total": len(flags), "pct": round(rate, 2)}
    return result


def save_report(report: dict, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  report saved -> {out_file}")
