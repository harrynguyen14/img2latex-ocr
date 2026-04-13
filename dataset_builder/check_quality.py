"""
Quick quality check for LLM-augmented LaTeX parquet files.
Usage:
    python check_quality.py path/to/file.parquet
    python check_quality.py path/to/dir/  # checks all parquet files in dir
"""
import re
import sys
from pathlib import Path

import pandas as pd

_PLAIN_SQRT_RE = re.compile(r"(?<!\\)sqrt\s*\(")
_PLAIN_FRAC_RE = re.compile(r"(?<![\\a-zA-Z])frac\s*\{")
_MARKDOWN_RE   = re.compile(r"\\\_|&amp;|&lt;|&gt;")
_WORD_RUN_RE   = re.compile(r"[a-z]{3,}(?:\s+[a-z]{3,}){3,}")
_LATEX_RE      = re.compile(r"[\\_{^}]")

CHECKS = {
    "short_lt5":      lambda s: len(s.strip()) < 5,
    "long_gt500":     lambda s: len(s.strip()) > 500,
    "plain_sqrt":     lambda s: bool(_PLAIN_SQRT_RE.search(s)),
    "plain_frac":     lambda s: bool(_PLAIN_FRAC_RE.search(s)),
    "markdown":       lambda s: bool(_MARKDOWN_RE.search(s)),
    "english_prose":  lambda s: bool(_WORD_RUN_RE.search(s)),
    "no_latex":       lambda s: not bool(_LATEX_RE.search(s)),
}


def check_file(path: Path):
    df = pd.read_parquet(path)
    print(f"\n{'='*60}")
    print(f"File : {path.name}")
    print(f"Rows : {len(df):,}  |  Cols: {list(df.columns)}")

    # transform distribution
    if "transform" in df.columns:
        print("\nTransform counts:")
        for t, c in df["transform"].value_counts().items():
            print(f"  {t:<22} {c:>5}")

    # run checks
    latex_col = df["latex"].fillna("").astype(str)
    flags = {}
    for name, fn in CHECKS.items():
        mask = latex_col.apply(fn)
        flags[name] = df[mask]

    any_bad = pd.concat(flags.values()).drop_duplicates()
    n_bad   = len(any_bad)
    n_total = len(df)

    print(f"\nQuality checks ({n_total} rows):")
    print(f"  {'Check':<22} {'Count':>6}  {'%':>6}  Examples (idx)")
    print(f"  {'-'*55}")
    for name, bad_df in flags.items():
        n = len(bad_df)
        pct = n / n_total * 100
        idxs = ", ".join(str(i) for i in bad_df["idx"].head(3).tolist()) if n else ""
        print(f"  {name:<22} {n:>6}  {pct:>5.1f}%  {idxs}")

    print(f"\n  Bad rows (any check) : {n_bad:>5}  ({n_bad/n_total*100:.1f}%)")
    print(f"  Clean rows           : {n_total-n_bad:>5}  ({(n_total-n_bad)/n_total*100:.1f}%)")

    if n_bad > 0:
        print(f"\nBad row samples (up to 10):")
        pd.set_option("display.max_colwidth", 120)
        for _, row in any_bad.head(10).iterrows():
            reasons = [k for k, v in flags.items() if row["idx"] in v["idx"].values]
            print(f"  [{row['idx']}] [{row.get('transform','')}] {reasons}")
            print(f"    {row['latex'][:120]}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_quality.py <file.parquet | directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        files = sorted(target.glob("*.parquet"))
        if not files:
            print(f"No parquet files found in {target}")
            sys.exit(1)
        for f in files:
            check_file(f)
    elif target.is_file():
        check_file(target)
    else:
        print(f"Path not found: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
