"""
Quality checker for LLM-augmented LaTeX parquet files.

Usage:
    # Single file
    python check_quality.py path/to/file.parquet

    # Directory with parquet files (flat)
    python check_quality.py path/to/dir/

    # Output directory with worker subdirs (worker0/, worker1/, ...)
    python check_quality.py path/to/output/ --workers

    # Limit number of files loaded per worker (for quick sampling)
    python check_quality.py path/to/output/ --workers --max-files 5

    # Show bad row samples
    python check_quality.py path/to/output/ --workers --show-samples
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
_PLAIN_SQRT_RE  = re.compile(r"(?<!\\)sqrt\s*\(")
_PLAIN_FRAC_RE  = re.compile(r"(?<![\\a-zA-Z])frac\s*\{")
_MARKDOWN_RE    = re.compile(r"\\\_|&amp;|&lt;|&gt;")
_WORD_RUN_RE    = re.compile(r"[a-z]{3,}(?:\s+[a-z]{3,}){3,}")
_LATEX_RE       = re.compile(r"[\\_{^}]")
_UNMATCHED_BRACE_RE = re.compile(r"(?<![\\])[{}]")
_DOUBLE_BACKSLASH_RE = re.compile(r"\\\\\\\\")  # 4 backslashes = escaped \\
_NULL_BYTES_RE  = re.compile(r"\x00")
_CONTROL_RE     = re.compile(r"[\x01-\x08\x0b-\x1f\x7f]")

CHECKS = {
    "empty":          lambda s: len(s.strip()) == 0,
    "short_lt5":      lambda s: 0 < len(s.strip()) < 5,
    "long_gt500":     lambda s: len(s.strip()) > 500,
    "plain_sqrt":     lambda s: bool(_PLAIN_SQRT_RE.search(s)),
    "plain_frac":     lambda s: bool(_PLAIN_FRAC_RE.search(s)),
    "markdown":       lambda s: bool(_MARKDOWN_RE.search(s)),
    "english_prose":  lambda s: bool(_WORD_RUN_RE.search(s)),
    "no_latex":       lambda s: not bool(_LATEX_RE.search(s)),
    "null_bytes":     lambda s: bool(_NULL_BYTES_RE.search(s)),
    "control_chars":  lambda s: bool(_CONTROL_RE.search(s)),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(frac: float, width: int = 20) -> str:
    filled = round(frac * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:5.1f}%" if total else "  N/A "


def _section(title: str):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_df(df: pd.DataFrame, label: str = "", show_samples: bool = False) -> dict:
    """Run all quality checks on a DataFrame and return summary dict."""
    n_total = len(df)
    latex_col = df["latex"].fillna("").astype(str)

    # -- basic checks --------------------------------------------------------
    flags: dict[str, pd.Series] = {}
    for name, fn in CHECKS.items():
        flags[name] = latex_col.apply(fn)

    bad_mask = pd.concat(flags.values(), axis=1).any(axis=1)
    n_bad = int(bad_mask.sum())

    # -- duplicates ----------------------------------------------------------
    dup_mask = latex_col.duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    n_exact_dup = int(latex_col.duplicated(keep="first").sum())

    # -- length stats --------------------------------------------------------
    lengths = latex_col.str.len()

    # -- per-transform -------------------------------------------------------
    transform_stats: dict = {}
    if "transform" in df.columns:
        for tfm, grp in df.groupby("transform", sort=False):
            g_latex = grp["latex"].fillna("").astype(str)
            g_bad = pd.concat(
                [g_latex.apply(fn) for fn in CHECKS.values()], axis=1
            ).any(axis=1).sum()
            transform_stats[tfm] = {
                "count": len(grp),
                "bad": int(g_bad),
                "med_len": int(g_latex.str.len().median()),
            }

    # -- per-source ----------------------------------------------------------
    source_stats: dict = {}
    if "source" in df.columns:
        for src, grp in df.groupby("source", sort=False):
            g_latex = grp["latex"].fillna("").astype(str)
            g_bad = pd.concat(
                [g_latex.apply(fn) for fn in CHECKS.values()], axis=1
            ).any(axis=1).sum()
            source_stats[src] = {
                "count": len(grp),
                "bad": int(g_bad),
            }

    summary = {
        "label": label,
        "n_total": n_total,
        "n_bad": n_bad,
        "n_dup": n_dup,
        "n_exact_dup": n_exact_dup,
        "flags": {k: int(v.sum()) for k, v in flags.items()},
        "bad_mask": bad_mask,
        "dup_mask": dup_mask,
        "lengths": lengths,
        "transform_stats": transform_stats,
        "source_stats": source_stats,
        "df": df if show_samples else None,
        "flags_series": flags if show_samples else None,
    }
    return summary


def print_summary(s: dict, show_samples: bool = False):
    n_total   = s["n_total"]
    n_bad     = s["n_bad"]
    n_dup     = s["n_dup"]
    n_exact   = s["n_exact_dup"]
    lengths   = s["lengths"]

    _section(f"DATASET: {s['label']}  ({n_total:,} rows)")

    # -- overview ------------------------------------------------------------
    print(f"\n  Rows         : {n_total:,}")
    print(f"  Clean rows   : {n_total - n_bad:,}  ({_pct(n_total - n_bad, n_total)})")
    print(f"  Bad rows     : {n_bad:,}  ({_pct(n_bad, n_total)})")
    print(f"  Duplicates   : {n_dup:,} rows involved  |  {n_exact:,} redundant copies")

    # -- length distribution -------------------------------------------------
    print(f"\n  LaTeX length distribution:")
    print(f"    min={lengths.min():.0f}  p5={lengths.quantile(.05):.0f}  "
          f"p25={lengths.quantile(.25):.0f}  median={lengths.median():.0f}  "
          f"p75={lengths.quantile(.75):.0f}  p95={lengths.quantile(.95):.0f}  "
          f"max={lengths.max():.0f}")

    bins = [0, 5, 20, 50, 100, 200, 500, 10_000]
    labels = ["<5", "5-20", "20-50", "50-100", "100-200", "200-500", ">500"]
    hist = pd.cut(lengths, bins=bins, labels=labels, right=False).value_counts().reindex(labels)
    print(f"    {'Range':<12} {'Count':>7}  {'%':>6}  Bar")
    for lbl, cnt in hist.items():
        cnt = int(cnt) if not np.isnan(cnt) else 0
        print(f"    {lbl:<12} {cnt:>7,}  {_pct(cnt, n_total)}  {_bar(cnt/n_total)}")

    # -- quality checks ------------------------------------------------------
    print(f"\n  Quality checks:")
    print(f"    {'Check':<22} {'Count':>7}  {'%':>6}  Bar")
    print(f"    {'-'*55}")
    for name, cnt in s["flags"].items():
        print(f"    {name:<22} {cnt:>7,}  {_pct(cnt, n_total)}  {_bar(cnt/n_total)}")

    # -- per transform -------------------------------------------------------
    if s["transform_stats"]:
        print(f"\n  Per-transform breakdown:")
        print(f"    {'Transform':<24} {'Count':>7}  {'Bad':>6}  {'Bad%':>6}  {'MedLen':>7}")
        print(f"    {'-'*56}")
        for tfm, st in sorted(s["transform_stats"].items(), key=lambda x: -x[1]["count"]):
            print(f"    {tfm:<24} {st['count']:>7,}  {st['bad']:>6,}  "
                  f"{_pct(st['bad'], st['count'])}  {st['med_len']:>7}")

    # -- per source ----------------------------------------------------------
    if s["source_stats"]:
        print(f"\n  Per-source breakdown:")
        print(f"    {'Source':<32} {'Count':>7}  {'Bad':>6}  {'Bad%':>6}")
        print(f"    {'-'*56}")
        for src, st in sorted(s["source_stats"].items(), key=lambda x: -x[1]["count"]):
            print(f"    {src:<32} {st['count']:>7,}  {st['bad']:>6,}  "
                  f"{_pct(st['bad'], st['count'])}")

    # -- samples -------------------------------------------------------------
    if show_samples and s["df"] is not None and n_bad > 0:
        df = s["df"]
        flags_series = s["flags_series"]
        bad_mask = s["bad_mask"]
        bad_df = df[bad_mask].copy()
        latex_col = df["latex"].fillna("").astype(str)

        print(f"\n  Bad row samples (up to 15):")
        pd.set_option("display.max_colwidth", 120)
        for _, row in bad_df.head(15).iterrows():
            row_latex = str(row.get("latex", ""))
            reasons = [
                k for k, v in flags_series.items()
                if v.iloc[df.index.get_loc(row.name)]
            ]
            src  = row.get("source", "")
            tfm  = row.get("transform", "")
            idx  = row.get("idx", row.name)
            print(f"\n    [{idx}] src={src} tfm={tfm}")
            print(f"    flags: {reasons}")
            print(f"    latex: {row_latex[:160]}")


# ---------------------------------------------------------------------------
# File / directory loaders
# ---------------------------------------------------------------------------

def load_flat_dir(path: Path, max_files: int | None = None) -> pd.DataFrame:
    files = sorted(path.glob("*.parquet"))
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No parquet files in {path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_worker_dir(path: Path, max_files: int | None = None) -> dict[str, pd.DataFrame]:
    """Load each worker subdirectory as a separate DataFrame."""
    worker_dirs = sorted(d for d in path.iterdir() if d.is_dir() and d.name.startswith("worker"))
    if not worker_dirs:
        raise FileNotFoundError(f"No worker* subdirs found in {path}")

    workers: dict[str, pd.DataFrame] = {}
    for wd in worker_dirs:
        files = sorted(wd.glob("*.parquet"))
        if max_files:
            files = files[:max_files]
        if not files:
            continue
        print(f"  Loading {wd.name}: {len(files)} files ...", flush=True)
        workers[wd.name] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return workers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quality checker for LaTeX parquet datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("target", help="Parquet file, flat dir, or output dir with workers")
    parser.add_argument(
        "--workers", action="store_true",
        help="Target is an output dir with worker0/, worker1/, ... subdirs"
    )
    parser.add_argument(
        "--max-files", type=int, default=None, metavar="N",
        help="Load only the first N parquet files per worker (for quick sampling)"
    )
    parser.add_argument(
        "--show-samples", action="store_true",
        help="Print example bad rows for each dataset"
    )
    parser.add_argument(
        "--combined", action="store_true",
        help="Also show combined analysis across all workers"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, metavar="FILE",
        help="Save report to a text file (e.g. report.txt). Output is written to both terminal and file."
    )
    args = parser.parse_args()

    target = Path(args.target)

    if not target.exists():
        print(f"ERROR: path not found: {target}")
        sys.exit(1)

    # -- setup tee output (terminal + file) ----------------------------------
    out_file = None
    if args.output:
        out_path = Path(args.output)
        out_file = open(out_path, "w", encoding="utf-8")

        class _Tee:
            def write(self, msg):
                sys.__stdout__.write(msg)
                out_file.write(msg)
            def flush(self):
                sys.__stdout__.flush()
                out_file.flush()

        sys.stdout = _Tee()
        print(f"(Report will also be saved to: {out_path.resolve()})")

    try:
        all_dfs: list[pd.DataFrame] = []

        if target.is_file():
            # single parquet file
            df = pd.read_parquet(target)
            s = analyze_df(df, label=target.name, show_samples=args.show_samples)
            print_summary(s, show_samples=args.show_samples)
            all_dfs.append(df)

        elif args.workers:
            print(f"\nLoading worker subdirs from: {target}")
            workers = load_worker_dir(target, max_files=args.max_files)
            for name, df in workers.items():
                s = analyze_df(df, label=name, show_samples=args.show_samples)
                print_summary(s, show_samples=args.show_samples)
                all_dfs.append(df)

            if args.combined and len(all_dfs) > 1:
                combined = pd.concat(all_dfs, ignore_index=True)
                s_all = analyze_df(combined, label="ALL WORKERS COMBINED", show_samples=False)
                print_summary(s_all, show_samples=False)

                # cross-worker duplicate check
                n_cross_dup = int(combined["latex"].fillna("").duplicated(keep="first").sum())
                _section("CROSS-WORKER DUPLICATE ANALYSIS")
                print(f"\n  Total rows across all workers : {len(combined):,}")
                print(f"  Cross-worker duplicate copies : {n_cross_dup:,}  ({_pct(n_cross_dup, len(combined))})")

        else:
            # flat directory of parquet files
            print(f"\nLoading parquet files from: {target}")
            df = load_flat_dir(target, max_files=args.max_files)
            s = analyze_df(df, label=str(target), show_samples=args.show_samples)
            print_summary(s, show_samples=args.show_samples)
            all_dfs.append(df)

        print(f"\n{'='*62}")
        print("  Done.")
        print(f"{'='*62}\n")

    finally:
        if out_file:
            sys.stdout = sys.__stdout__
            out_file.close()
            print(f"Report saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
