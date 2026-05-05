"""
Check two data quality issues across all subsets:
  1. Extreme aspect ratio (width/height > 20)
  2. Very short LaTeX (length < 5 chars)

Saves a visual grid PNG and prints a summary for each issue.
"""

import io
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_ROOT   = Path(r"D:\dataset-ocr-builder\latex-ocr-dataset\ocr-data\train")
OUT_DIR     = Path(r"D:\img2latex\dataset_builder\analysis_output")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SUBSETS = {
    "heavy":      sorted((DATA_ROOT / "heavy").glob("*.parquet")),
    "light":      sorted((DATA_ROOT / "light").glob("*.parquet")),
    "raw":        sorted((DATA_ROOT / "raw").glob("*.parquet")),
    "screenshot": sorted((DATA_ROOT / "screenshot").glob("*.parquet")),
}

ASPECT_THRESHOLD = 20.0
LATEX_MIN_LEN    = 5
GRID_SAMPLE      = 48   # rows to show in each visual grid


def scan_subset(name: str, paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """Returns (extreme_aspect_rows, short_latex_rows)."""
    extreme, short = [], []

    for p in tqdm(paths, desc=f"  {name}", leave=False):
        df = pd.read_parquet(p, columns=["idx", "image", "latex", "source"])
        for _, row in df.iterrows():
            latex = row["latex"] if isinstance(row["latex"], str) else ""
            raw   = row["image"]

            # ── check latex length ──
            if len(latex.strip()) < LATEX_MIN_LEN:
                short.append({
                    "subset": name,
                    "source": row["source"],
                    "latex":  latex,
                    "raw":    raw,
                })

            # ── check aspect ratio ──
            try:
                img    = Image.open(io.BytesIO(raw))
                aspect = img.width / max(img.height, 1)
                if aspect > ASPECT_THRESHOLD:
                    extreme.append({
                        "subset":  name,
                        "source":  row["source"],
                        "latex":   latex,
                        "aspect":  round(aspect, 1),
                        "width":   img.width,
                        "height":  img.height,
                        "raw":     raw,
                    })
            except Exception:
                pass

    return extreme, short


def save_grid(rows: list[dict], title: str, out_path: Path,
              label_fn, n: int = GRID_SAMPLE):
    sample = random.sample(rows, min(n, len(rows)))
    # sort so worst cases appear first
    sample.sort(key=lambda r: r.get("aspect", 0), reverse=True)

    COLS = 6
    ROWS = (len(sample) + COLS - 1) // COLS
    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3.5, ROWS * 2.8))
    fig.suptitle(f"{title}\n(total: {len(rows):,}  |  showing {len(sample)})",
                 fontsize=11, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i < len(sample):
            r = sample[i]
            try:
                img = Image.open(io.BytesIO(r["raw"]))
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "decode error", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_title(label_fn(r), fontsize=5.5, loc="left")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grid saved → {out_path}")


def print_summary(label: str, all_rows: list[dict]):
    print(f"\n{'─'*60}")
    print(f"  {label}  —  {len(all_rows):,} rows total")
    print(f"{'─'*60}")
    by_subset = Counter(r["subset"] for r in all_rows)
    by_source = Counter(r["source"] for r in all_rows)
    print("  By subset:")
    for k, v in by_subset.most_common():
        print(f"    {k:15s}  {v:>8,}")
    print("  By source (top 10):")
    for k, v in by_source.most_common(10):
        print(f"    {k:40s}  {v:>8,}")


# ── main ─────────────────────────────────────────────────────────────────────
all_extreme, all_short = [], []

for name, paths in SUBSETS.items():
    if not paths:
        continue
    print(f"\nScanning {name} ({len(paths)} shards) …")
    extreme, short = scan_subset(name, paths)
    all_extreme.extend(extreme)
    all_short.extend(short)
    print(f"  extreme aspect: {len(extreme):,}  |  short latex: {len(short):,}")

print_summary("Extreme aspect ratio (> 20)", all_extreme)
if all_extreme:
    save_grid(
        all_extreme,
        f"Extreme Aspect Ratio  (> {ASPECT_THRESHOLD})",
        OUT_DIR / "extreme_aspect_samples.png",
        label_fn=lambda r: (
            f"aspect={r['aspect']}  {r['width']}×{r['height']}\n"
            f"[{r['source'][:14]}]  {r['subset']}\n"
            f"{r['latex'][:55]}…" if len(r['latex']) > 55 else
            f"aspect={r['aspect']}  {r['width']}×{r['height']}\n"
            f"[{r['source'][:14]}]  {r['subset']}\n"
            f"{r['latex']}"
        ),
    )

print_summary(f"Very short LaTeX (< {LATEX_MIN_LEN} chars)", all_short)
if all_short:
    # print all unique short latex strings (usually few)
    unique_latex = Counter(r["latex"] for r in all_short)
    print("\n  Unique short latex values:")
    for lat, cnt in unique_latex.most_common(30):
        print(f"    {repr(lat):30s}  ×{cnt}")

    save_grid(
        all_short,
        f"Very Short LaTeX  (< {LATEX_MIN_LEN} chars)",
        OUT_DIR / "short_latex_samples.png",
        label_fn=lambda r: (
            f"latex={repr(r['latex'])}\n"
            f"[{r['source'][:14]}]  {r['subset']}"
        ),
    )

print("\nDone.")
