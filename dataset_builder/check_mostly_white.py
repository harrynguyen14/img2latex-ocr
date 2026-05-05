"""
Inspect "mostly white" images in the raw subset.
Samples ~60 rows where mean_px > 245, renders them to a grid PNG,
and prints the associated LaTeX + source for manual review.
"""

import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_ROOT   = Path(r"D:\dataset-ocr-builder\latex-ocr-dataset\ocr-data\train\raw")
OUT_DIR     = Path(r"D:\img2latex\dataset_builder\analysis_output")
OUT_PNG     = OUT_DIR / "mostly_white_raw_samples.png"
SAMPLE_N    = 60   # how many mostly-white rows to visualise
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── collect mostly-white rows across all shards ───────────────────────────────
shards = sorted(DATA_ROOT.glob("*.parquet"))
print(f"Scanning {len(shards)} raw shards …")

collected = []
for p in tqdm(shards):
    df = pd.read_parquet(p, columns=["idx", "image", "latex", "source"])
    for _, row in df.iterrows():
        try:
            img = Image.open(io.BytesIO(row["image"]))
            arr = np.array(img.convert("L"))
            mean_px = arr.mean()
            std_px  = arr.std()
            if mean_px > 245:
                collected.append({
                    "mean_px": round(mean_px, 1),
                    "std_px":  round(std_px, 2),
                    "width":   img.width,
                    "height":  img.height,
                    "latex":   row["latex"],
                    "source":  row["source"],
                    "raw":     row["image"],
                })
        except Exception:
            pass

print(f"\nFound {len(collected):,} mostly-white images in raw")

# ── breakdown by source ───────────────────────────────────────────────────────
from collections import Counter
src_counts = Counter(r["source"] for r in collected)
print("\nBy source:")
for src, cnt in src_counts.most_common():
    print(f"  {src:40s}  {cnt:>8,}")

# ── breakdown by pixel std (how much content is in the image) ─────────────────
stds = [r["std_px"] for r in collected]
print("\nPixel std-dev distribution (white bg with content has higher std):")
for threshold, label in [(0.5, "std < 0.5  → nearly blank"),
                         (5,   "std < 5    → very faint / almost blank"),
                         (15,  "std < 15   → faint content"),
                         (30,  "std < 30   → some content"),
                         (999, "std >= 30  → clearly has content")]:
    prev = 0 if threshold == 0.5 else [0.5, 5, 15, 30][([0.5,5,15,30,999].index(threshold)-1)]
    cnt  = sum(1 for s in stds if prev <= s < threshold)
    print(f"  {label:45s}  {cnt:>8,}  ({100*cnt/len(stds):.1f}%)")

# ── sample for visual grid ────────────────────────────────────────────────────
sample = random.sample(collected, min(SAMPLE_N, len(collected)))
sample.sort(key=lambda r: r["std_px"])  # low std (blank) first

COLS = 6
ROWS = (len(sample) + COLS - 1) // COLS
fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3.5, ROWS * 2.8))
fig.suptitle(
    f"'Mostly White' Raw Images  (mean_px > 245) — {len(collected):,} total\n"
    "sorted by pixel std-dev (left=blank → right=has content)",
    fontsize=11, fontweight="bold"
)

for i, ax in enumerate(axes.flat):
    if i < len(sample):
        r = sample[i]
        img = Image.open(io.BytesIO(r["raw"]))
        ax.imshow(img, cmap="gray" if img.mode == "L" else None, vmin=0, vmax=255)
        # truncate long latex for display
        latex_disp = r["latex"][:60] + "…" if len(r["latex"]) > 60 else r["latex"]
        ax.set_title(
            f"μ={r['mean_px']}  σ={r['std_px']}\n"
            f"{r['width']}×{r['height']}  [{r['source'][:12]}]\n"
            f"{latex_disp}",
            fontsize=5.5, loc="left"
        )
        ax.axis("off")
    else:
        ax.axis("off")

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"\nGrid saved → {OUT_PNG}")
