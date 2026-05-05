"""
Score entire train_filtered dataset using the current model,
then filter keeping:
  - 100% of hard examples   (edit_distance_ratio > 0.1)
  - 30%  of easy examples   (edit_distance_ratio == 0.0)
  - 100% of medium examples (0.0 < ratio <= 0.1)

Output: train_scored/{subset}/*.parquet  (same schema + "score" column)
        train_curriculum/{subset}/*.parquet  (filtered)
        score_report.txt
"""

import io
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(r"D:\img2latex")))
from nav2tex.pipeline_latex_ocr import Nav2TexPipeline

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = Path(r"D:\img2latex\latex_ocr")
DATA_ROOT    = Path(r"D:\dataset-ocr-builder\latex-ocr-dataset\ocr-data\train")
SCORED_ROOT  = Path(r"D:\dataset-ocr-builder\latex-ocr-dataset\ocr-data\train_scored")
OUT_ROOT     = Path(r"D:\dataset-ocr-builder\latex-ocr-dataset\ocr-data\train_curriculum")
REPORT_PATH  = Path(r"D:\img2latex\dataset_builder\analysis_output\score_report.txt")

BATCH_SIZE   = 32
EASY_KEEP    = 0.30   # fraction of easy samples to retain
RANDOM_SEED  = 42
SUBSETS      = ["heavy", "light", "raw", "screenshot"]

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ── Edit distance ─────────────────────────────────────────────────────────────
def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1,
                          curr[j-1] + 1,
                          prev[j-1] + (0 if ca == cb else 1))
        prev = curr
    return prev[lb]


def edit_distance_ratio(pred: str, gt: str) -> float:
    denom = max(len(gt), 1)
    return edit_distance(pred, gt) / denom


# ── Inference batch ───────────────────────────────────────────────────────────
def score_batch(pipeline: Nav2TexPipeline,
                image_bytes_list: list[bytes],
                latex_list: list[str]) -> list[float]:
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]
    try:
        preds = pipeline(imgs, max_new_tokens=200, num_beams=1)
        return [edit_distance_ratio(p, gt) for p, gt in zip(preds, latex_list)]
    except Exception:
        return [1.0] * len(latex_list)


# ── Per-shard scoring ─────────────────────────────────────────────────────────
def score_shard(pipeline, src: Path, dst: Path) -> pd.DataFrame:
    if dst.exists():
        return pd.read_parquet(dst)

    df = pd.read_parquet(src)
    all_scores = []
    n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    for start in tqdm(range(0, len(df), BATCH_SIZE), total=n_batches,
                      desc=f"    {src.name}", leave=False, unit="batch"):
        batch = df.iloc[start:start + BATCH_SIZE]
        scores = score_batch(
            pipeline,
            batch["image"].tolist(),
            batch["latex"].tolist(),
        )
        all_scores.extend(scores)

    df["score"] = all_scores
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return df


# ── Filter by score ───────────────────────────────────────────────────────────
def curriculum_filter(df: pd.DataFrame) -> pd.DataFrame:
    easy   = df[df["score"] == 0.0]
    medium = df[(df["score"] > 0.0) & (df["score"] <= 0.1)]
    hard   = df[df["score"] > 0.1]

    easy_keep = easy.sample(frac=EASY_KEEP, random_state=RANDOM_SEED) if len(easy) else easy
    return pd.concat([hard, medium, easy_keep], ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading model from {MODEL_PATH} …")
    pipeline = Nav2TexPipeline.from_pretrained(str(MODEL_PATH), device=DEVICE)
    pipeline.model.eval()

    report_lines = []
    grand = {"total": 0, "easy": 0, "medium": 0, "hard": 0, "kept": 0}

    for subset in SUBSETS:
        src_dir    = DATA_ROOT   / subset
        scored_dir = SCORED_ROOT / subset
        out_dir    = OUT_ROOT    / subset
        shards     = sorted(src_dir.glob("*.parquet"))

        if not shards:
            print(f"[SKIP] {subset} — no files")
            continue

        print(f"\n{'='*60}")
        print(f"  {subset.upper()}  ({len(shards)} shards)")
        print(f"{'='*60}")

        sub = {"total": 0, "easy": 0, "medium": 0, "hard": 0, "kept": 0}

        for shard in tqdm(shards, desc=f"  {subset}"):
            scored_path = scored_dir / shard.name
            df = score_shard(pipeline, shard, scored_path)

            easy_n   = int((df["score"] == 0.0).sum())
            medium_n = int(((df["score"] > 0.0) & (df["score"] <= 0.1)).sum())
            hard_n   = int((df["score"] > 0.1).sum())

            df_filt = curriculum_filter(df)
            out_path = out_dir / shard.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_filt.to_parquet(out_path, index=False)

            sub["total"]  += len(df)
            sub["easy"]   += easy_n
            sub["medium"] += medium_n
            sub["hard"]   += hard_n
            sub["kept"]   += len(df_filt)

        lines = [
            f"\n{'='*60}",
            f"  {subset.upper()}",
            f"{'='*60}",
            f"  Total   : {sub['total']:>10,}",
            f"  Easy  (score=0)       : {sub['easy']:>8,}  ({100*sub['easy']/max(sub['total'],1):.1f}%)",
            f"  Medium (0<score≤0.1)  : {sub['medium']:>8,}  ({100*sub['medium']/max(sub['total'],1):.1f}%)",
            f"  Hard  (score>0.1)     : {sub['hard']:>8,}  ({100*sub['hard']/max(sub['total'],1):.1f}%)",
            f"  Kept after filter     : {sub['kept']:>8,}  ({100*sub['kept']/max(sub['total'],1):.1f}%)",
        ]
        for l in lines:
            print(l)
            report_lines.append(l)

        for k in grand:
            grand[k] += sub[k]

    lines = [
        f"\n{'='*60}",
        "  OVERALL",
        f"{'='*60}",
        f"  Total   : {grand['total']:>10,}",
        f"  Easy    : {grand['easy']:>10,}  ({100*grand['easy']/max(grand['total'],1):.1f}%)",
        f"  Medium  : {grand['medium']:>10,}  ({100*grand['medium']/max(grand['total'],1):.1f}%)",
        f"  Hard    : {grand['hard']:>10,}  ({100*grand['hard']/max(grand['total'],1):.1f}%)",
        f"  Kept    : {grand['kept']:>10,}  ({100*grand['kept']/max(grand['total'],1):.1f}%)",
        f"\n  Scored  → {SCORED_ROOT}",
        f"  Output  → {OUT_ROOT}",
    ]
    for l in lines:
        print(l)
        report_lines.append(l)

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n  Report → {REPORT_PATH}")


if __name__ == "__main__":
    main()
