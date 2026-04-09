"""
run_split.py
------------
Load tất cả cleaned parquet → stratified split 90/5/5 theo source
→ lưu ra:
  latex-ocr-dataset/train/raw_train-*.parquet
  latex-ocr-dataset/validation/validation-*.parquet
  latex-ocr-dataset/test/test-*.parquet
"""

import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from data_aug import write_shards, ROWS_PER_SHARD

CLEANED_DIR = Path("D:/dataset-ocr-builder/cleaned")
OUT_DIR     = Path("D:/dataset-ocr-builder/latex-ocr-dataset")

TRAIN_RATIO = 0.90
VAL_RATIO   = 0.05
SEED        = 42


def stratified_split(records, train_r, val_r, seed):
    rng = random.Random(seed)
    by_source: dict[str, list] = {}
    for r in records:
        by_source.setdefault(r["source"], []).append(r)

    train, val, test = [], [], []
    for source, items in by_source.items():
        rng.shuffle(items)
        n       = len(items)
        n_val   = max(1, round(n * val_r))
        n_test  = max(1, round(n * (1 - train_r - val_r)))
        n_train = n - n_val - n_test
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])
        print(f"  {source:<35} total={n:>8,}  train={n_train:>8,}  val={n_val:>6,}  test={n_test:>6,}")

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


if __name__ == "__main__":
    # 1. Load
    print("Loading cleaned parquet files...")
    all_records = []
    for pfile in sorted(CLEANED_DIR.glob("*.parquet")):
        table   = pq.read_table(str(pfile))
        images  = table["image"].to_pylist()
        latexes = table["latex"].to_pylist()
        sources = table["source"].to_pylist()
        for img, lat, src in zip(images, latexes, sources):
            all_records.append({"image": img, "latex": lat, "source": src})
        print(f"  {pfile.name}: {len(images):,}")
    print(f"  total: {len(all_records):,}\n")

    # 2. Split
    print("Splitting 90/5/5 stratified by source...")
    train, val, test = stratified_split(all_records, TRAIN_RATIO, VAL_RATIO, SEED)
    print(f"\n  train={len(train):,}  val={len(val):,}  test={len(test):,}")

    # 3. Write
    print("\nWriting shards...")
    write_shards(train, OUT_DIR / "train" / "raw",   "raw_train",  ROWS_PER_SHARD)
    write_shards(val,   OUT_DIR / "validation", "validation", ROWS_PER_SHARD)
    write_shards(test,  OUT_DIR / "test",       "test",       ROWS_PER_SHARD)

    # 4. Save split stats
    stats = {"train": len(train), "validation": len(val), "test": len(test)}
    with open(OUT_DIR / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nDone. Stats saved to {OUT_DIR / 'split_stats.json'}")
