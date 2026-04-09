"""
run_aug.py
----------
Đọc raw_train shards đã split → apply light aug + heavy aug
→ lưu ra:
  latex-ocr-dataset/train/light_aug_train-*.parquet
  latex-ocr-dataset/train/heavy_aug_train-*.parquet

Chạy sau run_split.py.
"""

import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from data_aug import get_light_aug, get_heavy_aug, apply_aug, write_shards, ROWS_PER_SHARD


def diversify_im2latex(img_bytes: bytes, seed_offset: int = 0) -> bytes:
    """Resize im2latex fixed-size images to a random height in [48, 96]
    while preserving aspect ratio, to avoid model overfitting to 320x64."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    target_h = random.randint(48, 96)
    scale = target_h / img.height
    target_w = max(1, round(img.width * scale))
    img = img.resize((target_w, target_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

import pyarrow as pa

OUT_DIR   = Path("D:/dataset-ocr-builder/latex-ocr-dataset")
TRAIN_DIR = OUT_DIR / "train"


def get_raw_shards() -> list[Path]:
    files = sorted((TRAIN_DIR / "raw").glob("raw_train-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No raw_train shards found in {TRAIN_DIR / 'raw'}. Run run_split.py first."
        )
    return files


def aug_shard(pfile: Path, aug_fn, prefix: str, out_dir: Path,
              shard_idx: int, n_total_shards: int) -> int:
    """Load one raw shard, aug, write one output shard, then free RAM."""
    out_dir.mkdir(parents=True, exist_ok=True)

    table   = pq.read_table(str(pfile))
    images  = table["image"].to_pylist()
    latexes = table["latex"].to_pylist()
    sources = table["source"].to_pylist()
    del table  # free arrow table immediately

    augmented_imgs, augmented_lat, augmented_src = [], [], []
    for img_bytes, lat, src in tqdm(
        zip(images, latexes, sources),
        total=len(images),
        desc=f"  {prefix} shard {shard_idx+1}/{n_total_shards}",
        ncols=80,
        leave=False,
    ):
        if src == "im2latex":
            img_bytes = diversify_im2latex(img_bytes)
        aug       = aug_fn(src)
        aug_bytes = apply_aug(img_bytes, aug)
        augmented_imgs.append(aug_bytes)
        augmented_lat.append(lat)
        augmented_src.append(src)

    del images, latexes, sources  # free raw data

    fname = f"{prefix}-{str(shard_idx).zfill(5)}-of-{str(n_total_shards).zfill(5)}.parquet"
    out_table = pa.table({
        "idx":    pa.array(list(range(len(augmented_imgs))), type=pa.int64()),
        "image":  pa.array(augmented_imgs, type=pa.binary()),
        "latex":  pa.array(augmented_lat,  type=pa.string()),
        "source": pa.array(augmented_src,  type=pa.string()),
    })
    pq.write_table(out_table, str(out_dir / fname), compression="snappy")
    del augmented_imgs, augmented_lat, augmented_src, out_table

    return len(augmented_imgs)


if __name__ == "__main__":
    raw_shards = get_raw_shards()
    n_shards   = len(raw_shards)
    print(f"Found {n_shards} raw shards\n")

    for prefix, aug_fn, out_subdir in [
        ("light_aug_train", get_light_aug, TRAIN_DIR / "light"),
        ("heavy_aug_train", get_heavy_aug, TRAIN_DIR / "heavy"),
    ]:
        print(f"\n== {prefix} ==")
        total = 0
        for i, pfile in enumerate(raw_shards):
            n = aug_shard(pfile, aug_fn, prefix, out_subdir, i, n_shards)
            total += n
            print(f"  shard {i+1}/{n_shards}: {n:,} rows -> {out_subdir.name}/")
        print(f"  total {prefix}: {total:,}")

    # Update split_stats.json
    stats_path = OUT_DIR / "split_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        stats = {}
    light_files = sorted((TRAIN_DIR / "light").glob("light_aug_train-*.parquet"))
    heavy_files = sorted((TRAIN_DIR / "heavy").glob("heavy_aug_train-*.parquet"))
    stats["train_light"] = sum(pq.read_metadata(str(f)).num_rows for f in light_files)
    stats["train_heavy"] = sum(pq.read_metadata(str(f)).num_rows for f in heavy_files)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nDone. Stats saved to {stats_path}")
