"""
filter_raw.py

Filters and rewrites the merged raw parquet shards:
  1. Drop latex len < 3 or > 400
  2. Keep at most MAX_PER_LATEX rows per unique latex string
  3. Rebuild sequential idx

Usage:
    python dataset_builder/filter_raw.py \
        --data_dir D:/dataset-ocr-builder/latex-ocr-dataset/ocr-data/train/raw \
        --out_dir  D:/dataset-ocr-builder/latex-ocr-dataset/ocr-data/train/raw_filtered \
        --min_latex_len 3 \
        --max_latex_len 400 \
        --max_per_latex 3 \
        --rows_per_shard 50000
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",       type=str, default="D:/dataset-ocr-builder/latex-ocr-dataset/ocr-data/train/raw")
    ap.add_argument("--out_dir",        type=str, default="D:/dataset-ocr-builder/latex-ocr-dataset/ocr-data/train/raw_filtered")
    ap.add_argument("--min_latex_len",  type=int, default=3)
    ap.add_argument("--max_latex_len",  type=int, default=400)
    ap.add_argument("--max_per_latex",  type=int, default=3)
    ap.add_argument("--rows_per_shard", type=int, default=50_000)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.parquet"))
    print(f"Input: {len(files)} shards from {data_dir}")

    # Pass 1: count per-latex occurrences already seen (for cap enforcement)
    latex_seen: defaultdict[str, int] = defaultdict(int)

    # Stats
    n_total = n_short = n_long = n_capped = n_kept = 0

    # Buffer for output
    buf_idx: list[int]   = []
    buf_img: list[bytes] = []
    buf_lat: list[str]   = []
    buf_src: list[str]   = []

    shard_idx  = 0
    global_idx = 0

    def flush_shard(total_shards=999):
        nonlocal shard_idx
        fname = f"raw_train-{str(shard_idx).zfill(5)}-of-TOTAL.parquet"
        table = pa.table({
            "idx":    pa.array(buf_idx, type=pa.int64()),
            "image":  pa.array(buf_img, type=pa.binary()),
            "latex":  pa.array(buf_lat, type=pa.string()),
            "source": pa.array(buf_src, type=pa.string()),
        })
        pq.write_table(table, str(out_dir / fname), compression="snappy")
        print(f"  shard {shard_idx}: {len(buf_idx):,} rows "
              f"(idx {buf_idx[0]}..{buf_idx[-1]})")
        shard_idx += 1
        buf_idx.clear(); buf_img.clear(); buf_lat.clear(); buf_src.clear()

    for f in files:
        table = pq.read_table(str(f), columns=["image", "latex", "source"])
        images  = table["image"].to_pylist()
        latexs  = table["latex"].to_pylist()
        sources = table["source"].to_pylist()

        for img, lat, src in zip(images, latexs, sources):
            n_total += 1

            # Filter: skip bad rows
            if img is None or not lat or not str(lat).strip():
                n_short += 1
                continue

            lat = str(lat).strip()
            lat_len = len(lat)

            if lat_len < args.min_latex_len:
                n_short += 1
                continue
            if lat_len > args.max_latex_len:
                n_long += 1
                continue

            # Cap per-latex
            if latex_seen[lat] >= args.max_per_latex:
                n_capped += 1
                continue
            latex_seen[lat] += 1

            buf_idx.append(global_idx)
            buf_img.append(bytes(img))
            buf_lat.append(lat)
            buf_src.append(str(src))
            global_idx += 1
            n_kept += 1

            if len(buf_idx) >= args.rows_per_shard:
                flush_shard()

        print(f"  [{f.name}] running total kept={n_kept:,} "
              f"(short={n_short}, long={n_long}, capped={n_capped})", end="\r")

    if buf_idx:
        flush_shard()

    n_shards = shard_idx
    print(f"\n\nFilter complete:")
    print(f"  Input rows   : {n_total:,}")
    print(f"  Dropped short: {n_short:,}  (len < {args.min_latex_len})")
    print(f"  Dropped long : {n_long:,}  (len > {args.max_latex_len})")
    print(f"  Dropped cap  : {n_capped:,}  (>{args.max_per_latex}x same latex)")
    print(f"  Kept         : {n_kept:,}  ({100*n_kept/max(n_total,1):.1f}%)")
    print(f"  Output shards: {n_shards}")
    print(f"  Unique latex : {sum(1 for v in latex_seen.values() if v > 0):,}")

    # Rename TOTAL -> actual count
    print(f"\nRenaming shards ...")
    for fpath in sorted(out_dir.glob("raw_train-*-of-TOTAL.parquet")):
        new_name = fpath.name.replace("TOTAL", str(n_shards).zfill(5))
        fpath.rename(out_dir / new_name)
    print("Done.")


if __name__ == "__main__":
    main()
