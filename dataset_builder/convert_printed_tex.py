"""
convert_printed_tex.py

Converts PRINTED_TEX_230k dataset to parquet shards compatible with the
training pipeline (columns: image bytes, latex string).

Usage:
    python dataset_builder/convert_printed_tex.py \
        --dataset_dir D:/dataset-ocr-builder/PRINTED_TEX_230k/PRINTED_TEX_230k \
        --out_dir D:/dataset-ocr-builder/latex-ocr-dataset/ocr-data/train/printed_tex \
        --rows_per_shard 50000 \
        --n_workers 4
"""

import argparse
import io
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROWS_PER_SHARD = 50_000


def _detokenize(s: str) -> str:
    # Tokens are space-separated; joining removes the spaces.
    # e.g. "R _ { 1 2 }" -> "R_{12}"
    return s.replace(" ", "")


def _load_png_bytes(path: str) -> bytes | None:
    try:
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _process_batch(batch: list[tuple[int, str, str]]) -> list[dict]:
    records = []
    for idx, img_path, latex in batch:
        raw = _load_png_bytes(img_path)
        if raw is None:
            continue
        records.append({"idx": idx, "image": raw, "latex": latex, "source": "printed_tex"})
    return records


def _write_shard(records: list[dict], out_dir: Path, shard_idx: int, n_shards: int):
    fname = f"printed_tex-{str(shard_idx).zfill(5)}-of-{str(n_shards).zfill(5)}.parquet"
    table = pa.table({
        "idx":    pa.array([r["idx"]   for r in records], type=pa.int64()),
        "image":  pa.array([r["image"] for r in records], type=pa.binary()),
        "latex":  pa.array([r["latex"] for r in records], type=pa.string()),
        "source": pa.array([r["source"] for r in records], type=pa.string()),
    })
    pq.write_table(table, str(out_dir / fname), compression="snappy")
    print(f"  wrote {fname} ({len(records):,} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str,
                    default="D:/dataset-ocr-builder/PRINTED_TEX_230k/PRINTED_TEX_230k")
    ap.add_argument("--out_dir", type=str,
                    default="D:/dataset-ocr-builder/latex-ocr-dataset/ocr-data/train/printed_tex")
    ap.add_argument("--rows_per_shard", type=int, default=ROWS_PER_SHARD)
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--n_workers", type=int, default=4)
    ap.add_argument("--min_latex_len", type=int, default=2,
                    help="Skip formulas shorter than this (detokenized)")
    ap.add_argument("--max_latex_len", type=int, default=500)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    formulas_file = dataset_dir / "final_png_formulas.txt"
    images_file   = dataset_dir / "corresponding_png_images.txt"
    img_dir       = dataset_dir / "generated_png_images"

    print("Loading formula and image lists...")
    with open(formulas_file, encoding="utf-8") as f:
        formulas = [line.rstrip("\n") for line in f]
    with open(images_file, encoding="utf-8") as f:
        image_names = [line.strip() for line in f]

    assert len(formulas) == len(image_names), "Line count mismatch!"
    print(f"  Total entries: {len(formulas):,}")

    # Build work list, detokenize and filter
    pairs = []
    skipped = 0
    for idx, (tok, img_name) in enumerate(zip(formulas, image_names)):
        latex = _detokenize(tok.strip())
        if not (args.min_latex_len <= len(latex) <= args.max_latex_len):
            skipped += 1
            continue
        img_path = str(img_dir / img_name)
        if not os.path.exists(img_path):
            skipped += 1
            continue
        pairs.append((idx, img_path, latex))

    print(f"  Valid pairs: {len(pairs):,}  (skipped {skipped:,})")

    # Split into batches
    batches = [pairs[i:i + args.batch_size] for i in range(0, len(pairs), args.batch_size)]
    n_shards = max(1, math.ceil(len(pairs) / args.rows_per_shard))

    print(f"Processing {len(pairs):,} samples in {len(batches)} batches "
          f"({args.n_workers} workers) -> {n_shards} shards ...")

    all_records: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(_process_batch, b): len(b) for b in batches}
        for fut in as_completed(futures):
            all_records.extend(fut.result())
            done += futures[fut]
            print(f"  {done:,}/{len(pairs):,} processed, {len(all_records):,} ok", end="\r")

    print(f"\nTotal converted: {len(all_records):,}")
    print(f"Writing {n_shards} shard(s) to {out_dir} ...")
    n_shards = max(1, math.ceil(len(all_records) / args.rows_per_shard))
    for i in range(n_shards):
        chunk = all_records[i * args.rows_per_shard:(i + 1) * args.rows_per_shard]
        _write_shard(chunk, out_dir, i, n_shards)
    print("Done.")


if __name__ == "__main__":
    main()
