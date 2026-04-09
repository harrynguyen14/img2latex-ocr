"""
resize_aug_shards.py
--------------------
Post-process light/ và heavy/ shards: resize ảnh quá lớn xuống
theo threshold per-source, ghi đè lại shard.

Threshold (width x height):
  crohme                  : 800 x 500
  mathwriting             : 1500 x None  (height fixed 128, chỉ cap width)
  linxy_synthetic_handwrite: 1000 x 200
  default (tất cả còn lại): 1200 x 400

Chỉ resize khi ảnh vượt ngưỡng, giữ aspect ratio.
Ghi đè tại chỗ (backup shard cũ trước nếu muốn an toàn).

Chạy:
    python resize_aug_shards.py
    python resize_aug_shards.py --splits light heavy   # chỉ một số split
    python resize_aug_shards.py --dry_run              # chỉ đếm, không ghi
"""

import argparse
import io
import multiprocessing as mp
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

OUT_DIR = Path("D:/dataset-ocr-builder/latex-ocr-dataset")

# (max_w, max_h) — None = không cap chiều đó
SOURCE_CAPS = {
    "crohme":                     (800,  500),
    "mathwriting":                (1500, None),
    "linxy_synthetic_handwrite":  (1000, 200),
}
DEFAULT_CAP = (1200, 400)


def get_cap(source: str):
    return SOURCE_CAPS.get(source, DEFAULT_CAP)


def maybe_resize(img_bytes: bytes, source: str) -> tuple[bytes, bool]:
    """Trả về (new_bytes, was_resized)."""
    max_w, max_h = get_cap(source)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.width, img.height

    scale = 1.0
    if max_w and w > max_w:
        scale = min(scale, max_w / w)
    if max_h and h > max_h:
        scale = min(scale, max_h / h)

    if scale >= 1.0:
        return img_bytes, False

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), True


def process_shard(pfile: Path, dry_run: bool) -> dict:
    table   = pq.read_table(str(pfile))
    images  = table["image"].to_pylist()
    latexes = table["latex"].to_pylist()
    sources = table["source"].to_pylist()
    idxs    = table["idx"].to_pylist()

    resized_count = 0
    new_images = []

    for img_bytes, src in zip(images, sources):
        new_bytes, was_resized = maybe_resize(img_bytes, src)
        new_images.append(new_bytes)
        if was_resized:
            resized_count += 1

    stats = {"total": len(images), "resized": resized_count}

    if not dry_run and resized_count > 0:
        new_table = pa.table({
            "idx":    pa.array(idxs,       type=pa.int64()),
            "image":  pa.array(new_images, type=pa.binary()),
            "latex":  pa.array(latexes,    type=pa.string()),
            "source": pa.array(sources,    type=pa.string()),
        })
        pq.write_table(new_table, str(pfile), compression="snappy")

    return stats


def process_shard_worker(args):
    pfile, dry_run = args
    return process_shard(Path(pfile), dry_run)


def process_split(split_dir: Path, dry_run: bool, num_workers: int):
    files = sorted(split_dir.glob("*.parquet"))
    if not files:
        print(f"  [skip] no parquet in {split_dir}")
        return

    total, resized = 0, 0
    worker_args = [(str(f), dry_run) for f in files]

    with mp.Pool(num_workers) as pool:
        for s in tqdm(
            pool.imap_unordered(process_shard_worker, worker_args),
            total=len(files),
            desc=f"  {split_dir.name}",
            ncols=80,
        ):
            total   += s["total"]
            resized += s["resized"]

    pct = resized / total * 100 if total else 0
    action = "[dry-run]" if dry_run else "[done]"
    print(f"  {action} {split_dir.name}: {resized:,}/{total:,} resized ({pct:.2f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits",   nargs="*",
                    default=["light", "heavy"],
                    help="Subdirs under train/ to process (default: light heavy)")
    ap.add_argument("--extra",    nargs="+",
                    default=["validation", "test"],
                    help="Dirs directly under OUT_DIR to process (default: validation test)")
    ap.add_argument("--dry_run",    action="store_true",
                    help="Count only, do not write")
    ap.add_argument("--num_workers", type=int, default=mp.cpu_count(),
                    help="Parallel workers (default: all CPU cores)")
    args = ap.parse_args()

    print(f"Resize caps:")
    for src, cap in SOURCE_CAPS.items():
        print(f"  {src:<35} max_w={cap[0]}  max_h={cap[1]}")
    print(f"  {'default':<35} max_w={DEFAULT_CAP[0]}  max_h={DEFAULT_CAP[1]}")
    print(f"\n  workers: {args.num_workers}\n")

    for split_name in args.splits:
        process_split(OUT_DIR / "train" / split_name, args.dry_run, args.num_workers)

    for split_name in args.extra:
        process_split(OUT_DIR / split_name, args.dry_run, args.num_workers)
