"""
convert_crohme.py

Converts CROHME23 dataset (offline handwritten PNG images + INKML ground truth)
to parquet shards compatible with the training pipeline.

Handles:
  - train: CROHME2019 (10979) + CROHME2013 (1045, matched via CROHME2023_train INKML)
  - val:   all val split
  - test:  all test split

OffHME is skipped — no LaTeX ground truth in .lg files.

Usage:
    python dataset_builder/convert_crohme.py \
        --dataset_dir D:/dataset-ocr-builder/CROHME23/TC11_CROHME23 \
        --out_dir D:/dataset-ocr-builder/latex-ocr-dataset/ocr-data \
        --rows_per_shard 50000
"""

import argparse
import html
import io
import math
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def _parse_latex_from_inkml(inkml_path: Path) -> str | None:
    try:
        content = inkml_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    # First annotation with type="truth" that contains a $ or backslash (LaTeX, not MathML)
    matches = re.findall(r'<annotation type="truth">(.*?)</annotation>', content, re.DOTALL)
    for m in matches:
        s = m.strip()
        # Skip MathML blocks (contain <math or <mrow tags)
        if "<math" in s or "<mrow" in s:
            continue
        # Strip surrounding dollar signs, unescape HTML entities (&lt; etc.)
        s = html.unescape(s.strip("$").strip())
        if s:
            return s
    return None


def _load_png_bytes(path: Path) -> bytes | None:
    try:
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _build_split_records(
    img_dirs: list[Path],
    inkml_dirs: list[Path],
    source_tag: str,
    min_latex_len: int = 1,
    max_latex_len: int = 500,
) -> list[dict]:
    """
    Match PNG images to INKML ground truth by stem, extract LaTeX, return records.
    img_dirs and inkml_dirs can be multiple subdirectories merged together.
    """
    # Build stem -> inkml_path map from all inkml dirs
    inkml_map: dict[str, Path] = {}
    for d in inkml_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.inkml"):
            inkml_map[p.stem] = p

    records = []
    skipped_no_inkml = 0
    skipped_no_latex = 0
    skipped_bad_img  = 0
    skipped_len      = 0

    for img_dir in img_dirs:
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.rglob("*.png")):
            stem = img_path.stem
            inkml_path = inkml_map.get(stem)
            if inkml_path is None:
                skipped_no_inkml += 1
                continue

            latex = _parse_latex_from_inkml(inkml_path)
            if not latex:
                skipped_no_latex += 1
                continue

            if not (min_latex_len <= len(latex) <= max_latex_len):
                skipped_len += 1
                continue

            raw = _load_png_bytes(img_path)
            if raw is None:
                skipped_bad_img += 1
                continue

            records.append({"image": raw, "latex": latex, "source": source_tag})

    total = len(records) + skipped_no_inkml + skipped_no_latex + skipped_bad_img + skipped_len
    print(f"  {source_tag}: {len(records):,} ok / {total:,} total "
          f"(no_inkml={skipped_no_inkml}, no_latex={skipped_no_latex}, "
          f"bad_img={skipped_bad_img}, len_filter={skipped_len})")
    return records


def _write_shards(records: list[dict], out_dir: Path, prefix: str, rows_per_shard: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(records)
    n_shards = max(1, math.ceil(n / rows_per_shard))
    for i in range(n_shards):
        chunk = records[i * rows_per_shard:(i + 1) * rows_per_shard]
        fname = f"{prefix}-{str(i).zfill(5)}-of-{str(n_shards).zfill(5)}.parquet"
        table = pa.table({
            "image":  pa.array([r["image"]  for r in chunk], type=pa.binary()),
            "latex":  pa.array([r["latex"]  for r in chunk], type=pa.string()),
            "source": pa.array([r["source"] for r in chunk], type=pa.string()),
        })
        pq.write_table(table, str(out_dir / fname), compression="snappy")
        print(f"    wrote {fname} ({len(chunk):,} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str,
                    default="D:/dataset-ocr-builder/CROHME23/TC11_CROHME23")
    ap.add_argument("--out_dir", type=str,
                    default="D:/dataset-ocr-builder/latex-ocr-dataset/ocr-data")
    ap.add_argument("--rows_per_shard", type=int, default=50_000)
    ap.add_argument("--min_latex_len", type=int, default=1)
    ap.add_argument("--max_latex_len", type=int, default=500)
    args = ap.parse_args()

    base = Path(args.dataset_dir)
    out  = Path(args.out_dir)

    splits = {
        "train": {
            "img_dirs": [
                base / "IMG" / "train" / "CROHME2019",
                base / "IMG" / "train" / "CROHME2013_train",
                # OffHME excluded — no LaTeX ground truth
            ],
            "inkml_dirs": [
                base / "INKML" / "train" / "CROHME2019",
                base / "INKML" / "train" / "CROHME2023_train",  # covers CROHME2013 images
            ],
            "out_subdir": "train/crohme",
            "prefix": "crohme_train",
            "source_tag": "crohme_handwritten",
        },
        "val": {
            "img_dirs":   [base / "IMG" / "val"],
            "inkml_dirs": [base / "INKML" / "val"],
            "out_subdir": "val/crohme",
            "prefix": "crohme_val",
            "source_tag": "crohme_handwritten",
        },
        "test": {
            "img_dirs":   [base / "IMG" / "test"],
            "inkml_dirs": [base / "INKML" / "test"],
            "out_subdir": "test/crohme",
            "prefix": "crohme_test",
            "source_tag": "crohme_handwritten",
        },
    }

    for split_name, cfg in splits.items():
        print(f"\n=== {split_name} ===")
        records = _build_split_records(
            img_dirs=cfg["img_dirs"],
            inkml_dirs=cfg["inkml_dirs"],
            source_tag=cfg["source_tag"],
            min_latex_len=args.min_latex_len,
            max_latex_len=args.max_latex_len,
        )
        if records:
            out_dir = out / cfg["out_subdir"]
            print(f"  Writing to {out_dir} ...")
            _write_shards(records, out_dir, cfg["prefix"], args.rows_per_shard)
        else:
            print(f"  No records — skipping write.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
