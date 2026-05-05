"""
Filter all 4 subsets, removing rows where:
  1. LaTeX length < 5 chars
  2. LaTeX braces unbalanced ({ count != } count)
  3. Image aspect ratio > 20

Reads : DATA_ROOT/{subset}/*.parquet
Writes: OUT_ROOT/{subset}/*.parquet  (same filenames, same schema)
"""

import io
import re
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

DATA_ROOT = Path(r"D:\dataset-ocr-builder\latex-ocr-dataset\ocr-data\train")
OUT_ROOT  = Path(r"D:\dataset-ocr-builder\latex-ocr-dataset\ocr-data\train_filtered")

ASPECT_MAX = 20.0
LATEX_MIN  = 5

SUBSETS = ["heavy", "light", "raw", "screenshot"]

BRACE_OPEN  = re.compile(r"\{")
BRACE_CLOSE = re.compile(r"\}")


def should_keep(image_bytes: bytes, latex: str) -> tuple[bool, str]:
    if not isinstance(latex, str) or len(latex.strip()) < LATEX_MIN:
        return False, "short_latex"

    if len(BRACE_OPEN.findall(latex)) != len(BRACE_CLOSE.findall(latex)):
        return False, "unbal_brace"

    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.width / max(img.height, 1) > ASPECT_MAX:
            return False, "extreme_aspect"
    except Exception:
        return False, "decode_error"

    return True, ""


def filter_shard(src: Path, dst: Path) -> dict:
    df = pd.read_parquet(src)
    counts = {"short_latex": 0, "unbal_brace": 0,
              "extreme_aspect": 0, "decode_error": 0}

    mask = []
    for _, row in df.iterrows():
        keep, reason = should_keep(row["image"], row["latex"])
        mask.append(keep)
        if not keep:
            counts[reason] += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    df[mask].reset_index(drop=True).to_parquet(dst, index=False)
    return {"total": len(df), "kept": int(sum(mask)), **counts}


def main():
    grand = {"total": 0, "kept": 0,
             "short_latex": 0, "unbal_brace": 0,
             "extreme_aspect": 0, "decode_error": 0}

    for subset in SUBSETS:
        src_dir = DATA_ROOT / subset
        dst_dir = OUT_ROOT  / subset
        shards  = sorted(src_dir.glob("*.parquet"))

        if not shards:
            print(f"[SKIP] {subset} — no files found")
            continue

        if dst_dir.exists():
            print(f"[SKIP] {subset} — {dst_dir} already exists")
            continue

        print(f"\n{'='*60}")
        print(f"  {subset.upper()}  ({len(shards)} shards)")
        print(f"{'='*60}")

        sub = {"total": 0, "kept": 0,
               "short_latex": 0, "unbal_brace": 0,
               "extreme_aspect": 0, "decode_error": 0}

        for shard in tqdm(shards, desc=f"  {subset}"):
            s = filter_shard(shard, dst_dir / shard.name)
            for k in sub:
                sub[k] += s[k]

        dropped  = sub["total"] - sub["kept"]
        drop_pct = 100 * dropped / max(sub["total"], 1)
        print(f"  Total          : {sub['total']:>10,}")
        print(f"  Kept           : {sub['kept']:>10,}  ({100-drop_pct:.2f}%)")
        print(f"  Dropped        : {dropped:>10,}  ({drop_pct:.2f}%)")
        print(f"    short_latex    : {sub['short_latex']:>8,}")
        print(f"    unbal_brace    : {sub['unbal_brace']:>8,}")
        print(f"    extreme_aspect : {sub['extreme_aspect']:>8,}")
        print(f"    decode_error   : {sub['decode_error']:>8,}")

        for k in grand:
            grand[k] += sub[k]

    dropped  = grand["total"] - grand["kept"]
    drop_pct = 100 * dropped / max(grand["total"], 1)
    print(f"\n{'='*60}")
    print("  OVERALL")
    print(f"{'='*60}")
    print(f"  Total          : {grand['total']:>10,}")
    print(f"  Kept           : {grand['kept']:>10,}  ({100-drop_pct:.2f}%)")
    print(f"  Dropped        : {dropped:>10,}  ({drop_pct:.2f}%)")
    print(f"    short_latex    : {grand['short_latex']:>8,}")
    print(f"    unbal_brace    : {grand['unbal_brace']:>8,}")
    print(f"    extreme_aspect : {grand['extreme_aspect']:>8,}")
    print(f"    decode_error   : {grand['decode_error']:>8,}")
    print(f"\n  Output → {OUT_ROOT}")


if __name__ == "__main__":
    main()
