from datasets import load_dataset
from pathlib import Path

BASE_DIR = "D:/dataset-ocr-builder"

# Already downloaded — skip:
#   linxy/LaTeX_OCR       -> D:/dataset-ocr-builder/linxy
#   OleehyO/latex-formulas -> D:/dataset-ocr-builder/oleehyo
#   mathwriting-2024       -> D:/dataset-ocr-builder/mathwriting-2024


def should_skip(path):
    return Path(path).exists()


# ── CROHME ────────────────────────────────────────────────────────────────────
# Neeze/CROHME-full: ~12k samples, splits: train / 2014 / 2016 / 2019
# Fields: image, label (LaTeX)
crohme_dir = f"{BASE_DIR}/crohme"
if should_skip(crohme_dir):
    print("CROHME already exists, skipping")
else:
    ds = load_dataset("Neeze/CROHME-full", cache_dir=crohme_dir)
    print(f"Saved CROHME to {crohme_dir}")

# ── IM2LATEX-100K ─────────────────────────────────────────────────────────────
# yuntian-deng/im2latex-100k: ~68k samples, splits: train / val / test
# Fields: image, formula (LaTeX), filename
im2latex_dir = f"{BASE_DIR}/im2latex"
if should_skip(im2latex_dir):
    print("IM2LATEX already exists, skipping")
else:
    ds = load_dataset("yuntian-deng/im2latex-100k", cache_dir=im2latex_dir)
    print(f"Saved IM2LATEX to {im2latex_dir}")

# ── HME100K ───────────────────────────────────────────────────────────────────
# lmms-lab/LLaVA-OneVision-Data subset hme100k: ~100k samples
# Fields: image, latex_formula
# NOTE: large parent dataset, only the hme100k subset is downloaded
hme_dir = f"{BASE_DIR}/hme100k"
if should_skip(hme_dir):
    print("HME100K already exists, skipping")
else:
    ds = load_dataset(
        "lmms-lab/LLaVA-OneVision-Data",
        "hme100k",
        cache_dir=hme_dir,
    )
    print(f"Saved HME100K to {hme_dir}")
