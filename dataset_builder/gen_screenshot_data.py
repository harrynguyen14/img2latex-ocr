"""
gen_screenshot_data.py

Generates synthetic screenshot-domain LaTeX OCR data by:
1. Rendering LaTeX strings via pdflatex+pdf2image (best) or dvipng or matplotlib fallback
2. Random math fonts (Times, Palatino, Euler, sans-serif, etc.)
3. Screenshot/camera-realistic augmentations
4. Saving as parquet shards compatible with the training pipeline

Renderer priority (auto-detected):
  pdflatex + pdf2image  ->  dvipng  ->  matplotlib

Usage:
    python dataset_builder/gen_screenshot_data.py `
        --source_dir D:/dataset-ocr-builder/latex-ocr-dataset/ocr-data/train/raw `
        --out_dir    D:/dataset-ocr-builder/latex-ocr-dataset/ocr-data/train/screenshot `
        --n_samples  200000 `
        --n_workers  4 `
        --tex_bin_dir C:/texlive/2026/bin/windows `
        --renderer   auto
"""

import argparse
import io
import math
import os
import random
import shutil
import subprocess
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageEnhance, ImageFilter

warnings.filterwarnings("ignore")

ROWS_PER_SHARD = 50_000

# Set by worker init — full path to TeX Live bin dir so subprocess can find binaries
_TEX_BIN_DIR: str = ""
_RENDERER: str = "matplotlib"


# ── Renderer detection ────────────────────────────────────────────────────────

def _has_cmd(*cmds, extra_path: str = "") -> bool:
    search_path = (extra_path + os.pathsep + os.environ.get("PATH", "")) if extra_path else None
    return all(shutil.which(c, path=search_path) is not None for c in cmds)


def _detect_renderer(tex_bin_dir: str = "") -> str:
    try:
        import pdf2image  # noqa: F401
        if _has_cmd("pdflatex", extra_path=tex_bin_dir):
            return "pdflatex"
    except ImportError:
        pass
    if _has_cmd("latex", "dvipng", extra_path=tex_bin_dir):
        return "dvipng"
    return "matplotlib"


def _subprocess_env() -> dict:
    """Return os.environ copy with TeX bin dir prepended to PATH."""
    env = os.environ.copy()
    if _TEX_BIN_DIR:
        env["PATH"] = _TEX_BIN_DIR + os.pathsep + env.get("PATH", "")
    return env


def _tex_cmd(name: str) -> str:
    """Resolve a TeX binary to its full path using _TEX_BIN_DIR."""
    if _TEX_BIN_DIR:
        search = _TEX_BIN_DIR + os.pathsep + os.environ.get("PATH", "")
        found = shutil.which(name, path=search)
        if found:
            return found
    return name


# ── Math font packages ────────────────────────────────────────────────────────

_MATH_FONTS = [
    # (packages,                                   weight)
    (r"\usepackage{amsmath,amssymb}",              30),  # Computer Modern (default)
    (r"\usepackage{amsmath,amssymb,lmodern}",      15),  # Latin Modern
    (r"\usepackage{amsmath,amssymb,mathptmx}",     10),  # Times
    (r"\usepackage{amsmath,amssymb,newpxmath}",    10),  # Palatino-like
    (r"\usepackage{amsmath,amssymb,newtxmath}",    10),  # TX fonts
    (r"\usepackage{amsmath,amssymb,fourier}",       8),  # Utopia
    (r"\usepackage{amsmath,amssymb,cmbright}",      8),  # CM Bright (sans)
    (r"\usepackage{amsmath,amssymb,euler}",         6),  # Euler (handwriting-like)
    (r"\usepackage{amsmath,amssymb,mathpazo}",      8),  # Palatino
    (r"\usepackage{amsmath,amssymb,kpfonts}",       5),  # KP fonts
]

_FONT_PACKAGES = [pkg for pkg, _ in _MATH_FONTS]
_FONT_WEIGHTS  = [w   for _,   w in _MATH_FONTS]


def _random_font_pkg() -> str:
    return random.choices(_FONT_PACKAGES, weights=_FONT_WEIGHTS, k=1)[0]


# ── Background styles ─────────────────────────────────────────────────────────

_BG_STYLES = [
    # ((bg_r, bg_g, bg_b), (fg_r, fg_g, fg_b)), weight
    (((1.00, 1.00, 1.00), (0.00, 0.00, 0.00)), 30),  # white
    (((1.00, 0.99, 0.90), (0.00, 0.00, 0.00)), 20),  # cream
    (((0.94, 0.94, 0.94), (0.00, 0.00, 0.00)), 15),  # light gray
    (((0.12, 0.12, 0.15), (0.95, 0.95, 0.95)), 10),  # dark
    (((1.00, 0.97, 0.80), (0.00, 0.00, 0.00)),  8),  # yellow
    (((0.93, 0.95, 1.00), (0.00, 0.00, 0.00)),  7),  # blue tint
    (((0.93, 1.00, 0.95), (0.00, 0.00, 0.00)),  5),  # green tint
    (((1.00, 0.94, 0.96), (0.00, 0.00, 0.00)),  5),  # pink tint
]

_BG_COLORS  = [c for c, _ in _BG_STYLES]
_BG_WEIGHTS = [w for _, w in _BG_STYLES]


def _random_bg():
    return random.choices(_BG_COLORS, weights=_BG_WEIGHTS, k=1)[0]


def _build_tex(latex: str, font_pkg: str, bg_r, bg_g, bg_b, fg_r, fg_g, fg_b, pad: int) -> str:
    return (
        r"\documentclass[preview]{standalone}" + "\n"
        + font_pkg + "\n"
        + r"\usepackage{xcolor}" + "\n"
        + rf"\pagecolor[rgb]{{{bg_r:.4f},{bg_g:.4f},{bg_b:.4f}}}" + "\n"
        + rf"\color[rgb]{{{fg_r:.4f},{fg_g:.4f},{fg_b:.4f}}}" + "\n"
        + r"\begin{document}" + "\n"
        + rf"\hspace{{{pad}pt}}$\displaystyle {latex}$\hspace{{{pad}pt}}" + "\n"
        + r"\end{document}"
    )


# ── pdflatex + pdf2image renderer ────────────────────────────────────────────

def _render_pdflatex(latex: str) -> "Image.Image | None":
    from pdf2image import convert_from_path

    (bg_r, bg_g, bg_b), (fg_r, fg_g, fg_b) = _random_bg()
    font_pkg = _random_font_pkg()
    dpi = random.randint(150, 300)
    pad = random.randint(4, 16)
    tex_src = _build_tex(latex, font_pkg, bg_r, bg_g, bg_b, fg_r, fg_g, fg_b, pad)

    with tempfile.TemporaryDirectory() as tmp:
        tex_path = os.path.join(tmp, "formula.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_src)

        r = subprocess.run(
            [_tex_cmd("pdflatex"), "-interaction=nonstopmode", "-halt-on-error", "formula.tex"],
            cwd=tmp, capture_output=True, timeout=20, env=_subprocess_env(),
        )
        if r.returncode != 0:
            return None

        pdf_path = os.path.join(tmp, "formula.pdf")
        if not os.path.exists(pdf_path):
            return None

        pages = convert_from_path(pdf_path, dpi=dpi, fmt="png")
        if not pages:
            return None

        img = pages[0].convert("RGB")
        if np.array(img, dtype=np.float32).std() < 2.0:
            return None
        return img


# ── latex + dvipng renderer ───────────────────────────────────────────────────

def _render_dvipng(latex: str) -> "Image.Image | None":
    (bg_r, bg_g, bg_b), (fg_r, fg_g, fg_b) = _random_bg()
    font_pkg = _random_font_pkg()
    dpi = random.randint(120, 220)
    pad = random.randint(4, 16)
    tex_src = _build_tex(latex, font_pkg, bg_r, bg_g, bg_b, fg_r, fg_g, fg_b, pad)

    with tempfile.TemporaryDirectory() as tmp:
        tex_path = os.path.join(tmp, "formula.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_src)

        r = subprocess.run(
            [_tex_cmd("latex"), "-interaction=nonstopmode", "-halt-on-error", "formula.tex"],
            cwd=tmp, capture_output=True, timeout=15, env=_subprocess_env(),
        )
        if r.returncode != 0:
            return None

        dvi_path = os.path.join(tmp, "formula.dvi")
        if not os.path.exists(dvi_path):
            return None

        png_path = os.path.join(tmp, "formula.png")
        bg_hex = "{:02x}{:02x}{:02x}".format(int(bg_r*255), int(bg_g*255), int(bg_b*255))
        r2 = subprocess.run(
            [_tex_cmd("dvipng"), "-D", str(dpi), "-bg", f"#{bg_hex}",
             "-T", "tight", "-o", png_path, dvi_path],
            capture_output=True, timeout=15, env=_subprocess_env(),
        )
        if r2.returncode != 0 or not os.path.exists(png_path):
            return None

        img = Image.open(png_path).convert("RGB")
        if np.array(img, dtype=np.float32).std() < 2.0:
            return None
        return img


# ── matplotlib fallback renderer ──────────────────────────────────────────────

_MPL_BG = [
    ((1.0,  1.0,  1.0 ), "black"),
    ((1.0,  0.99, 0.90), "black"),
    ((0.94, 0.94, 0.94), "black"),
    ((0.12, 0.12, 0.15), "white"),
    ((1.0,  0.97, 0.80), "black"),
    ((0.93, 0.95, 1.0 ), "black"),
]


def _render_matplotlib(latex: str) -> "Image.Image | None":
    bg, fg = random.choice(_MPL_BG)
    dpi = random.randint(120, 200)
    fontsize = random.randint(16, 26)
    fig = plt.figure(figsize=(random.uniform(5, 10), random.uniform(1.0, 2.0)))
    fig.patch.set_facecolor(bg)
    try:
        fig.text(random.uniform(0.02, 0.06), random.uniform(0.3, 0.7),
                 f"${latex}$", fontsize=fontsize, ha="left", va="center", color=fg)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor=bg, edgecolor="none")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        plt.close(fig)
        if np.array(img, dtype=np.float32).std() < 2.0:
            return None
        return img
    except Exception:
        plt.close(fig)
        return None


# ── Unified render entry point ────────────────────────────────────────────────

def _render(latex: str) -> "Image.Image | None":
    if _RENDERER == "pdflatex":
        return _render_pdflatex(latex)
    if _RENDERER == "dvipng":
        return _render_dvipng(latex)
    if _RENDERER == "matplotlib":
        return _render_matplotlib(latex)
    # auto: pdflatex -> dvipng -> matplotlib
    img = _render_pdflatex(latex)
    if img is None:
        img = _render_dvipng(latex)
    if img is None:
        img = _render_matplotlib(latex)
    return img


# ── Augmentations ─────────────────────────────────────────────────────────────

def _aug_jpeg(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(35, 85))
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def _aug_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.8)))

def _aug_noise(img):
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr + np.random.normal(0, random.uniform(5, 30), arr.shape), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def _aug_brightness_contrast(img):
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.55, 1.45))
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.55, 1.6))

def _aug_shadow(img):
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    if random.random() < 0.5:
        grad = np.linspace(random.uniform(0.6, 0.95), 1.0, w, dtype=np.float32)
        arr *= grad[np.newaxis, :, np.newaxis]
    else:
        grad = np.linspace(random.uniform(0.6, 0.95), 1.0, h, dtype=np.float32)
        arr *= grad[:, np.newaxis, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def _aug_color_tint(img):
    arr = np.array(img, dtype=np.float32)
    tint = np.array([random.uniform(0.80, 1.0) for _ in range(3)], dtype=np.float32)
    return Image.fromarray(np.clip(arr * tint, 0, 255).astype(np.uint8))

def _aug_chromatic_aberration(img):
    arr = np.array(img)
    shift = random.randint(1, 4)
    r = np.roll(arr[:, :, 0], shift, axis=1)
    b = np.roll(arr[:, :, 2], -shift, axis=1)
    return Image.fromarray(np.stack([r, arr[:, :, 1], b], axis=2))

def _aug_moire(img):
    arr = np.array(img, dtype=np.int16)
    w = arr.shape[1]
    pattern = (np.sin(2 * np.pi * random.uniform(0.04, 0.12) * np.arange(w))
               * random.uniform(0.02, 0.06) * 255).astype(np.int16)
    arr[:, :, :] += pattern[np.newaxis, :, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def _aug_paper_texture(img):
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    texture = np.random.normal(1.0, 0.015, (h, w)).astype(np.float32)
    return Image.fromarray(np.clip(arr * texture[:, :, np.newaxis], 0, 255).astype(np.uint8))

def _aug_screenshot_border(img):
    pad = random.randint(3, 18)
    bg = tuple(random.randint(180, 255) for _ in range(3))
    out = Image.new("RGB", (img.width + 2*pad, img.height + 2*pad), bg)
    out.paste(img, (pad, pad))
    return out

def _aug_rotation(img):
    angle = random.uniform(-3, 3)
    bg = (255, 255, 255) if np.array(img).mean() > 128 else (0, 0, 0)
    return img.rotate(angle, fillcolor=bg, expand=False)

def _aug_downscale_upscale(img):
    scale = random.uniform(0.4, 0.75)
    small = img.resize((max(1, int(img.width*scale)), max(1, int(img.height*scale))), Image.BILINEAR)
    return small.resize(img.size, Image.BILINEAR)


def apply_screenshot_aug(img: Image.Image) -> Image.Image:
    if random.random() < 0.65: img = _aug_jpeg(img)
    if random.random() < 0.60: img = _aug_brightness_contrast(img)
    if random.random() < 0.50: img = _aug_blur(img)
    if random.random() < 0.45: img = _aug_noise(img)
    if random.random() < 0.35: img = _aug_shadow(img)
    if random.random() < 0.25: img = _aug_color_tint(img)
    if random.random() < 0.20: img = _aug_chromatic_aberration(img)
    if random.random() < 0.15: img = _aug_moire(img)
    if random.random() < 0.30: img = _aug_paper_texture(img)
    if random.random() < 0.20: img = _aug_screenshot_border(img)
    if random.random() < 0.30: img = _aug_rotation(img)
    if random.random() < 0.25: img = _aug_downscale_upscale(img)
    return img


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_latex_strings(source_dir: Path, min_len: int = 20, max_len: int = 400) -> list[str]:
    strings = []
    for f in sorted(source_dir.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["latex"])
        for s in df["latex"].dropna():
            s = s.strip()
            if min_len <= len(s) <= max_len:
                strings.append(s)
    return strings


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker_init(renderer: str, tex_bin_dir: str):
    global _RENDERER, _TEX_BIN_DIR
    _RENDERER = renderer
    _TEX_BIN_DIR = tex_bin_dir


def _worker(batch: list[tuple[int, str]]) -> list[dict]:
    random.seed()
    np.random.seed()
    records = []
    for idx, latex in batch:
        try:
            img = _render(latex)
            if img is None:
                continue
            img = apply_screenshot_aug(img)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            records.append({
                "idx":    idx,
                "image":  buf.getvalue(),
                "latex":  latex,
                "source": "screenshot_synthetic",
            })
        except Exception:
            pass
    return records


# ── Shard writing ─────────────────────────────────────────────────────────────

def _write_shards(records: list[dict], out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(records)
    n_shards = max(1, math.ceil(n / ROWS_PER_SHARD))
    for i in range(n_shards):
        chunk = records[i * ROWS_PER_SHARD:(i + 1) * ROWS_PER_SHARD]
        fname = f"{prefix}-{str(i).zfill(5)}-of-{str(n_shards).zfill(5)}.parquet"
        table = pa.table({
            "idx":    pa.array([r["idx"]   for r in chunk], type=pa.int64()),
            "image":  pa.array([r["image"] for r in chunk], type=pa.binary()),
            "latex":  pa.array([r["latex"] for r in chunk], type=pa.string()),
            "source": pa.array([r["source"] for r in chunk], type=pa.string()),
        })
        pq.write_table(table, str(out_dir / fname), compression="snappy")
        print(f"  wrote {fname} ({len(chunk):,} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir",    type=str, default="D:/dataset-ocr-builder/latex-ocr-dataset/ocr-data/train/raw")
    ap.add_argument("--out_dir",       type=str, default="D:/dataset-ocr-builder/latex-ocr-dataset/ocr-data/train/screenshot")
    ap.add_argument("--n_samples",     type=int, default=200_000)
    ap.add_argument("--n_workers",     type=int, default=4)
    ap.add_argument("--batch_size",    type=int, default=50)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--prefix",        type=str, default="screenshot_train")
    ap.add_argument("--renderer",      type=str, default="auto",
                    choices=["auto", "pdflatex", "dvipng", "matplotlib"],
                    help="auto: pdflatex > dvipng > matplotlib")
    ap.add_argument("--tex_bin_dir",   type=str, default="",
                    help="TeX Live bin dir, e.g. C:/texlive/2026/bin/windows")
    ap.add_argument("--min_latex_len", type=int, default=20)
    ap.add_argument("--max_latex_len", type=int, default=400)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    tex_bin_dir = args.tex_bin_dir.strip()

    effective = _detect_renderer(tex_bin_dir) if args.renderer == "auto" else args.renderer

    if effective == "pdflatex" and not _has_cmd("pdflatex", extra_path=tex_bin_dir):
        print("ERROR: pdflatex not found. Use --tex_bin_dir C:/texlive/2026/bin/windows")
        return
    if effective == "dvipng" and not _has_cmd("latex", "dvipng", extra_path=tex_bin_dir):
        print("ERROR: latex/dvipng not found. Use --tex_bin_dir C:/texlive/2026/bin/windows")
        return
    if effective == "pdflatex":
        try:
            import pdf2image  # noqa: F401
        except ImportError:
            print("ERROR: pdf2image not installed. Run: pip install pdf2image")
            return

    print(f"Renderer    : {effective}")
    print(f"TeX bin dir : {tex_bin_dir or '(from system PATH)'}")
    print(f"Fonts       : {len(_MATH_FONTS)} math font packages (random per sample)")

    print(f"Loading LaTeX strings from {args.source_dir} ...")
    latex_pool = _load_latex_strings(Path(args.source_dir), args.min_latex_len, args.max_latex_len)
    print(f"  pool size: {len(latex_pool):,} strings")

    if len(latex_pool) < args.n_samples:
        latex_pool = latex_pool * (args.n_samples // len(latex_pool) + 1)

    selected = random.sample(latex_pool, args.n_samples)
    indexed  = list(enumerate(selected))
    batches  = [indexed[i:i + args.batch_size] for i in range(0, len(indexed), args.batch_size)]

    print(f"Rendering {args.n_samples:,} samples in {len(batches)} batches ({args.n_workers} workers) ...")

    all_records: list[dict] = []
    done = 0
    with ProcessPoolExecutor(
        max_workers=args.n_workers,
        initializer=_worker_init,
        initargs=(effective, tex_bin_dir),
    ) as pool:
        futures = {pool.submit(_worker, b): b for b in batches}
        for fut in as_completed(futures):
            all_records.extend(fut.result())
            done += len(futures[fut])
            print(f"  {done:,}/{args.n_samples:,} submitted, {len(all_records):,} rendered ok", end="\r")

    pct = 100 * len(all_records) / max(args.n_samples, 1)
    print(f"\nTotal rendered: {len(all_records):,} / {args.n_samples:,} ({pct:.1f}%)")
    print(f"Writing shards to {args.out_dir} ...")
    _write_shards(all_records, Path(args.out_dir), args.prefix)
    print("Done.")


if __name__ == "__main__":
    main()
