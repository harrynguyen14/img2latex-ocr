import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_DIR       = Path("D:/dataset-ocr-builder")
MATHWRITING_DIR = BASE_DIR / "mathwriting-2024" / "mathwriting-2024"
INKML_NS       = {"ink": "http://www.w3.org/2003/InkML"}

MIN_LABEL_CHARS = 2
MAX_LABEL_CHARS = 512
MIN_IMG_PX      = 16       # bỏ ảnh quá nhỏ (cả width lẫn height)
INK_RENDER_H    = 128      # chiều cao cố định khi render ink (px)
INK_STROKE_W    = 2        # độ dày nét vẽ (px)
INK_PADDING     = 8        # padding xung quanh ink khi render (px)


# ── LaTeX normalization ───────────────────────────────────────────────────────

# strip \begin{align*} ... \end{align*} và các môi trường tương tự (OleehyO)
_ALIGN_WRAP = re.compile(
    r"\\begin\s*\{(align|gather|equation|eqnarray)\*?\}(.*?)\\end\s*\{(align|gather|equation|eqnarray)\*?\}",
    re.DOTALL
)
_MULTI_SPACE = re.compile(r" {2,}")

def normalize_latex(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()

    # strip environment wrappers — lấy nội dung bên trong
    m = _ALIGN_WRAP.search(text)
    if m:
        text = m.group(2).strip()

    # strip & (alignment marker trong align/eqnarray)
    text = re.sub(r"(?<!\\)&", " ", text)

    # strip \\ (line break trong align)
    text = re.sub(r"\\\\", " ", text)

    # strip \nonumber, \label{...}, \tag{...}
    text = re.sub(r"\\nonumber", "", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\tag\{[^}]*\}", "", text)

    # LINXY space-separated: "\ frac { a } { b }" -> "\frac{a}{b}"
    text = re.sub(r"\\ ([a-zA-Z]+)", r"\\\1", text)
    text = re.sub(r"\s*\{\s*", "{", text)
    text = re.sub(r"\s*\}\s*", "}", text)
    text = re.sub(r"\s*([_^])\s*", r"\1", text)

    # strip trailing backslash (artifact from \\ line breaks at end of string)
    text = text.rstrip()
    while text.endswith("\\"):
        text = text[:-1].rstrip()

    # collapse whitespace
    text = _MULTI_SPACE.sub(" ", text).strip()

    return text


def is_valid_label(text: str) -> bool:
    if not text or len(text.strip()) < MIN_LABEL_CHARS:
        return False
    if len(text) > MAX_LABEL_CHARS:
        return False
    return True


# ── Image cleaning ────────────────────────────────────────────────────────────

def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    if img.mode == "L":
        return img.convert("RGB")
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def is_valid_image(img: Image.Image) -> bool:
    return img.width >= MIN_IMG_PX and img.height >= MIN_IMG_PX


def crop_content(img: Image.Image, padding: int = 8) -> Image.Image:
    """
    Crop vùng chứa nội dung, xử lý cả 2 trường hợp:
    - Background trắng, foreground tối  (CROHME dark-on-white)
    - Background đen,  foreground sáng  (CROHME white-on-black)
    """
    img = to_rgb(img)
    gray = np.array(img.convert("L"))
    mean = gray.mean()

    if mean < 128:
        # dark background → foreground là pixel sáng (> threshold)
        threshold = min(50, int(mean) + 20)
        mask = gray > threshold
    else:
        # light background → foreground là pixel tối (< threshold)
        threshold = max(200, int(mean) - 30)
        mask = gray < threshold

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    if (rmax - rmin) < MIN_IMG_PX or (cmax - cmin) < MIN_IMG_PX:
        return img

    rmin = max(0, rmin - padding)
    rmax = min(img.height - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(img.width - 1, cmax + padding)
    cropped = img.crop((cmin, rmin, cmax + 1, rmax + 1))

    # Nếu background đen → invert về trắng (chuẩn hóa về light background)
    if mean < 128:
        cropped_arr = 255 - np.array(cropped)
        cropped = Image.fromarray(cropped_arr.astype(np.uint8))

    return cropped


# ── InkML rendering ───────────────────────────────────────────────────────────

def parse_inkml(filepath: Path):
    tree = ET.parse(filepath)
    root = tree.getroot()
    anns = {
        a.get("type"): a.text
        for a in root.findall("ink:annotation", INKML_NS)
    }
    strokes = []
    for t in root.findall("ink:trace", INKML_NS):
        if not t.text:
            continue
        pts = []
        for p in t.text.split(","):
            parts = p.strip().split()
            if len(parts) >= 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        if pts:
            strokes.append(pts)
    return anns, strokes


def render_ink(strokes: list, target_h: int = INK_RENDER_H,
               padding: int = INK_PADDING,
               stroke_width: int = INK_STROKE_W) -> Image.Image:
    if not strokes:
        return None

    all_x = [p[0] for s in strokes for p in s]
    all_y = [p[1] for s in strokes for p in s]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    ink_w = x_max - x_min
    ink_h = y_max - y_min

    if ink_h < 1:
        ink_h = 1

    # scale để chiều cao = target_h - 2*padding
    scale = (target_h - 2 * padding) / ink_h
    render_w = max(int(ink_w * scale) + 2 * padding, target_h)
    render_h = target_h

    img = Image.new("RGB", (render_w, render_h), (255, 255, 255))
    pixels = img.load()

    def draw_line(x0, y0, x1, y1):
        # Bresenham's line
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            for dw in range(-stroke_width, stroke_width + 1):
                for dh in range(-stroke_width, stroke_width + 1):
                    px, py = x0 + dw, y0 + dh
                    if 0 <= px < render_w and 0 <= py < render_h:
                        pixels[px, py] = (0, 0, 0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    for stroke in strokes:
        scaled = [
            (
                (p[0] - x_min) * scale + padding,
                (p[1] - y_min) * scale + padding,
            )
            for p in stroke
        ]
        for i in range(len(scaled) - 1):
            draw_line(*scaled[i], *scaled[i + 1])

    return img


# ── Per-dataset clean functions ───────────────────────────────────────────────

def clean_linxy_sample(sample: dict) -> dict | None:
    label = normalize_latex(sample["text"])
    if not is_valid_label(label):
        return None
    img = to_rgb(sample["image"])
    if not is_valid_image(img):
        return None
    return {"image": img, "latex": label}


def clean_oleehyo_sample(sample: dict) -> dict | None:
    label = normalize_latex(sample["latex_formula"])
    if not is_valid_label(label):
        return None
    img = to_rgb(sample["image"])
    if not is_valid_image(img):
        return None
    return {"image": img, "latex": label}


def clean_crohme_sample(sample: dict) -> dict | None:
    label = normalize_latex(sample["label"])
    if not is_valid_label(label):
        return None
    img = to_rgb(sample["image"])
    img = crop_content(img, padding=8)
    if not is_valid_image(img):
        return None
    return {"image": img, "latex": label}


def clean_im2latex_sample(sample: dict) -> dict | None:
    label = normalize_latex(sample["formula"])
    if not is_valid_label(label):
        return None
    img = to_rgb(sample["image"])
    if not is_valid_image(img):
        return None
    return {"image": img, "latex": label}


def clean_hme100k_sample(sample: dict) -> dict | None:
    label = ""
    for turn in sample["conversations"]:
        if turn["from"] == "gpt":
            # strip prompt prefix nếu có
            val = turn["value"]
            val = re.sub(r"^.*?LaTeX[:\s]*", "", val, flags=re.IGNORECASE).strip()
            label = normalize_latex(val)
            break
    if not is_valid_label(label):
        return None
    img = to_rgb(sample["image"])
    if not is_valid_image(img):
        return None
    return {"image": img, "latex": label}


def clean_mathwriting_inkml(filepath: Path) -> dict | None:
    try:
        anns, strokes = parse_inkml(filepath)
    except Exception:
        return None

    label = anns.get("normalizedLabel", anns.get("label", ""))
    label = normalize_latex(label)
    if not is_valid_label(label):
        return None

    img = render_ink(strokes)
    if img is None or not is_valid_image(img):
        return None

    return {"image": img, "latex": label}


# ── Parquet writer ────────────────────────────────────────────────────────────

def img_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def write_parquet(records: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images = pa.array([r["image"] for r in records], type=pa.binary())
    labels = pa.array([r["latex"] for r in records], type=pa.string())
    sources = pa.array([r["source"] for r in records], type=pa.string())
    idxs = pa.array(list(range(len(records))), type=pa.int64())
    table = pa.table({"idx": idxs, "image": images, "latex": labels, "source": sources})
    pq.write_table(table, str(out_path), compression="snappy")
    print(f"  saved {len(records):,} rows -> {out_path}")


# ── Per-dataset pipeline ──────────────────────────────────────────────────────

def process_hf_dataset(ds_name: str, hf_id: str, hf_config: str | None,
                        cache_dir: Path, clean_fn, splits: list,
                        out_dir: Path):
    print(f"\n{'='*60}")
    print(f"  {ds_name}")
    print(f"{'='*60}")

    kwargs = {"cache_dir": str(cache_dir)}
    if hf_config:
        ds = load_dataset(hf_id, hf_config, **kwargs)
    else:
        ds = load_dataset(hf_id, **kwargs)

    records = []
    skipped = 0
    for split in splits:
        if split not in ds:
            continue
        data = ds[split]
        for i in tqdm(range(len(data)), desc=f"  [{split}]", ncols=80):
            cleaned = clean_fn(data[i])
            if cleaned is None:
                skipped += 1
                continue
            records.append({
                "image":  img_to_bytes(cleaned["image"]),
                "latex":  cleaned["latex"],
                "source": ds_name,
            })

    print(f"  kept={len(records):,}  skipped={skipped:,}")
    if records:
        write_parquet(records, out_dir / f"{ds_name}.parquet")


def process_mathwriting(out_dir: Path):
    print(f"\n{'='*60}")
    print(f"  mathwriting")
    print(f"{'='*60}")

    inkml_splits = ["train", "synthetic", "valid", "test"]
    records = []
    skipped = 0

    for mw_split in inkml_splits:
        split_dir = MATHWRITING_DIR / mw_split
        files = list(split_dir.glob("*.inkml"))
        for f in tqdm(files, desc=f"  [{mw_split}]", ncols=80):
            cleaned = clean_mathwriting_inkml(f)
            if cleaned is None:
                skipped += 1
                continue
            records.append({
                "image":  img_to_bytes(cleaned["image"]),
                "latex":  cleaned["latex"],
                "source": "mathwriting",
            })

    print(f"  kept={len(records):,}  skipped={skipped:,}")
    if records:
        write_parquet(records, out_dir / "mathwriting.parquet")


# ── Main ──────────────────────────────────────────────────────────────────────

def should_skip(out_dir: Path, ds_name: str) -> bool:
    p = out_dir / f"{ds_name}.parquet"
    if p.exists():
        print(f"  [skip] {ds_name}.parquet already exists")
        return True
    return False


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", nargs="*", default=[],
                    help="Dataset names to re-run even if parquet exists. "
                         "Use --force all to re-run everything.")
    args = ap.parse_args()
    force_all = "all" in args.force
    force_set = set(args.force)

    OUT_DIR = BASE_DIR / "cleaned"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def need_run(name: str) -> bool:
        if force_all or name in force_set:
            return True
        return not should_skip(OUT_DIR, name)

    LINXY_CONFIGS = ["full", "synthetic_handwrite", "human_handwrite"]
    for config in LINXY_CONFIGS:
        name = f"linxy_{config}"
        if need_run(name):
            process_hf_dataset(
                ds_name=name,
                hf_id="linxy/LaTeX_OCR",
                hf_config=config,
                cache_dir=BASE_DIR / "linxy",
                clean_fn=clean_linxy_sample,
                splits=["train", "validation", "test"],
                out_dir=OUT_DIR,
            )

    if need_run("oleehyo"):
        process_hf_dataset(
            ds_name="oleehyo",
            hf_id="OleehyO/latex-formulas",
            hf_config="cleaned_formulas",
            cache_dir=BASE_DIR / "oleehyo",
            clean_fn=clean_oleehyo_sample,
            splits=["train"],
            out_dir=OUT_DIR,
        )

    if need_run("crohme"):
        process_hf_dataset(
            ds_name="crohme",
            hf_id="Neeze/CROHME-full",
            hf_config=None,
            cache_dir=BASE_DIR / "crohme",
            clean_fn=clean_crohme_sample,
            splits=["train", "2014", "2016", "2019"],
            out_dir=OUT_DIR,
        )

    if need_run("im2latex"):
        process_hf_dataset(
            ds_name="im2latex",
            hf_id="yuntian-deng/im2latex-100k",
            hf_config=None,
            cache_dir=BASE_DIR / "im2latex",
            clean_fn=clean_im2latex_sample,
            splits=["train", "val", "test"],
            out_dir=OUT_DIR,
        )

    if need_run("hme100k"):
        process_hf_dataset(
            ds_name="hme100k",
            hf_id="lmms-lab/LLaVA-OneVision-Data",
            hf_config="hme100k",
            cache_dir=BASE_DIR / "hme100k",
            clean_fn=clean_hme100k_sample,
            splits=["train"],
            out_dir=OUT_DIR,
        )

    if need_run("mathwriting"):
        process_mathwriting(OUT_DIR)
