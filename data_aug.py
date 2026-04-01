import argparse
import random
import io
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from datasets import load_dataset, concatenate_datasets

from utils import is_valid_sample, strip_align, resize_image

MAX_TOKEN_LEN = 200
CLEAN_RATIO = 0.35

LINXY_CONFIGS = ["full", "synthetic_handwrite", "human_handwrite"]
OLEEHYO_CONFIG = "cleaned_formulas"


def load_and_filter():
    all_train, all_val, all_test = [], [], []

    for cfg_name in tqdm(LINXY_CONFIGS, desc="Loading linxy configs"):
        ds = load_dataset("linxy/LaTeX_OCR", cfg_name, trust_remote_code=True)
        for split_name, split_data in ds.items():
            filtered = split_data.filter(
                lambda x: is_valid_sample(x["text"], MAX_TOKEN_LEN),
                desc=f"Filtering linxy/{cfg_name}/{split_name}",
            )
            renamed = filtered.select_columns(["image", "text"]).rename_column("text", "label")
            if split_name == "train":
                all_train.append(renamed)
            elif split_name == "validation":
                all_val.append(renamed)
            elif split_name == "test":
                all_test.append(renamed)

    print("Loading OleehyO/latex-formulas ...")
    ds_olee = load_dataset("OleehyO/latex-formulas", OLEEHYO_CONFIG, trust_remote_code=True)
    for split_name, split_data in ds_olee.items():
        filtered = split_data.map(
            lambda x: {"label": strip_align(x["latex_formula"])},
            desc=f"Processing OleehyO/{split_name}",
        )
        filtered = filtered.filter(
            lambda x: is_valid_sample(x["label"], MAX_TOKEN_LEN),
            desc=f"Filtering OleehyO/{split_name}",
        )
        filtered = filtered.select_columns(["image", "label"])
        if split_name == "train":
            all_train.append(filtered)

    print("Concatenating and shuffling ...")
    train_ds = concatenate_datasets(all_train).shuffle(seed=42)
    val_ds = concatenate_datasets(all_val) if all_val else None
    test_ds = concatenate_datasets(all_test) if all_test else None

    return train_ds, val_ds, test_ds


def aug_jpeg_compression(img):
    quality = random.randint(30, 75)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def aug_low_resolution(img):
    scale = random.uniform(0.2, 0.6)
    w, h = img.size
    sw, sh = max(16, int(w * scale)), max(8, int(h * scale))
    return img.resize((sw, sh), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def aug_gaussian_noise(img):
    std = random.uniform(5, 25)
    arr = np.array(img).astype(np.float32)
    arr = np.clip(arr + np.random.normal(0, std, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def aug_salt_pepper(img):
    amount = random.uniform(0.01, 0.05)
    arr = np.array(img).copy()
    n = int(arr.size * amount)
    coords = [np.random.randint(0, i, n // 2) for i in arr.shape[:2]]
    arr[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i, n // 2) for i in arr.shape[:2]]
    arr[coords[0], coords[1]] = 0
    return Image.fromarray(arr)


def aug_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))


def aug_background(img):
    arr = np.array(img.convert("RGBA")).astype(np.float32)
    bg = np.full_like(arr, [random.randint(230, 255), random.randint(220, 255), random.randint(180, 240), 255], dtype=np.float32)
    alpha = arr[:, :, 3:4] / 255.0
    blended = arr[:, :, :3] * alpha + bg[:, :, :3] * (1 - alpha)
    return Image.fromarray(blended.astype(np.uint8), mode="RGB")


def aug_rotation(img):
    return img.rotate(random.uniform(-3, 3), expand=False, fillcolor=(255, 255, 255))


def aug_perspective(img):
    tensor = TF.to_tensor(img)
    _, h, w = tensor.shape
    d = random.uniform(0.02, 0.06)
    dx, dy = int(w * d), int(h * d)
    sp = [[0, 0], [w, 0], [w, h], [0, h]]
    ep = [
        [random.randint(0, dx), random.randint(0, dy)],
        [random.randint(w - dx, w), random.randint(0, dy)],
        [random.randint(w - dx, w), random.randint(h - dy, h)],
        [random.randint(0, dx), random.randint(h - dy, h)],
    ]
    tensor = TF.perspective(tensor, sp, ep, interpolation=T.InterpolationMode.BILINEAR)
    return TF.to_pil_image(tensor)


def aug_random_erase(img):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(random.randint(1, 3)):
        rw = random.randint(int(w * 0.03), int(w * 0.12))
        rh = random.randint(int(h * 0.1), int(h * 0.4))
        rx = random.randint(0, w - rw)
        ry = random.randint(0, h - rh)
        draw.rectangle([rx, ry, rx + rw, ry + rh], fill=random.choice([(255, 255, 255), (0, 0, 0), (200, 200, 200)]))
    return img


def aug_shadow(img):
    img = img.copy().convert("RGB")
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    sw = random.randint(int(w * 0.05), int(w * 0.15))
    alpha = np.linspace(0.6, 1.0, sw)
    side = random.choice(["left", "right", "top", "bottom"])
    if side == "left":
        arr[:, :sw] *= alpha[np.newaxis, :, np.newaxis]
    elif side == "right":
        arr[:, -sw:] *= alpha[::-1][np.newaxis, :, np.newaxis]
    elif side == "top":
        arr[:sw, :] *= alpha[:, np.newaxis, np.newaxis]
    else:
        arr[-sw:, :] *= alpha[::-1][:, np.newaxis, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


LIGHT_AUGS = [
    (aug_blur, 0.3),
    (aug_rotation, 0.3),
    (aug_background, 0.4),
    (aug_shadow, 0.2),
    (aug_low_resolution, 0.2),
]

HEAVY_AUGS = [
    (aug_jpeg_compression, 0.4),
    (aug_low_resolution, 0.4),
    (aug_gaussian_noise, 0.35),
    (aug_salt_pepper, 0.2),
    (aug_blur, 0.3),
    (aug_background, 0.4),
    (aug_rotation, 0.35),
    (aug_perspective, 0.25),
    (aug_random_erase, 0.3),
    (aug_shadow, 0.25),
]


def apply_augmentation(img: Image.Image, stage: int = 1) -> Image.Image:
    if random.random() < CLEAN_RATIO:
        return img.convert("RGB")
    img = img.convert("RGB")
    for aug_fn, prob in (LIGHT_AUGS if stage == 1 else HEAVY_AUGS):
        if random.random() < prob:
            try:
                img = aug_fn(img)
            except Exception:
                pass
    return img


def process_one(args):
    idx, sample, out_dir, stage = args
    img = sample["image"].convert("RGB")
    label = sample["label"]
    aug_img = apply_augmentation(img, stage=stage)
    aug_img = resize_image(aug_img)
    img_path = out_dir / f"{idx:08d}.png"
    aug_img.save(img_path, format="PNG")
    return {"img_path": str(img_path), "label": label}


def run_augmentation(ds, out_dir: Path, stage: int, num_workers: int = 4):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Augmenting {len(ds)} samples (stage={stage}) -> {out_dir}")

    manifest = []
    tasks = [(i, ds[i], out_dir, stage) for i in range(len(ds))]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Stage {stage}"):
            manifest.append(future.result())

    manifest.sort(key=lambda x: x["img_path"])
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Manifest: {manifest_path}")
    return manifest_path


def save_val_test(val_ds, test_ds, out_dir: Path):
    for split_name, ds in [("val", val_ds), ("test", test_ds)]:
        if ds is None:
            continue
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for i in tqdm(range(len(ds)), desc=f"Saving {split_name}"):
            sample = ds[i]
            img = sample["image"].convert("RGB")
            img = resize_image(img)
            img_path = split_dir / f"{i:08d}.png"
            img.save(img_path, format="PNG")
            manifest.append({"img_path": str(img_path), "label": sample["label"]})
        with open(split_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"Saved {split_name}: {split_dir / 'manifest.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download, filter, augment và lưu ra disk")
    parser.add_argument("--out_dir", type=str, default="data/augmented")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)

    print("Downloading and filtering datasets...")
    train_ds, val_ds, test_ds = load_and_filter()

    for stage in [1, 2]:
        run_augmentation(train_ds, out_dir / f"stage{stage}", stage, args.num_workers)

    save_val_test(val_ds, test_ds, out_dir)
