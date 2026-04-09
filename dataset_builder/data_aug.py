"""
data_aug.py — Aug transforms + IO helpers (shared module)
"""

import io
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from PIL import Image, ImageFilter
import kornia.geometry.transform as KGT
import kornia.augmentation as KA

ROWS_PER_SHARD = 50_000


# ── Custom transforms ─────────────────────────────────────────────────────────

class PaperTexture(ImageOnlyTransform):
    def apply(self, img, **params):
        h, w = img.shape[:2]
        texture_type = random.choice(["plain", "lined", "grid"])
        overlay = np.ones((h, w, 3), dtype=np.uint8) * 255
        if texture_type == "lined":
            spacing = random.randint(15, 30)
            color = random.randint(180, 220)
            for y in range(0, h, spacing):
                cv2.line(overlay, (0, y), (w, y), (color, color, 230), 1)
        elif texture_type == "grid":
            spacing = random.randint(20, 40)
            color = random.randint(190, 220)
            for y in range(0, h, spacing):
                cv2.line(overlay, (0, y), (w, y), (color, color, 230), 1)
            for x in range(0, w, spacing):
                cv2.line(overlay, (x, 0), (x, h), (color, color, 230), 1)
        alpha = random.uniform(0.05, 0.20)
        return cv2.addWeighted(img, 1.0, overlay, alpha, 0)

    def get_transform_init_args_names(self):
        return ()


class BackgroundTint(ImageOnlyTransform):
    def apply(self, img, **params):
        tint = random.choice([
            (255, 255, 255),
            (255, 253, 230),
            (240, 240, 240),
            (230, 245, 255),
        ])
        mask = np.all(img > 200, axis=2)
        result = img.copy()
        result[mask] = np.clip(
            img[mask].astype(int) * np.array(tint) / 255, 0, 255
        ).astype(np.uint8)
        return result

    def get_transform_init_args_names(self):
        return ()


class DarkMode(ImageOnlyTransform):
    def apply(self, img, **params):
        return 255 - img

    def get_transform_init_args_names(self):
        return ()


class InkBleed(ImageOnlyTransform):
    def apply(self, img, **params):
        pil = Image.fromarray(img)
        pil = pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        return np.array(pil)

    def get_transform_init_args_names(self):
        return ()


class PencilEffect(ImageOnlyTransform):
    def apply(self, img, **params):
        noise = np.random.normal(0, random.uniform(5, 15), img.shape).astype(np.int16)
        result = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        pil = Image.fromarray(result)
        pil = pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        return np.array(pil)

    def get_transform_init_args_names(self):
        return ()


class FingerShadow(ImageOnlyTransform):
    def apply(self, img, **params):
        h, w = img.shape[:2]
        shadow = np.ones((h, w), dtype=np.float32)
        corner = random.choice(["tl", "tr", "bl", "br"])
        radius = random.randint(w // 4, w // 2)
        cx = 0 if corner[1] == "l" else w
        cy = 0 if corner[0] == "t" else h
        ys, xs = np.ogrid[:h, :w]
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        shadow = np.where(dist < radius, np.maximum(0.5, dist / radius), 1.0)
        return (img * shadow[:, :, np.newaxis]).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ()


class BarrelDistortion(ImageOnlyTransform):
    """Dùng kornia.geometry.transform.barrel_pincushion."""

    def apply(self, img, **params):
        h, w = img.shape[:2]
        k1 = random.uniform(-0.3, -0.05)
        k2 = random.uniform(-0.1,  0.05)
        # kornia nhận BCHW float32 [0,1]
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        try:
            out = KGT.barrel_pincushion(t, k1, k2, k1 * 0.1)
            result = (out.squeeze(0).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        except Exception:
            # fallback: OpenCV undistort
            fx = fy = float(max(w, h))
            cam  = np.array([[fx, 0, w/2.0], [0, fy, h/2.0], [0, 0, 1]], dtype=np.float32)
            dist = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
            result = cv2.undistort(img, cam, dist)
        return result

    def get_transform_init_args_names(self):
        return ()


class ChromaticAberration(ImageOnlyTransform):
    """Dùng kornia.augmentation.RandomChromaticAberration."""

    def apply(self, img, **params):
        # kornia nhận BCHW float32 [0,1]
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        try:
            aug = KA.RandomChromaticAberration(
                fringe=(0.01, 0.03),
                p=1.0,
            )
            out    = aug(t)
            result = (out.squeeze(0).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        except Exception:
            # fallback: numpy channel shift
            shift  = random.randint(1, 3)
            r = np.roll(img[:, :, 0],  shift, axis=1)
            g = img[:, :, 1]
            b = np.roll(img[:, :, 2], -shift, axis=1)
            result = np.stack([r, g, b], axis=2).astype(np.uint8)
        return result

    def get_transform_init_args_names(self):
        return ()


class MoirePattern(ImageOnlyTransform):
    def apply(self, img, **params):
        w = img.shape[1]
        freq      = random.uniform(0.05, 0.15)
        intensity = random.uniform(0.03, 0.08)
        pattern   = (np.sin(2 * np.pi * freq * np.arange(w)) * intensity * 255).astype(np.int16)
        result    = img.astype(np.int16)
        result[:, :, :] += pattern[np.newaxis, :, np.newaxis]
        return np.clip(result, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ()


class StrokeWidthVariation(ImageOnlyTransform):
    def apply(self, img, **params):
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            return cv2.erode(img, kernel, iterations=1)
        return cv2.dilate(img, kernel, iterations=1)

    def get_transform_init_args_names(self):
        return ()


class UnevenLighting(ImageOnlyTransform):
    def apply(self, img, **params):
        h, w   = img.shape[:2]
        direction = random.choice(["lr", "tb", "diag"])
        strength  = random.uniform(0.1, 0.35)
        if direction == "lr":
            grad = np.linspace(1 - strength, 1 + strength, w, dtype=np.float32)
            gradient = np.tile(grad, (h, 1))
        elif direction == "tb":
            grad = np.linspace(1 - strength, 1 + strength, h, dtype=np.float32)
            gradient = np.tile(grad[:, np.newaxis], (1, w))
        else:
            gx = np.linspace(1 - strength, 1.0, w, dtype=np.float32)
            gy = np.linspace(1.0, 1 + strength, h, dtype=np.float32)
            gradient = np.outer(gy, gx)
        return np.clip(img * gradient[:, :, np.newaxis], 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ()


class PartialOcclusion(ImageOnlyTransform):
    def apply(self, img, **params):
        h, w   = img.shape[:2]
        result = img.copy()
        for _ in range(random.randint(1, 2)):
            oh = random.randint(h // 10, h // 4)
            ow = random.randint(w // 10, w // 4)
            y  = random.randint(0, h - oh)
            x  = random.randint(0, w - ow)
            result[y:y+oh, x:x+ow] = random.randint(150, 220)
        return result

    def get_transform_init_args_names(self):
        return ()


# ── Aug pipelines ─────────────────────────────────────────────────────────────

HANDWRITING_SOURCES = {"crohme", "hme100k", "linxy_human_handwrite", "mathwriting"}


def get_light_aug(source: str) -> A.Compose:
    rot = 15 if source in HANDWRITING_SOURCES else 5
    return A.Compose([
        A.Rotate(limit=rot, border_mode=cv2.BORDER_CONSTANT, fill=255, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(std_range=(5.0 / 255, 15.0 / 255), p=0.3),
        A.ImageCompression(quality_range=(75, 95), p=0.3),
        BackgroundTint(p=0.3),
        PaperTexture(p=0.3),
        UnevenLighting(p=0.2),
        InkBleed(p=0.2),
    ])


def get_heavy_aug(source: str) -> A.Compose:
    rot = 15 if source in HANDWRITING_SOURCES else 5
    return A.Compose([
        A.Rotate(limit=rot, border_mode=cv2.BORDER_CONSTANT, fill=255, p=0.6),
        A.Perspective(scale=(0.02, 0.06), p=0.3),
        A.Affine(shear=(-10, 10), p=0.3),
        A.ElasticTransform(alpha=30, sigma=5, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.RandomShadow(p=0.2),
        UnevenLighting(p=0.3),
        A.OneOf([
            A.GaussNoise(std_range=(10.0 / 255, 30.0 / 255)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3)),
        ], p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.Defocus(radius=(1, 3)),
        ], p=0.4),
        A.ImageCompression(quality_range=(50, 90), p=0.4),
        BackgroundTint(p=0.4),
        PaperTexture(p=0.4),
        A.OneOf([DarkMode(), A.ToGray(p=1.0)], p=0.2),
        StrokeWidthVariation(p=0.3),
        InkBleed(p=0.3),
        PencilEffect(p=0.15),
        BarrelDistortion(p=0.2),
        ChromaticAberration(p=0.15),
        MoirePattern(p=0.1),
        FingerShadow(p=0.15),
        PartialOcclusion(p=0.15),
    ])


# ── IO helpers ────────────────────────────────────────────────────────────────

def bytes_to_np(b: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))


def np_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def apply_aug(img_bytes: bytes, aug: A.Compose) -> bytes:
    arr = bytes_to_np(img_bytes)
    try:
        result = aug(image=arr)["image"]
    except Exception:
        result = arr
    return np_to_bytes(result)


def write_shards(records: list[dict], out_dir: Path, prefix: str,
                 rows_per_shard: int = ROWS_PER_SHARD):
    out_dir.mkdir(parents=True, exist_ok=True)
    n        = len(records)
    n_shards = max(1, math.ceil(n / rows_per_shard))

    for idx in range(n_shards):
        start = idx * rows_per_shard
        chunk = records[start : start + rows_per_shard]
        fname = f"{prefix}-{str(idx).zfill(5)}-of-{str(n_shards).zfill(5)}.parquet"
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.table({
            "idx":    pa.array(list(range(start, start + len(chunk))), type=pa.int64()),
            "image":  pa.array([r["image"]  for r in chunk], type=pa.binary()),
            "latex":  pa.array([r["latex"]  for r in chunk], type=pa.string()),
            "source": pa.array([r["source"] for r in chunk], type=pa.string()),
        })
        pq.write_table(table, str(out_dir / fname), compression="snappy")

    print(f"  {prefix}: {n:,} rows -> {n_shards} shards")
