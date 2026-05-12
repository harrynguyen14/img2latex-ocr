import io
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import IterableDataset


def _resize_to_token_budget(
    img:          Image.Image,
    max_tokens:   int,
    patch_stride: int = 4,
    max_side:     int = 2048,
    min_side:     int = 16,
) -> Image.Image:
    w, h = img.size

    if w > max_side or h > max_side:
        scale = min(max_side / w, max_side / h)
        w = max(int(w * scale), min_side)
        h = max(int(h * scale), min_side)
        img = img.resize((w, h), Image.LANCZOS)

    ph = max(h // patch_stride, 1)
    pw = max(w // patch_stride, 1)
    if ph * pw > max_tokens:
        scale  = (max_tokens / (ph * pw)) ** 0.5
        new_ph = max(int(ph * scale), min_side // patch_stride)
        new_pw = max(int(pw * scale), min_side // patch_stride)
        if new_ph * new_pw > max_tokens:
            if new_ph == min_side // patch_stride:
                new_pw = max_tokens // new_ph
            else:
                new_ph = max_tokens // new_pw
        h = max(new_ph * patch_stride, min_side)
        w = max(new_pw * patch_stride, min_side)
        img = img.resize((w, h), Image.LANCZOS)

    w = max(round(img.size[0] / patch_stride) * patch_stride, min_side)
    h = max(round(img.size[1] / patch_stride) * patch_stride, min_side)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.LANCZOS)
    return img


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32)  # H W C, no copy if already uint8
    t = torch.from_numpy(arr).permute(2, 0, 1).mul_(1.0 / 127.5).sub_(1.0)
    return t


def _aug_jpeg(img: Image.Image, quality_range=(40, 85)) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(*quality_range))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _aug_gaussian_noise(img: Image.Image, std_range=(5, 25)) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, random.uniform(*std_range), arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def _aug_blur(img: Image.Image, radius_range=(0.3, 1.2)) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*radius_range)))


def _aug_brightness_contrast(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.5))


def _aug_dark_mode(img: Image.Image) -> Image.Image:
    return ImageOps.invert(img)


def _aug_color_tint(img: Image.Image) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    tint = np.array([random.uniform(0.85, 1.0) for _ in range(3)], dtype=np.float32)
    return Image.fromarray(np.clip(arr * tint, 0, 255).astype(np.uint8))


def _aug_shadow_gradient(img: Image.Image) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    if random.random() < 0.5:
        grad = np.linspace(random.uniform(0.75, 1.0), 1.0, w, dtype=np.float32)
        arr *= grad[np.newaxis, :, np.newaxis]
    else:
        grad = np.linspace(random.uniform(0.75, 1.0), 1.0, h, dtype=np.float32)
        arr *= grad[:, np.newaxis, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _aug_screenshot_border(img: Image.Image) -> Image.Image:
    pad = random.randint(4, 20)
    bg = (random.randint(200, 255),) * 3
    out = Image.new("RGB", (img.width + 2 * pad, img.height + 2 * pad), bg)
    out.paste(img, (pad, pad))
    return out


def apply_augmentation(img: Image.Image, aug_mode: str = "none") -> Image.Image:
    if aug_mode == "none":
        return img

    if aug_mode == "light":
        if random.random() < 0.5:
            img = _aug_brightness_contrast(img)
        if random.random() < 0.3:
            img = _aug_blur(img, (0.2, 0.7))
        return img

    if aug_mode == "heavy":
        if random.random() < 0.5:
            img = _aug_jpeg(img, (35, 80))
        if random.random() < 0.7:
            img = _aug_brightness_contrast(img)
        if random.random() < 0.5:
            img = _aug_blur(img, (0.3, 1.5))
        if random.random() < 0.4:
            img = _aug_gaussian_noise(img, (5, 25))
        if random.random() < 0.2:
            img = _aug_dark_mode(img)
        if random.random() < 0.3:
            img = _aug_shadow_gradient(img)
        if random.random() < 0.25:
            img = _aug_color_tint(img)
        if random.random() < 0.2:
            img = _aug_screenshot_border(img)
        return img

    return img


def _decode_image(raw) -> Image.Image:
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    if isinstance(raw, dict):
        if raw.get("bytes"):
            return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        if raw.get("path"):
            return Image.open(raw["path"]).convert("RGB")
    raise ValueError(f"Cannot decode image from {type(raw)}")


def _process(sample: dict, tokenizer, args) -> dict:
    pil = _decode_image(sample["image"])
    aug_mode = getattr(args, "aug_mode", "none")
    if aug_mode != "none":
        pil = apply_augmentation(pil, aug_mode)
    # FGE applies 2× conv stride-2, so effective token stride = patch_size × 4
    patch_size = getattr(args, "patch_size", 4)
    effective_stride = patch_size * 4
    pil = _resize_to_token_budget(
        pil,
        max_tokens=getattr(args, "max_visual_tokens", 1024),
        patch_stride=effective_stride,
        max_side=getattr(args, "max_side", 2048),
    )
    w, h = pil.size
    num_tokens = (h // effective_stride) * (w // effective_stride)

    tensor = _to_tensor(pil)
    label = sample.get("latex") or sample.get("label") or ""
    ids = tokenizer.encode(label)
    if len(ids) > args.max_token_len:
        ids = ids[:args.max_token_len]
    pad_id         = tokenizer.pad_token_id
    pad_len        = args.max_token_len - len(ids)
    input_ids      = torch.tensor(ids + [pad_id] * pad_len, dtype=torch.long)
    attention_mask = torch.tensor([1] * len(ids) + [0] * pad_len, dtype=torch.long)
    lab            = input_ids.clone()
    lab[attention_mask == 0] = -100
    return {
        "pixel_values":   tensor,
        "num_tokens":     num_tokens,
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         lab,
    }


class Nav2TexTrainDataset(IterableDataset):
    def __init__(
        self,
        train_dir: str,
        sources: list[str],
        weights: list[float],
        tokenizer,
        args,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.tokenizer  = tokenizer
        self.args       = args
        self.rank       = rank
        self.world_size = world_size
        self.seed       = seed

        self.source_files: dict[str, list[Path]] = {}
        self.weights: dict[str, float] = {}
        for src, w in zip(sources, weights):
            files = sorted((Path(train_dir) / src).glob("*.parquet"))
            if files:
                self.source_files[src] = files
                self.weights[src] = w

    def set_weights(self, new_weights: dict[str, float]) -> None:
        for src in self.source_files.keys():
            if src in new_weights:
                self.weights[src] = float(new_weights[src])

    def _stream_source(self, files: list[Path], rng):
        import pyarrow.parquet as pq
        for pfile in files:
            table = pq.read_table(str(pfile), columns=["image", "latex"])
            indices = list(range(len(table)))
            rng.shuffle(indices)
            images = table["image"].to_pylist()
            latexs = table["latex"].to_pylist()
            for i in indices:
                img_raw = images[i]
                lat = latexs[i]
                if not lat or not isinstance(lat, str) or not lat.strip() or img_raw is None:
                    continue
                yield {"image": img_raw, "latex": lat.strip()}

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id   = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rng = random.Random(self.seed + worker_id + self.rank * 1000)

        iters  = {src: self._stream_source(files, rng) for src, files in self.source_files.items()}
        active = set(iters.keys())
        names  = list(iters.keys())
        global_idx = 0

        while active:
            avail   = [s for s in names if s in active]
            w_avail = [self.weights[s] for s in avail]
            chosen  = rng.choices(avail, weights=w_avail, k=1)[0]
            try:
                sample = next(iters[chosen])
            except StopIteration:
                active.discard(chosen)
                continue

            if global_idx % num_workers == worker_id:
                try:
                    yield _process(sample, self.tokenizer, self.args)
                except Exception:
                    pass
            global_idx += 1


class Nav2TexValDataset(IterableDataset):
    def __init__(
        self,
        val_dir: str,
        tokenizer,
        args,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.files      = sorted(Path(val_dir).glob("*.parquet"))
        self.tokenizer  = tokenizer
        self.args       = args
        self.rank       = rank
        self.world_size = world_size
        self.seed       = seed

    def __iter__(self):
        import pyarrow.parquet as pq
        worker_info = torch.utils.data.get_worker_info()
        worker_id   = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rng = random.Random(self.seed + worker_id + self.rank * 1000)

        global_idx = 0
        for pfile in self.files:
            table = pq.read_table(str(pfile), columns=["image", "latex"])
            indices = list(range(len(table)))
            rng.shuffle(indices)
            images = table["image"].to_pylist()
            latexs = table["latex"].to_pylist()
            for i in indices:
                if global_idx % (num_workers * self.world_size) == (worker_id * self.world_size + self.rank):
                    img_raw = images[i]
                    lat = latexs[i]
                    if not lat or not isinstance(lat, str) or not lat.strip() or img_raw is None:
                        global_idx += 1
                        continue
                    try:
                        yield _process({"image": img_raw, "latex": lat.strip()}, self.tokenizer, self.args)
                    except Exception:
                        pass
                global_idx += 1
