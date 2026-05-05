import io
import json
import random
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import IterableDataset
from huggingface_hub import hf_hub_download

def get_tokenizer(repo_id: str):
    path = hf_hub_download(repo_id=repo_id, filename="tokenizer/tokenizer.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return LaTeXTokenizerV2(
        token2id=data["token2id"],
        id2token={int(k): v for k, v in data["id2token"].items()},
        merges=[tuple(m) for m in data["merges"]],
    )


def _resize(img: Image.Image, image_height: int, max_image_width: int, patch_size: int) -> Image.Image:
    w, h = img.size
    new_w = int(round(w * image_height / max(h, 1)))
    new_w = min(new_w, max_image_width)
    new_w = max((new_w // patch_size) * patch_size, patch_size)
    if (w, h) != (new_w, image_height):
        img = img.resize((new_w, image_height), Image.BILINEAR)
    return img


def _pad_to_patch_grid(img: Image.Image, patch_size: int, max_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    w = min(w, max_w)
    h = min(h, max_h)
    if w < img.size[0] or h < img.size[1]:
        img = img.crop((0, 0, w, h))
    tw = min((w + patch_size - 1) // patch_size * patch_size, max_w)
    th = min((h + patch_size - 1) // patch_size * patch_size, max_h)
    tw = max(tw, patch_size)
    th = max(th, patch_size)
    if tw == w and th == h:
        return img
    out = Image.new("RGB", (tw, th), (255, 255, 255))
    out.paste(img, (0, 0))
    return out


def _to_tensor(img: Image.Image) -> torch.Tensor:
    t = TF.to_tensor(img)
    return TF.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def _aug_jpeg(img: Image.Image, quality_range=(40, 85)) -> Image.Image:
    buf = io.BytesIO()
    q = random.randint(*quality_range)
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _aug_gaussian_noise(img: Image.Image, std_range=(5, 25)) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    std = random.uniform(*std_range)
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _aug_blur(img: Image.Image, radius_range=(0.3, 1.2)) -> Image.Image:
    r = random.uniform(*radius_range)
    return img.filter(ImageFilter.GaussianBlur(radius=r))


def _aug_brightness_contrast(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.5))
    return img


def _aug_dark_mode(img: Image.Image) -> Image.Image:
    return ImageOps.invert(img)


def _aug_color_tint(img: Image.Image) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    tint = np.array([random.uniform(0.85, 1.0) for _ in range(3)], dtype=np.float32)
    arr = np.clip(arr * tint, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


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
    """
    aug_mode: 'none' | 'light' | 'heavy' | 'screenshot'
    screenshot simulates real-world screenshot/camera domain
    """
    if aug_mode == "none":
        return img

    if aug_mode == "light":
        if random.random() < 0.5:
            img = _aug_brightness_contrast(img)
        if random.random() < 0.3:
            img = _aug_blur(img, (0.2, 0.7))
        return img

    if aug_mode == "heavy":
        if random.random() < 0.7:
            img = _aug_brightness_contrast(img)
        if random.random() < 0.4:
            img = _aug_blur(img, (0.3, 1.0))
        if random.random() < 0.3:
            img = _aug_gaussian_noise(img, (3, 15))
        if random.random() < 0.15:
            img = _aug_dark_mode(img)
        if random.random() < 0.2:
            img = _aug_color_tint(img)
        return img

    if aug_mode == "screenshot":
        # simulate screenshots, camera shots, PDF crops
        if random.random() < 0.5:
            img = _aug_jpeg(img, (35, 80))
        if random.random() < 0.6:
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
    if getattr(args, "resize_in_dataset", True):
        img = _resize(pil, args.image_height, args.max_image_width, args.patch_size)
    else:
        img = _pad_to_patch_grid(
            pil, args.patch_size, args.max_image_width,
            getattr(args, "max_image_height", args.image_height),
        )
    tensor = _to_tensor(img)
    label = sample.get("latex") or sample.get("label") or ""
    ids = tokenizer.encode(label)
    if len(ids) > args.max_token_len:
        ids = ids[:args.max_token_len]
    pad_len        = args.max_token_len - len(ids)
    input_ids      = torch.tensor(ids + [0] * pad_len, dtype=torch.long)
    attention_mask = torch.tensor([1] * len(ids) + [0] * pad_len, dtype=torch.long)
    lab            = input_ids.clone()
    lab[attention_mask == 0] = -100
    return {
        "pixel_values":   tensor,
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         lab,
    }


class Nav2TexHFDataset(IterableDataset):
    def __init__(self, dataset_id: str, split: str, tokenizer, args,
                 rank: int = 0, world_size: int = 1,
                 names: list[str] | None = None,
                 weights: list[float] | None = None,
                 seed: int = 42):
        from datasets import load_dataset, interleave_datasets
        self.tokenizer = tokenizer
        self.args      = args
        self.rank      = rank
        self.world_size = world_size
        self.seed      = seed

        if names and split == "train":
            subsets = []
            for name in names:
                ds = load_dataset(
                    dataset_id,
                    data_files={"train": f"train/{name}/*.parquet"},
                    split="train",
                    streaming=True,
                )
                subsets.append(ds)
            probs = None
            if weights:
                total = sum(weights)
                probs = [w / total for w in weights]
            self.ds = interleave_datasets(subsets, probabilities=probs, seed=seed)
        else:
            self.ds = load_dataset(dataset_id, split=split, streaming=True)

        if world_size > 1:
            self.ds = self.ds.filter(lambda _, idx: idx % world_size == rank, with_indices=True)

    def __iter__(self):
        for sample in self.ds:
            try:
                yield _process(sample, self.tokenizer, self.args)
            except Exception:
                pass


class Nav2TexDiskDataset(IterableDataset):
    def __init__(self, cache_path: str, tokenizer, args, rank: int = 0, world_size: int = 1):
        from datasets import load_from_disk
        self.ds          = load_from_disk(cache_path)
        self.tokenizer   = tokenizer
        self.args        = args
        self.rank        = rank
        self.world_size  = world_size
        self.num_samples = len(self.ds) // world_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for i, sample in enumerate(self.ds):
            if i % self.world_size == self.rank:
                yield _process(sample, self.tokenizer, self.args)


class Nav2TexFlatParquetDataset(IterableDataset):
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
        import random
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


class Nav2TexParquetDataset(IterableDataset):

    def __init__(
        self,
        data_dir: str,
        sources: list[str],
        weights: list[float],
        tokenizer,
        args,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.data_dir   = Path(data_dir)
        self.tokenizer  = tokenizer
        self.args       = args
        self.rank       = rank
        self.world_size = world_size
        self.seed       = seed

        self.source_files: dict[str, list[Path]] = {}
        self.weights: dict[str, float] = {}
        for src, w in zip(sources, weights):
            files = sorted((self.data_dir / src).glob("*.parquet"))
            if files:
                self.source_files[src] = files
                self.weights[src] = w

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
                if not lat or not isinstance(lat, str) or not lat.strip():
                    continue
                if img_raw is None:
                    continue
                yield {"image": img_raw, "latex": lat.strip()}

    def __iter__(self):
        import random
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
        
