import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
    patch_size = getattr(args, "patch_size", 4)
    effective_stride = patch_size
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

        shard_files: dict[str, list[Path]] = {
            src: [f for i, f in enumerate(files) if i % num_workers == worker_id]
            for src, files in self.source_files.items()
        }
        shard_files = {src: files for src, files in shard_files.items() if files}

        iters  = {src: self._stream_source(files, rng) for src, files in shard_files.items()}
        active = set(iters.keys())
        names  = list(iters.keys())

        while active:
            avail   = [s for s in names if s in active]
            w_avail = [self.weights[s] for s in avail]
            chosen  = rng.choices(avail, weights=w_avail, k=1)[0]
            try:
                sample = next(iters[chosen])
            except StopIteration:
                active.discard(chosen)
                continue

            try:
                yield _process(sample, self.tokenizer, self.args)
            except Exception:
                pass


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