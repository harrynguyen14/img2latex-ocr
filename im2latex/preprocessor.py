import io
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import IterableDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from pretrain_decoder.tokenizer import load_tokenizer as _load_tokenizer


def get_tokenizer(tokenizer_dir: str):
    return _load_tokenizer(tokenizer_dir)


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
    if getattr(args, "resize_in_dataset", True):
        img = _resize(pil, args.image_height, args.max_image_width, args.patch_size)
    else:
        img = _pad_to_patch_grid(
            pil, args.patch_size, args.max_image_width,
            getattr(args, "max_image_height", args.image_height),
        )
    tensor = _to_tensor(img)
    label = sample.get("latex") or sample.get("label") or ""
    ids = tokenizer.encode(label).ids
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


class LaTeXOCRHFDataset(IterableDataset):
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


class LaTeXOCRDiskDataset(IterableDataset):
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


class LaTeXOCRFlatParquetDataset(IterableDataset):
    """Stream image+latex from a flat directory of parquet files (e.g. validation/).

    No subfolders, no weighted interleaving — just stream all files sequentially.
    """

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
        worker_info = torch.utils.data.get_worker_info()
        worker_id   = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rng = random.Random(self.seed + worker_id + self.rank * 1000)

        import pyarrow.parquet as pq
        global_idx = 0
        for pfile in self.files:
            table   = pq.read_table(str(pfile), columns=["image", "latex"])
            indices = list(range(len(table)))
            rng.shuffle(indices)
            images = table["image"].to_pylist()
            latexs = table["latex"].to_pylist()
            for i in indices:
                if global_idx % (num_workers * self.world_size) == (worker_id * self.world_size + self.rank):
                    img_raw = images[i]
                    lat     = latexs[i]
                    if not lat or not isinstance(lat, str) or not lat.strip() or img_raw is None:
                        global_idx += 1
                        continue
                    try:
                        yield _process({"image": img_raw, "latex": lat.strip()}, self.tokenizer, self.args)
                    except Exception:
                        pass
                global_idx += 1


class LaTeXOCRParquetDataset(IterableDataset):
    """Stream image+latex pairs directly from local parquet files.

    Expects layout: data_dir/{src}/*.parquet  (harryrobert/ocr-latex-filter format)
    Columns: image (bytes), latex (str), source, idx
    """

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
        import random
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

    def _stream_source(self, files: list[Path], rng) -> "Iterator[dict]":
        import random
        for pfile in files:
            import pyarrow.parquet as pq
            table = pq.read_table(str(pfile), columns=["image", "latex"])
            indices = list(range(len(table)))
            rng.shuffle(indices)
            images = table["image"].to_pylist()
            latexs = table["latex"].to_pylist()
            for i in indices:
                img_raw = images[i]
                lat     = latexs[i]
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
        ws     = [self.weights[s] for s in names]

        global_idx = 0
        while active:
            avail = [s for s in names if s in active]
            if not avail:
                break
            w_avail = [self.weights[s] for s in avail]
            chosen  = rng.choices(avail, weights=w_avail, k=1)[0]
            try:
                sample = next(iters[chosen])
            except StopIteration:
                active.discard(chosen)
                continue

            # shard across workers and DDP ranks
            if global_idx % (num_workers * self.world_size) == (worker_id * self.world_size + self.rank):
                try:
                    yield _process(sample, self.tokenizer, self.args)
                except Exception:
                    pass
            global_idx += 1
