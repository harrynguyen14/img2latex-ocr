import io
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from transformers import AutoTokenizer

from constants import (
    IMAGE_HEIGHT, MAX_IMAGE_WIDTH, PATCH_SIZE,
    MAX_TOKEN_LEN, TOKENIZER_NAME,
)


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _resize(img: Image.Image) -> Image.Image:
    w, h = img.size
    # Compute target width in one step: scale by height ratio, then snap to patch grid
    new_w = int(w * IMAGE_HEIGHT / h) if h != IMAGE_HEIGHT else w
    new_w = max((min(new_w, MAX_IMAGE_WIDTH) // PATCH_SIZE) * PATCH_SIZE, PATCH_SIZE)
    if (w, h) != (new_w, IMAGE_HEIGHT):
        img = img.resize((new_w, IMAGE_HEIGHT), Image.BILINEAR)
    return img


def _to_tensor(img: Image.Image) -> torch.Tensor:
    t = TF.to_tensor(img)
    return TF.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def _decode_image(raw) -> Image.Image:
    if isinstance(raw, dict):
        return Image.open(io.BytesIO(raw["bytes"]))
    return raw


def _process(sample: dict, tokenizer) -> dict:
    img    = _resize(_decode_image(sample["image"]).convert("RGB"))
    tensor = _to_tensor(img)

    enc = tokenizer(
        sample["label"],
        max_length=MAX_TOKEN_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)
    labels         = input_ids.clone()
    labels[attention_mask == 0] = -100

    return {
        "pixel_values":   tensor,
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


class LaTeXDataset(IterableDataset):
    def __init__(self, data_path: str, split: str, tokenizer,
                 rank: int = 0, world_size: int = 1):
        from datasets import load_dataset
        from pathlib import Path

        self.tokenizer  = tokenizer
        self.split      = split
        self.rank       = rank
        self.world_size = world_size
        self.data_path  = data_path
        self.streaming_ds = None  # fallback for non-parquet paths

        p = Path(data_path)
        if p.exists():
            import pyarrow.parquet as pq
            all_files = sorted(p.glob(f"{split}-*.parquet"))
            self.num_samples = sum(pq.read_metadata(f).num_rows for f in all_files)

            # File-level DDP sharding: each rank owns a subset of parquet files
            self.files = [f for i, f in enumerate(all_files) if i % world_size == rank]
            if not self.files:
                # Fewer files than ranks — fall back to row-level filter on all files
                self.files = all_files
                self._row_filter = True
            else:
                self._row_filter = False
        else:
            self.files = None
            ds = load_dataset(data_path, split=split, streaming=True)
            if world_size > 1:
                ds = ds.filter(
                    lambda _, idx: idx % world_size == rank,
                    with_indices=True,
                )
            self.streaming_ds = ds
            try:
                from datasets import load_dataset_builder
                builder = load_dataset_builder(data_path)
                builder.download_and_prepare()
                self.num_samples = builder.info.splits[split].num_examples
            except Exception:
                self.num_samples = None

    def _iter_files(self, files):
        """Iterate over a list of parquet files, further sharding among DataLoader workers."""
        from datasets import load_dataset

        worker_info = get_worker_info()
        if worker_info is not None:
            # Split assigned files across DataLoader workers
            worker_files = [f for i, f in enumerate(files)
                            if i % worker_info.num_workers == worker_info.id]
        else:
            worker_files = files

        if not worker_files:
            return

        ds = load_dataset(
            "parquet",
            data_files={self.split: [str(f) for f in worker_files]},
            split=self.split,
            streaming=True,
        )
        for sample in ds:
            yield _process(sample, self.tokenizer)

    def __iter__(self):
        if self.files is not None:
            yield from self._iter_files(self.files)
        else:
            # HF Hub streaming fallback — workers get same stream, no further splitting
            for sample in self.streaming_ds:
                yield _process(sample, self.tokenizer)
