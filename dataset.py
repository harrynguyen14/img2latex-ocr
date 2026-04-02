import io
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import IterableDataset
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
    if h != IMAGE_HEIGHT:
        w = int(w * IMAGE_HEIGHT / h)
        img = img.resize((w, IMAGE_HEIGHT), Image.LANCZOS)
    w = min(img.size[0], MAX_IMAGE_WIDTH)
    w = max((w // PATCH_SIZE) * PATCH_SIZE, PATCH_SIZE)
    if img.size[0] != w:
        img = img.resize((w, IMAGE_HEIGHT), Image.LANCZOS)
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

        p = Path(data_path)
        if p.exists():
            import pyarrow.parquet as pq
            files = sorted(p.glob(f"{split}-*.parquet"))
            ds    = load_dataset(
                "parquet",
                data_files={split: [str(f) for f in files]},
                split=split,
                streaming=True,
            )
            self.num_samples = sum(pq.read_metadata(f).num_rows for f in files)
        else:
            ds = load_dataset(data_path, split=split, streaming=True)
            try:
                from datasets import load_dataset_builder
                builder = load_dataset_builder(data_path)
                builder.download_and_prepare()
                self.num_samples = builder.info.splits[split].num_examples
            except Exception:
                self.num_samples = None

        # Shard tại HF dataset level — mỗi rank chỉ đọc phần của mình
        if world_size > 1:
            self.ds = ds.filter(
                lambda _, idx: idx % world_size == rank,
                with_indices=True,
            )
        else:
            self.ds = ds

    def __iter__(self):
        for sample in self.ds:
            yield _process(sample, self.tokenizer)
