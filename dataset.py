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


class LaTeXDataset(IterableDataset):
    def __init__(self, data_path: str, split: str, tokenizer):
        from datasets import load_dataset
        from pathlib import Path

        self.tokenizer = tokenizer
        self.split = split

        p = Path(data_path)
        if p.exists():
            import pyarrow.parquet as pq
            files = sorted(p.glob(f"{split}-*.parquet"))
            self.ds = load_dataset(
                "parquet",
                data_files={split: [str(f) for f in files]},
                split=split,
                streaming=True,
            )
            self.num_samples = sum(pq.read_metadata(f).num_rows for f in files)
        else:
            self.ds = load_dataset(data_path, split=split, streaming=True)
            try:
                from datasets import load_dataset_builder
                builder = load_dataset_builder(data_path)
                builder.download_and_prepare()
                self.num_samples = builder.info.splits[split].num_examples
            except Exception:
                self.num_samples = None

    def __iter__(self):
        for sample in self.ds:
            img = _resize(_decode_image(sample["image"]).convert("RGB"))
            tensor = _to_tensor(img)

            enc = self.tokenizer(
                sample["label"],
                max_length=MAX_TOKEN_LEN,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            yield {
                "pixel_values": tensor,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }


class IterableDatasetShard(IterableDataset):
    def __init__(self, dataset: IterableDataset, num_processes: int, process_index: int):
        self.dataset = dataset
        self.num_processes = num_processes
        self.process_index = process_index
        self.num_samples = getattr(dataset, "num_samples", None)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id   = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id   = 0
            num_workers = 1

        stride = self.num_processes * num_workers
        start  = self.process_index * num_workers + worker_id

        for i, sample in enumerate(self.dataset):
            if i % stride == start:
                yield sample
