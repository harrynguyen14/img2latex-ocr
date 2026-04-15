import random
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer

from config import DecoderConfig


HF_REPO = "harryrobert/latex-ocr-aug"

SOURCE_VALUES = {
    "raw":        ["raw"],
    "light_text": ["light_text"],
    "heavy_text": ["heavy_text"],
}

VAL_SOURCE_MAP = {
    "val":       None,
    "val_raw":   "raw",
    "val_light": "light_text",
    "val_heavy": "heavy_text",
}


def _hf_stream(hf_split: str, seed: int, source_filter: list[str] | None = None) -> Iterator[str]:
    ds = load_dataset(HF_REPO, split=hf_split, streaming=True, trust_remote_code=True)
    if source_filter:
        ds = ds.filter(lambda row: row["source"] in source_filter)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for row in ds:
        val = row.get("latex") or ""
        if val and isinstance(val, str) and val.strip():
            yield val.strip()


def _interleaved_stream(
    weights: dict[str, float],
    hf_split: str,
    rng: random.Random,
    seed: int,
) -> Iterator[str]:
    iters = {
        name: _hf_stream(hf_split, seed, SOURCE_VALUES[name])
        for name in weights
    }
    active = set(weights.keys())

    while active:
        available = [s for s in weights if s in active]
        if not available:
            break
        w_active = [weights[s] for s in available]
        chosen   = rng.choices(available, weights=w_active, k=1)[0]
        try:
            yield next(iters[chosen])
        except StopIteration:
            active.discard(chosen)


class PretrainDataset(IterableDataset):
    def __init__(self, tokenizer: Tokenizer, cfg: DecoderConfig, seed: int = 42, split: str = "train"):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.seed      = seed
        self.split     = split

    def __iter__(self) -> Iterator[dict]:
        rng = random.Random(self.seed)

        if self.split == "train":
            weights = {
                "raw":        self.cfg.raw_ratio,
                "light_text": self.cfg.light_ratio,
                "heavy_text": self.cfg.heavy_ratio,
            }
            stream = _interleaved_stream(weights, "train", rng, self.seed)
        else:
            source_key = VAL_SOURCE_MAP.get(self.split)
            source_filter = SOURCE_VALUES[source_key] if source_key else None
            stream = _hf_stream("validation", self.seed + 1, source_filter)

        current: list[int] = []
        for text in stream:
            ids = self.tokenizer.encode(text).ids
            if len(ids) > self.cfg.max_seq_len:
                ids = ids[:self.cfg.max_seq_len]
            if not ids:
                continue

            if len(current) + len(ids) <= self.cfg.max_seq_len:
                current.extend(ids)
            else:
                if current:
                    padded    = current + [self.cfg.pad_id] * (self.cfg.max_seq_len - len(current))
                    input_ids = torch.tensor(padded, dtype=torch.long)
                    yield {"input_ids": input_ids, "attention_mask": input_ids != self.cfg.pad_id}
                current = ids

        if current:
            padded    = current + [self.cfg.pad_id] * (self.cfg.max_seq_len - len(current))
            input_ids = torch.tensor(padded, dtype=torch.long)
            yield {"input_ids": input_ids, "attention_mask": input_ids != self.cfg.pad_id}


def build_dataloader(
    dataset: PretrainDataset,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        generator=g,
        drop_last=True,
    )
