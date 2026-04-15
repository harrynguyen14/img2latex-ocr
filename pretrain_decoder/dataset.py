import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from config import DecoderConfig


def _split_files(path: Path, ratio: float, seed: int, val_files: int = 1) -> tuple[list[Path], list[Path]]:
    files = sorted(path.glob("*.parquet"))
    n_keep = max(1, round(len(files) * ratio))
    sampled = random.Random(seed).sample(files, n_keep)
    return sampled[val_files:], sampled[:val_files]


def _stream_files(files: list[Path], rng: random.Random) -> Iterator[str]:
    for pfile in files:
        table = pq.read_table(str(pfile), columns=["latex"])
        rows  = table["latex"].to_pylist()
        rng.shuffle(rows)
        for val in rows:
            if val and isinstance(val, str) and val.strip():
                yield val.strip()


def _get_file_pools(cfg: DecoderConfig, seed: int) -> tuple[list[Path], list[Path]]:
    splits = {
        "raw":        (Path(cfg.raw_dir),   cfg.raw_ratio),
        "light_text": (Path(cfg.light_dir), cfg.light_ratio),
        "heavy_text": (Path(cfg.heavy_dir), cfg.heavy_ratio),
    }
    train_pool, val_pool = [], []
    for name, (path, ratio) in splits.items():
        tr, va = _split_files(path, ratio, seed)
        train_pool.extend(tr)
        val_pool.extend(va)
    return train_pool, val_pool


class PretrainDataset(IterableDataset):
    def __init__(self, tokenizer: Tokenizer, cfg: DecoderConfig, seed: int = 42, split: str = "train"):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.seed      = seed
        self.split     = split

    def __iter__(self) -> Iterator[dict]:
        rng = random.Random(self.seed)
        train_pool, val_pool = _get_file_pools(self.cfg, self.seed)
        file_pool = train_pool if self.split == "train" else val_pool
        rng.shuffle(file_pool)

        current: list[int] = []
        for text in _stream_files(file_pool, rng):
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
