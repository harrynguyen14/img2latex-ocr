import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "tokenizer_v2"))
from tokenizer_v2 import LaTeXTokenizerV2 as LaTeXTokenizer

from config import DecoderConfig


def _split_files(path: Path, ratio: float, seed: int, val_files: int = 3) -> tuple[list[Path], list[Path]]:
    files = sorted(path.glob("*.parquet"))
    n_keep = max(1, round(len(files) * ratio))
    sampled = random.Random(seed).sample(files, n_keep)
    return sampled[val_files:], sampled[:val_files]


def _stream_parquet(files: list[Path], rng: random.Random) -> Iterator[str]:
    for pfile in files:
        table = pq.read_table(str(pfile), columns=["latex"])
        rows  = table["latex"].to_pylist()
        rng.shuffle(rows)
        for val in rows:
            if val and isinstance(val, str) and val.strip():
                yield val.strip()


def _get_pools(cfg: DecoderConfig, seed: int) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    sources = {
        "raw":        (Path(cfg.data_dir) / "train" / "raw",        cfg.raw_ratio),
        "light_text": (Path(cfg.data_dir) / "train" / "light_text", cfg.light_ratio),
        "heavy_text": (Path(cfg.data_dir) / "train" / "heavy_text", cfg.heavy_ratio),
    }
    train_pools, val_pools = {}, {}
    for name, (path, ratio) in sources.items():
        tr, va = _split_files(path, ratio, seed)
        train_pools[name] = tr
        val_pools[name]   = va
    return train_pools, val_pools


def _interleaved_stream(
    pools: dict[str, list[Path]],
    weights: dict[str, float],
    rng: random.Random,
) -> Iterator[str]:
    iters  = {name: _stream_parquet(files, rng) for name, files in pools.items()}
    active = set(pools.keys())

    while active:
        available = [s for s in pools if s in active]
        if not available:
            break
        w_active = [weights[s] for s in available]
        chosen   = rng.choices(available, weights=w_active, k=1)[0]
        try:
            yield next(iters[chosen])
        except StopIteration:
            active.discard(chosen)


class PretrainDataset(IterableDataset):
    def __init__(self, tokenizer: LaTeXTokenizer, cfg: DecoderConfig, seed: int = 42, split: str = "train"):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.seed      = seed
        self.split     = split

    def __iter__(self) -> Iterator[dict]:
        rng = random.Random(self.seed)
        train_pools, val_pools = _get_pools(self.cfg, self.seed)

        if self.split == "train":
            weights = {
                "raw":        self.cfg.raw_weight,
                "light_text": self.cfg.light_weight,
                "heavy_text": self.cfg.heavy_weight,
            }
            stream = _interleaved_stream(train_pools, weights, rng)
        elif self.split == "val_raw":
            stream = _stream_parquet(val_pools["raw"], rng)
        elif self.split == "val_light":
            stream = _stream_parquet(val_pools["light_text"], rng)
        elif self.split == "val_heavy":
            stream = _stream_parquet(val_pools["heavy_text"], rng)
        else:
            all_val = [f for files in val_pools.values() for f in files]
            stream  = _stream_parquet(all_val, rng)

        current: list[int] = []
        for text in stream:
            ids = self.tokenizer.encode(text)
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
