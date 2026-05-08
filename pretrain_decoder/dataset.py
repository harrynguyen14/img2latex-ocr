import glob
import random
import re
from typing import Iterator

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pyarrow.parquet as pq
from transformers import NougatTokenizerFast

from normalize import normalize

_CPE_PATTERNS = re.compile(
    r'\\(frac|int|sum|prod|matrix|pmatrix|bmatrix|cases|align|begin|sqrt'
    r'|underbrace|overbrace|overset|underset|substack|bigoplus|bigotimes'
    r'|lim|sup|inf|max|min)\b'
)

def _complexity_score(s: str) -> float:
    char_len   = len(s)
    cmd_count  = len(_CPE_PATTERNS.findall(s))
    nest_depth = s.count('{')
    return char_len + cmd_count * 15 + nest_depth * 10


class LaTeXDataset(Dataset):
    def __init__(self, config):
        self.config    = config
        self.tokenizer = NougatTokenizerFast.from_pretrained(config.tokenizer_dir)

        files = sorted(glob.glob(config.data_glob))
        if not files:
            raise FileNotFoundError(f"No parquet files found: {config.data_glob}")

        rows = []
        for f in files:
            table = pq.read_table(f, columns=["latex"])
            rows.extend(table["latex"].to_pylist())

        self.samples = [r for r in rows if r and isinstance(r, str) and r.strip()]
        self._scores = [_complexity_score(s) for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        text = normalize(self.samples[idx])
        ids  = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)

        max_tokens = self.config.max_seq_len - 1
        ids = ids[:max_tokens]

        input_ids = [self.config.bos_token_id] + ids
        labels    = ids + [self.config.eos_token_id]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
            "true_len":  torch.tensor(len(ids),  dtype=torch.float),
        }

    def normal_indices(self) -> list[int]:
        t = self.config.cpe_score_threshold
        return [i for i, sc in enumerate(self._scores) if sc <= t]

    def cpe_indices(self) -> list[int]:
        t = self.config.cpe_score_threshold
        return [i for i, sc in enumerate(self._scores) if sc > t]

    def score_stats(self) -> dict:
        import statistics
        scores = self._scores
        thresh = self.config.cpe_score_threshold
        n_cpe  = sum(1 for sc in scores if sc > thresh)
        return {
            "total":   len(scores),
            "n_cpe":   n_cpe,
            "n_spe":   len(scores) - n_cpe,
            "cpe_pct": round(n_cpe / len(scores) * 100, 2),
            "median":  round(statistics.median(scores), 1),
            "p95":     round(sorted(scores)[int(len(scores) * 0.95)], 1),
        }


class CPEInterleaveSampler(Sampler):
    def __init__(self, dataset: LaTeXDataset, batch_size: int, cpe_ratio: float, seed: int = 42):
        self.normal_idx = dataset.normal_indices()
        self.cpe_idx    = dataset.cpe_indices()
        self.batch_size = batch_size
        self.cpe_ratio  = cpe_ratio
        self.seed       = seed

        self.n_cpe_per_batch    = max(1, round(batch_size * cpe_ratio))
        self.n_normal_per_batch = batch_size - self.n_cpe_per_batch
        self.n_batches          = len(self.normal_idx) // self.n_normal_per_batch

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)

        normal_pool = self.normal_idx.copy()
        rng.shuffle(normal_pool)

        cpe_pool = self.cpe_idx.copy()
        rng.shuffle(cpe_pool)
        if len(cpe_pool) < self.n_batches * self.n_cpe_per_batch:
            repeats  = (self.n_batches * self.n_cpe_per_batch) // max(len(cpe_pool), 1) + 1
            cpe_pool = (cpe_pool * repeats)[: self.n_batches * self.n_cpe_per_batch]
            rng.shuffle(cpe_pool)

        for b in range(self.n_batches):
            n_start   = b * self.n_normal_per_batch
            c_start   = b * self.n_cpe_per_batch
            batch_idx = (
                normal_pool[n_start : n_start + self.n_normal_per_batch]
                + cpe_pool[c_start : c_start + self.n_cpe_per_batch]
            )
            rng.shuffle(batch_idx)
            yield batch_idx


def collate_fn(batch: list[dict], pad_token_id: int = 1) -> dict:
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_list, labels_list, mask_list = [], [], []
    for item in batch:
        n   = item["input_ids"].size(0)
        pad = max_len - n
        input_ids_list.append(
            torch.cat([item["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)])
        )
        mask_list.append(
            torch.cat([torch.ones(n, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)])
        )

    return {
        "input_ids":      torch.stack(input_ids_list),
        "labels":         torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
        "true_len":       torch.stack([item["true_len"] for item in batch]),
    }


def build_dataloader(config, split: str = "train") -> DataLoader:
    dataset = LaTeXDataset(config)
    pw = getattr(config, "persistent_workers", False) and config.num_workers > 0
    pf = getattr(config, "prefetch_factor", 2) if config.num_workers > 0 else None

    if split == "train" and getattr(config, "cpe_ratio", 0) > 0:
        sampler = CPEInterleaveSampler(
            dataset,
            batch_size=config.batch_size,
            cpe_ratio=config.cpe_ratio,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=lambda b: collate_fn(b, pad_token_id=config.pad_token_id),
            persistent_workers=pw,
            prefetch_factor=pf,
        )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=config.pad_token_id),
        persistent_workers=pw,
        prefetch_factor=pf,
    )