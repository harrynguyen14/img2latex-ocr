from pathlib import Path
from torch.utils.data import DataLoader, IterableDataset

from .preprocessor import LaTeXOCRHFDataset, LaTeXOCRDiskDataset

DISK_CACHE_DIR = "/kaggle/working/cache"


def build_datasets(args, data_source: str, rank: int, world_size: int, tokenizer):
    train_cache = Path(DISK_CACHE_DIR) / args.train_split
    val_cache   = Path(DISK_CACHE_DIR) / args.val_split

    if train_cache.exists() and val_cache.exists():
        print(f"[dataset] disk cache found → {DISK_CACHE_DIR}")
        return (
            LaTeXOCRDiskDataset(str(train_cache), tokenizer, args, rank=rank, world_size=world_size),
            LaTeXOCRDiskDataset(str(val_cache),   tokenizer, args, rank=rank, world_size=world_size),
        )

    print(f"[dataset] HF streaming → {data_source}")
    return (
        LaTeXOCRHFDataset(data_source, args.train_split, tokenizer, args, rank=rank, world_size=world_size),
        LaTeXOCRHFDataset(data_source, args.val_split,   tokenizer, args, rank=rank, world_size=world_size),
    )


def build_dataloader(ds, bs: int, nw: int, collate_fn, pin: bool, prefetch: int, persistent: bool):
    kw = {
        "batch_size": bs,
        "num_workers": nw,
        "collate_fn": collate_fn,
        "pin_memory": pin,
        "shuffle": False if isinstance(ds, IterableDataset) else True,
    }
    if nw > 0:
        kw["prefetch_factor"] = prefetch
        kw["persistent_workers"] = persistent
    return DataLoader(ds, **kw)
