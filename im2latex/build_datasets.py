from pathlib import Path
from torch.utils.data import DataLoader, IterableDataset

from .preprocessor import LaTeXOCRHFDataset, LaTeXOCRDiskDataset, LaTeXOCRParquetDataset, LaTeXOCRFlatParquetDataset

DISK_CACHE_DIR = "/kaggle/working/cache"

_DEFAULT_SOURCES = ["raw", "light", "heavy"]
_DEFAULT_WEIGHTS = [1.0, 1.0, 1.0]


def build_datasets(args, data_source: str, rank: int, world_size: int, tokenizer):
    data_path = getattr(args, "data_path", "").strip()

    # Local parquet layout: data_path/train/{raw,light,heavy}/ + data_path/validation/
    if data_path and Path(data_path).exists():
        train_dir = Path(data_path) / "train"
        val_dir   = Path(data_path) / "validation"

        if train_dir.exists():
            sources = getattr(args, "sources", _DEFAULT_SOURCES)
            weights = getattr(args, "weights", _DEFAULT_WEIGHTS)

            train_ds = LaTeXOCRParquetDataset(
                str(train_dir), sources, weights, tokenizer, args,
                rank=rank, world_size=world_size,
                val_files=0, is_val=False,
            )

            if val_dir.exists():
                val_ds = LaTeXOCRFlatParquetDataset(
                    str(val_dir), tokenizer, args,
                    rank=rank, world_size=world_size,
                )
                print(f"[dataset] train={train_dir}  val={val_dir}")
            else:
                # fallback: split last few files from train as val
                val_ds = LaTeXOCRParquetDataset(
                    str(train_dir), sources, [1.0] * len(sources), tokenizer, args,
                    rank=rank, world_size=world_size,
                    val_files=3, is_val=True,
                )
                print(f"[dataset] train={train_dir}  val=split-from-train")

            return train_ds, val_ds

    # HF disk cache
    train_cache = Path(DISK_CACHE_DIR) / args.train_split
    val_cache   = Path(DISK_CACHE_DIR) / args.val_split
    if train_cache.exists() and val_cache.exists():
        print(f"[dataset] disk cache found → {DISK_CACHE_DIR}")
        return (
            LaTeXOCRDiskDataset(str(train_cache), tokenizer, args, rank=rank, world_size=world_size),
            LaTeXOCRDiskDataset(str(val_cache),   tokenizer, args, rank=rank, world_size=world_size),
        )

    # HF streaming fallback
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
