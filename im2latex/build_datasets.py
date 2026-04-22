from pathlib import Path
from torch.utils.data import DataLoader, IterableDataset

from .preprocessor import LaTeXOCRHFDataset, LaTeXOCRParquetDataset, LaTeXOCRFlatParquetDataset

# Repo layout (harryrobert/ocr-latex-filter):
#   train/{raw,light,heavy}/*.parquet
#   validation/*.parquet
#   test/*.parquet

_DEFAULT_SOURCES = ["raw", "light", "heavy"]
_DEFAULT_WEIGHTS = [1.0, 1.0, 1.0]


def build_datasets(args, data_source: str, tokenizer):
    data_path = getattr(args, "data_path", "").strip()

    # Local layout: data_path/train/{raw,light,heavy}/ + data_path/validation/
    if data_path and Path(data_path).exists():
        train_dir = Path(data_path) / "train"
        val_dir   = Path(data_path) / "validation"

        if train_dir.exists():
            sources = getattr(args, "sources", _DEFAULT_SOURCES)
            weights = getattr(args, "weights", _DEFAULT_WEIGHTS)

            train_ds = LaTeXOCRParquetDataset(
                str(train_dir), sources, weights, tokenizer, args,
            )

            if val_dir.exists():
                val_ds = LaTeXOCRFlatParquetDataset(
                    str(val_dir), tokenizer, args,
                )
                print(f"[dataset] train={train_dir}  val={val_dir}")
            else:
                raise FileNotFoundError(
                    f"validation/ not found under {data_path}. "
                    f"Download the full repo: huggingface-cli download harryrobert/ocr-latex-filter --repo-type dataset --local-dir {data_path}"
                )

            return train_ds, val_ds

    # HF streaming fallback (harryrobert/ocr-latex-filter)
    sources = getattr(args, "sources", _DEFAULT_SOURCES)
    weights = getattr(args, "weights", _DEFAULT_WEIGHTS)
    print(f"[dataset] HF streaming → {data_source}  sources={sources}")
    return (
        LaTeXOCRHFDataset(data_source, "train", tokenizer, args,
                          names=sources, weights=weights),
        LaTeXOCRHFDataset(data_source, "validation", tokenizer, args,),
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
