from pathlib import Path
from torch.utils.data import DataLoader

from .preprocessor import Nav2TexTrainDataset, Nav2TexValDataset
from .utils import TokenBudgetBatcher, collate_fn

_DEFAULT_SOURCES = ["raw", "light", "heavy"]
_DEFAULT_WEIGHTS = [1.0, 1.0, 1.0]


def build_datasets(args, tokenizer):
    data_path = Path(args.data_path)
    train_dir = data_path / "train"
    val_dir   = data_path / "validation"

    if not train_dir.exists():
        raise FileNotFoundError(f"train/ not found under {data_path}")
    if not val_dir.exists():
        raise FileNotFoundError(f"validation/ not found under {data_path}")

    sources = getattr(args, "sources", _DEFAULT_SOURCES)
    weights = getattr(args, "weights", _DEFAULT_WEIGHTS)

    train_ds = Nav2TexTrainDataset(str(train_dir), sources, weights, tokenizer, args)
    val_ds   = Nav2TexValDataset(str(val_dir), tokenizer, args)

    print(f"[dataset] train={train_dir}  val={val_dir}  sources={sources}")
    return train_ds, val_ds


def build_dataloader(ds, token_budget: int, nw: int, pin: bool, prefetch: int, persistent: bool):
    batched_ds = TokenBudgetBatcher(ds, token_budget)
    kw = {
        "batch_size":  1,           # each item from batcher is already a full batch (list)
        "num_workers": nw,
        "collate_fn":  lambda x: collate_fn(x[0]),  # unwrap the outer list added by DataLoader
        "pin_memory":  pin,
        "shuffle":     False,
    }
    if nw > 0:
        kw["prefetch_factor"]    = prefetch
        kw["persistent_workers"] = persistent
    return DataLoader(batched_ds, **kw)
