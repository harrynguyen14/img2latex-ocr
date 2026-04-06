import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .collator import LaTeXOCRCollator
from .config import load_config
from .dataset import LaTeXOCRDataset, get_tokenizer, resolve_data_source
from .evaluate import compute_metrics, print_metrics
from .model import LaTeXOCRModel


def move_batched_images(bi, device):
    return [[t.to(device, non_blocking=True) for t in imgs] for imgs in bi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_path", type=str, default=None)
    ap.add_argument("--split", type=str, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    args = ap.parse_args()

    cfg_path = args.config or str(Path(__file__).resolve().parent / "config.yaml")
    cfg = load_config(cfg_path)
    data_source = resolve_data_source(cfg, args.data_path)
    split = args.split or cfg.get("test_split", "test")
    bs = args.batch_size or cfg.get("batch_size", 4)
    nw = args.num_workers or cfg.get("num_workers", 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(cfg["tokenizer_name"])
    ds = LaTeXOCRDataset(data_source, split, tokenizer, cfg, rank=0, world_size=1)
    loader = DataLoader(
        ds,
        batch_size=bs,
        num_workers=nw,
        collate_fn=LaTeXOCRCollator(),
        pin_memory=device.type == "cuda",
    )

    model = LaTeXOCRModel.from_checkpoint(args.checkpoint, device=str(device))
    model.eval()

    preds, refs = [], []
    for batch in tqdm(loader, desc="test"):
        bi = move_batched_images(batch["batched_images"], device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pr = model.generate(bi)
        preds.extend(pr)
        lid = batch["labels"].numpy()
        lid = np.where(lid == -100, tokenizer.pad_token_id, lid)
        refs.extend(tokenizer.batch_decode(lid, skip_special_tokens=True))

    mets = compute_metrics(preds, refs)
    print_metrics(mets, prefix=split)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for p, r in zip(preds, refs):
                f.write(f"REF: {r}\nPRED: {p}\nEXACT: {p.strip() == r.strip()}\n---\n")


if __name__ == "__main__":
    main()
