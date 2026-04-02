import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LaTeXDataset, get_tokenizer
from collator import LaTeXDataCollator
from modeling import LaTeXOCRModel
from evaluate import compute_metrics, print_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_loader(data_path: str, split: str, tokenizer, batch_size: int, num_workers: int):
    ds = LaTeXDataset(data_path, split, tokenizer)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=LaTeXDataCollator())


@torch.no_grad()
def run_test(model: LaTeXOCRModel, loader: DataLoader, tokenizer, output_file: str = None):
    model.eval()
    all_preds, all_refs = [], []

    for batch in tqdm(loader, desc="Testing"):
        pv = batch["pixel_values"].to(DEVICE)
        pm = batch["patch_mask"].to(DEVICE)
        all_preds.extend(model.generate(pv, pm))

        ref_ids = batch["labels"].cpu().numpy()
        ref_ids = np.where(ref_ids == -100, tokenizer.pad_token_id, ref_ids)
        all_refs.extend(tokenizer.batch_decode(ref_ids, skip_special_tokens=True))

    metrics = compute_metrics(all_preds, all_refs)
    print_metrics(metrics, prefix="Test")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for pred, ref in zip(all_preds, all_refs):
                f.write(f"REF : {ref}\n")
                f.write(f"PRED: {pred}\n")
                f.write(f"EXACT: {pred.strip() == ref.strip()}\n")
                f.write("-" * 80 + "\n")
        print(f"Saved to {output_file}")

    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   type=str, default="checkpoints/stage2/best")
    p.add_argument("--data_path",    type=str, required=True)
    p.add_argument("--split",        type=str, default="test")
    p.add_argument("--output",       type=str, default="test_results.txt")
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--num_workers",  type=int, default=2)
    return p.parse_args()


def main():
    args      = parse_args()
    tokenizer = get_tokenizer()
    model, _  = LaTeXOCRModel.from_checkpoint(args.checkpoint, device=DEVICE)
    model.to(DEVICE)
    loader    = build_loader(args.data_path, args.split, tokenizer,
                             args.batch_size, args.num_workers)
    run_test(model, loader, tokenizer, output_file=args.output)


if __name__ == "__main__":
    main()
