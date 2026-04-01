import torch
import argparse
from pathlib import Path
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocess import LaTeXDataset, get_tokenizer
from encode import collate_images
from modeling_latex_ocr import LaTeXOCRModel
from evaluate import compute_metrics, print_metrics
from utils import get_device

DATA_DIR = Path("data")
DEVICE = get_device()


def collate_fn(batch, device):
    images = [s["image"] for s in batch]
    input_ids = torch.stack([s["input_ids"] for s in batch])
    attention_mask = torch.stack([s["attention_mask"] for s in batch])
    pixel_values, patch_mask = collate_images(images, device=device)
    return {
        "pixel_values": pixel_values,
        "patch_mask": patch_mask,
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "raw_labels": [s["label"] for s in batch],
    }


def load_model(checkpoint_dir: str) -> LaTeXOCRModel:
    model, _ = LaTeXOCRModel.from_checkpoint(checkpoint_dir, device=DEVICE)
    model.eval()
    return model


def run_test(model, test_loader, tokenizer, output_file: str = None):
    all_preds, all_refs = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            preds = model.generate(batch["pixel_values"], batch["patch_mask"])
            all_preds.extend(preds)
            all_refs.extend(batch["raw_labels"])

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_stage2")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--output", type=str, default="test_results.txt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = get_tokenizer()

    split_path = DATA_DIR / args.split
    if not split_path.exists():
        print(f"Split '{args.split}' not found. Run preprocess.py first.")
        return

    ds_raw = load_from_disk(str(split_path))
    if args.limit:
        ds_raw = ds_raw.select(range(min(args.limit, len(ds_raw))))

    dataset = LaTeXDataset(ds_raw, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, DEVICE),
    )

    model = load_model(args.checkpoint)
    run_test(model, loader, tokenizer, output_file=args.output)


if __name__ == "__main__":
    main()
