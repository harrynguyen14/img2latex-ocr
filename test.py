import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocess import LaTeXDataset, get_tokenizer
from trainer import LaTeXDataCollator
from modeling_latex_ocr import LaTeXOCRModel
from evaluate import compute_metrics, print_metrics
from utils import get_device

DEVICE = get_device()


def load_model(checkpoint_dir: str) -> LaTeXOCRModel:
    model, _ = LaTeXOCRModel.from_checkpoint(checkpoint_dir, device=DEVICE)
    model.eval()
    return model


def run_test(model, test_loader, output_file: str = None):
    all_preds, all_refs = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            pv = batch["pixel_values"].to(DEVICE)
            pm = batch["patch_mask"].to(DEVICE)
            preds = model.generate(pv, pm)
            all_preds.extend(preds)
            ref_ids = batch["labels"].cpu().numpy()
            ref_ids = np.where(ref_ids == -100, model.tokenizer.pad_token_id, ref_ids)
            ref_strs = model.tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
            all_refs.extend(ref_strs)

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
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage2/best_model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="HF repo id hoặc local Parquet folder")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default="test_results.txt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = get_tokenizer()

    dataset = LaTeXDataset(args.data_path, args.split, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=LaTeXDataCollator(),
    )

    model = load_model(args.checkpoint)
    run_test(model, loader, output_file=args.output)


if __name__ == "__main__":
    main()
