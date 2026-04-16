import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import collate_fn, move_batch
from .preprocessor import LaTeXOCRHFDataset
from pretrain_decoder.tokenizer import load_fast_tokenizer as get_tokenizer
from .evaluate import compute_metrics, print_metrics
from .latex_ocr_model import LaTeXOCRModel


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",   type=str, required=True)
    ap.add_argument("--dataset_id",   type=str, default="harryrobert/latex-ocr-aug")
    ap.add_argument("--data_path",    type=str, default="")
    ap.add_argument("--split",        type=str, default="test")
    ap.add_argument("--tokenizer_dir", type=str, default="D:/img2latex/tokenizer")
    ap.add_argument("--batch_size",   type=int, default=4)
    ap.add_argument("--num_workers",  type=int, default=2)
    ap.add_argument("--max_token_len",    type=int,   default=150)
    ap.add_argument("--image_height",     type=int,   default=64)
    ap.add_argument("--max_image_width",  type=int,   default=672)
    ap.add_argument("--max_image_height", type=int,   default=640)
    ap.add_argument("--patch_size",       type=int,   default=16)
    ap.add_argument("--resize_in_dataset", action="store_true", default=True)
    ap.add_argument("--output",       type=str, default=None)
    return ap.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer   = get_tokenizer(args.tokenizer_dir)
    data_source = args.data_path.strip() or args.dataset_id
    ds     = LaTeXOCRHFDataset(data_source, args.split, tokenizer, args)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=collate_fn, pin_memory=device.type == "cuda")

    import json
    from pathlib import Path
    from safetensors.torch import load_file

    ckpt_dir = Path(args.checkpoint)
    with open(ckpt_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    model = LaTeXOCRModel(config).to(device)
    state = load_file(str(ckpt_dir / "model.safetensors"), device=str(device))
    ve_state = {k.replace("visual_encoder.", ""): v for k, v in state.items() if k.startswith("visual_encoder.")}
    model.visual_encoder.load_state_dict(ve_state, strict=True)
    model.eval()

    preds, refs = [], []
    for batch in tqdm(loader, desc="test"):
        batch = move_batch(batch, device)
        with torch.no_grad():
            pr = model.generate(batch["batched_images"])
        preds.extend(pr)
        lid = batch["labels"].cpu().numpy()
        lid = np.where(lid == -100, tokenizer.pad_token_id, lid)
        refs.extend(tokenizer.batch_decode(lid, skip_special_tokens=True))

    mets = compute_metrics(preds, refs)
    print_metrics(mets, prefix=args.split)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for p, r in zip(preds, refs):
                f.write(f"REF: {r}\nPRED: {p}\nEXACT: {p.strip() == r.strip()}\n---\n")


if __name__ == "__main__":
    main()
