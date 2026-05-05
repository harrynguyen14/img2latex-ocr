import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from .utils import collate_fn, configure_runtime
from .build_datasets import build_datasets, build_dataloader
from .preprocessor import get_tokenizer
from .trainer import Trainer
from tokenizer import LaTeXTokenizerV2


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_id",                  type=str,   default="harryrobert/ocr-latex-filter")
    ap.add_argument("--data_path",                   type=str,   default="/workspace/data")
    ap.add_argument("--sources",     nargs="+",      type=str,   default=["raw", "light", "heavy"])
    ap.add_argument("--weights",     nargs="+",      type=float, default=[1.0, 1.0, 1.0])
    ap.add_argument("--max_token_len",               type=int,   default=200)
    ap.add_argument("--image_height",                type=int,   default=64)
    ap.add_argument("--max_image_width",             type=int,   default=1024)
    ap.add_argument("--max_image_height",            type=int,   default=640)
    ap.add_argument("--patch_size",                  type=int,   default=16)
    ap.add_argument("--resize_in_dataset",           action="store_true", default=True)

    ap.add_argument("--decoder_ckpt",                type=str,   default="harryrobert/pretrain-decoder")
    ap.add_argument("--qat",                         action="store_true", default=False)
    ap.add_argument("--sparsity_lambda",             type=float, default=0.0)
    ap.add_argument("--tokenizer_dir",               type=str,   default="D:/img2latex/tokenizer")
    ap.add_argument("--navit_dim",                   type=int,   default=512)
    ap.add_argument("--navit_depth",                 type=int,   default=8)
    ap.add_argument("--navit_heads",                 type=int,   default=8)
    ap.add_argument("--navit_dim_head",              type=int,   default=64)
    ap.add_argument("--navit_mlp_dim",               type=int,   default=2048)
    ap.add_argument("--navit_dropout",               type=float, default=0.0)
    ap.add_argument("--navit_emb_dropout",           type=float, default=0.0)
    ap.add_argument("--vision_hidden_size",          type=int,   default=512)
    ap.add_argument("--llm_hidden_size",             type=int,   default=512)
    ap.add_argument("--projector_intermediate_size", type=int,   default=1024)
    ap.add_argument("--max_visual_tokens",           type=int,   default=256)

    ap.add_argument("--decoder_warmup_steps",        type=int,   default=5000)
    ap.add_argument("--batch_size",                  type=int,   default=1)
    ap.add_argument("--eval_batch_size",             type=int,   default=1)
    ap.add_argument("--grad_accum",                  type=int,   default=32)
    ap.add_argument("--lr",                          type=float, default=1e-4)
    ap.add_argument("--weight_decay",                type=float, default=0.01)
    ap.add_argument("--max_grad_norm",               type=float, default=1.0)
    ap.add_argument("--warmup_ratio",                type=float, default=0.05)
    ap.add_argument("--max_steps",                   type=int,   default=10000)
    ap.add_argument("--log_steps",                   type=int,   default=50)
    ap.add_argument("--val_loss_steps",              type=int,   default=2500)
    ap.add_argument("--eval_steps",                  type=int,   default=10000)
    ap.add_argument("--save_steps",                  type=int,   default=10000)
    ap.add_argument("--eval_samples",                type=int,   default=512)
    ap.add_argument("--bleu_samples",                type=int,   default=1500)
    ap.add_argument("--final_eval_samples",          type=int,   default=0)
    ap.add_argument("--num_workers",                 type=int,   default=1)
    ap.add_argument("--prefetch_factor",             type=int,   default=4)
    ap.add_argument("--persistent_workers",          action="store_true", default=False)
    ap.add_argument("--cuda_benchmark",              action="store_true", default=True)
    ap.add_argument("--torch_compile",               action="store_true", default=False)
    ap.add_argument("--seed",                        type=int,   default=42)

    ap.add_argument("--ckpt_dir",                    type=str,   default="/kaggle/working/checkpoints")
    ap.add_argument("--resume",                      type=str,   default=None)

    ap.add_argument("--max_new_tokens",              type=int,   default=200)
    ap.add_argument("--num_beams",                   type=int,   default=4)

    return ap.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    configure_runtime(args, device)

    tokenizer_dir = Path(args.tokenizer_dir)
    if tokenizer_dir.exists():
        tokenizer = LaTeXTokenizerV2.load(tokenizer_dir)
    else:
        tokenizer = get_tokenizer(args.decoder_ckpt)
    data_source = args.data_path.strip() or args.dataset_id
    train_ds, val_ds = build_datasets(args, data_source, tokenizer)

    nw         = args.num_workers
    prefetch   = args.prefetch_factor
    persistent = args.persistent_workers and nw > 0
    train_loader = build_dataloader(train_ds, args.batch_size,      nw, collate_fn, device.type == "cuda", prefetch, persistent)
    val_loader   = build_dataloader(val_ds,   args.eval_batch_size, nw, collate_fn, device.type == "cuda", prefetch, persistent)

    trainer = Trainer(args, train_loader, val_loader, device, tokenizer)
    trainer.train()


if __name__ == "__main__":
    main()
