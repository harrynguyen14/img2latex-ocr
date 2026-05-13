import random
import sys
from pathlib import Path

import numpy as np
import torch


def _ensure_package_importable():
    """Add parent of im2latex/ to sys.path so relative imports work when run directly."""
    pkg_dir = Path(__file__).resolve().parent
    root = str(pkg_dir.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_path",           type=str,   default="/workspace/data")
    ap.add_argument("--sources", nargs="+",  type=str,   default=["raw", "light", "heavy"])
    ap.add_argument("--weights", nargs="+",  type=float, default=[1.0, 1.0, 1.0])
    ap.add_argument(
        "--weight_stages",
        type=str,
        default="",
        help=(
            "Optional schedule for train sampling weights. "
            "Format: step:w1,w2,w3;step:w1,w2,w3 (aligned with --sources). "
            "Example: 0:0.8,0.15,0.05;40000:0.6,0.25,0.15;80000:0.45,0.3,0.25"
        ),
    )
    ap.add_argument("--max_token_len",       type=int,   default=512)
    ap.add_argument("--max_visual_tokens",   type=int,   default=1024)
    ap.add_argument("--max_side",            type=int,   default=2048)
    ap.add_argument("--patch_size",          type=int,   default=4)

    ap.add_argument("--decoder_repo_id",     type=str,   default="harryrobert/nav2tex-decoder")
    ap.add_argument("--tokenizer_repo_id",   type=str,   default="harryrobert/nav2tex-decoder")

    ap.add_argument("--navit_dim",           type=int,   default=1024)
    ap.add_argument("--navit_depth",         type=int,   default=12)
    ap.add_argument("--navit_heads",         type=int,   default=16)
    ap.add_argument("--navit_dim_head",      type=int,   default=64)
    ap.add_argument("--navit_mlp_dim",       type=int,   default=4096)
    ap.add_argument("--navit_dropout",       type=float, default=0.0)
    ap.add_argument("--navit_emb_dropout",   type=float, default=0.0)

    ap.add_argument("--decoder_warmup_steps", type=int,   default=2000)
    ap.add_argument("--len_loss_start_step",  type=int,   default=15000,
                    help="Step at which to enable len_loss (LAM). Should be well after decoder unfreeze.")
    ap.add_argument("--batch_size",           type=int,   default=16)
    ap.add_argument("--grad_accum",           type=int,   default=32)
    ap.add_argument("--lr",                   type=float, default=1e-4)
    ap.add_argument("--encoder_lr",           type=float, default=1e-4,
                    help="LR for visual encoder and LAM (all non-decoder params).")
    ap.add_argument("--decoder_lr",           type=float, default=1e-5,
                    help="LR for decoder after unfreeze.")
    ap.add_argument("--weight_decay",         type=float, default=0.01)
    ap.add_argument("--max_grad_norm",        type=float, default=1.0)
    ap.add_argument("--warmup_ratio",         type=float, default=0.05)
    ap.add_argument("--max_steps",            type=int,   default=10000)
    ap.add_argument("--log_steps",            type=int,   default=50)
    ap.add_argument("--val_loss_steps",       type=int,   default=2500)
    ap.add_argument("--eval_steps",           type=int,   default=10000)
    ap.add_argument("--save_steps",           type=int,   default=10000)
    ap.add_argument("--eval_samples",         type=int,   default=1024)
    ap.add_argument("--bleu_samples",         type=int,   default=512)
    ap.add_argument("--final_eval_samples",   type=int,   default=0)
    ap.add_argument("--num_workers",          type=int,   default=1)
    ap.add_argument("--prefetch_factor",      type=int,   default=4)
    ap.add_argument("--persistent_workers",   action="store_true", default=False)
    ap.add_argument("--no_cuda_benchmark",    action="store_true", default=False,
                    help="Disable cudnn.benchmark (enabled by default)")
    ap.add_argument("--grad_checkpoint",      action="store_true", default=False)
    ap.add_argument("--decoder_grad_checkpoint", action="store_true", default=False,
                    help="Enable gradient checkpointing on the decoder after unfreeze to save activation memory.")
    ap.add_argument("--unfreeze_grad_accum_divisor", type=int, default=1,
                    help="Divide grad_accum by this factor when decoder is unfrozen to keep memory stable. "
                         "E.g. 2 halves micro-batch count, keeping effective batch size unchanged.")
    ap.add_argument("--torch_compile",        action="store_true", default=False)
    ap.add_argument("--seed",                 type=int,   default=42)

    ap.add_argument("--ckpt_dir",             type=str,   default="/workspace/checkpoints")
    ap.add_argument("--resume",               type=str,   default=None)

    ap.add_argument("--max_new_tokens",       type=int,   default=512)
    ap.add_argument("--early_stopping_patience", type=int, default=0,
                    help="Stop if val_ppl does not improve for this many val checks. 0 = disabled.")

    args = ap.parse_args()
    args.cuda_benchmark = not args.no_cuda_benchmark
    return args


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    from im2latex.utils import configure_runtime
    from im2latex.build_datasets import build_datasets, build_dataloader
    from im2latex.trainer import Trainer

    configure_runtime(args, device)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_repo_id)

    train_ds, val_ds = build_datasets(args, tokenizer)

    nw         = args.num_workers
    prefetch   = args.prefetch_factor
    persistent = args.persistent_workers and nw > 0
    train_loader = build_dataloader(train_ds, args.batch_size, nw, device.type == "cuda", prefetch, persistent, args.max_token_len)
    val_loader   = build_dataloader(val_ds,   args.batch_size, nw, device.type == "cuda", prefetch, persistent, args.max_token_len)

    trainer = Trainer(args, train_loader, val_loader, device, tokenizer)
    trainer.train()


if __name__ == "__main__":
    _ensure_package_importable()
    main()