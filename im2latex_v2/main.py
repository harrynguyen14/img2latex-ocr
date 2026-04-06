import argparse
import os
import random

import numpy as np
import torch

from .utils import collate_fn, configure_runtime, setup_distributed
from .build_datasets import build_datasets, build_dataloader
from .preprocessor import get_tokenizer
from .trainer_ddp import DDPTrainer


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_id",                  type=str,   default="harryrobert/latex-ocr-v3")
    ap.add_argument("--data_path",                   type=str,   default="")
    ap.add_argument("--train_split",                 type=str,   default="full_train")
    ap.add_argument("--val_split",                   type=str,   default="dev")
    ap.add_argument("--max_token_len",               type=int,   default=150)
    ap.add_argument("--image_height",                type=int,   default=64)
    ap.add_argument("--max_image_width",             type=int,   default=672)
    ap.add_argument("--max_image_height",            type=int,   default=640)
    ap.add_argument("--patch_size",                  type=int,   default=16)
    ap.add_argument("--resize_in_dataset",           action="store_true", default=True)

    ap.add_argument("--tokenizer_name",              type=str,   default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--navit_dim",                   type=int,   default=1024)
    ap.add_argument("--navit_depth",                 type=int,   default=12)
    ap.add_argument("--navit_heads",                 type=int,   default=16)
    ap.add_argument("--navit_dim_head",              type=int,   default=64)
    ap.add_argument("--navit_mlp_dim",               type=int,   default=4096)
    ap.add_argument("--navit_dropout",               type=float, default=0.0)
    ap.add_argument("--navit_emb_dropout",           type=float, default=0.0)
    ap.add_argument("--vision_hidden_size",          type=int,   default=1024)
    ap.add_argument("--llm_hidden_size",             type=int,   default=1536)
    ap.add_argument("--projector_intermediate_size", type=int,   default=4096)
    ap.add_argument("--max_visual_tokens",           type=int,   default=256)

    ap.add_argument("--stage",                       type=int,   default=2)
    ap.add_argument("--epochs",                      type=int,   default=1)
    ap.add_argument("--batch_size",                  type=int,   default=1)
    ap.add_argument("--grad_accum",                  type=int,   default=32)
    ap.add_argument("--lr_stage1",                   type=float, default=1e-4)
    ap.add_argument("--lr_stage2",                   type=float, default=2e-5)
    ap.add_argument("--weight_decay",                type=float, default=0.01)
    ap.add_argument("--max_grad_norm",               type=float, default=1.0)
    ap.add_argument("--warmup_ratio",                type=float, default=0.05)
    ap.add_argument("--max_steps",                   type=int,   default=10000)
    ap.add_argument("--log_steps",                   type=int,   default=50)
    ap.add_argument("--eval_steps",                  type=int,   default=500)
    ap.add_argument("--save_steps",                  type=int,   default=10000)
    ap.add_argument("--eval_samples",                type=int,   default=200)
    ap.add_argument("--num_workers",                 type=int,   default=1)
    ap.add_argument("--prefetch_factor",             type=int,   default=4)
    ap.add_argument("--persistent_workers",          action="store_true", default=False)
    ap.add_argument("--amp_dtype",                   type=str,   default="float16")
    ap.add_argument("--gradient_checkpointing",      action="store_true", default=False)
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

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        rank, local_rank, world_size = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    configure_runtime(args, device)

    tokenizer    = get_tokenizer(args.tokenizer_name)
    data_source  = args.data_path.strip() or args.dataset_id
    train_ds, val_ds = build_datasets(args, data_source, rank, world_size, tokenizer)

    bs         = args.batch_size
    nw         = args.num_workers
    prefetch   = args.prefetch_factor
    persistent = args.persistent_workers and nw > 0
    train_loader = build_dataloader(train_ds, bs, nw, collate_fn, device.type == "cuda", prefetch, persistent)
    val_loader   = build_dataloader(val_ds,   bs, nw, collate_fn, device.type == "cuda", prefetch, persistent)

    TrainerCls = DDPTrainer
    trainer = TrainerCls(args, train_loader, val_loader, device, tokenizer, distributed, rank, local_rank, world_size)
    trainer.train()


if __name__ == "__main__":
    main()