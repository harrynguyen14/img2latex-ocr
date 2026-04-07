from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
import bitsandbytes as bnb

from .utils import move_batch, lr_cosine, cleanup_distributed
from .evaluate import compute_metrics, print_metrics
from .latex_ocr_model import LaTeXOCRModel


def save_training_state(ckpt_dir: Path, optimizer, global_step: int, best_bleu: float):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"optimizer": optimizer.state_dict(), "global_step": global_step, "best_bleu": best_bleu},
        ckpt_dir / "training_state.pt",
    )


def load_training_state(path: Path, map_location) -> dict | None:
    p = path / "training_state.pt"
    if not p.is_file():
        return None
    return torch.load(p, map_location=map_location, weights_only=False)


@torch.no_grad()
def run_eval(model, loader, device, tokenizer, max_batches: int):
    model.eval()
    preds, refs = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch(batch, device)
        pr = model.generate(batch["batched_images"], num_beams=1)
        preds.extend(pr)
        lid = batch["labels"].cpu().numpy()
        lid = np.where(lid == -100, tokenizer.pad_token_id, lid)
        refs.extend(tokenizer.batch_decode(lid, skip_special_tokens=True))
    model.train()
    return compute_metrics(preds, refs)


class BaseTrainer:
    def __init__(self, args, train_loader, val_loader, device, tokenizer, distributed, rank, local_rank, world_size):
        self.args         = args
        self.device       = device
        self.tokenizer    = tokenizer
        self.distributed  = distributed
        self.is_master    = rank == 0
        self.rank         = rank
        self.local_rank   = local_rank
        self.world_size   = world_size
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.amp_dtype    = torch.bfloat16 if args.amp_dtype in ("bf16", "bfloat16") else torch.float16

        ns = getattr(train_loader.dataset, "num_samples", None)
        bs = args.batch_size
        if ns:
            spe = max(ns // max(world_size, 1) // max(bs, 1), 1)
            self.total_steps = min(spe * args.epochs // max(args.grad_accum, 1), args.max_steps)
        else:
            self.total_steps = args.max_steps
        self.warmup = max(int(self.total_steps * args.warmup_ratio), 1)

        self.global_step = 0
        self.best_bleu   = -1.0
        self.ckpt_dir    = Path(args.ckpt_dir) / f"stage{args.stage}"
        if self.is_master:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self, trainable):
        lr = self.args.lr_stage1 if self.args.stage == 1 else self.args.lr_stage2
        self.lr = lr
        self.opt = bnb.optim.AdamW8bit(trainable, lr=lr, weight_decay=self.args.weight_decay)
        print("[optimizer] Using AdamW8bit")

    def _resume(self, resume_dir: Path):
        ts = load_training_state(resume_dir, self.device)
        if ts:
            try:
                self.opt.load_state_dict(ts["optimizer"])
            except ValueError:
                print("[resume] Optimizer param groups mismatch — skipping optimizer state")
            self.global_step = int(ts.get("global_step", 0))
            self.best_bleu   = float(ts.get("best_bleu", -1.0))

    def _step_lr(self):
        t = lr_cosine(self.global_step, self.total_steps, self.lr, self.warmup)
        for g in self.opt.param_groups:
            g["lr"] = t
        return t

    def _barrier(self):
        if self.distributed:
            dist.barrier()

    def _cleanup(self):
        if self.distributed:
            cleanup_distributed()

    def train(self):
        raise NotImplementedError

    def _save(self, path: Path):
        raise NotImplementedError
