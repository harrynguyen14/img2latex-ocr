import argparse
import contextlib
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from im2latex_v2.collator import LaTeXOCRCollator
from im2latex_v2.config import load_config
from im2latex_v2.dataset import (
    LaTeXOCRDataset,
    LaTeXOCRParquetMapDataset,
    get_tokenizer,
    resolve_data_source,
)
from im2latex_v2.evaluate import compute_metrics, print_metrics
from im2latex_v2.model import LaTeXOCRModel, alignment_loss


def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size()


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def move_batched_images(bi, device):
    return [[t.to(device, non_blocking=True) for t in imgs] for imgs in bi]


def move_batch(batch, device):
    return {
        "batched_images": move_batched_images(batch["batched_images"], device),
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
    }


def lr_cosine(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return peak * 0.5 * (1.0 + math.cos(math.pi * t))


def amp_dtype_from_cfg(cfg: dict) -> torch.dtype:
    name = str(cfg.get("amp_dtype", "float16")).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


def configure_runtime(cfg: dict, device: torch.device):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if device.type == "cuda" and cfg.get("cuda_benchmark", True):
        torch.backends.cudnn.benchmark = True


def build_dataloader(
    ds,
    bs: int,
    nw: int,
    collate_fn,
    pin: bool,
    prefetch: int,
    persistent: bool,
    sampler=None,
    shuffle: bool = True,
):
    kw = {
        "batch_size": bs,
        "num_workers": nw,
        "collate_fn": collate_fn,
        "pin_memory": pin,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
    }
    if nw > 0:
        kw["prefetch_factor"] = prefetch
        kw["persistent_workers"] = persistent
    return DataLoader(ds, **kw)


def save_training_state(ckpt_dir: Path, optimizer: torch.optim.Optimizer, global_step: int, best_bleu: float):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "best_bleu": best_bleu,
        },
        ckpt_dir / "training_state.pt",
    )


def load_training_state(path: Path, map_location) -> dict | None:
    p = path / "training_state.pt"
    if not p.is_file():
        return None
    return torch.load(p, map_location=map_location, weights_only=False)


@torch.no_grad()
def run_eval(module, loader, device, tokenizer, max_batches: int, amp_dtype: torch.dtype):
    module.eval()
    preds, refs = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch(batch, device)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            pr = module.generate(batch["batched_images"])
        preds.extend(pr)
        lid = batch["labels"].cpu().numpy()
        lid = np.where(lid == -100, tokenizer.pad_token_id, lid)
        refs.extend(tokenizer.batch_decode(lid, skip_special_tokens=True))
    module.train()
    return compute_metrics(preds, refs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--data_path", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    cfg_path = args.config or str(Path(__file__).resolve().parent / "config.yaml")
    cfg = load_config(cfg_path)
    data_source = resolve_data_source(cfg, args.data_path)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        rank, local_rank, world_size = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
        is_master = rank == 0
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_master = True

    seed = int(cfg.get("seed", 42)) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    amp_dtype = amp_dtype_from_cfg(cfg)
    configure_runtime(cfg, device)

    stage = int(cfg.get("stage", 1))

    tokenizer = get_tokenizer(cfg["tokenizer_name"])
    data_path = Path(data_source)
    use_map = bool(cfg.get("parquet_map_mode", False)) and data_path.is_dir()
    train_sampler = None
    val_sampler = None
    if use_map:
        try:
            train_ds = LaTeXOCRParquetMapDataset(
                str(data_path), cfg["train_split"], tokenizer, cfg
            )
            val_ds = LaTeXOCRParquetMapDataset(str(data_path), cfg["val_split"], tokenizer, cfg)
        except FileNotFoundError:
            use_map = False
    if not use_map:
        train_ds = LaTeXOCRDataset(
            data_source, cfg["train_split"], tokenizer, cfg, rank=rank, world_size=world_size
        )
        val_ds = LaTeXOCRDataset(
            data_source, cfg["val_split"], tokenizer, cfg, rank=rank, world_size=world_size
        )
    if use_map and distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )

    model = LaTeXOCRModel(cfg).to(device)
    resume_dir = Path(args.resume).resolve() if args.resume else None
    if resume_dir is not None and resume_dir.is_dir():
        ck = LaTeXOCRModel.from_checkpoint(str(resume_dir), device=str(device))
        model.load_state_dict(ck.state_dict(), strict=True)

    model.set_train_stage(stage)

    if cfg.get("gradient_checkpointing", False) and stage == 2:
        model.gradient_checkpointing_enable()

    if cfg.get("torch_compile", False) and hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead")

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    module = model.module if isinstance(model, DDP) else model
    trainable = [p for p in model.parameters() if p.requires_grad]
    lr = float(cfg["lr_stage1"] if stage == 1 else cfg["lr_stage2"])
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=float(cfg["weight_decay"]))

    global_step = 0
    best_bleu = -1.0
    ts = load_training_state(resume_dir, device) if resume_dir else None
    if ts is not None:
        try:
            opt.load_state_dict(ts["optimizer"])
        except ValueError:
            print("[resume] Optimizer param groups mismatch — skipping optimizer state (cross-stage resume)")
        global_step = int(ts.get("global_step", 0))
        best_bleu = float(ts.get("best_bleu", -1.0))

    bs = int(cfg["batch_size"])
    nw = int(cfg["num_workers"])
    prefetch = int(cfg.get("prefetch_factor", 2))
    persistent = bool(cfg.get("persistent_workers", True)) and nw > 0
    train_shuffle = train_sampler is None and not distributed
    val_shuffle = False
    train_loader = build_dataloader(
        train_ds,
        bs,
        nw,
        LaTeXOCRCollator(),
        device.type == "cuda",
        prefetch,
        persistent,
        sampler=train_sampler,
        shuffle=train_shuffle,
    )
    val_loader = build_dataloader(
        val_ds,
        bs,
        nw,
        LaTeXOCRCollator(),
        device.type == "cuda",
        prefetch,
        persistent,
        sampler=val_sampler,
        shuffle=val_shuffle,
    )

    ckpt_root = Path(cfg.get("ckpt_dir", "checkpoints"))
    if is_master:
        ckpt_root.mkdir(parents=True, exist_ok=True)
    sub = f"stage{stage}"
    ckpt_dir = ckpt_root / sub

    epochs = int(cfg["epochs"])
    accum = int(cfg["grad_accum"])
    max_steps_cap = int(cfg.get("max_steps", 100000))
    log_every = int(cfg["log_steps"])
    eval_every = int(cfg["eval_steps"])
    save_every = int(cfg["save_steps"])
    eval_samples = int(cfg.get("eval_samples", 200))

    ns = len(train_ds) if use_map else train_ds.num_samples
    if ns:
        spe = max(ns // max(world_size, 1) // max(bs, 1), 1)
        total_steps = min(spe * epochs // max(accum, 1), max_steps_cap)
    else:
        total_steps = max_steps_cap
    warmup = max(int(total_steps * float(cfg.get("warmup_ratio", 0.05))), 1)

    micro = 0
    accum_loss = 0.0
    pbar = tqdm(total=total_steps, initial=global_step, disable=not is_master)

    model.train()
    epoch_idx = 0
    while global_step < total_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_idx)
        for batch in train_loader:
            if global_step >= total_steps:
                break
            batch = move_batch(batch, device)
            is_sync = micro == accum - 1
            sync_ctx = model.no_sync() if isinstance(model, DDP) and not is_sync else contextlib.nullcontext()

            with sync_ctx:
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    if stage == 1:
                        loss = alignment_loss(module, batch["batched_images"], batch["labels"]) / accum
                    else:
                        loss = model(
                            batch["batched_images"],
                            batch["input_ids"],
                            batch["attention_mask"],
                            batch["labels"],
                        ).loss / accum
                loss.backward()
            accum_loss += loss.item()
            micro += 1

            if not is_sync:
                continue

            t = lr_cosine(global_step, total_steps, lr, warmup)
            for g in opt.param_groups:
                g["lr"] = t
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(cfg["max_grad_norm"]))
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            micro = 0
            pbar.update(1)
            if is_master and global_step % log_every == 0:
                ep = global_step / max(total_steps, 1)
                # Print a single line per logging step (avoid tqdm postfix clutter).
                msg = (
                    f"loss={accum_loss:.4f} "
                    f"grad_norm={grad_norm:.4f} "
                    f"learning_rate={t:.3e} "
                    f"epoch={ep:.4f}"
                )
                pbar.write(msg)
            accum_loss = 0.0

            if global_step > 0 and global_step % eval_every == 0 and stage == 2:
                if distributed:
                    dist.barrier()
                if is_master:
                    mb = max(eval_samples // bs, 1)
                    mets = run_eval(module, val_loader, device, tokenizer, mb, amp_dtype)
                    print_metrics(mets, prefix=f"step {global_step}")
                    if mets["bleu4"] > best_bleu:
                        best_bleu = mets["bleu4"]
                        best_path = ckpt_dir / "best"
                        module.save_checkpoint(str(best_path), step=global_step, metrics=mets)
                        save_training_state(best_path, opt, global_step, best_bleu)
                if distributed:
                    dist.barrier()

            if global_step > 0 and global_step % save_every == 0:
                if distributed:
                    dist.barrier()
                if is_master:
                    sp = ckpt_dir / f"step-{global_step}"
                    module.save_checkpoint(str(sp), step=global_step)
                    save_training_state(sp, opt, global_step, best_bleu)
                if distributed:
                    dist.barrier()

        epoch_idx += 1

    pbar.close()
    if distributed:
        dist.barrier()
    if is_master:
        final_path = ckpt_dir / "final"
        module.save_checkpoint(str(final_path), step=global_step)
        save_training_state(final_path, opt, global_step, best_bleu)
    if distributed:
        dist.barrier()
        cleanup_ddp()


if __name__ == "__main__":
    main()
