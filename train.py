import os
import argparse
import contextlib
import yaml
import math
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from pathlib import Path

from constants import TOKENIZER_NAME
from dataset import LaTeXDataset, IterableDatasetShard, get_tokenizer
from collator import LaTeXDataCollator
from modeling import LaTeXOCRModel
from evaluate import compute_metrics, print_metrics


def load_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = {}
    for section in raw.values():
        if isinstance(section, dict):
            cfg.update(section)
    return cfg


def get_lr(step: int, max_steps: int, lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def build_loader(data_path: str, split: str, tokenizer, cfg: dict,
                 rank: int, world_size: int) -> DataLoader:
    ds    = LaTeXDataset(data_path, split, tokenizer)
    shard = IterableDatasetShard(ds, num_processes=world_size, process_index=rank)
    return DataLoader(
        shard,
        batch_size=cfg["batch_size"],
        collate_fn=LaTeXDataCollator(),
        num_workers=cfg["num_workers"],
        pin_memory=True,
    ), ds.num_samples


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def eval_loop(model, loader, device: torch.device, world_size: int,
              tokenizer, max_batches: int = None) -> dict:
    raw = model.module if isinstance(model, DDP) else model
    raw.eval()

    total_loss  = torch.tensor(0.0, device=device)
    total_count = torch.tensor(0,   device=device)
    all_preds, all_refs = [], []

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = raw(**batch)
        total_loss  += out.loss.detach()
        total_count += 1

        preds = raw.generate(batch["pixel_values"], batch["patch_mask"])
        all_preds.extend(preds)
        ref_ids = batch["labels"].cpu().numpy()
        ref_ids = np.where(ref_ids == -100, tokenizer.pad_token_id, ref_ids)
        all_refs.extend(tokenizer.batch_decode(ref_ids, skip_special_tokens=True))

    dist.all_reduce(total_loss,  op=dist.ReduceOp.SUM)
    dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
    avg_loss = (total_loss / total_count).item()

    metrics = compute_metrics(all_preds, all_refs)
    metrics["loss"] = round(avg_loss, 4)
    raw.train()
    return metrics


def run_stage(stage: int, cfg: dict, data_path: str,
              rank: int, local_rank: int, world_size: int,
              resume: str = None):
    device     = torch.device(f"cuda:{local_rank}")
    is_master  = rank == 0
    use_lora   = stage == 2
    stage_name = f"stage{stage}"
    ckpt_dir   = Path(cfg["ckpt_dir"]) / stage_name

    model_cfg  = {**cfg, "use_lora": use_lora, "max_new_tokens": cfg["max_token_len"]}
    model      = LaTeXOCRModel(model_cfg).to(device)

    if stage == 2:
        best = Path(cfg["ckpt_dir"]) / "stage1" / "best"
        if best.exists():
            loaded, _ = LaTeXOCRModel.from_checkpoint(str(best), device=str(device))
            model.visual_encoder.load_state_dict(loaded.visual_encoder.state_dict())
            if is_master:
                print(f"Loaded stage1 weights from {best}")

    if resume:
        loaded, _ = LaTeXOCRModel.from_checkpoint(resume, device=str(device))
        model.visual_encoder.load_state_dict(loaded.visual_encoder.state_dict())
        if is_master:
            print(f"Resumed from {resume}")

    if stage == 1:
        model.freeze_decoder()
        if is_master:
            print("[Stage 1] Decoder frozen.")
    else:
        model.unfreeze_lora()

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=use_lora)
    raw   = model.module

    tokenizer = raw.tokenizer
    train_loader, num_samples = build_loader(data_path, stage_name, tokenizer, cfg, rank, world_size)
    val_loader,   _           = build_loader(data_path, "validation",  tokenizer, cfg, rank, world_size)

    num_samples   = num_samples or cfg.get("num_samples", 659658)
    num_epochs    = cfg[f"stage{stage}_epochs"]
    lr            = cfg[f"lr_stage{stage}"]
    steps_per_ep  = (num_samples // world_size) // cfg["batch_size"]
    max_steps     = steps_per_ep * num_epochs // cfg["grad_accum"]
    warmup_steps  = max(int(max_steps * 0.05), 1)
    eval_every    = cfg["eval_steps"]
    save_every    = cfg["save_steps"]

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=cfg["weight_decay"])
    scaler    = torch.amp.GradScaler("cuda")

    global_step  = 0
    best_bleu    = -1.0
    t0           = time.time()
    grad_accum   = cfg["grad_accum"]
    micro_step   = 0
    accum_loss   = 0.0

    data_iter = iter(train_loader)

    while global_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            try:
                batch = next(data_iter)
            except StopIteration:
                break

        batch   = move_batch(batch, device)
        is_sync = (micro_step == grad_accum - 1)

        with model.no_sync() if not is_sync else contextlib.nullcontext():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out  = model(**batch)
                loss = out.loss / grad_accum
            scaler.scale(loss).backward()
        accum_loss += loss.item()
        micro_step += 1

        if not is_sync:
            continue

        cur_lr = get_lr(global_step, max_steps, lr, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        micro_step   = 0
        step_loss    = accum_loss
        accum_loss   = 0.0

        if is_master and global_step % 50 == 0:
            dt = time.time() - t0
            print(f"[Stage {stage}] step={global_step}/{max_steps}  "
                  f"loss={step_loss:.4f}  lr={cur_lr:.2e}  t={dt:.1f}s")
            t0 = time.time()

        if global_step % eval_every == 0:
            max_eval = cfg.get("eval_samples", 200) // cfg["batch_size"]
            metrics  = eval_loop(model, val_loader, device, world_size,
                                 tokenizer, max_batches=max_eval)
            if is_master:
                print_metrics(metrics, prefix=f"Stage{stage} step={global_step}")
                if metrics["bleu4"] > best_bleu:
                    best_bleu = metrics["bleu4"]
                    raw.save_checkpoint(
                        str(ckpt_dir / "best"),
                        step=global_step, metrics=metrics,
                    )
                    print(f"  New best saved (bleu4={best_bleu:.4f})")
            dist.barrier()

        if is_master and global_step % save_every == 0:
            raw.save_checkpoint(
                str(ckpt_dir / f"step-{global_step}"),
                step=global_step, optimizer=optimizer,
            )
        dist.barrier()

    if is_master:
        raw.save_checkpoint(str(ckpt_dir / "final"), step=global_step, optimizer=optimizer)
        tokenizer.save_pretrained(str(ckpt_dir / "final"))
        print(f"[Stage {stage}] Done. Final checkpoint saved.")
    dist.barrier()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     type=str, default="config.yaml")
    p.add_argument("--data_path",  type=str, required=True)
    p.add_argument("--resume",     type=str, default=None)
    p.add_argument("--only_stage", type=int, choices=[1, 2], default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    rank, local_rank, world_size = setup_ddp()
    torch.manual_seed(cfg.get("seed", 42) + rank)

    if rank == 0:
        Path(cfg["ckpt_dir"]).mkdir(parents=True, exist_ok=True)

    only = args.only_stage
    if only in (None, 1):
        if rank == 0:
            print("\n=== STAGE 1: Freeze decoder, train encoder + bridge ===")
        run_stage(1, cfg, args.data_path, rank, local_rank, world_size,
                  resume=args.resume if only == 1 else None)

    if only in (None, 2):
        if rank == 0:
            print("\n=== STAGE 2: Train encoder + bridge + LoRA decoder ===")
        run_stage(2, cfg, args.data_path, rank, local_rank, world_size,
                  resume=args.resume if only == 2 else None)

    cleanup_ddp()
    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    main()
