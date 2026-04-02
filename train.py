import os
import argparse
import contextlib
import yaml
import math
import numpy as np
import torch
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from pathlib import Path

from constants import TOKENIZER_NAME
from dataset import LaTeXDataset, get_tokenizer
from collator import LaTeXDataCollator
from modeling import LaTeXOCRModel
from evaluate import compute_metrics, print_metrics


def load_config(path: str) -> dict:
    with open(path) as f:
        nested = yaml.safe_load(f)
    config = {}
    for section in nested.values():
        if isinstance(section, dict):
            config.update(section)
    return config


def get_lr(step: int, max_steps: int, peak_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def setup_ddp():
    # Suppress hostname resolution warnings in container environments
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "eth0")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def build_dataloader(data_path: str, split: str, tokenizer, config: dict,
                     rank: int, world_size: int) -> DataLoader:
    dataset = LaTeXDataset(data_path, split, tokenizer, rank=rank, world_size=world_size)
    loader  = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        collate_fn=LaTeXDataCollator(),
        num_workers=config["num_workers"],
        pin_memory=True,
        prefetch_factor=2 if config["num_workers"] > 0 else None,
        persistent_workers=config["num_workers"] > 0,
    )
    return loader, dataset.num_samples


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _gather_strings(local_strings: list, world_size: int, device: torch.device) -> list:
    """Gather string lists from all DDP ranks onto every rank."""
    # Encode to bytes, pad to same length, all_gather, decode
    encoded   = [s.encode("utf-8") for s in local_strings]
    max_len   = max((len(b) for b in encoded), default=1)

    # Sync max_len across ranks
    max_len_t = torch.tensor(max_len, device=device)
    dist.all_reduce(max_len_t, op=dist.ReduceOp.MAX)
    max_len = max_len_t.item()

    padded = torch.zeros(len(encoded), max_len, dtype=torch.uint8, device=device)
    for i, b in enumerate(encoded):
        padded[i, :len(b)] = torch.frombuffer(b, dtype=torch.uint8)

    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    results = []
    for rank_tensor in gathered:
        for row in rank_tensor:
            byte_seq = row.cpu().numpy().tobytes().rstrip(b"\x00")
            results.append(byte_seq.decode("utf-8"))
    return results


@torch.no_grad()
def evaluate(model, loader, device: torch.device,
             tokenizer, world_size: int, max_batches: int = None) -> dict:
    unwrapped_model = model.module if isinstance(model, DDP) else model
    unwrapped_model.eval()

    total_loss  = torch.tensor(0.0, device=device)
    total_count = torch.tensor(0,   device=device)
    local_preds, local_refs = [], []

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = unwrapped_model(**batch)
        total_loss  += output.loss.detach()
        total_count += 1

        preds     = unwrapped_model.generate(batch["pixel_values"], batch["patch_mask"])
        label_ids = batch["labels"].cpu().numpy()
        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
        refs      = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        local_preds.extend(preds)
        local_refs.extend(refs)

    # Aggregate loss across all ranks
    dist.all_reduce(total_loss,  op=dist.ReduceOp.SUM)
    dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
    avg_loss = (total_loss / total_count).item()

    # Gather predictions and references from all ranks for consistent metrics
    all_preds = _gather_strings(local_preds, world_size, device)
    all_refs  = _gather_strings(local_refs,  world_size, device)

    metrics = compute_metrics(all_preds, all_refs)
    metrics["loss"] = round(avg_loss, 4)
    unwrapped_model.train()
    return metrics


def run_training_phase(phase: str, config: dict, data_path: str,
                       rank: int, local_rank: int, world_size: int,
                       resume: str = None):
    """
    phase: "pretrain" (encoder + bridge + LoRA decoder) or "lora_finetune" (LoRA adapters only).
    Both phases use LoRA on the decoder to keep optimizer state memory within T4 VRAM budget.
    """
    device    = torch.device(f"cuda:{local_rank}")
    is_master = rank == 0
    ckpt_dir  = Path(config["ckpt_dir"]) / phase

    # Always use LoRA: full decoder AdamW states (~6GB) exceed T4 15GB budget
    model_config = {**config, "use_lora": True, "max_new_tokens": config["max_token_len"]}
    model        = LaTeXOCRModel(model_config).to(device)

    if phase == "pretrain" and is_master:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Pretrain] Trainable: {trainable_params / 1e6:.1f}M params "
              f"(encoder + bridge + LoRA adapters).")

    if phase == "lora_finetune":
        pretrain_best = Path(config["ckpt_dir"]) / "pretrain" / "best"
        if pretrain_best.exists():
            pretrained, _ = LaTeXOCRModel.from_checkpoint(str(pretrain_best), device=str(device))
            model.visual_encoder.load_state_dict(pretrained.visual_encoder.state_dict())
            pretrained_lora = {k: v for k, v in pretrained.decoder.state_dict().items()
                               if "lora_" in k}
            if pretrained_lora:
                model.decoder.load_state_dict(pretrained_lora, strict=False)
            if is_master:
                print(f"Loaded pretrain weights from {pretrain_best}")
        model.freeze_for_lora_finetuning()
        if is_master:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[LoRA finetune] Encoder + decoder base frozen. "
                  f"Trainable: {trainable_params / 1e6:.1f}M params (LoRA adapters only).")

    if resume:
        resumed, _ = LaTeXOCRModel.from_checkpoint(resume, device=str(device))
        model.visual_encoder.load_state_dict(resumed.visual_encoder.state_dict())
        if is_master:
            print(f"Resumed from {resume}")

    model           = DDP(model, device_ids=[local_rank], find_unused_parameters=True,
                          gradient_as_bucket_view=True)
    model._set_static_graph()
    unwrapped_model = model.module
    tokenizer       = unwrapped_model.tokenizer

    training_split = "stage1" if phase == "pretrain" else "stage2"
    train_loader, num_samples = build_dataloader(data_path, training_split, tokenizer, config, rank, world_size)
    val_loader,   _           = build_dataloader(data_path, "validation", tokenizer, config, rank, world_size)

    num_samples  = num_samples or config.get("num_samples", 659658)
    num_epochs   = config["pretrain_epochs"] if phase == "pretrain" else config["lora_finetune_epochs"]
    peak_lr      = config["pretrain_lr"]     if phase == "pretrain" else config["lora_finetune_lr"]
    grad_accum   = config["grad_accum"]
    steps_per_ep = (num_samples // world_size) // config["batch_size"]
    max_steps    = steps_per_ep * num_epochs // grad_accum
    warmup_steps = max(int(max_steps * 0.05), 1)
    eval_every   = config["eval_steps"]
    save_every   = config["save_steps"]

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=peak_lr, weight_decay=config["weight_decay"])
    scaler    = torch.amp.GradScaler("cuda")

    log_every   = config.get("log_steps", 10)
    global_step = 0
    best_bleu   = -1.0
    micro_step  = 0
    accum_loss  = 0.0
    ema_loss    = 0.0
    epoch       = 0

    data_iter = iter(train_loader)
    pbar = tqdm(total=max_steps, desc=f"{phase} epoch 1/{num_epochs}",
                disable=not is_master, dynamic_ncols=True)

    while global_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch    += 1
            data_iter = iter(train_loader)
            if is_master:
                pbar.set_description(f"{phase} epoch {epoch + 1}/{num_epochs}")
            try:
                batch = next(data_iter)
            except StopIteration:
                break

        batch   = move_batch_to_device(batch, device)
        is_sync = (micro_step == grad_accum - 1)

        with model.no_sync() if not is_sync else contextlib.nullcontext():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(**batch)
                loss   = output.loss / grad_accum
            scaler.scale(loss).backward()
        accum_loss += loss.item()
        micro_step += 1

        if not is_sync:
            continue

        cur_lr = get_lr(global_step, max_steps, peak_lr, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, config["max_grad_norm"])

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scaler_skipped = scaler.get_scale() < scale_before

        global_step += 1
        micro_step   = 0
        cur_loss     = accum_loss * grad_accum   # restore true loss scale
        accum_loss   = 0.0
        ema_loss     = 0.9 * ema_loss + 0.1 * cur_loss if global_step > 1 else cur_loss

        cur_epoch   = global_step / max_steps * num_epochs
        update_norm = (grad_norm * cur_lr).item() if not scaler_skipped else 0.0
        pbar.update(1)

        if is_master and global_step % log_every == 0:
            log_dict = {
                "loss":        round(ema_loss, 3),
                "grad_norm":   round(grad_norm.item(), 2) if not scaler_skipped else "nan",
                "update_norm": f"{update_norm:.3e}",
                "lr":          f"{cur_lr:.2e}",
                "epoch":       f"{cur_epoch:.6f}",
            }
            print(log_dict)

        if global_step % eval_every == 0:
            max_eval_batches = config.get("eval_samples", 200) // config["batch_size"]
            metrics = evaluate(model, val_loader, device,
                               tokenizer, world_size, max_batches=max_eval_batches)
            if is_master:
                print_metrics(metrics, prefix=f"{phase} step={global_step}")
                if metrics["bleu4"] > best_bleu:
                    best_bleu = metrics["bleu4"]
                    unwrapped_model.save_checkpoint(
                        str(ckpt_dir / "best"),
                        step=global_step, metrics=metrics,
                    )
                    print(f"  New best saved (bleu4={best_bleu:.4f})")
            dist.barrier()

        if is_master and global_step % save_every == 0:
            unwrapped_model.save_checkpoint(
                str(ckpt_dir / f"step-{global_step}"),
                step=global_step, optimizer=optimizer,
            )
        dist.barrier()

    pbar.close()

    if is_master:
        unwrapped_model.save_checkpoint(
            str(ckpt_dir / "final"), step=global_step, optimizer=optimizer
        )
        tokenizer.save_pretrained(str(ckpt_dir / "final"))
        print(f"[{phase}] Done. Final checkpoint saved.")
    dist.barrier()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       type=str, default="config.yaml")
    p.add_argument("--data_path",    type=str, required=True)
    p.add_argument("--resume",       type=str, default=None)
    p.add_argument("--only_phase",   type=str, choices=["pretrain", "lora_finetune"], default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cudnn.benchmark = True

    rank, local_rank, world_size = setup_ddp()
    torch.manual_seed(config.get("seed", 42) + rank)

    print(f"[rank={rank}] local_rank={local_rank} world_size={world_size} "
          f"device=cuda:{local_rank} ({torch.cuda.get_device_name(local_rank)})")

    if rank == 0:
        Path(config["ckpt_dir"]).mkdir(parents=True, exist_ok=True)

    only_phase = args.only_phase
    if only_phase in (None, "pretrain"):
        if rank == 0:
            print("\n=== PHASE 1: End-to-end pretraining (encoder + bridge + decoder) ===")
        run_training_phase("pretrain", config, args.data_path, rank, local_rank, world_size,
                           resume=args.resume if only_phase == "pretrain" else None)

    if only_phase in (None, "lora_finetune"):
        if rank == 0:
            print("\n=== PHASE 2: LoRA finetuning (frozen encoder + decoder base, LoRA adapters only) ===")
        run_training_phase("lora_finetune", config, args.data_path, rank, local_rank, world_size,
                           resume=args.resume if only_phase == "lora_finetune" else None)

    cleanup_ddp()
    if rank == 0:
        print("Training complete.")


if __name__ == "__main__":
    main()
