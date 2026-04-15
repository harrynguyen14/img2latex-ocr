import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

try:
    from safetensors.torch import save_model, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from config import DecoderConfig
from model import LaTeXDecoder
from dataset import PretrainDataset, build_dataloader
from tokenizer import load_tokenizer


def cosine_with_warmup(optimizer: AdamW, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def _make_optimizer(model: LaTeXDecoder, cfg: DecoderConfig) -> AdamW:
    no_decay = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) or "norm" in name:
            for pname, _ in module.named_parameters(prefix=name):
                no_decay.add(pname)
    return AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if n not in no_decay and p.requires_grad],
             "weight_decay": cfg.weight_decay},
            {"params": [p for n, p in model.named_parameters() if n in no_decay and p.requires_grad],
             "weight_decay": 0.0},
        ],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
    )


def save_checkpoint(model, optimizer, scheduler, step, loss, out_dir, keep_last_n):
    ckpt_dir = out_dir / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if HAS_SAFETENSORS:
        save_model(model, str(ckpt_dir / "model.safetensors"))
    else:
        torch.save(model.state_dict(), str(ckpt_dir / "model.pt"))

    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step, "loss": loss},
        str(ckpt_dir / "trainer.pt"),
    )
    with open(ckpt_dir / "train_info.json", "w") as f:
        json.dump({"step": step, "loss": round(loss, 6)}, f)

    all_ckpts = sorted(out_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    while len(all_ckpts) > keep_last_n:
        old = all_ckpts.pop(0)
        for fi in old.iterdir():
            fi.unlink()
        old.rmdir()


def load_checkpoint(model, optimizer, scheduler, ckpt_dir) -> int:
    if HAS_SAFETENSORS and (ckpt_dir / "model.safetensors").exists():
        model.load_state_dict(load_file(str(ckpt_dir / "model.safetensors")), strict=False)
    else:
        model.load_state_dict(torch.load(str(ckpt_dir / "model.pt"), map_location="cpu"), strict=False)
    trainer = torch.load(str(ckpt_dir / "trainer.pt"), map_location="cpu")
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    return trainer["step"]


def find_latest_checkpoint(out_dir: Path) -> Path | None:
    ckpts = sorted(out_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    return ckpts[-1] if ckpts else None


@torch.no_grad()
def evaluate(model: LaTeXDecoder, val_loader, device, amp_dtype, cfg: DecoderConfig) -> dict:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in val_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                loss = model.compute_loss(input_ids, attention_mask)
        else:
            loss = model.compute_loss(input_ids, attention_mask)

        total_loss += loss.item()
        total_batches += 1

    model.train()

    if total_batches == 0:
        return {"val_loss": float("inf"), "val_ppl": float("inf")}

    avg_loss = total_loss / total_batches
    ppl = math.exp(min(avg_loss, 20.0))
    return {"val_loss": round(avg_loss, 4), "val_ppl": round(ppl, 2)}


def train(cfg: DecoderConfig, resume: bool = True):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16  = cfg.dtype == "bfloat16" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if cfg.dtype == "float16" else None)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True

    print(f"device={device}  amp_dtype={amp_dtype}")

    tok          = load_tokenizer(cfg.tokenizer_dir)
    train_ds     = PretrainDataset(tok, cfg, split="train")
    train_loader = build_dataloader(train_ds, cfg.batch_size, num_workers=cfg.num_workers)
    val_loaders  = {
        name: build_dataloader(PretrainDataset(tok, cfg, split=name), cfg.batch_size, num_workers=cfg.num_workers)
        for name in ("val_raw", "val_light", "val_heavy")
    }

    model     = LaTeXDecoder(cfg).to(device)
    optimizer = _make_optimizer(model, cfg)
    scheduler = cosine_with_warmup(optimizer, cfg.warmup_steps, cfg.max_steps)
    scaler    = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    # Load checkpoint BEFORE torch.compile so optimizer state refs match
    start_step = 0
    if resume:
        ckpt = Path(resume) if (isinstance(resume, str) and resume) else find_latest_checkpoint(out_dir)
        if ckpt and Path(ckpt).exists():
            print(f"Resuming from {ckpt}")
            start_step = load_checkpoint(model, optimizer, scheduler, Path(ckpt))
        else:
            print("No checkpoint found, starting from scratch")

    if cfg.compile:
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    print(model)

    cfg.save(out_dir / "config.json")

    model.train()
    step          = start_step
    data_iter     = iter(train_loader)
    running_loss  = 0.0
    running_gnorm = 0.0
    accum_loss    = 0.0

    best_val_ppl      = float("inf")
    no_improve_count  = 0
    early_stop        = False

    pbar = tqdm(total=cfg.max_steps, initial=start_step, desc="Training",
                unit="step", dynamic_ncols=True, file=sys.stdout,
                position=0, leave=True)

    while step < cfg.max_steps and not early_stop:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(cfg.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    loss = model.compute_loss(input_ids, attention_mask) / cfg.grad_accum_steps
            else:
                loss = model.compute_loss(input_ids, attention_mask) / cfg.grad_accum_steps

            if amp_dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()

        if amp_dtype == torch.float16:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        scheduler.step()
        step += 1
        running_loss  += accum_loss
        running_gnorm += grad_norm.item()
        pbar.update(1)

        do_log  = (step % cfg.log_every_steps == 0)
        do_eval = (step % cfg.eval_every_steps == 0)
        do_save = (step % cfg.save_every_steps == 0)

        if do_eval:
            eval_metrics = {}
            avg_val_ppl  = 0.0
            for name, vloader in val_loaders.items():
                m = evaluate(model, vloader, device, amp_dtype, cfg)
                short = name.replace("val_", "")
                eval_metrics[f"ppl_{short}"]  = m["val_ppl"]
                eval_metrics[f"loss_{short}"] = m["val_loss"]
                if name != "val_heavy":
                    avg_val_ppl += m["val_ppl"]
            avg_val_ppl /= 2

            if avg_val_ppl < best_val_ppl:
                best_val_ppl     = avg_val_ppl
                no_improve_count = 0
                save_checkpoint(model, optimizer, scheduler, step, accum_loss, out_dir, cfg.keep_last_n_ckpt)
                tqdm.write(f"  [best] avg_val_ppl={avg_val_ppl:.2f} — checkpoint saved")
                do_save = False
            else:
                no_improve_count += 1
                if no_improve_count >= cfg.early_stopping_patience:
                    tqdm.write(f"  [early stop] val_ppl did not improve for {cfg.early_stopping_patience} evals. Best={best_val_ppl:.2f}")
                    early_stop = True
        else:
            eval_metrics = {}

        if do_log:
            avg_loss  = running_loss  / cfg.log_every_steps
            avg_gnorm = running_gnorm / cfg.log_every_steps
            lr_now    = scheduler.get_last_lr()[0]
            train_ppl = math.exp(min(avg_loss, 20.0))
            logs = {
                "ppl":       round(train_ppl, 2),
                "loss":      round(avg_loss,  4),
                "grad_norm": round(avg_gnorm, 4),
                "lr":        f"{lr_now:.2e}",
                "step":      step,
            }
            if eval_metrics:
                logs.update(eval_metrics)
            if do_save:
                logs["saved"] = True
            tqdm.write(str(logs))
            running_loss  = 0.0
            running_gnorm = 0.0
        elif do_eval and eval_metrics:
            lr_now = scheduler.get_last_lr()[0]
            logs = {"step": step, "lr": f"{lr_now:.2e}"}
            logs.update(eval_metrics)
            tqdm.write(str(logs))

        if do_save:
            save_checkpoint(model, optimizer, scheduler, step, accum_loss, out_dir, cfg.keep_last_n_ckpt)
            if not do_log:
                tqdm.write(f"  checkpoint saved at step {step}")

    pbar.close()
    if not early_stop:
        save_checkpoint(model, optimizer, scheduler, step, accum_loss, out_dir, 999)
    print(f"Training done at step {step}. Best val_ppl={best_val_ppl:.2f}")
    return model
