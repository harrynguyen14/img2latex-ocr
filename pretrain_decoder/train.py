import contextlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from safetensors.torch import save_file, load_file
from tqdm import tqdm

from config import get_config
from dataset import build_dataloader
from model import DecoderLM

def _flatten_tensors(d: dict, prefix: str) -> tuple[dict, dict]:
    """Recursively flatten nested dict into {prefix/key: tensor}, non-tensors into scalars dict."""
    tensors, scalars = {}, {}
    for k, v in d.items():
        full_key = f"{prefix}/{k}"
        if isinstance(v, torch.Tensor):
            tensors[full_key] = v.cpu()
        elif isinstance(v, dict):
            t, s = _flatten_tensors(v, full_key)
            tensors.update(t)
            scalars.update(s)
        else:
            scalars[full_key] = v
    return tensors, scalars

def _unflatten_tensors(tensors: dict, scalars: dict, prefix: str) -> dict:
    """Reconstruct nested dict from flat tensors + scalars under a given prefix."""
    result = {}
    sub_prefix = prefix + "/"
    for key, val in tensors.items():
        if key.startswith(sub_prefix):
            parts = key[len(sub_prefix):].split("/")
            node = result
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = val
    for key, val in scalars.items():
        if key.startswith(sub_prefix):
            parts = key[len(sub_prefix):].split("/")
            node = result
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = val
    return result

def _make_scheduler(optimizer, config, total_steps) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def _make_optimizer(model: DecoderLM, config) -> AdamW:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name or (name.endswith(".weight") and param.dim() == 1):
            no_decay.append(param)
        else:
            decay.append(param)
    return AdamW(
        [
            {"params": decay,    "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.lr,
    )

def _find_latest_checkpoint(save_dir: Path) -> Path | None:
    ckpts = sorted(save_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    return ckpts[-1] if ckpts else None

def _save_checkpoint(model, optimizer, scheduler, config, step: int, loss: float, save_dir: Path):
    ckpt_dir = save_dir / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    model_sd = {k: v.cpu() for k, v in raw_model.state_dict().items()}
    save_file(model_sd, ckpt_dir / "model.safetensors")

    opt_tensors, opt_scalars = _flatten_tensors(optimizer.state_dict(), "optimizer")
    sch_tensors, sch_scalars = _flatten_tensors({"state": scheduler.state_dict()}, "scheduler")
    trainer_tensors = {**opt_tensors, **sch_tensors}
    trainer_scalars = {**opt_scalars, **sch_scalars, "step": step, "loss": loss}
    metadata = {k: json.dumps(v) for k, v in trainer_scalars.items()}
    if not trainer_tensors:
        trainer_tensors["_sentinel"] = torch.zeros(1)
    save_file(trainer_tensors, ckpt_dir / "trainer.safetensors", metadata=metadata)

    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2)

    import shutil
    tok_src = Path(config.tokenizer_dir)
    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "preprocessor_config.json"):
        src = tok_src / fname
        if src.exists():
            shutil.copy2(src, ckpt_dir / fname)

def _load_checkpoint(model, optimizer, scheduler, ckpt_dir: Path, device) -> int:
    sd = load_file(ckpt_dir / "model.safetensors", device="cpu")
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd)

    trainer_tensors = load_file(ckpt_dir / "trainer.safetensors", device="cpu")
    trainer_tensors.pop("_sentinel", None)
    from safetensors import safe_open
    with safe_open(ckpt_dir / "trainer.safetensors", framework="pt", device="cpu") as f:
        metadata = f.metadata()
    trainer_scalars = {k: json.loads(v) for k, v in metadata.items()}

    opt_sd  = _unflatten_tensors(trainer_tensors, trainer_scalars, "optimizer")
    sch_sd  = _unflatten_tensors(trainer_tensors, trainer_scalars, "scheduler")
    sch_sd  = sch_sd.get("state", sch_sd)

    for state in opt_sd.get("state", {}).values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    optimizer.load_state_dict(opt_sd)
    scheduler.load_state_dict(sch_sd)
    return int(trainer_scalars["step"])

def train():
    config   = get_config()
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.cuda_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device={device}  bf16={config.bf16}  cudnn_benchmark={config.cuda_benchmark}")

    loader = build_dataloader(config, split="train")

    stats = loader.dataset.score_stats()
    print(
        f"dataset: total={stats['total']}  "
        f"spe={stats['n_spe']} ({100-stats['cpe_pct']}%)  "
        f"cpe={stats['n_cpe']} ({stats['cpe_pct']}%)  "
        f"score_median={stats['median']}  score_p95={stats['p95']}"
    )

    steps_per_epoch = len(loader) // config.grad_accum
    total_steps     = steps_per_epoch * config.max_epochs
    print(f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}")

    model     = DecoderLM(config).to(device)
    print(f"parameters: {model.num_parameters() / 1e6:.1f}M")

    optimizer = _make_optimizer(model, config)
    scheduler = _make_scheduler(optimizer, config, total_steps)
    use_bf16  = config.bf16 and device.type == "cuda"
    amp_ctx   = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)

    start_step = 0
    latest = _find_latest_checkpoint(save_dir)
    if latest is not None:
        print(f"resuming from {latest}")
        start_step = _load_checkpoint(model, optimizer, scheduler, latest, device)
        print(f"resumed at step {start_step}")

    if config.compile:
        print("torch.compile: compiling model...")
        model = torch.compile(model)
        print("torch.compile: done")

    model.train()
    step              = start_step
    data_iter         = iter(loader)
    running_loss      = 0.0
    running_lm_loss   = 0.0
    running_len_loss  = 0.0
    running_gnorm     = 0.0
    tokens_seen       = 0
    t0                = time.perf_counter()

    pbar = tqdm(total=total_steps, initial=start_step, desc="train", dynamic_ncols=True)

    while step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss     = 0.0
        accum_lm_loss  = 0.0
        accum_len_loss = 0.0
        batch_tokens   = 0

        for _ in range(config.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids      = batch["input_ids"].to(device)
            labels         = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            true_len       = batch["true_len"].to(device)
            batch_tokens  += attention_mask.sum().item()

            with amp_ctx:
                _sdp_backends = (
                    [SDPBackend.FLASH_ATTENTION]
                    if config.flash_attn
                    else [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
                )
                _sdp_ctx = (
                    sdpa_kernel(_sdp_backends)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with _sdp_ctx:
                    loss, lm_loss, len_loss = model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        true_len=true_len,
                    )
                    loss = loss / config.grad_accum

            loss.backward()
            accum_loss     += loss.item()
            accum_lm_loss  += lm_loss.item() / config.grad_accum
            accum_len_loss += len_loss.item() / config.grad_accum

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm).item()
        optimizer.step()
        scheduler.step()

        step             += 1
        running_loss     += accum_loss
        running_lm_loss  += accum_lm_loss
        running_len_loss += accum_len_loss
        running_gnorm    += grad_norm
        tokens_seen      += batch_tokens

        pbar.update(1)

        if step % config.log_every_n_steps == 0:
            elapsed     = time.perf_counter() - t0
            tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
            n           = config.log_every_n_steps
            avg_loss    = running_loss     / n
            avg_lm      = running_lm_loss  / n
            avg_len     = running_len_loss / n
            avg_gnorm   = running_gnorm    / n
            lr_now      = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}", lm=f"{avg_lm:.4f}", len=f"{avg_len:.4f}",
                gnorm=f"{avg_gnorm:.3f}", lr=f"{lr_now:.2e}", tok_s=f"{tok_per_sec:,.0f}",
            )
            tqdm.write(
                f"step={step:>7d}  loss={avg_loss:.4f}  lm={avg_lm:.4f}  len={avg_len:.4f}"
                f"  gnorm={avg_gnorm:.3f}  lr={lr_now:.2e}  tok/s={tok_per_sec:,.0f}"
            )
            running_loss     = 0.0
            running_lm_loss  = 0.0
            running_len_loss = 0.0
            running_gnorm    = 0.0
            tokens_seen      = 0
            t0               = time.perf_counter()

        if step % config.save_every_n_steps == 0:
            _save_checkpoint(model, optimizer, scheduler, config, step, accum_loss, save_dir)
            tqdm.write(f"  checkpoint saved at step {step}")

    pbar.close()
    _save_checkpoint(model, optimizer, scheduler, config, step, accum_loss, save_dir)
    print(f"training complete at step {step}")

if __name__ == "__main__":
    train()