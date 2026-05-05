"""
Fine-tune Nav2TexModel (from_pretrained format) on train_filtered data.
Reuses im2latex dataset/dataloader pipeline, no decoder freezing.

Usage:
    python train.py \
        --model_path D:/img2latex/latex_ocr \
        --data_path  D:/dataset-ocr-builder/latex-ocr-dataset/ocr-data \
        --ckpt_dir   D:/img2latex/checkpoints \
        --resume     D:/img2latex/checkpoints/step_0050000  # optional
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Token indices sequence length is longer")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

try:
    from safetensors.torch import save_file as st_save, load_file as st_load
    HAS_ST = True
except ImportError:
    HAS_ST = False

sys.path.insert(0, str(Path(__file__).parent))
from im2latex.preprocessor import Nav2TexParquetDataset, Nav2TexFlatParquetDataset
from im2latex.build_datasets import build_dataloader
from im2latex.utils import collate_fn, move_batch
from nav2tex.tokenization_latex_ocr import LaTeXTokenizer


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",    type=str, required=True)
    ap.add_argument("--data_path",     type=str, required=True)
    ap.add_argument("--ckpt_dir",      type=str, default="checkpoints")
    ap.add_argument("--resume",        type=str, default=None)

    ap.add_argument("--sources",       nargs="+", default=["raw", "light", "heavy", "screenshot"])
    ap.add_argument("--weights",       nargs="+", type=float, default=[1.0, 1.0, 1.0, 0.5])
    ap.add_argument("--resize_in_dataset", action="store_true", default=True)

    ap.add_argument("--batch_size",       type=int,   default=4)
    ap.add_argument("--eval_batch_size",  type=int,   default=4)
    ap.add_argument("--grad_accum",       type=int,   default=8)
    ap.add_argument("--lr",               type=float, default=3e-5)
    ap.add_argument("--weight_decay",     type=float, default=0.01)
    ap.add_argument("--max_grad_norm",    type=float, default=1.0)
    ap.add_argument("--warmup_ratio",     type=float, default=0.03)
    ap.add_argument("--max_steps",        type=int,   default=50000)
    ap.add_argument("--log_steps",        type=int,   default=50)
    ap.add_argument("--val_loss_steps",   type=int,   default=1000)
    ap.add_argument("--save_steps",       type=int,   default=5000)
    ap.add_argument("--eval_samples",     type=int,   default=512)
    ap.add_argument("--num_workers",      type=int,   default=2)
    ap.add_argument("--prefetch_factor",  type=int,   default=4)
    ap.add_argument("--persistent_workers", action="store_true", default=True)
    ap.add_argument("--seed",             type=int,   default=42)
    ap.add_argument("--keep_last_n",      type=int,   default=3)
    return ap.parse_args()


# ── Scheduler ─────────────────────────────────────────────────────────────────
def cosine_with_warmup(optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ── Optimizer ─────────────────────────────────────────────────────────────────
def make_optimizer(model, lr, weight_decay):
    no_decay = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) or "norm" in name:
            for pname, _ in module.named_parameters(prefix=name):
                no_decay.add(pname)
    params = list(model.named_parameters())
    return AdamW(
        [
            {"params": [p for n, p in params if n not in no_decay], "weight_decay": weight_decay},
            {"params": [p for n, p in params if n in no_decay],     "weight_decay": 0.0},
        ],
        lr=lr, betas=(0.9, 0.95), eps=1e-8,
    )


# ── Checkpoint ────────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, step, ckpt_dir, keep_last_n, model_path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {f"visual_encoder.{k}": v.contiguous().cpu()
             for k, v in model.visual_encoder.state_dict().items()}
    state.update({f"decoder.{k}": v.contiguous().cpu()
                  for k, v in model.decoder.state_dict().items()})

    if HAS_ST:
        st_save(state, ckpt_dir / "model.safetensors")
    else:
        torch.save(state, ckpt_dir / "model.pt")

    torch.save({"optimizer": optimizer.state_dict(), "step": step}, ckpt_dir / "optimizer.pt")
    torch.save({"scheduler": scheduler.state_dict(), "step": step}, ckpt_dir / "scheduler.pt")
    (ckpt_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": step}, indent=2), encoding="utf-8"
    )

    # copy model configs so checkpoint is self-contained
    for fname in ("config.json", "preprocessor_config.json",
                  "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"):
        src = Path(model_path) / fname
        if src.exists():
            shutil.copy2(src, ckpt_dir / fname)

    # prune old checkpoints
    parent = ckpt_dir.parent
    all_ckpts = sorted(parent.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    while len(all_ckpts) > keep_last_n:
        old = all_ckpts.pop(0)
        shutil.rmtree(old)
    tqdm.write(f"[ckpt] saved → {ckpt_dir}")


def load_resume(model, optimizer, scheduler, resume_dir):
    resume_dir = Path(resume_dir)
    sf = resume_dir / "model.safetensors"
    step = 0

    if not sf.exists():
        print(f"[resume] no model.safetensors in {resume_dir}, starting fresh")
        return 0
    state = st_load(str(sf))

    # strip prefix and load
    ve_state  = {k[len("visual_encoder."):]: v for k, v in state.items() if k.startswith("visual_encoder.")}
    dec_state = {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}
    if ve_state:
        model.visual_encoder.load_state_dict(ve_state, strict=True)
    if dec_state:
        model.decoder.load_state_dict(dec_state, strict=True)
        model.decoder.tie_weights()
    tqdm.write(f"[resume] model weights loaded from {resume_dir}")

    opt_pt = resume_dir / "optimizer.pt"
    if opt_pt.exists():
        ts = torch.load(str(opt_pt), map_location="cpu")
        step = ts.get("step", 0)
        device = next(model.parameters()).device
        for s in ts["optimizer"]["state"].values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(device)
        try:
            optimizer.load_state_dict(ts["optimizer"])
        except Exception as e:
            tqdm.write(f"[resume] optimizer mismatch ({e}), using fresh")

    sched_pt = resume_dir / "scheduler.pt"
    if sched_pt.exists():
        ts = torch.load(str(sched_pt), map_location="cpu")
        try:
            scheduler.load_state_dict(ts.get("scheduler", ts))
        except Exception:
            tqdm.write("[resume] scheduler mismatch, skipping")

    tqdm.write(f"[resume] step={step}")
    return step


# ── Val metrics ───────────────────────────────────────────────────────────────
@torch.no_grad()
def run_val(model, loader, device, max_batches, tokenizer=None, gen_batches=8):
    from im2latex.evaluate import compute_metrics
    model.eval()
    amp = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                         enabled=device.type == "cuda")
    total_loss = 0.0
    correct = 0
    total_tokens = 0
    n = 0
    predictions, references = [], []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch(batch, device)
        with amp:
            out = model(batch["batched_images"], batch["input_ids"],
                        batch["attention_mask"], batch["labels"])

        total_loss += out.hidden_states[0].item()

        # token_acc: slice text portion only (logits includes visual tokens prefix)
        logits = out.last_hidden_state                   # (B, vis_len+text_len, vocab)
        labels = batch["labels"]                         # (B, text_len)
        text_logits  = logits[:, -labels.shape[1]:]     # (B, text_len, vocab)
        shift_logits = text_logits[:, :-1]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        preds = shift_logits.argmax(dim=-1)
        correct      += (preds[mask] == shift_labels[mask]).sum().item()
        total_tokens += mask.sum().item()

        # collect predictions for BLEU/exact_match (only first gen_batches batches)
        if tokenizer is not None and i < gen_batches:
            with amp:
                gen_ids = model.generate(batch["batched_images"], num_beams=1)
            skip_ids = {tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id}
            for j in range(gen_ids.shape[0]):
                pred = "".join(tokenizer.convert_ids_to_tokens(
                               [t for t in gen_ids[j].tolist() if t not in skip_ids]))
                ref_ids = [t for t in batch["labels"][j].tolist() if t >= 0 and t not in skip_ids]
                ref = "".join(tokenizer.convert_ids_to_tokens(ref_ids))
                if i == 0 and j < 2:
                    tqdm.write(f"[debug] pred: {pred[:100]}")
                    tqdm.write(f"[debug] ref:  {ref[:100]}")
                predictions.append(pred)
                references.append(ref)
        n += 1

    model.train()
    if n == 0:
        return {"val_loss": float("inf"), "val_ppl": float("inf"), "token_acc": 0.0}
    avg = total_loss / n
    token_acc = correct / max(total_tokens, 1)
    metrics = {
        "val_loss":  round(avg, 4),
        "val_ppl":   round(math.exp(min(avg, 20.0)), 2),
        "token_acc": round(token_acc, 4),
    }
    if predictions:
        gen_metrics = compute_metrics(predictions, references)
        metrics["bleu4"]        = gen_metrics["bleu4"]
        metrics["exact_match"]  = gen_metrics["exact_match"]
        metrics["edit_distance"]= gen_metrics["edit_distance"]
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ── Load model ──
    print(f"Loading model from {args.model_path} …")
    model_path = Path(args.model_path)
    if not model_path.exists():
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(args.model_path))
        print(f"Downloaded to {model_path}")

    sys.path.insert(0, str(model_path))
    from nav2tex.modeling_latex_ocr import Nav2TexModel
    from nav2tex.configuration_latex_ocr import Nav2TexConfig

    config = Nav2TexConfig.from_pretrained(str(model_path))
    model  = Nav2TexModel.from_pretrained(str(model_path), config=config).to(device)
    model.train()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


    # inject image/token params from model config so dataset preprocessing is always in sync
    args.image_height    = config.image_height
    args.max_image_width = config.max_image_width
    args.patch_size      = config.patch_size
    args.max_token_len   = config.decoder_arch["max_seq_len"]
    print(f"[config] image_height={args.image_height}  max_image_width={args.max_image_width}"
          f"  patch_size={args.patch_size}  max_token_len={args.max_token_len}")

    # ── Tokenizer ──
    tokenizer = LaTeXTokenizer(str(model_path / "tokenizer.json"),
                               model_max_length=args.max_token_len)

    # ── Datasets ──
    local_data = Path(args.data_path)
    is_local   = local_data.exists()

    if is_local:
        train_dir = local_data / "train"
        val_dir   = local_data / "validation"
        if not train_dir.exists():
            raise FileNotFoundError(f"train/ not found at {train_dir}")
        if not val_dir.exists():
            raise FileNotFoundError(f"validation/ not found at {val_dir}")
        train_ds = Nav2TexParquetDataset(
            str(train_dir), args.sources, args.weights, tokenizer, args,
        )
        val_ds = Nav2TexFlatParquetDataset(str(val_dir), tokenizer, args)
        print(f"[dataset] local  → {local_data}")
    else:
        from im2latex.preprocessor import Nav2TexHFDataset
        train_ds = Nav2TexHFDataset(
            args.data_path, "train", tokenizer, args,
            names=args.sources, weights=args.weights,
        )
        val_ds = Nav2TexHFDataset(args.data_path, "validation", tokenizer, args)
        print(f"[dataset] streaming → {args.data_path}")

    nw = args.num_workers
    pf = args.prefetch_factor if nw > 0 else None
    pw = args.persistent_workers and nw > 0
    train_loader = build_dataloader(train_ds, args.batch_size,      nw, collate_fn, device.type == "cuda", pf, pw)
    val_loader   = build_dataloader(val_ds,   args.eval_batch_size, nw, collate_fn, device.type == "cuda", pf, pw)

    # ── Optimizer / scheduler ──
    warmup_steps = max(1, int(args.max_steps * args.warmup_ratio))
    optimizer  = make_optimizer(model, args.lr, args.weight_decay)
    scheduler  = cosine_with_warmup(optimizer, warmup_steps, args.max_steps)
    ckpt_dir   = Path(args.ckpt_dir)

    global_step = 0
    if args.resume:
        global_step = load_resume(model, optimizer, scheduler, args.resume)

    # ── Training loop ──
    accum      = args.grad_accum
    micro      = 0
    accum_loss = 0.0
    best_ppl   = float("inf")
    data_iter  = iter(train_loader)
    amp        = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda")

    pbar = tqdm(total=args.max_steps, initial=global_step,
                desc="Train", unit="step", dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = move_batch(batch, device)
        with amp:
            out  = model(batch["batched_images"], batch["input_ids"],
                         batch["attention_mask"], batch["labels"])
            loss = out.hidden_states[0] / accum

        loss.backward()
        accum_loss += out.hidden_states[0].item()
        micro += 1

        if micro < accum:
            continue

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        micro = 0
        pbar.update(1)

        if global_step % args.log_steps == 0:
            lr_now = scheduler.get_last_lr()[0]
            tqdm.write(str({
                "step":      global_step,
                "loss":      round(accum_loss, 4),
                "ppl":       round(math.exp(min(accum_loss, 20.0)), 2),
                "grad_norm": round(grad_norm, 4),
                "lr":        f"{lr_now:.2e}",
            }))
        accum_loss = 0.0

        if global_step % args.val_loss_steps == 0:
            max_val = max(args.eval_samples // args.eval_batch_size, 1)
            val_metrics = run_val(model, val_loader, device, max_val, tokenizer=tokenizer)
            tqdm.write(str({"step": global_step, **val_metrics}))
            if val_metrics["val_ppl"] < best_ppl:
                best_ppl = val_metrics["val_ppl"]
                save_checkpoint(model, optimizer, scheduler, global_step,
                                ckpt_dir / "best", keep_last_n=999, model_path=args.model_path)
                tqdm.write(f"  [best] val_ppl={best_ppl:.2f}  token_acc={val_metrics['token_acc']:.4f}")

        if global_step % args.save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, global_step,
                            ckpt_dir / f"step_{global_step:07d}",
                            keep_last_n=args.keep_last_n, model_path=args.model_path)

    pbar.close()
    save_checkpoint(model, optimizer, scheduler, global_step,
                    ckpt_dir / "final", keep_last_n=999, model_path=args.model_path)
    print(f"Done. best val_ppl={best_ppl:.2f}")


if __name__ == "__main__":
    main()
