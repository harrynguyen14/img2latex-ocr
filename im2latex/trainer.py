import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

try:
    from safetensors.torch import save_file as st_save_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from .utils import move_batch
from .latex_ocr_model import LaTeXOCRModel
from .evaluate import compute_metrics, print_metrics


def cosine_with_warmup(optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def _make_optimizer(model: LaTeXOCRModel, lr: float, weight_decay: float) -> AdamW:
    no_decay = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) or "norm" in name:
            for pname, _ in module.named_parameters(prefix=name):
                no_decay.add(pname)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    return AdamW(
        [
            {"params": [p for n, p in trainable if n not in no_decay], "weight_decay": weight_decay},
            {"params": [p for n, p in trainable if n in no_decay],     "weight_decay": 0.0},
        ],
        lr=lr, betas=(0.9, 0.95), eps=1e-8,
    )


def _save_checkpoint(model: LaTeXOCRModel, optimizer, scheduler, step, ckpt_dir: Path, keep_last_n: int):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {f"visual_encoder.{k}": v.contiguous().cpu()
             for k, v in model.visual_encoder.state_dict().items()}

    if HAS_SAFETENSORS:
        st_save_file(state, ckpt_dir / "model.safetensors")
    else:
        torch.save(state, ckpt_dir / "model.pt")

    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step},
        ckpt_dir / "trainer.pt",
    )

    parent = ckpt_dir.parent
    all_ckpts = sorted(parent.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    while len(all_ckpts) > keep_last_n:
        old = all_ckpts.pop(0)
        for fi in old.iterdir():
            fi.unlink()
        old.rmdir()


@torch.no_grad()
def run_val_loss(model: LaTeXOCRModel, loader, device, max_batches: int) -> dict:
    """Fast validation: CE loss only, no generation."""
    model.eval()
    total_loss, total_batches = 0.0, 0
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                             enabled=device.type == "cuda")
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch(batch, device)
        with amp_ctx:
            out = model(
                batch["batched_images"],
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
        total_loss   += out.loss.item()
        total_batches += 1
    model.train()
    if total_batches == 0:
        return {"val_loss": float("inf"), "val_ppl": float("inf")}
    avg = total_loss / total_batches
    return {"val_loss": round(avg, 4), "val_ppl": round(math.exp(min(avg, 20.0)), 2)}


@torch.no_grad()
def run_bleu_eval(model: LaTeXOCRModel, loader, device, tokenizer, max_batches: int) -> dict:
    """Slow validation: generate + BLEU/exact/edit-distance."""
    model.eval()
    preds, refs = [], []
    pad_id = tokenizer.token_to_id("<pad>") or 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch(batch, device)
        gen   = model.generate(batch["batched_images"])   # list[str]
        preds.extend(gen)
        for ids in batch["labels"].cpu().tolist():
            clean = [x for x in ids if x not in (-100, pad_id)]
            refs.append(tokenizer.decode(clean))
    model.train()
    return compute_metrics(preds, refs)


class Trainer:
    def __init__(self, args, train_loader, val_loader, device, tokenizer,
                 distributed=False, rank=0, local_rank=0, world_size=1):
        self.args         = args
        self.device       = device
        self.tokenizer    = tokenizer
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.ckpt_dir     = Path(args.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        warmup_steps     = max(1, int(args.max_steps * args.warmup_ratio))
        self.total_steps = args.max_steps
        self.global_step = 0
        self.best_val_ppl = float("inf")

        self.model = LaTeXOCRModel(vars(args) if not isinstance(args, dict) else args).to(device)
        self.model.set_train_stage(1)

        # GPU optimizations
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True
            torch.backends.cudnn.benchmark        = True

        if getattr(args, "torch_compile", False):
            print("Compiling visual_encoder with torch.compile ...")
            self.model.visual_encoder = torch.compile(self.model.visual_encoder)

        self.optimizer = _make_optimizer(self.model, args.lr_stage1, args.weight_decay)
        self.scheduler = cosine_with_warmup(self.optimizer, warmup_steps, self.total_steps)

        self.decoder_warmup_steps = getattr(args, "decoder_warmup_steps", 0)

        if getattr(args, "resume", None):
            self._load_resume(Path(args.resume))

        if self.decoder_warmup_steps > 0 and self.global_step < self.decoder_warmup_steps:
            self.model.freeze_decoder()
            print(f"Decoder frozen for first {self.decoder_warmup_steps} steps")
        elif self.decoder_warmup_steps > 0 and self.global_step >= self.decoder_warmup_steps:
            print(f"Resuming at step {self.global_step} — decoder already unfrozen")

    def _load_resume(self, resume_dir: Path):
        sf = resume_dir / "model.safetensors"
        pt = resume_dir / "model.pt"
        if sf.exists():
            from safetensors.torch import load_file
            state = load_file(str(sf))
        elif pt.exists():
            state = torch.load(str(pt), map_location="cpu")
        else:
            print(f"[resume] No model file in {resume_dir}")
            return

        ve_state = {k.replace("visual_encoder.", ""): v for k, v in state.items() if k.startswith("visual_encoder.")}
        if ve_state:
            self.model.visual_encoder.load_state_dict(ve_state, strict=True)
            print(f"[resume] visual_encoder loaded")

        trainer_pt = resume_dir / "trainer.pt"
        if trainer_pt.exists():
            ts = torch.load(str(trainer_pt), map_location="cpu")
            opt_state = ts["optimizer"]
            device = next(self.model.parameters()).device
            for state in opt_state["state"].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            self.optimizer.load_state_dict(opt_state)
            self.scheduler.load_state_dict(ts["scheduler"])
            self.global_step = ts.get("step", 0)
            print(f"[resume] step={self.global_step}")

    def _forward_loss(self, batch) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            return self.model(
                batch["batched_images"],
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            ).loss

    def train(self):
        args       = self.args
        accum      = args.grad_accum
        micro      = 0
        accum_loss = 0.0
        data_iter  = iter(self.train_loader)

        # BLEU eval every N eval intervals (expensive, less frequent)
        bleu_every = getattr(args, "bleu_every", 5)  # every 5 * eval_steps
        bleu_counter = 0

        pbar = tqdm(total=self.total_steps, initial=self.global_step,
                    desc="Train", unit="step",
                    dynamic_ncols=True, file=sys.stdout, position=0, leave=True)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        while self.global_step < self.total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = move_batch(batch, self.device)
            loss  = self._forward_loss(batch) / accum
            loss.backward()
            accum_loss += loss.item()
            micro += 1

            if micro < accum:
                continue

            grad_norm = nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                args.max_grad_norm,
            ).item()

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            micro = 0
            pbar.update(1)

            # unfreeze decoder after warmup
            if self.decoder_warmup_steps > 0 and self.global_step == self.decoder_warmup_steps:
                self.model.unfreeze_all()
                # rebuild optimizer to include decoder params
                self.optimizer = _make_optimizer(self.model, args.lr_stage1, args.weight_decay)
                self.scheduler = cosine_with_warmup(
                    self.optimizer,
                    warmup_steps=0,
                    max_steps=self.total_steps - self.global_step,
                )
                tqdm.write(f"  [unfreeze] decoder unfrozen at step {self.global_step}, optimizer reset")

            if self.global_step % args.log_steps == 0:
                lr_now    = self.scheduler.get_last_lr()[0]
                train_ppl = math.exp(min(accum_loss, 20.0))
                tqdm.write(str({
                    "ppl":       round(train_ppl, 2),
                    "loss":      round(accum_loss, 4),
                    "grad_norm": round(grad_norm,  4),
                    "lr":        f"{lr_now:.2e}",
                    "step":      self.global_step,
                }))
            accum_loss = 0.0

            if self.global_step % args.eval_steps == 0:
                max_val_batches = max(args.eval_samples // args.batch_size, 1)

                # Always run fast val_loss
                val_metrics = run_val_loss(self.model, self.val_loader, self.device, max_val_batches)
                log = {"step": self.global_step, **val_metrics}

                # BLEU every bleu_every eval intervals
                bleu_counter += 1
                if bleu_counter % bleu_every == 0:
                    bleu_metrics = run_bleu_eval(
                        self.model, self.val_loader, self.device, self.tokenizer, max_val_batches
                    )
                    log.update(bleu_metrics)
                    print_metrics(bleu_metrics, prefix=f"step {self.global_step}")

                tqdm.write(str(log))

                # Save best by val_ppl
                if val_metrics["val_ppl"] < self.best_val_ppl:
                    self.best_val_ppl = val_metrics["val_ppl"]
                    _save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        self.global_step, self.ckpt_dir / "best", keep_last_n=999,
                    )
                    tqdm.write(f"  [best] val_ppl={self.best_val_ppl:.2f} — checkpoint saved")

            if self.global_step % args.save_steps == 0:
                _save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.global_step,
                    self.ckpt_dir / f"step_{self.global_step:07d}",
                    keep_last_n=3,
                )

        pbar.close()
        _save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            self.global_step, self.ckpt_dir / "final", keep_last_n=999,
        )
        print(f"Training done at step {self.global_step}. Best val_ppl={self.best_val_ppl:.2f}")
