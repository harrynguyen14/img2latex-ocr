import contextlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from tqdm import tqdm
import bitsandbytes as bnb

from .utils import move_batch, lr_cosine, wrap_fsdp, cleanup_distributed
from .evaluate import compute_metrics, print_metrics
from .latex_ocr_model import LaTeXOCRModel, alignment_loss


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
def run_eval(model, loader, device, tokenizer, max_batches: int, amp_dtype: torch.dtype):
    model.eval()
    preds, refs = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch(batch, device)
        pr = model.generate(batch["batched_images"])
        preds.extend(pr)
        lid = batch["labels"].cpu().numpy()
        lid = np.where(lid == -100, tokenizer.pad_token_id, lid)
        refs.extend(tokenizer.batch_decode(lid, skip_special_tokens=True))
    model.train()
    return compute_metrics(preds, refs)


class Trainer:
    def __init__(self, args, train_loader, val_loader, device, tokenizer, distributed, rank, local_rank, world_size):
        self.args         = args
        self.device       = device
        self.tokenizer    = tokenizer
        self.distributed  = distributed
        self.is_master    = rank == 0
        self.world_size   = world_size
        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.amp_dtype = torch.bfloat16 if args.amp_dtype in ("bf16", "bfloat16") else torch.float16

        self.model = LaTeXOCRModel(args)
        self.model.to(device)

        resume_dir = Path(args.resume).resolve() if args.resume else None
        if resume_dir is not None and resume_dir.is_dir():
            model_pt = resume_dir / "model.pt"
            if model_pt.is_file():
                state = torch.load(str(model_pt), map_location=device, weights_only=False)
                ve_state = {k.replace("visual_encoder.", ""): v for k, v in state.items() if k.startswith("visual_encoder.")}
                self.model.visual_encoder.load_state_dict(ve_state, strict=True)
                print("[resume] Loaded visual_encoder weights from stage1 checkpoint")

        if args.stage == 2:
            self.model.decoder.apply_lora()
            print("[stage2] LoRA applied to decoder")

        self.model.set_train_stage(args.stage)

        if args.gradient_checkpointing and args.stage == 2:
            self.model.gradient_checkpointing_enable()

        if args.torch_compile and hasattr(torch, "compile") and device.type == "cuda":
            self.model.visual_encoder = torch.compile(self.model.visual_encoder, mode="reduce-overhead", fullgraph=False)
            print("[compile] visual_encoder compiled")

        if distributed:
            self.model = wrap_fsdp(self.model, self.amp_dtype)
            print("[FSDP] Model wrapped")

        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.lr        = args.lr_stage1 if args.stage == 1 else args.lr_stage2

        self.opt = bnb.optim.AdamW8bit(self.trainable, lr=self.lr, weight_decay=args.weight_decay)
        print("[optimizer] Using AdamW8bit")

        self.global_step = 0
        self.best_bleu   = -1.0
        if resume_dir:
            ts = load_training_state(resume_dir, device)
            if ts:
                try:
                    self.opt.load_state_dict(ts["optimizer"])
                except ValueError:
                    print("[resume] Optimizer param groups mismatch — skipping optimizer state")
                self.global_step = int(ts.get("global_step", 0))
                self.best_bleu   = float(ts.get("best_bleu", -1.0))

        ns = getattr(train_loader.dataset, "num_samples", None)
        bs = args.batch_size
        if ns:
            spe = max(ns // max(world_size, 1) // max(bs, 1), 1)
            self.total_steps = min(spe * args.epochs // max(args.grad_accum, 1), args.max_steps)
        else:
            self.total_steps = args.max_steps
        self.warmup = max(int(self.total_steps * args.warmup_ratio), 1)

        self.ckpt_dir = Path(args.ckpt_dir) / f"stage{args.stage}"
        if self.is_master:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _no_sync_ctx(self, is_sync: bool):
        if not is_sync and isinstance(self.model, FSDP):
            return self.model.no_sync()
        return contextlib.nullcontext()

    def train(self):
        args       = self.args
        accum      = args.grad_accum
        micro      = 0
        accum_loss = 0.0
        pbar       = tqdm(total=self.total_steps, initial=self.global_step, disable=not self.is_master)

        self.model.train()
        while self.global_step < self.total_steps:
            for batch in self.train_loader:
                if self.global_step >= self.total_steps:
                    break

                batch   = move_batch(batch, self.device)
                is_sync = micro == accum - 1

                with self._no_sync_ctx(is_sync):
                    if args.stage == 1:
                        loss = alignment_loss(self.model, batch["batched_images"], batch["labels"]) / accum
                    else:
                        loss = self.model(
                            batch["batched_images"],
                            batch["input_ids"],
                            batch["attention_mask"],
                            batch["labels"],
                        ).loss / accum
                    loss.backward()
                    torch.cuda.empty_cache()

                accum_loss += loss.item()
                micro += 1

                if not is_sync:
                    continue

                t = lr_cosine(self.global_step, self.total_steps, self.lr, self.warmup)
                for g in self.opt.param_groups:
                    g["lr"] = t

                if isinstance(self.model, FSDP):
                    self.model.clip_grad_norm_(args.max_grad_norm)
                    grad_norm = args.max_grad_norm
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable, args.max_grad_norm)
                    if isinstance(grad_norm, torch.Tensor):
                        grad_norm = grad_norm.item()

                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                self.global_step += 1
                micro = 0
                pbar.update(1)

                if self.is_master and self.global_step % args.log_steps == 0:
                    pbar.write(
                        f"loss={accum_loss:.4f} grad_norm={grad_norm:.4f} "
                        f"lr={t:.3e} epoch={self.global_step / max(self.total_steps, 1):.4f}"
                    )
                accum_loss = 0.0

                if self.global_step % args.eval_steps == 0 and args.stage == 2:
                    self._eval_and_save(pbar)

                if self.global_step % args.save_steps == 0:
                    self._save(self.ckpt_dir / f"step-{self.global_step}")

        pbar.close()
        self._barrier()
        self._save(self.ckpt_dir / "final")
        self._barrier()
        if self.distributed:
            cleanup_distributed()

    def _eval_and_save(self, pbar):
        self._barrier()
        mb   = max(self.args.eval_samples // self.args.batch_size, 1)
        mets = run_eval(self.model, self.val_loader, self.device, self.tokenizer, mb, self.amp_dtype)
        if self.is_master:
            print_metrics(mets, prefix=f"step {self.global_step}")
        is_best = mets["bleu4"] > self.best_bleu
        if is_best:
            self.best_bleu = mets["bleu4"]
            self._save(self.ckpt_dir / "best")
        self._barrier()

    def _save(self, path: Path):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        if isinstance(self.model, FSDP):
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
                state = self.model.state_dict()
        else:
            state = self.model.state_dict()

        if self.is_master:
            path.mkdir(parents=True, exist_ok=True)
            torch.save(state, path / "model.pt")
            save_training_state(path, self.opt, self.global_step, self.best_bleu)

    def _barrier(self):
        if self.distributed:
            dist.barrier()
