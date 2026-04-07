import contextlib
from pathlib import Path

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from tqdm import tqdm

from .utils import move_batch, wrap_fsdp
from .latex_ocr_model import LaTeXOCRModel
from .trainer_base import BaseTrainer, save_training_state, load_training_state, run_eval
from .evaluate import print_metrics


class FSDPTrainer(BaseTrainer):
    def __init__(self, args, train_loader, val_loader, device, tokenizer,
                 distributed, rank, local_rank, world_size):
        super().__init__(args, train_loader, val_loader, device, tokenizer,
                         distributed, rank, local_rank, world_size)

        self.model = LaTeXOCRModel(args)

        resume_dir = Path(args.resume).resolve() if args.resume else None
        if resume_dir is not None and resume_dir.is_dir():
            model_pt = resume_dir / "model.pt"
            if model_pt.is_file():
                state = torch.load(str(model_pt), map_location="cpu", weights_only=False)
                ve_state = {
                    k.replace("visual_encoder.", ""): v
                    for k, v in state.items()
                    if k.startswith("visual_encoder.")
                }
                if ve_state:
                    self.model.visual_encoder.load_state_dict(ve_state, strict=True)
                    print("[resume] Loaded visual_encoder weights")
                # Nếu resume cùng stage 2 (có decoder weights), load luôn
                dec_state = {
                    k.replace("decoder.model.", ""): v
                    for k, v in state.items()
                    if k.startswith("decoder.model.")
                }
                if dec_state:
                    self.model.decoder.model.load_state_dict(dec_state, strict=False)
                    print("[resume] Loaded decoder weights")

        # Stage 2: apply LoRA (không dùng QLoRA vì FSDP không compatible với 4-bit)
        if args.stage == 2:
            self.model.decoder.apply_lora(use_qlora=False)
            print("[stage2] LoRA applied to decoder (plain LoRA for FSDP)")
            if args.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()

        self.model.set_train_stage(args.stage)

        # Move toàn bộ model lên GPU trước khi wrap FSDP
        self.model = self.model.to(device)

        if args.torch_compile and hasattr(torch, "compile") and device.type == "cuda":
            self.model.visual_encoder = torch.compile(
                self.model.visual_encoder, mode="reduce-overhead", fullgraph=False
            )
            print("[compile] visual_encoder compiled")

        # FSDP wrap sau cùng — không giữ reference đến submodule gốc
        if distributed:
            self.model = wrap_fsdp(self.model, self.amp_dtype)
            print("[FSDP] Model wrapped")

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self._build_optimizer(trainable)

        # Resume: chỉ restore global_step nếu cùng stage
        if resume_dir is not None and resume_dir.is_dir():
            ts = load_training_state(resume_dir, device)
            if ts:
                is_same_stage = f"stage{args.stage}" in str(resume_dir)
                if is_same_stage:
                    try:
                        self.opt.load_state_dict(ts["optimizer"])
                    except ValueError:
                        print("[resume] Optimizer mismatch — skipping optimizer state")
                    self.global_step = int(ts.get("global_step", 0))
                    self.best_bleu   = float(ts.get("best_bleu", -1.0))
                    print(f"[resume] Restored global_step={self.global_step}")
                else:
                    print("[resume] Cross-stage — resetting global_step=0")

    def _no_sync_ctx(self, is_sync: bool):
        if not is_sync and isinstance(self.model, FSDP):
            return self.model.no_sync()
        return contextlib.nullcontext()

    def _forward_loss(self, batch) -> torch.Tensor:
        if self.args.stage == 1:
            # stage1_forward là method của LaTeXOCRModel — FSDP dispatch an toàn
            return self.model.stage1_forward(
                batch["batched_images"], batch["labels"]
            )
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.device.type == "cuda",
        ):
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
        pbar = tqdm(total=self.total_steps, initial=self.global_step,
                    disable=not self.is_master)

        self.model.train()
        while self.global_step < self.total_steps:
            for batch in self.train_loader:
                if self.global_step >= self.total_steps:
                    break

                batch   = move_batch(batch, self.device)
                is_sync = micro == accum - 1

                with self._no_sync_ctx(is_sync):
                    loss = self._forward_loss(batch) / accum
                    loss.backward()

                accum_loss += loss.item()
                micro += 1

                if not is_sync:
                    continue

                t = self._step_lr()

                if isinstance(self.model, FSDP):
                    self.model.clip_grad_norm_(args.max_grad_norm)
                    grad_norm = args.max_grad_norm
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )
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
                        f"lr={t:.3e} step={self.global_step}/{self.total_steps}"
                    )
                accum_loss = 0.0

                # Eval chỉ ở stage 2
                if self.global_step % args.eval_steps == 0 and args.stage == 2:
                    self._eval_and_save()

                if self.global_step % args.save_steps == 0:
                    self._save(self.ckpt_dir / f"step-{self.global_step}")

        pbar.close()
        self._barrier()
        self._save(self.ckpt_dir / "final")
        self._barrier()
        self._cleanup()

    def _eval_and_save(self):
        self._barrier()
        if self.is_master:
            mb   = max(self.args.eval_samples // self.args.batch_size, 1)
            mets = run_eval(self.model, self.val_loader, self.device, self.tokenizer, mb)
            print_metrics(mets, prefix=f"step {self.global_step}")
            if mets["bleu4"] > self.best_bleu:
                self.best_bleu = mets["bleu4"]
                self._save(self.ckpt_dir / "best")
        self._barrier()

    def _save(self, path: Path):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        if isinstance(self.model, FSDP):
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
                full_state = self.model.state_dict()
        else:
            full_state = self.model.state_dict()

        if self.is_master:
            path.mkdir(parents=True, exist_ok=True)
            save_state = {}
            for k, v in full_state.items():
                if k.startswith("visual_encoder."):
                    save_state[k] = v.cpu()
                elif self.args.stage == 2 and k.startswith("decoder."):
                    save_state[k] = v.cpu()
            torch.save(save_state, path / "model.pt")
            save_training_state(path, self.opt, self.global_step, self.best_bleu)
