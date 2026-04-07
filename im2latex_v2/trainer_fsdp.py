"""
FSDPTrainer — dùng cho stage 2 multi-GPU.

Chiến lược:
- Stage 1: FSDP wrap toàn bộ model (an toàn vì không có PEFT).
- Stage 2: PEFT (LoRA) + FSDP không tương thích (FSDP shard lm_head,
  PEFT gọi lm_head trực tiếp → size mismatch).
  Thay vào đó dùng Hybrid:
    * visual_encoder.projector  → DDP (trainable, cần sync grad)
    * decoder (LoRA)            → replicate mỗi GPU, NO DDP wrap
  Decoder nhỏ (~1.5B bf16 = ~3GB), chia 2 GPU chỉ cần ~3GB/GPU → fit T4.
"""
import contextlib
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
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

        # Load checkpoint weights trước khi apply LoRA
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
                dec_state = {
                    k.replace("decoder.model.", ""): v
                    for k, v in state.items()
                    if k.startswith("decoder.model.")
                }
                if dec_state:
                    self.model.decoder.model.load_state_dict(dec_state, strict=False)
                    print("[resume] Loaded decoder weights")

        if args.stage == 2:
            self.model.decoder.apply_lora(use_qlora=False)
            print("[stage2] LoRA applied to decoder (plain LoRA)")
            if args.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()

        self.model.set_train_stage(args.stage)

        if args.stage == 2:
            self._setup_stage2(args, distributed, local_rank, device)
        else:
            self._setup_stage1(args, distributed, device)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self._build_optimizer(trainable)

        # Resume training state
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

    def _setup_stage1(self, args, distributed, device):
        """Stage 1: FSDP wrap toàn bộ model."""
        self.model = self.model.to(dtype=self.amp_dtype).to(device)
        if args.torch_compile and hasattr(torch, "compile") and device.type == "cuda":
            self.model.visual_encoder = torch.compile(
                self.model.visual_encoder, mode="reduce-overhead", fullgraph=False
            )
            print("[compile] visual_encoder compiled")
        if distributed:
            self.model = wrap_fsdp(self.model, self.amp_dtype)
            print("[FSDP] Model wrapped (stage1)")
        self._fsdp_stage1 = distributed

    def _setup_stage2(self, args, distributed, local_rank, device):
        """
        Stage 2 Hybrid:
        - decoder: bf16, đặt lên GPU của process này (không DDP/FSDP)
        - visual_encoder: đặt lên GPU, projector wrap DDP để sync grad
        """
        # Cast về bf16 trước khi move GPU
        self.model = self.model.to(dtype=torch.bfloat16)

        # Đặt decoder lên GPU của process này
        self.model.decoder.model = self.model.decoder.model.to(device)

        # Đặt visual_encoder lên GPU
        self.model.visual_encoder = self.model.visual_encoder.to(device)

        if distributed:
            # Chỉ wrap projector bằng DDP (trainable ở stage 2)
            # navit bị frozen nên không cần sync grad
            self.model.visual_encoder.projector = DDP(
                self.model.visual_encoder.projector,
                device_ids=[local_rank],
                find_unused_parameters=False,
            )
            print("[DDP] visual_encoder.projector wrapped (stage2)")

        # Reference bỏ qua DDP wrapper để dùng khi save
        self._proj_module = (
            self.model.visual_encoder.projector.module
            if isinstance(self.model.visual_encoder.projector, DDP)
            else self.model.visual_encoder.projector
        )
        self._ve_module = self.model.visual_encoder
        self._fsdp_stage1 = False

    def _no_sync_ctx(self, is_sync: bool):
        if is_sync:
            return contextlib.nullcontext()
        if self.args.stage == 1 and self._fsdp_stage1 and isinstance(self.model, FSDP):
            return self.model.no_sync()
        if self.args.stage == 2 and isinstance(self.model.visual_encoder.projector, DDP):
            return self.model.visual_encoder.projector.no_sync()
        return contextlib.nullcontext()

    def _forward_loss(self, batch) -> torch.Tensor:
        if self.args.stage == 1:
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

                if self._fsdp_stage1 and isinstance(self.model, FSDP):
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
        if self.args.stage == 1 and self._fsdp_stage1 and isinstance(self.model, FSDP):
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
                full_state = self.model.state_dict()
        else:
            full_state = None  # dùng riêng từng submodule

        if self.is_master:
            path.mkdir(parents=True, exist_ok=True)

            if full_state is not None:
                # Stage 1 FSDP
                save_state = {k: v.cpu() for k, v in full_state.items()
                              if k.startswith("visual_encoder.")}
            else:
                # Stage 2 Hybrid: lấy từ submodules trực tiếp
                save_state = {}
                # visual_encoder (navit + projector)
                ve = self._ve_module
                for k, v in ve.navit.state_dict().items():
                    save_state[f"visual_encoder.navit.{k}"] = v.cpu()
                for k, v in self._proj_module.state_dict().items():
                    save_state[f"visual_encoder.projector.{k}"] = v.cpu()
                # decoder
                if self.args.stage == 2:
                    for k, v in self.model.decoder.model.state_dict().items():
                        save_state[f"decoder.model.{k}"] = v.cpu()

            torch.save(save_state, path / "model.pt")
            save_training_state(path, self.opt, self.global_step, self.best_bleu)
