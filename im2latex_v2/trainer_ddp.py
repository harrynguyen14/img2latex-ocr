import contextlib
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .utils import move_batch
from .latex_ocr_model import LaTeXOCRModel, alignment_loss
from .trainer_base import BaseTrainer, save_training_state, load_training_state, run_eval
from .evaluate import print_metrics


class DDPTrainer(BaseTrainer):
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
                self.model.visual_encoder.load_state_dict(ve_state, strict=True)
                print("[resume] Loaded visual_encoder weights from stage1 checkpoint")

        # LoRA và gradient checkpointing chỉ dùng ở stage 2
        if args.stage == 2:
            self.model.decoder.apply_lora(use_qlora=True)
            print("[stage2] LoRA applied to decoder (QLoRA)")
            if args.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()

        self.model.set_train_stage(args.stage)

        # visual_encoder luôn lên GPU
        self.model.visual_encoder = self.model.visual_encoder.to(device)

        # Stage 1: decoder load bình thường (không 4-bit) → cần move lên GPU
        # vì alignment_loss gọi get_input_embeddings() cùng device với labels.
        # Stage 2: decoder dùng device_map="auto" (4-bit) → bitsandbytes tự đặt lên GPU.
        if args.stage == 1:
            self.model.decoder.model = self.model.decoder.model.to(device)

        if distributed:
            self.model.visual_encoder = DDP(
                self.model.visual_encoder,
                device_ids=[local_rank],
                find_unused_parameters=False,
            )
            print(f"[DDP] visual_encoder wrapped (stage{args.stage})")

        # ve_module: reference bỏ qua DDP wrapper, dùng khi save checkpoint
        self.ve_module = (
            self.model.visual_encoder.module
            if isinstance(self.model.visual_encoder, DDP)
            else self.model.visual_encoder
        )

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self._build_optimizer(trainable)

        # Chỉ load global_step khi resume CÙNG stage.
        # Nếu cross-stage (vd: resume stage1/final để train stage2)
        # thì chỉ lấy model weights ở trên, reset global_step=0.
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
        if not is_sync and isinstance(self.model.visual_encoder, DDP):
            return self.model.visual_encoder.no_sync()
        return contextlib.nullcontext()

    def _forward_loss(self, batch) -> torch.Tensor:
        if self.args.stage == 1:
            return alignment_loss(
                self.model, batch["batched_images"], batch["labels"]
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

                # eval chỉ chạy ở stage 2 (stage 1 decoder chưa generate được)
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
        if self.is_master:
            path.mkdir(parents=True, exist_ok=True)
            state = {}
            for k, v in self.ve_module.state_dict().items():
                state[f"visual_encoder.{k}"] = v.cpu()
            # Stage 2: lưu thêm decoder (có LoRA weights)
            if self.args.stage == 2:
                for k, v in self.model.decoder.model.state_dict().items():
                    state[f"decoder.model.{k}"] = v.cpu()
            torch.save(state, path / "model.pt")
            save_training_state(path, self.opt, self.global_step, self.best_bleu)