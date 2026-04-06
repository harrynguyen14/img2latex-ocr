import contextlib
from pathlib import Path

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from tqdm import tqdm

from .utils import move_batch, wrap_fsdp
from .latex_ocr_model import LaTeXOCRModel
from .trainer_base import BaseTrainer, save_training_state


class FSDPTrainer(BaseTrainer):
    def __init__(self, args, train_loader, val_loader, device, tokenizer,
                 distributed, rank, local_rank, world_size):
        super().__init__(args, train_loader, val_loader, device, tokenizer,
                         distributed, rank, local_rank, world_size)

        self.model = LaTeXOCRModel(args)
        self.model.to(device)

        resume_dir = Path(args.resume).resolve() if args.resume else None
        if resume_dir is not None and resume_dir.is_dir():
            model_pt = resume_dir / "model.pt"
            if model_pt.is_file():
                state = torch.load(str(model_pt), map_location=device, weights_only=False)
                ve_state = {
                    k.replace("visual_encoder.", ""): v
                    for k, v in state.items()
                    if k.startswith("visual_encoder.")
                }
                self.model.visual_encoder.load_state_dict(ve_state, strict=True)
                print("[resume] Loaded visual_encoder weights")

        self.model.set_train_stage(args.stage)
        self.model = self.model.to(dtype=self.amp_dtype)

        if args.torch_compile and hasattr(torch, "compile") and device.type == "cuda":
            self.model.visual_encoder = torch.compile(
                self.model.visual_encoder, mode="reduce-overhead", fullgraph=False
            )
            print("[compile] visual_encoder compiled")

        # QUAN TRỌNG: wrap FSDP sau cùng. KHÔNG giữ raw_model reference vì
        # FSDP shard parameter in-place — mọi reference đến submodule gốc
        # sẽ thấy weight shape [0] khi gọi ngoài FSDP forward context.
        if distributed:
            self.model = wrap_fsdp(self.model, self.amp_dtype)
            print("[FSDP] Model wrapped")

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self._build_optimizer(trainable)

        if resume_dir:
            self._resume(resume_dir)

    def _no_sync_ctx(self, is_sync: bool):
        if not is_sync and isinstance(self.model, FSDP):
            return self.model.no_sync()
        return contextlib.nullcontext()

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
                    # stage1_forward() là method của LaTeXOCRModel — FSDP sẽ
                    # unshard tất cả parameter trước khi dispatch vào method này,
                    # giống hệt cách nó unshard cho forward(). Đây là cách
                    # duy nhất an toàn với FSDP.
                    loss = self.model.stage1_forward(
                        batch["batched_images"], batch["labels"]
                    ) / accum
                    loss.backward()
                    torch.cuda.empty_cache()

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

                if self.global_step % args.save_steps == 0:
                    self._save(self.ckpt_dir / f"step-{self.global_step}")

        pbar.close()
        self._barrier()
        self._save(self.ckpt_dir / "final")
        self._barrier()
        self._cleanup()

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