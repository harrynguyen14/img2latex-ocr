import json
import math
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
from tqdm import tqdm

from safetensors.torch import save_file as st_save_file, load_file as st_load_file

from .build_datasets import build_dataloader
from .utils import move_batch
from .nav2tex import LaTeXOCRModel
from .nav2tex.model import decode_ids
from .evaluate import compute_metrics, print_metrics


def cosine_with_warmup(optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def _make_optimizer(model: LaTeXOCRModel, encoder_lr: float, decoder_lr: float, weight_decay: float):
    no_decay = {
        n for n, p in model.named_parameters()
        if p.requires_grad and (p.ndim < 2 or "norm" in n.lower() or n.endswith(".bias"))
    }
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    decoder_wd = [p for n, p in trainable if n.startswith("decoder.") and n not in no_decay]
    decoder_nd = [p for n, p in trainable if n.startswith("decoder.") and n in no_decay]
    encoder_wd = [p for n, p in trainable if not n.startswith("decoder.") and n not in no_decay]
    encoder_nd = [p for n, p in trainable if not n.startswith("decoder.") and n in no_decay]

    opt_cls = bnb.optim.AdamW8bit if HAS_BNB else AdamW

    param_groups = []
    if encoder_wd: param_groups.append({"params": encoder_wd, "lr": encoder_lr, "weight_decay": weight_decay})
    if encoder_nd: param_groups.append({"params": encoder_nd, "lr": encoder_lr, "weight_decay": 0.0})
    if decoder_wd: param_groups.append({"params": decoder_wd, "lr": decoder_lr, "weight_decay": weight_decay})
    if decoder_nd: param_groups.append({"params": decoder_nd, "lr": decoder_lr, "weight_decay": 0.0})

    return opt_cls(param_groups, lr=encoder_lr, betas=(0.9, 0.95), eps=1e-8)


def _load_model_state(model: LaTeXOCRModel, state: dict, strict: bool = True) -> None:
    ve_state  = {k[len("visual_encoder."):]: v for k, v in state.items() if k.startswith("visual_encoder.")}
    dec_state = {k[len("decoder."):]: v       for k, v in state.items() if k.startswith("decoder.")}

    if ve_state:
        model.visual_encoder.load_state_dict(ve_state, strict=strict)
        tqdm.write(f"[ckpt] visual_encoder loaded ({len(ve_state)} tensors)")
    if dec_state:
        model.decoder.load_state_dict(dec_state, strict=strict)
        tqdm.write(f"[ckpt] decoder loaded ({len(dec_state)} tensors)")
    if not ve_state and not dec_state:
        model.load_state_dict(state, strict=strict)
        tqdm.write("[ckpt] model loaded (flat state dict)")


def _flatten_tensors(d: dict, prefix: str) -> tuple[dict, dict]:
    tensors, scalars = {}, {}
    for k, v in d.items():
        full_key = f"{prefix}/{k}"
        if isinstance(v, torch.Tensor):
            tensors[full_key] = v.contiguous().cpu()
        elif isinstance(v, dict):
            t, s = _flatten_tensors(v, full_key)
            tensors.update(t); scalars.update(s)
        else:
            scalars[full_key] = v
    return tensors, scalars


def _unflatten_tensors(tensors: dict, scalars: dict, prefix: str) -> dict:
    result = {}
    sub = prefix + "/"
    for key, val in {**tensors, **scalars}.items():
        if not key.startswith(sub):
            continue
        parts = key[len(sub):].split("/")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    return result


def _parse_weight_stages(stages_str: str, sources: list[str]) -> list[tuple[int, dict[str, float]]]:
    if not stages_str:
        return []
    parsed: list[tuple[int, dict[str, float]]] = []
    for chunk in stages_str.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid weight stage '{chunk}'. Expected 'step:w1,w2,...'")
        step_text, weights_text = chunk.split(":", 1)
        step = int(step_text.strip())
        vals = [float(x.strip()) for x in weights_text.split(",") if x.strip()]
        if len(vals) != len(sources):
            raise ValueError(f"Stage '{chunk}' has {len(vals)} weights but {len(sources)} sources")
        parsed.append((step, {src: val for src, val in zip(sources, vals)}))
    parsed.sort(key=lambda x: x[0])
    return parsed


def _write_ckpt(model: LaTeXOCRModel, optimizer, scheduler, step: int, ckpt_dir: Path, tokenizer=None):
    tmp_dir = ckpt_dir.parent / (ckpt_dir.name + ".tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    state = {f"visual_encoder.{k}": v.contiguous().cpu() for k, v in model.visual_encoder.state_dict().items()}
    state.update({f"decoder.{k}": v.contiguous().cpu() for k, v in model.decoder.state_dict().items()})
    st_save_file(state, tmp_dir / "model.safetensors")

    opt_tensors, opt_scalars = _flatten_tensors(optimizer.state_dict(), "optimizer")
    sch_tensors, sch_scalars = _flatten_tensors({"state": scheduler.state_dict()}, "scheduler")
    trainer_tensors = {**opt_tensors, **sch_tensors}
    trainer_scalars = {**opt_scalars, **sch_scalars, "step": step}
    if not trainer_tensors:
        trainer_tensors["_sentinel"] = torch.zeros(1)
    metadata = {k: json.dumps(v) for k, v in trainer_scalars.items()}
    st_save_file(trainer_tensors, tmp_dir / "trainer.safetensors", metadata=metadata)

    with open(tmp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(model.config, f, indent=2, ensure_ascii=False)

    if tokenizer is not None:
        tokenizer.save_pretrained(str(tmp_dir / "tokenizer"))

    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    try:
        tmp_dir.rename(ckpt_dir)
    except OSError:
        shutil.move(str(tmp_dir), str(ckpt_dir))


def _save_best(model, optimizer, scheduler, step, ckpt_dir: Path, tokenizer=None):
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    _write_ckpt(model, optimizer, scheduler, step, ckpt_dir, tokenizer)
    tqdm.write(f"[ckpt] best/ overwritten at step {step}")


def _save_periodic(model, optimizer, scheduler, step, base_dir: Path, keep_last_n: int, tokenizer=None):
    ckpt_dir = base_dir / f"step_{step:07d}"
    _write_ckpt(model, optimizer, scheduler, step, ckpt_dir, tokenizer)
    tqdm.write(f"[ckpt] {ckpt_dir.name} saved")

    all_ckpts = sorted(base_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    while len(all_ckpts) > keep_last_n:
        old = all_ckpts.pop(0)
        shutil.rmtree(old)
        tqdm.write(f"[ckpt] {old.name} removed")


@torch.no_grad()
def run_val_loss(model: LaTeXOCRModel, loader, device, max_batches: int) -> dict:
    model.eval()
    total_loss, total_batches = 0.0, 0
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    try:
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = move_batch(batch, device)
            with amp_ctx:
                out = model(batch["batched_images"], batch["input_ids"], batch["attention_mask"], batch["labels"],
                            true_len=batch.get("true_len"))
            total_loss    += out.loss.item()
            total_batches += 1
    finally:
        model.train()
    if total_batches == 0:
        return {"val_loss": float("inf"), "val_ppl": float("inf")}
    avg = total_loss / total_batches
    return {"val_loss": round(avg, 4), "val_ppl": round(math.exp(min(avg, 20.0)), 2)}


@torch.no_grad()
def run_bleu_eval(model: LaTeXOCRModel, loader, device, n_samples: int) -> dict:
    import random as _random
    import pyarrow.parquet as pq
    from .preprocessor import _process
    from .utils import make_collate_fn

    ds = loader.dataset
    underlying = ds.dataset if hasattr(ds, "dataset") else ds
    files = getattr(underlying, "files", [])

    all_rows: list[dict] = []
    for pfile in files:
        table = pq.read_table(str(pfile), columns=["image", "latex"])
        images = table["image"].to_pylist()
        latexs = table["latex"].to_pylist()
        for img_raw, lat in zip(images, latexs):
            if lat and isinstance(lat, str) and lat.strip() and img_raw is not None:
                all_rows.append({"image": img_raw, "latex": lat.strip()})

    chosen = _random.sample(all_rows, min(n_samples, len(all_rows)))

    args = underlying.args
    tokenizer = underlying.tokenizer
    collate = make_collate_fn(args.max_token_len)
    batch_size = loader.batch_size or 20

    model.eval()
    preds, refs = [], []
    skip_ids = {model.decoder.pad_token_id, model.decoder.eos_token_id, model.decoder.bos_token_id}
    n_batches = math.ceil(len(chosen) / batch_size)
    try:
        for start in tqdm(range(0, len(chosen), batch_size), total=n_batches,
                          desc="BLEU eval", unit="batch", leave=False,
                          file=sys.stdout, position=1, dynamic_ncols=True):
            items = []
            for row in chosen[start: start + batch_size]:
                try:
                    items.append(_process(row, tokenizer, args))
                except Exception:
                    pass
            if not items:
                continue
            batch = move_batch(collate(items), device)
            gen = model.generate(batch["batched_images"])
            preds.extend(gen)
            for ids in batch["labels"].cpu().tolist():
                refs.append(decode_ids(model.tokenizer, [x for x in ids if x >= 0], skip_ids=skip_ids))
    finally:
        model.train()

    if not preds:
        return {"bleu4": 0.0, "exact_match": 0.0, "edit_distance": 1.0, "n_samples": 0}
    return compute_metrics(preds, refs)


class Trainer:
    def __init__(self, args, train_loader, val_loader, device, tokenizer):
        self.args         = args
        self.device       = device
        self.tokenizer    = tokenizer
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.ckpt_dir     = Path(args.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.total_steps          = args.max_steps
        self.global_step          = 0
        self.best_val_ppl         = float("inf")
        self.decoder_warmup_steps = args.decoder_warmup_steps
        self.len_loss_start_step  = getattr(args, "len_loss_start_step", 15000)
        self.sources              = list(getattr(args, "sources", []))
        self.weight_stages        = _parse_weight_stages(getattr(args, "weight_stages", ""), self.sources)
        self.active_weight_stage  = -1
        self.early_stopping_patience = getattr(args, "early_stopping_patience", 0)
        self.early_stopping_counter  = 0

        self.model = LaTeXOCRModel(
            vars(args) if not isinstance(args, dict) else args,
            tokenizer=tokenizer,
        ).to(device)

        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True
            torch.backends.cudnn.benchmark        = True
            self.model.visual_encoder.to(memory_format=torch.channels_last)

        if getattr(args, "torch_compile", False):
            print("Compiling visual_encoder with torch.compile ...")
            self.model.visual_encoder = torch.compile(self.model.visual_encoder)

        if self.decoder_warmup_steps > 0:
            self.model.freeze_decoder()
            print(f"Decoder frozen for first {self.decoder_warmup_steps} steps")

        warmup_steps   = max(1, int(args.max_steps * args.warmup_ratio))
        self.optimizer = _make_optimizer(
            self.model,
            encoder_lr=getattr(args, "encoder_lr", args.lr),
            decoder_lr=getattr(args, "decoder_lr", args.lr),
            weight_decay=args.weight_decay,
        )
        self.scheduler = cosine_with_warmup(self.optimizer, warmup_steps, self.total_steps)

        if getattr(args, "resume", None):
            self._load_resume(Path(args.resume))

        if self.decoder_warmup_steps > 0 and self.global_step >= self.decoder_warmup_steps:
            self.model.unfreeze_all()
            print(f"Resuming at step {self.global_step} — decoder already unfrozen")
            # Rebuild optimizer with full param groups (encoder + decoder) before loading state
            self.optimizer = _make_optimizer(
                self.model, getattr(args, "encoder_lr", args.lr),
                getattr(args, "decoder_lr", args.lr), args.weight_decay,
            )
            remaining = self.total_steps - self.global_step
            self.scheduler = cosine_with_warmup(self.optimizer, warmup_steps=0, max_steps=max(remaining, 1))

        self._apply_resume_opt()

        self._maybe_switch_weight_stage(force=True)

        self._accum_after_unfreeze: int = self.args.grad_accum

        self._trainable_params: list[nn.Parameter] = [
            p for p in self.model.parameters() if p.requires_grad
        ]

    def _refresh_trainable_params(self):
        self._trainable_params = [p for p in self.model.parameters() if p.requires_grad]

    def _load_resume(self, resume_dir: Path):
        """Load model weights + step. Returns raw opt/sch state dicts for deferred loading."""
        sf = resume_dir / "model.safetensors"
        if not sf.exists():
            print(f"[resume] No model.safetensors in {resume_dir}")
            return

        _load_model_state(self.model, st_load_file(str(sf)), strict=False)

        trainer_sf = resume_dir / "trainer.safetensors"
        if not trainer_sf.exists():
            return

        from safetensors import safe_open
        trainer_tensors = st_load_file(str(trainer_sf), device="cpu")
        trainer_tensors.pop("_sentinel", None)
        with safe_open(str(trainer_sf), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
        trainer_scalars = {k: json.loads(v) for k, v in metadata.items()}

        self.global_step = int(trainer_scalars.get("step", 0))
        self._resume_opt_sd  = _unflatten_tensors(trainer_tensors, trainer_scalars, "optimizer")
        self._resume_sch_sd  = _unflatten_tensors(trainer_tensors, trainer_scalars, "scheduler").get("state", {})

    def _apply_resume_opt(self):
        """Load optimizer+scheduler state saved by _load_resume, after param groups are final."""
        opt_sd = getattr(self, "_resume_opt_sd", None)
        sch_sd = getattr(self, "_resume_sch_sd", None)
        if opt_sd is None:
            return
        device = next(self.model.parameters()).device
        for s in opt_sd.get("state", {}).values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(device)
        try:
            self.optimizer.load_state_dict(opt_sd)
            if sch_sd:
                self.scheduler.load_state_dict(sch_sd)
            print(f"[resume] optimizer+scheduler loaded, step={self.global_step}")
        except Exception as e:
            print(f"[resume] WARNING: Could not load optimizer/scheduler: {e}")
        self._resume_opt_sd = None
        self._resume_sch_sd = None

    def _forward_loss(self, batch) -> tuple:
        true_len = (
            batch.get("true_len")
            if self.global_step >= self.len_loss_start_step
            else None
        )
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            out = self.model(
                batch["batched_images"],
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
                true_len=true_len,
            )
        return out.loss, out.lm_loss, out.len_loss

    def _rebuild_train_loader(self):
        args = self.args
        nw = args.num_workers
        prefetch = args.prefetch_factor
        persistent = args.persistent_workers and nw > 0
        self.train_loader = build_dataloader(
            self.train_loader.dataset,
            args.batch_size,
            nw,
            self.device.type == "cuda",
            prefetch,
            persistent,
            args.max_token_len,
        )

    def _maybe_switch_weight_stage(self, force: bool = False) -> bool:
        if not self.weight_stages:
            return False
        ds = self.train_loader.dataset
        underlying = ds.dataset if hasattr(ds, "dataset") else ds
        if not hasattr(underlying, "set_weights"):
            return False

        target_idx = -1
        for i, (start_step, _weights) in enumerate(self.weight_stages):
            if self.global_step >= start_step:
                target_idx = i
            else:
                break

        if target_idx < 0:
            return False
        if not force and target_idx == self.active_weight_stage:
            return False

        stage_step, stage_weights = self.weight_stages[target_idx]
        underlying.set_weights(stage_weights)
        self._rebuild_train_loader()
        self.active_weight_stage = target_idx
        tqdm.write(
            f"[data] switched source weights at step={self.global_step} "
            f"(stage start={stage_step}): {stage_weights}"
        )
        return True

    def _unfreeze_decoder(self):
        old_opt_state = self.optimizer.state_dict()
        old_param_to_idx = {
            id(p): i
            for i, p in enumerate(
                p for g in self.optimizer.param_groups for p in g["params"]
            )
        }

        self.model.unfreeze_all()

        if getattr(self.args, "decoder_grad_checkpoint", False):
            self.model.enable_decoder_grad_checkpoint()
            tqdm.write("  [unfreeze] decoder gradient checkpointing enabled")

        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        divisor = max(1, getattr(self.args, "unfreeze_grad_accum_divisor", 1))
        if divisor > 1:
            self._accum_after_unfreeze = max(1, self.args.grad_accum // divisor)
            tqdm.write(
                f"  [unfreeze] grad_accum {self.args.grad_accum} → {self._accum_after_unfreeze} "
                f"(divisor={divisor}, effective batch unchanged)"
            )
        else:
            self._accum_after_unfreeze = self.args.grad_accum

        self.optimizer = _make_optimizer(
            self.model, self.args.encoder_lr, self.args.decoder_lr, self.args.weight_decay
        )

        new_state = self.optimizer.state_dict()
        transferred = 0
        flat_new_params = [p for g in self.optimizer.param_groups for p in g["params"]]
        for new_flat_idx, p in enumerate(flat_new_params):
            old_flat_idx = old_param_to_idx.get(id(p))
            if old_flat_idx is not None and old_flat_idx in old_opt_state.get("state", {}):
                new_state["state"][new_flat_idx] = old_opt_state["state"][old_flat_idx]
                transferred += 1
        self.optimizer.load_state_dict(new_state)

        remaining      = self.total_steps - self.global_step
        decoder_rewarm = min(500, remaining // 20)
        self.scheduler = cosine_with_warmup(self.optimizer, warmup_steps=decoder_rewarm, max_steps=remaining)
        self._refresh_trainable_params()
        tqdm.write(
            f"  [unfreeze] decoder unfrozen at step {self.global_step}, "
            f"encoder_lr={self.args.encoder_lr:.1e}, decoder_lr={self.args.decoder_lr:.1e}, "
            f"rewarmup={decoder_rewarm}, transferred_opt_states={transferred}"
        )

    def train(self):
        args       = self.args
        micro      = 0
        data_iter  = iter(self.train_loader)

        val_loss_steps  = getattr(args, "val_loss_steps", 2500)
        eval_steps      = getattr(args, "eval_steps", 10000)
        max_val_batches = max(math.ceil(args.eval_samples / args.batch_size), 1)
        bleu_n_samples  = max(getattr(args, "bleu_samples", 512), 1)

        pbar = tqdm(total=self.total_steps, initial=self.global_step,
                    desc="Train", unit="step",
                    dynamic_ncols=True, file=sys.stdout, position=0, leave=True)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        device = self.device
        accum_loss     = torch.zeros(1, device=device)
        accum_lm_loss  = torch.zeros(1, device=device)
        accum_len_loss = torch.zeros(1, device=device)

        while self.global_step < self.total_steps:
            accum = self._accum_after_unfreeze

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = move_batch(batch, device)
            loss, lm_loss, len_loss = self._forward_loss(batch)

            scaled = loss / accum
            scaled.backward()

            with torch.no_grad():
                accum_loss     += loss     / accum
                accum_lm_loss  += lm_loss  / accum
                accum_len_loss += len_loss / accum

            micro += 1
            if micro < accum:
                continue

            grad_norm = nn.utils.clip_grad_norm_(
                self._trainable_params,
                args.max_grad_norm,
            ).item()

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            micro = 0
            pbar.update(1)

            if self._maybe_switch_weight_stage():
                data_iter = iter(self.train_loader)

            if self.decoder_warmup_steps > 0 and self.global_step == self.decoder_warmup_steps:
                self._unfreeze_decoder()
                micro = 0
                accum_loss.zero_()
                accum_lm_loss.zero_()
                accum_len_loss.zero_()

            if self.global_step % args.log_steps == 0:
                loss_val     = accum_loss.item()
                lm_loss_val  = accum_lm_loss.item()
                len_loss_val = accum_len_loss.item()
                lr_now       = self.scheduler.get_last_lr()[0]
                if self.global_step < self.decoder_warmup_steps:
                    phase = "freeze"
                elif self.global_step < self.len_loss_start_step:
                    phase = "joint"
                else:
                    phase = "lam"
                tqdm.write(str({
                    "phase":     phase,
                    "ppl":       round(math.exp(min(loss_val, 20.0)), 2),
                    "loss":      round(loss_val,     4),
                    "lm":        round(lm_loss_val,  4),
                    "len":       round(len_loss_val, 4),
                    "grad_norm": round(grad_norm,    4),
                    "lr":        f"{lr_now:.2e}",
                    "step":      self.global_step,
                }))

            accum_loss.zero_()
            accum_lm_loss.zero_()
            accum_len_loss.zero_()

            if self.global_step % val_loss_steps == 0:
                val_metrics = run_val_loss(self.model, self.val_loader, self.device, max_val_batches)
                tqdm.write(str({"step": self.global_step, **val_metrics}))

                if val_metrics["val_ppl"] < self.best_val_ppl:
                    self.best_val_ppl = val_metrics["val_ppl"]
                    self.early_stopping_counter = 0
                    _save_best(self.model, self.optimizer, self.scheduler,
                               self.global_step, self.ckpt_dir / "best", self.tokenizer)
                    tqdm.write(f"  [best] val_ppl={self.best_val_ppl:.2f}")
                elif self.early_stopping_patience > 0:
                    self.early_stopping_counter += 1
                    tqdm.write(f"  [early_stop] no improvement {self.early_stopping_counter}/{self.early_stopping_patience}")
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        tqdm.write(f"  [early_stop] stopping at step {self.global_step}")
                        break

            if self.global_step % eval_steps == 0:
                try:
                    bleu_metrics = run_bleu_eval(self.model, self.val_loader, self.device, bleu_n_samples)
                    print_metrics(bleu_metrics, prefix=f"step {self.global_step}")
                    tqdm.write(str({"step": self.global_step, **bleu_metrics}))
                except Exception as e:
                    tqdm.write(f"[eval] BLEU eval failed at step {self.global_step}: {e}")
                    self.model.train()

            if self.global_step % args.save_steps == 0:
                _save_periodic(self.model, self.optimizer, self.scheduler,
                               self.global_step, self.ckpt_dir, keep_last_n=3, tokenizer=self.tokenizer)

        pbar.close()
        _save_periodic(self.model, self.optimizer, self.scheduler,
                       self.global_step, self.ckpt_dir, keep_last_n=3, tokenizer=self.tokenizer)
        print(f"Training done at step {self.global_step}. Best val_ppl={self.best_val_ppl:.2f}")

        final_samples = getattr(args, "final_eval_samples", 0)
        final_n       = final_samples if final_samples > 0 else bleu_n_samples
        final_batches = max(math.ceil(final_n / args.batch_size), 1)
        print(f"Running final eval on {final_n} samples ...")
        final_loss = run_val_loss(self.model, self.val_loader, self.device, final_batches)
        final_bleu = run_bleu_eval(self.model, self.val_loader, self.device, final_n)
        print_metrics({**final_loss, **final_bleu}, prefix="final")