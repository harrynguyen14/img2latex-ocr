import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from datasets import load_from_disk
from pathlib import Path
from tqdm import tqdm

from preprocess import LaTeXDataset, get_tokenizer
from encode import collate_images
from evaluate import compute_metrics
from modeling_latex_ocr import LaTeXOCRConfig, LaTeXOCRModel
from utils import load_yaml, get_device


def merge_args(cfg: dict, args: argparse.Namespace) -> dict:
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    flat = {k: v for section in cfg.values() if isinstance(section, dict) for k, v in section.items()}
    flat.update(overrides)
    return flat


def parse_args():
    parser = argparse.ArgumentParser(description="Train NaViT + Qwen2.5-Coder LaTeX OCR")
    parser.add_argument("--config", type=str, default="config.yaml")

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--max_token_len", type=int)

    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--load_in_8bit", action="store_true", default=None)

    parser.add_argument("--stage1_epochs", type=int)
    parser.add_argument("--stage2_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--grad_accum", type=int)
    parser.add_argument("--lr_stage1", type=float)
    parser.add_argument("--lr_stage2", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--eval_every", type=int)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--eval_samples", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--only_stage", type=int, choices=[1, 2], default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Tên folder checkpoint để resume (vd: checkpoint-1000)")
    parser.add_argument("--aug_manifest_stage1", type=str, default=None)
    parser.add_argument("--aug_manifest_stage2", type=str, default=None)

    return parser.parse_args()


DEVICE = get_device()


def build_ocr_config(cfg: dict, use_lora: bool = False) -> LaTeXOCRConfig:
    return LaTeXOCRConfig(
        patch_size=cfg["patch_size"],
        image_height=cfg["image_height"],
        max_image_width=cfg["max_image_width"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        mlp_ratio=cfg["mlp_ratio"],
        encoder_dropout=cfg["dropout"],
        bridge_out_dim=cfg["bridge_out_dim"],
        decoder_name=cfg["tokenizer_name"],
        tokenizer_name=cfg["tokenizer_name"],
        max_new_tokens=cfg["max_token_len"],
        use_lora=use_lora,
        lora_rank=cfg.get("lora_rank", 64),
        lora_alpha=cfg.get("lora_alpha", 128),
        lora_dropout=cfg.get("lora_dropout", 0.05),
    )


def collate_fn(batch, device):
    images = [s["image"] for s in batch]
    input_ids = torch.stack([s["input_ids"] for s in batch])
    attention_mask = torch.stack([s["attention_mask"] for s in batch])
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    pixel_values, patch_mask = collate_images(images, device=device)
    return {
        "pixel_values": pixel_values,
        "patch_mask": patch_mask,
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
        "raw_labels": [s["label"] for s in batch],
    }


def make_loader(dataset, cfg, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, DEVICE),
    )


def run_eval(model: LaTeXOCRModel, val_loader, n_samples):
    model.eval()
    preds, refs, count = [], [], 0
    with torch.no_grad():
        for batch in val_loader:
            generated = model.generate(batch["pixel_values"], batch["patch_mask"])
            preds.extend(generated)
            refs.extend(batch["raw_labels"])
            count += len(generated)
            if count >= n_samples:
                break
    return compute_metrics(preds, refs)


def train_stage(model: LaTeXOCRModel, train_loader, val_loader, optimizer, scheduler, cfg, stage_name, start_step=0):
    scaler = GradScaler("cuda")
    step = start_step
    best_bleu = 0.0
    epochs = cfg[f"{stage_name}_epochs"]
    ckpt_dir = Path(cfg["ckpt_dir"])
    merge_lora = stage_name == "stage2"

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"[{stage_name}] Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(pbar):
            with autocast("cuda"):
                out = model(
                    batch["pixel_values"],
                    batch["patch_mask"],
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )
                loss = out.loss / cfg["grad_accum"]

            scaler.scale(loss).backward()

            if (i + 1) % cfg["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg["max_grad_norm"],
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
                step += 1

            total_loss += loss.item() * cfg["grad_accum"]
            pbar.set_postfix({"loss": f"{loss.item() * cfg['grad_accum']:.4f}", "step": step})

            if step > 0 and step % cfg["eval_every"] == 0 and val_loader:
                metrics = run_eval(model, val_loader, cfg["eval_samples"])
                print(f"\n[Eval step {step}] BLEU4={metrics['bleu4']:.4f}  ExactMatch={metrics['exact_match']:.4f}  EditDist={metrics['edit_distance']:.4f}")

                if metrics["bleu4"] > best_bleu:
                    best_bleu = metrics["bleu4"]
                    model.save_checkpoint(
                        ckpt_dir / f"best_{stage_name}",
                        step=step,
                        optimizer=optimizer,
                        metrics=metrics,
                        merge_lora=False,
                    )
                model.train()

            if step > 0 and step % cfg["save_every"] == 0:
                model.save_checkpoint(
                    ckpt_dir / f"checkpoint-{step}",
                    step=step,
                    optimizer=optimizer,
                    metrics={},
                    merge_lora=False,
                )

        avg_loss = total_loss / len(train_loader)
        print(f"[{stage_name}] Epoch {epoch+1} avg_loss={avg_loss:.4f}")

    model.save_checkpoint(
        ckpt_dir / f"{stage_name}_final",
        step=step,
        optimizer=optimizer,
        metrics={"best_bleu": best_bleu},
        merge_lora=merge_lora,
    )

    return step


def main():
    args = parse_args()
    cfg_raw = load_yaml(args.config)
    cfg = merge_args(cfg_raw, args)

    torch.manual_seed(cfg.get("seed", 42))
    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(exist_ok=True)
    data_dir = Path(cfg["data_dir"])

    tokenizer = get_tokenizer()
    train_ds_raw = load_from_disk(str(data_dir / "train"))
    val_ds_raw = load_from_disk(str(data_dir / "val")) if (data_dir / "val").exists() else None

    only_stage = args.only_stage

    if only_stage in (None, 1):
        print("\n=== STAGE 1: Freeze decoder, train encoder + bridge ===")

        ocr_config = build_ocr_config(cfg, use_lora=False)
        model = LaTeXOCRModel(ocr_config).to(DEVICE)
        model.freeze_decoder()

        start_step = 0
        if args.resume:
            resume_dir = ckpt_dir / args.resume
            loaded, trainer_state = LaTeXOCRModel.from_checkpoint(str(resume_dir), device=DEVICE)
            model.visual_encoder.load_state_dict(loaded.visual_encoder.state_dict())
            start_step = trainer_state.get("step", 0)

        train_ds = LaTeXDataset(train_ds_raw, tokenizer, augmented_manifest=args.aug_manifest_stage1)
        val_ds = LaTeXDataset(val_ds_raw, tokenizer) if val_ds_raw else None
        train_loader = make_loader(train_ds, cfg)
        val_loader = make_loader(val_ds, cfg, shuffle=False) if val_ds else None

        opt1 = torch.optim.AdamW(model.stage1_params(), lr=cfg["lr_stage1"], weight_decay=cfg["weight_decay"])
        if args.resume:
            model.load_optimizer(str(ckpt_dir / args.resume), opt1)
        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=cfg["stage1_epochs"] * len(train_loader) // cfg["grad_accum"])
        step = train_stage(model, train_loader, val_loader, opt1, sch1, cfg, "stage1", start_step)

    if only_stage in (None, 2):
        print("\n=== STAGE 2: Unfreeze decoder với LoRA ===")

        ocr_config2 = build_ocr_config(cfg, use_lora=True)
        model2 = LaTeXOCRModel(ocr_config2).to(DEVICE)

        start_step = 0
        stage1_dir = ckpt_dir / "stage1_final"
        if stage1_dir.exists():
            loaded, trainer_state = LaTeXOCRModel.from_checkpoint(str(stage1_dir), device=DEVICE)
            model2.visual_encoder.load_state_dict(loaded.visual_encoder.state_dict())
            start_step = trainer_state.get("step", 0)
            print(f"Loaded stage1 weights (step={start_step})")

        if args.resume and only_stage == 2:
            resume_dir = ckpt_dir / args.resume
            loaded, trainer_state = LaTeXOCRModel.from_checkpoint(str(resume_dir), device=DEVICE)
            model2.visual_encoder.load_state_dict(loaded.visual_encoder.state_dict())
            start_step = trainer_state.get("step", 0)

        train_ds2 = LaTeXDataset(train_ds_raw, tokenizer, augmented_manifest=args.aug_manifest_stage2)
        val_ds2 = LaTeXDataset(val_ds_raw, tokenizer) if val_ds_raw else None
        train_loader2 = make_loader(train_ds2, cfg)
        val_loader2 = make_loader(val_ds2, cfg, shuffle=False) if val_ds2 else None

        opt2 = torch.optim.AdamW(model2.stage2_params(), lr=cfg["lr_stage2"], weight_decay=cfg["weight_decay"])
        if args.resume and only_stage == 2:
            model2.load_optimizer(str(ckpt_dir / args.resume), opt2)

        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=cfg["stage2_epochs"] * len(train_loader2) // cfg["grad_accum"])
        train_stage(model2, train_loader2, val_loader2, opt2, sch2, cfg, "stage2", start_step)

    print("Done.")


if __name__ == "__main__":
    main()
