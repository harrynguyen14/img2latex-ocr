import argparse
import torch
from pathlib import Path

from transformers import TrainingArguments

from preprocess import LaTeXDataset, get_tokenizer
from modeling_latex_ocr import LaTeXOCRConfig, LaTeXOCRModel
from trainer import LaTeXOCRTrainer, LaTeXDataCollator, make_compute_metrics
from utils import load_yaml, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train NaViT + Qwen2.5-Coder LaTeX OCR")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_path", type=str, required=True,
                        help="HF repo id hoặc local Parquet folder (vd: harryrobert/latex-ocr)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path checkpoint để resume (vd: checkpoints/checkpoint-1000)")
    parser.add_argument("--only_stage", type=int, choices=[1, 2], default=None)

    # Override config từ CLI
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr_stage1", type=float)
    parser.add_argument("--lr_stage2", type=float)
    parser.add_argument("--stage1_epochs", type=int)
    parser.add_argument("--stage2_epochs", type=int)
    parser.add_argument("--grad_accum", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--eval_samples", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ckpt_dir", type=str)
    return parser.parse_args()


def merge_args(cfg: dict, args: argparse.Namespace) -> dict:
    overrides = {k: v for k, v in vars(args).items() if v is not None and k not in ("config", "data_path", "resume", "only_stage")}
    flat = {k: v for section in cfg.values() if isinstance(section, dict) for k, v in section.items()}
    flat.update(overrides)
    return flat


def build_model_config(cfg: dict, use_lora: bool) -> LaTeXOCRConfig:
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


def build_training_args(cfg: dict, output_dir: str, lr: float, num_epochs: int, num_samples: int) -> TrainingArguments:
    steps_per_epoch = num_samples // cfg["batch_size"]
    max_steps = (steps_per_epoch * num_epochs) // cfg["grad_accum"]
    warmup_steps = int(max_steps * 0.05)
    return TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=lr,
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="bleu4",
        greater_is_better=True,
        logging_steps=50,
        report_to=cfg.get("report_to", "tensorboard"),
        dataloader_num_workers=cfg["num_workers"],
        remove_unused_columns=False,   # giữ pixel_values, patch_mask, raw_labels
        seed=cfg.get("seed", 42),
        label_names=["labels"],
    )


def run_stage(
    stage: int,
    cfg: dict,
    data_path: str,
    tokenizer,
    resume: str = None,
):
    use_lora = (stage == 2)
    stage_name = f"stage{stage}"
    output_dir = str(Path(cfg["ckpt_dir"]) / stage_name)

    lr = cfg["lr_stage1"] if stage == 1 else cfg["lr_stage2"]
    num_epochs = cfg["stage1_epochs"] if stage == 1 else cfg["stage2_epochs"]

    model_cfg = build_model_config(cfg, use_lora=use_lora)
    model = LaTeXOCRModel(model_cfg).to(get_device())

    # Load weights từ stage trước hoặc resume
    if stage == 2:
        stage1_dir = Path(cfg["ckpt_dir"]) / "stage1"
        best_stage1 = stage1_dir / "best_model" if (stage1_dir / "best_model").exists() else stage1_dir
        if best_stage1.exists():
            loaded, _ = LaTeXOCRModel.from_checkpoint(str(best_stage1), device=get_device())
            model.visual_encoder.load_state_dict(loaded.visual_encoder.state_dict())
            print(f"Loaded stage1 weights from {best_stage1}")

    if resume:
        loaded, _ = LaTeXOCRModel.from_checkpoint(resume, device=get_device())
        model.visual_encoder.load_state_dict(loaded.visual_encoder.state_dict())
        print(f"Resumed from {resume}")

    train_ds = LaTeXDataset(data_path, stage_name, tokenizer)
    val_ds   = LaTeXDataset(data_path, "validation", tokenizer)

    num_samples = train_ds.num_samples or cfg.get("num_samples", 659658)
    training_args = build_training_args(cfg, output_dir, lr, num_epochs, num_samples)
    collator = LaTeXDataCollator()
    compute_metrics = make_compute_metrics(tokenizer)

    # Stage 1: freeze decoder từ epoch 0 → unfreeze không cần
    # Stage 2: LoRA đã được bật khi use_lora=True, không cần callback freeze/unfreeze
    callbacks = []
    if stage == 1:
        from transformers import TrainerCallback

        class FreezeDecoderCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, model=None, **kwargs):
                model.freeze_decoder()
                print("[Stage 1] Decoder frozen.")

        callbacks.append(FreezeDecoderCallback())

    trainer = LaTeXOCRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=resume if resume else None)

    # Lưu best model theo chuẩn HF
    best_dir = Path(output_dir) / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"[Stage {stage}] Best model saved to {best_dir}")

    return trainer


def main():
    args = parse_args()
    cfg_raw = load_yaml(args.config)
    cfg = merge_args(cfg_raw, args)

    torch.manual_seed(cfg.get("seed", 42))
    Path(cfg["ckpt_dir"]).mkdir(exist_ok=True)

    tokenizer = get_tokenizer()
    only_stage = args.only_stage

    if only_stage in (None, 1):
        print("\n=== STAGE 1: Freeze decoder, train encoder + bridge ===")
        run_stage(1, cfg, args.data_path, tokenizer, resume=args.resume if args.only_stage == 1 else None)

    if only_stage in (None, 2):
        print("\n=== STAGE 2: Train encoder + bridge + LoRA decoder ===")
        run_stage(2, cfg, args.data_path, tokenizer, resume=args.resume if args.only_stage == 2 else None)

    print("Done.")


if __name__ == "__main__":
    main()
