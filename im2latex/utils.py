import os
import torch
from typing import Any


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | list]:
    return {
        "batched_images": [[item["pixel_values"]] for item in batch],
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def move_batch(batch, device):
    return {
        "batched_images": [[t.to(device, non_blocking=True) for t in imgs] for imgs in batch["batched_images"]],
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
    }


def configure_runtime(cfg, device: torch.device):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cuda_benchmark = cfg.get("cuda_benchmark", True) if isinstance(cfg, dict) else getattr(cfg, "cuda_benchmark", True)
    if device.type == "cuda" and cuda_benchmark:
        torch.backends.cudnn.benchmark = True
