import os
import torch
from typing import Any
from torch.utils.data import IterableDataset


class TokenBudgetBatcher(IterableDataset):
    def __init__(self, dataset, token_budget: int):
        self.dataset      = dataset
        self.token_budget = token_budget

    def __iter__(self):
        batch        = []
        batch_tokens = 0

        for sample in self.dataset:
            n = sample["num_tokens"]
            if batch and batch_tokens + n > self.token_budget:
                yield batch
                batch        = []
                batch_tokens = 0
            batch.append(sample)
            batch_tokens += n

        if batch:
            yield batch


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | list]:
    labels = torch.stack([item["labels"] for item in batch])
    true_len = (labels != -100).sum(dim=1).float()
    return {
        "batched_images": [[item["pixel_values"]] for item in batch],
        "input_ids":      torch.stack([item["input_ids"]      for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels":         labels,
        "true_len":       true_len,
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    """
    FIX: pixel_values tensors are float — pin_memory=True in the DataLoader
    makes them page-locked, so .to(device, non_blocking=True) triggers an async
    DMA transfer and the CPU continues immediately. The original code already used
    non_blocking=True here, but the encoder was calling image.to(device) again
    inside its forward loop, which issued redundant (and blocking) copies.
    That second call has been removed from encoder.py.
    """
    return {
        "batched_images": [
            [t.to(device, non_blocking=True) for t in imgs]
            for imgs in batch["batched_images"]
        ],
        "input_ids":      batch["input_ids"].to(device,      non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device,  non_blocking=True),
        "labels":         batch["labels"].to(device,          non_blocking=True),
        "true_len":       batch["true_len"].to(device,        non_blocking=True),
    }


def configure_runtime(cfg, device: torch.device):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cuda_benchmark = (
        cfg.get("cuda_benchmark", True) if isinstance(cfg, dict)
        else getattr(cfg, "cuda_benchmark", True)
    )
    if device.type == "cuda" and cuda_benchmark:
        torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        # RTX 5090 / Blackwell: enable CUDNN frontend for fused attention kernels
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True