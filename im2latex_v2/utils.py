import os
import math
import functools
import torch
from typing import Any
import torch.distributed as dist


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | list]:
    return {
        "batched_images": [[item["pixel_values"]] for item in batch],
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size()


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_fsdp(model, amp_dtype: torch.dtype):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, BackwardPrefetch
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    mp = MixedPrecision(
        param_dtype=amp_dtype,
        reduce_dtype=torch.float32,
        buffer_dtype=amp_dtype,
    )

    auto_wrap = functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000)

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )


def move_batched_images(bi, device):
    return [[t.to(device, non_blocking=True) for t in imgs] for imgs in bi]


def move_batch(batch, device):
    return {
        "batched_images": move_batched_images(batch["batched_images"], device),
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
    }


def lr_cosine(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return peak * 0.5 * (1.0 + math.cos(math.pi * t))


def amp_dtype_from_cfg(cfg: dict) -> torch.dtype:
    name = str(cfg.get("amp_dtype", "float16")).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


def configure_runtime(cfg, device: torch.device):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cuda_benchmark = cfg.get("cuda_benchmark", True) if isinstance(cfg, dict) else getattr(cfg, "cuda_benchmark", True)
    if device.type == "cuda" and cuda_benchmark:
        torch.backends.cudnn.benchmark = True
