from typing import Any

import torch


class LaTeXOCRCollator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | list]:
        batched_images = [[item["pixel_values"]] for item in batch]
        return {
            "batched_images": batched_images,
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }
