import torch
from typing import Any, Dict, List

from constants import IMAGE_HEIGHT, PATCH_SIZE


class LaTeXDataCollator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_list = [item["pixel_values"] for item in batch]
        max_w = max(t.shape[-1] for t in pixel_list)

        padded, masks = [], []
        for t in pixel_list:
            w = t.shape[-1]
            t_pad = torch.nn.functional.pad(t, (0, max_w - w), value=1.0)
            padded.append(t_pad)
            ph = IMAGE_HEIGHT // PATCH_SIZE
            pw_total = max_w // PATCH_SIZE
            mask = torch.zeros(ph, pw_total, dtype=torch.bool)
            mask[:, : w // PATCH_SIZE] = True
            masks.append(mask.reshape(-1))

        return {
            "pixel_values":   torch.stack(padded),
            "patch_mask":     torch.stack(masks),
            "input_ids":      torch.stack([item["input_ids"]      for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels":         torch.stack([item["labels"]         for item in batch]),
        }
