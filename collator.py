import torch
from typing import Any, Dict, List

from constants import IMAGE_HEIGHT, PATCH_SIZE


class LaTeXDataCollator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_pixel_tensors = [item["pixel_values"] for item in batch]
        max_image_width     = max(t.shape[-1] for t in batch_pixel_tensors)

        padded_images = []
        valid_patch_masks = []
        for image_tensor in batch_pixel_tensors:
            image_width = image_tensor.shape[-1]
            padded_image = torch.nn.functional.pad(
                image_tensor, (0, max_image_width - image_width), value=1.0
            )
            padded_images.append(padded_image)

            patch_height     = IMAGE_HEIGHT // PATCH_SIZE
            max_patch_width  = max_image_width // PATCH_SIZE
            valid_patch_mask = torch.zeros(patch_height, max_patch_width, dtype=torch.bool)
            # Mark patches that correspond to actual image pixels (not padding)
            valid_patch_mask[:, : image_width // PATCH_SIZE] = True
            valid_patch_masks.append(valid_patch_mask.reshape(-1))

        return {
            "pixel_values":   torch.stack(padded_images),
            "patch_mask":     torch.stack(valid_patch_masks),
            "input_ids":      torch.stack([item["input_ids"]      for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels":         torch.stack([item["labels"]         for item in batch]),
        }
