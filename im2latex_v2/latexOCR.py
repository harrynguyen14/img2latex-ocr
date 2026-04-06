import torch
import torch.nn as nn

from .mlp_projector import MLPProjector
from .encoder import NaViT_Encoder


class VisualEncoder(nn.Module):
    def __init__(self, encoder: NaViT_Encoder, bridge: MLPProjector, max_visual_tokens: int):
        super().__init__()
        self.navit = encoder
        self.projector = bridge
        self.max_visual_tokens = max_visual_tokens

    def forward(self, batched_images):
        x, mask = self.navit(batched_images)
        if x.shape[1] > self.max_visual_tokens:
            x = x[:, : self.max_visual_tokens]
            mask = mask[:, : self.max_visual_tokens]
        x = self.projector(x)
        return x, mask
