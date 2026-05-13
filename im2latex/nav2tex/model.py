import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .decoder import Nav2TexDecoder
from .encoder import NaViT_Encoder


def decode_ids(tokenizer, ids: list[int], skip_ids: set[int] | None = None) -> str:
    if skip_ids is None:
        skip_ids = set()
    filtered = [i for i in ids if i not in skip_ids]
    return tokenizer.decode(filtered, skip_special_tokens=False)


class VisualEncoder(nn.Module):
    def __init__(self, encoder: NaViT_Encoder, max_visual_tokens: int):
        super().__init__()
        self.navit = encoder
        self.max_visual_tokens = max_visual_tokens

    def _pool_to_budget(self, x: torch.Tensor, pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        if T <= self.max_visual_tokens:
            return x, pad_mask

        grid_side = math.isqrt(T)
        if grid_side * grid_side == T:
            ph = pw = grid_side
        else:
            ph = math.isqrt(T)
            pw = T // ph

        target = self.max_visual_tokens
        out_ph = max(1, math.isqrt(target))
        out_pw = max(1, target // out_ph)

        x_grid = x.view(B, ph, pw, D).permute(0, 3, 1, 2)
        x_grid = F.adaptive_avg_pool2d(x_grid, (out_ph, out_pw))
        x_out  = x_grid.permute(0, 2, 3, 1).reshape(B, out_ph * out_pw, D)

        mask_grid = pad_mask.float().view(B, 1, ph, pw)
        mask_grid = F.adaptive_avg_pool2d(mask_grid, (out_ph, out_pw))
        mask_out  = (mask_grid.squeeze(1).reshape(B, out_ph * out_pw) > 0.0)

        return x_out, mask_out

    def forward(self, batched_images):
        x, pad_mask = self.navit(batched_images)
        x, pad_mask = self._pool_to_budget(x, pad_mask)
        x = x * pad_mask.unsqueeze(-1)
        return x, pad_mask


class LaTeXOCRModel(nn.Module):
    def __init__(self, config, tokenizer=None):
        super().__init__()
        if not isinstance(config, dict):
            config = vars(config)
        self.config = dict(config)

        self.visual_encoder = VisualEncoder(
            NaViT_Encoder(
                dim=config["navit_dim"],
                depth=config["navit_depth"],
                heads=config["navit_heads"],
                mlp_dim=config["navit_mlp_dim"],
                dim_head=config["navit_dim_head"],
                dropout=config["navit_dropout"],
                emb_dropout=config["navit_emb_dropout"],
                grad_checkpoint=config.get("grad_checkpoint", False),
            ),
            max_visual_tokens=config["max_visual_tokens"],
        )

        self.decoder   = Nav2TexDecoder(repo_id=config.get("decoder_repo_id", "harryrobert/nav2tex-decoder"))
        self.tokenizer = tokenizer

    def freeze_decoder(self):
        for p in self.decoder.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            if p.dtype.is_floating_point:
                p.requires_grad = True

    def enable_decoder_grad_checkpoint(self):
        dec = self.decoder._model
        if hasattr(dec, "enable_gradient_checkpointing"):
            dec.enable_gradient_checkpointing()
        elif hasattr(dec, "transformer") and hasattr(dec.transformer, "grad_checkpoint"):
            dec.transformer.grad_checkpoint = True

    def forward(self, batched_images, input_ids, attention_mask, labels, true_len=None):
        ve, _ = self.visual_encoder(batched_images)
        loss, lm_loss, len_loss = self.decoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_output=ve,
            labels=labels,
            true_len=true_len,
        )
        return type("Out", (), {"loss": loss, "lm_loss": lm_loss, "len_loss": len_loss})()

    @torch.no_grad()
    def generate(self, batched_images, max_new_tokens=None):
        self.eval()
        max_new_tokens = max_new_tokens or self.config.get("max_new_tokens", 256)

        ve, _ = self.visual_encoder(batched_images)
        generated = self.decoder.generate(
            encoder_output=ve,
            max_new_tokens=max_new_tokens,
        )

        skip = {self.decoder.pad_token_id, self.decoder.eos_token_id, self.decoder.bos_token_id}
        return [decode_ids(self.tokenizer, ids, skip_ids=skip) for ids in generated]

    @classmethod
    def from_pretrained(cls, ckpt_dir: str, device: str = "cpu", tokenizer=None):
        from safetensors.torch import load_file
        ckpt = Path(ckpt_dir)
        with open(ckpt / "config.json", encoding="utf-8") as f:
            config = json.load(f)
        model = cls(config, tokenizer=tokenizer)
        state = load_file(str(ckpt / "model.safetensors"), device=device)
        ve_state  = {k[len("visual_encoder."):]: v for k, v in state.items() if k.startswith("visual_encoder.")}
        dec_state = {k[len("decoder."):]: v       for k, v in state.items() if k.startswith("decoder.")}
        if ve_state:
            model.visual_encoder.load_state_dict(ve_state, strict=True)
        if dec_state:
            model.decoder.load_state_dict(dec_state, strict=True)
        model.to(device)
        model.eval()
        return model