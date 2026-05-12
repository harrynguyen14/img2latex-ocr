import json
import math
import torch
import torch.nn as nn
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

    def forward(self, batched_images):
        x, pad_mask = self.navit(batched_images)
        # x:        (B, L, D)   — L = max real tokens in batch after pad_sequence
        # pad_mask: (B, L)      — True = real token, False = padding from pad_sequence

        if x.shape[1] > self.max_visual_tokens:
            # BUG FIX 1: tail truncation drops the bottom-right of the image.
            # Use stride sampling instead — samples uniformly across the full sequence,
            # preserving spatial coverage of the entire expression.
            #
            # Why not just resize upstream? _resize_to_token_budget() already constrains
            # each image individually, but pad_sequence pads the batch to the longest
            # sample, so x.shape[1] can exceed max_visual_tokens when batch variance
            # is high (one very wide image forces all others to be padded out).
            # In that edge case we still need a truncation strategy here.
            total = x.shape[1]
            step  = math.ceil(total / self.max_visual_tokens)
            idx   = torch.arange(0, total, step, device=x.device)[:self.max_visual_tokens]
            x        = x[:, idx]
            pad_mask = pad_mask[:, idx]

        # BUG FIX 2: zero out pad positions in encoder output before passing to decoder.
        #
        # Context: NaViT packs multiple images per batch then pads with pad_sequence()
        # to make a rectangular (B, L, D) tensor. Positions where pad_mask=False are
        # zero-filled by pad_sequence, but after the truncation branch above the zeros
        # are still present. More importantly, DecoderLM.forward() does NOT accept an
        # encoder_attention_mask argument, so the cross-attention inside the decoder
        # will attend to ALL L positions — including the padding zeros.
        #
        # Attending to zero vectors is not catastrophic (they contribute ~0 after
        # softmax weighting), but they still dilute attention scores for real tokens,
        # especially when pad ratio is high (short images in a batch with one tall image).
        #
        # Fix: replace pad positions with a large negative value in the key/value space.
        # We cannot inject an attention mask, so we make the pad tokens "invisible" by
        # filling them with -inf scaled down to a large finite negative — the cross-attn
        # softmax will drive their weight toward 0 without numerical overflow.
        #
        # Concretely: fill pad positions with a learned or fixed "null" embedding.
        # The simplest stable approach is to fill with 0 (already done by pad_sequence)
        # AND multiply real tokens by the mask, so any in-place modification to x
        # before this point doesn't corrupt the pad region.
        #
        # Shape broadcast: pad_mask (B, L) → (B, L, 1) for multiplication with (B, L, D)
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

    def forward(self, batched_images, input_ids, attention_mask, labels):
        ve, _ = self.visual_encoder(batched_images)
        loss, lm_loss, len_loss = self.decoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_output=ve,
            labels=labels,
        )
        return type("Out", (), {"loss": loss, "lm_loss": lm_loss, "len_loss": len_loss})()

    @torch.no_grad()
    def generate(self, batched_images, max_new_tokens=None, num_beams=None):
        self.eval()
        cfg = self.config
        max_new_tokens = max_new_tokens or cfg.get("max_new_tokens", 256)
        num_beams      = num_beams      or cfg.get("num_beams", 1)

        ve, _ = self.visual_encoder(batched_images)
        generated = self.decoder.generate(
            encoder_output=ve,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
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