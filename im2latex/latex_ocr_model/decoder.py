import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pretrain_decoder.model import LaTeXDecoder
from pretrain_decoder.config import DecoderConfig
from pretrain_decoder.tokenizer import load_tokenizer


class CustomDecoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        decoder_ckpt = config.get("decoder_ckpt")
        cfg_path     = Path(decoder_ckpt) / "config.json" if decoder_ckpt else None
        decoder_cfg  = DecoderConfig.load(cfg_path) if (cfg_path and cfg_path.exists()) else DecoderConfig()

        self.decoder_cfg = decoder_cfg
        self.model       = LaTeXDecoder(decoder_cfg)

        if decoder_ckpt:
            ckpt_dir = Path(decoder_ckpt)
            sf = ckpt_dir / "model.safetensors"
            pt = ckpt_dir / "model.pt"
            if sf.exists():
                from safetensors.torch import load_file
                state = load_file(str(sf))
            elif pt.exists():
                state = torch.load(str(pt), map_location="cpu")
            else:
                raise FileNotFoundError(f"No model weights found in {ckpt_dir}")
            # strip _orig_mod. prefix if saved from torch.compile
            state = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                     for k, v in state.items()}
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if unexpected:
                print(f"  [decoder] unexpected keys: {unexpected[:3]}...")
            print(f"  Loaded decoder weights from {ckpt_dir.name} (missing={len(missing)}, unexpected={len(unexpected)})")

        self.tokenizer = load_tokenizer(config["tokenizer_dir"])
        self.pad_token_id = decoder_cfg.pad_id
        self.eos_token_id = decoder_cfg.eos_id

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        B, T, _ = inputs_embeds.shape
        logits   = self._forward_embeds(inputs_embeds, attention_mask)

        if labels is None:
            return type("Out", (), {"logits": logits, "loss": None})()

        import torch.nn.functional as F
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_labels == self.decoder_cfg.pad_id, -100)
        loss = F.cross_entropy(
            shift_logits.view(-1, self.decoder_cfg.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return type("Out", (), {"logits": logits, "loss": loss})()

    def _forward_embeds(self, inputs_embeds: torch.Tensor, attention_mask=None) -> torch.Tensor:
        x = self.model.embed_drop(inputs_embeds)
        for layer in self.model.layers:
            x = layer(x, attention_mask)
        return self.model.lm_head(self.model.norm_final(x))

    @torch.no_grad()
    def generate(self, inputs_embeds, attention_mask, max_new_tokens, num_beams=1,
                 pad_token_id=None, eos_token_id=None):
        eos = eos_token_id if eos_token_id is not None else self.eos_token_id
        B   = inputs_embeds.shape[0]

        generated_ids = []
        past_embeds   = inputs_embeds

        for _ in range(max_new_tokens):
            logits   = self._forward_embeds(past_embeds, attention_mask)
            next_id  = logits[:, -1, :].argmax(dim=-1)
            generated_ids.append(next_id)

            next_emb     = self.model.embed_tokens(next_id).unsqueeze(1)
            past_embeds  = torch.cat([past_embeds, next_emb], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones(B, 1)], dim=1
                )

            if (next_id == eos).all():
                break

        if not generated_ids:
            return torch.zeros(B, 0, dtype=torch.long, device=inputs_embeds.device)

        return torch.stack(generated_ids, dim=1)
