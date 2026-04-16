import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

from pretrain_decoder.tokenizer import load_tokenizer

HF_REPO_ID = "harryrobert/latexOCR"


class CustomDecoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        repo_id   = config.get("decoder_ckpt") or HF_REPO_ID
        hf_config = AutoConfig.from_pretrained(
            repo_id,
            trust_remote_code=True,
            force_download=True
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            trust_remote_code=True,
            force_download=True
        )
        print(f"  Loaded decoder from HF: {repo_id}")

        self._model       = hf_model
        self.pad_token_id = hf_config.pad_id
        self.eos_token_id = hf_config.eos_id
        self._vocab_size  = hf_config.vocab_size
        self._pad_id      = hf_config.pad_id
        self.tokenizer    = load_tokenizer(config["tokenizer_dir"])

    def get_input_embeddings(self) -> nn.Embedding:
        return self._model.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        logits = self._forward_embeds(inputs_embeds, attention_mask)

        if labels is None:
            return type("Out", (), {"logits": logits, "loss": None})()

        import torch.nn.functional as F
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_labels == self._pad_id, -100)
        loss = F.cross_entropy(
            shift_logits.view(-1, self._vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return type("Out", (), {"logits": logits, "loss": loss})()

    def _forward_embeds(self, inputs_embeds: torch.Tensor, attention_mask=None) -> torch.Tensor:
        m = self._model
        x = m.embed_drop(inputs_embeds)
        mask = attention_mask.bool() if attention_mask is not None else None
        for layer in m.layers:
            x = layer(x, mask)
        return m.lm_head(m.norm_final(x))

    @torch.no_grad()
    def generate(self, inputs_embeds, attention_mask, max_new_tokens, num_beams=1,
                 pad_token_id=None, eos_token_id=None):
        eos = eos_token_id if eos_token_id is not None else self.eos_token_id
        B   = inputs_embeds.shape[0]

        generated_ids = []
        past_embeds   = inputs_embeds

        for _ in range(max_new_tokens):
            logits  = self._forward_embeds(past_embeds, attention_mask)
            next_id = logits[:, -1, :].argmax(dim=-1)
            generated_ids.append(next_id)

            next_emb    = self._model.embed_tokens(next_id).unsqueeze(1)
            past_embeds = torch.cat([past_embeds, next_emb], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones(B, 1)], dim=1
                )

            if (next_id == eos).all():
                break

        if not generated_ids:
            return torch.zeros(B, 0, dtype=torch.long, device=inputs_embeds.device)

        return torch.stack(generated_ids, dim=1)
