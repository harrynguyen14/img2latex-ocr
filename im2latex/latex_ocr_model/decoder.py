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
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            trust_remote_code=True,
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
        pad = pad_token_id if pad_token_id is not None else self.pad_token_id
        return self._beam_search(inputs_embeds, attention_mask, max_new_tokens, num_beams, eos, pad)

    @torch.no_grad()
    def _beam_search(self, inputs_embeds, attention_mask, max_new_tokens, num_beams, eos_id, pad_id):
        """Beam search decode (num_beams=1 = greedy), batch_size=1 only."""
        device = inputs_embeds.device
        B      = inputs_embeds.shape[0]
        assert B == 1, "beam search only supports batch_size=1"

        # expand to (num_beams, seq_len, D)
        embeds = inputs_embeds.expand(num_beams, -1, -1)          # (K, S, D)
        mask   = attention_mask.expand(num_beams, -1) if attention_mask is not None else None

        beam_scores = torch.zeros(num_beams, device=device)       # log-probs
        beam_ids    = [[] for _ in range(num_beams)]              # generated token ids per beam
        done        = [False] * num_beams

        for _ in range(max_new_tokens):
            logits = self._forward_embeds(embeds, mask)           # (K, S, V)
            log_p  = torch.log_softmax(logits[:, -1, :], dim=-1)  # (K, V)

            # scores for each (beam, token) combination
            scores = beam_scores.unsqueeze(1) + log_p             # (K, V)
            # flatten and pick top-K
            flat   = scores.view(-1)                              # (K*V,)
            topk_scores, topk_idx = flat.topk(num_beams, sorted=True)

            vocab_size  = log_p.shape[-1]
            beam_origin = topk_idx // vocab_size                  # which beam each came from
            next_tokens = topk_idx %  vocab_size                  # which token

            new_embeds     = []
            new_mask       = []
            new_beam_ids   = []
            new_beam_scores = []

            for k in range(num_beams):
                src   = beam_origin[k].item()
                tok   = next_tokens[k].item()
                score = topk_scores[k].item()

                if done[src]:
                    new_beam_ids.append(beam_ids[src])
                    new_beam_scores.append(beam_scores[src].item())
                    new_embeds.append(embeds[src])
                    if mask is not None:
                        new_mask.append(mask[src])
                    continue

                new_ids = beam_ids[src] + [tok]
                new_beam_ids.append(new_ids)
                new_beam_scores.append(score)

                tok_emb    = self._model.embed_tokens(
                    torch.tensor([tok], device=device)
                ).unsqueeze(0)                                    # (1, 1, D)
                new_embeds.append(torch.cat([embeds[src:src+1], tok_emb], dim=1).squeeze(0))
                if mask is not None:
                    new_mask.append(torch.cat([mask[src:src+1],
                                               mask.new_ones(1, 1)], dim=1).squeeze(0))

                if tok == eos_id:
                    done[k] = True

            embeds      = torch.stack(new_embeds, dim=0)
            beam_ids    = new_beam_ids
            beam_scores = torch.tensor(new_beam_scores, device=device)
            if mask is not None:
                mask = torch.stack(new_mask, dim=0)

            if all(done):
                break

        # return best beam (index 0 = highest score)
        best = beam_ids[0]
        if not best:
            return torch.zeros(B, 0, dtype=torch.long, device=device)
        return torch.tensor(best, dtype=torch.long, device=device).unsqueeze(0)
