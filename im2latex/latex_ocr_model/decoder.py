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
        device   = inputs_embeds.device
        B        = inputs_embeds.shape[0]
        assert B == 1, "beam search only supports batch_size=1"

        vis_emb  = inputs_embeds[0]                               # (S, D)
        vis_len  = vis_emb.shape[0]
        vis_mask = attention_mask[0] if attention_mask is not None else None

        # Each beam: (score, [token_ids], done)
        beams = [(0.0, [], False)] * num_beams

        for _ in range(max_new_tokens):
            # Build batch: visual embeds + generated token embeds for each beam
            all_embeds = []
            all_masks  = []
            active_idx = []  # beams that are not done

            for k, (score, ids, done) in enumerate(beams):
                if ids:
                    tok_tensor = torch.tensor(ids, device=device, dtype=torch.long)
                    tok_emb    = self._model.embed_tokens(tok_tensor)  # (T, D)
                    seq_emb    = torch.cat([vis_emb, tok_emb], dim=0)  # (S+T, D)
                else:
                    seq_emb = vis_emb                                  # (S, D)

                all_embeds.append(seq_emb)
                if vis_mask is not None:
                    if ids:
                        tok_mask = vis_mask.new_ones(len(ids))
                        all_masks.append(torch.cat([vis_mask, tok_mask]))
                    else:
                        all_masks.append(vis_mask)
                active_idx.append(k)

            # Pad to same length for batched forward
            max_len = max(e.shape[0] for e in all_embeds)
            D       = all_embeds[0].shape[-1]
            padded_embeds = vis_emb.new_zeros(num_beams, max_len, D)
            padded_mask   = None
            if vis_mask is not None:
                padded_mask = vis_mask.new_zeros(num_beams, max_len)

            for k, (emb, msk) in enumerate(zip(all_embeds, all_masks if vis_mask is not None else [None]*num_beams)):
                L = emb.shape[0]
                padded_embeds[k, :L] = emb
                if padded_mask is not None:
                    padded_mask[k, :L] = msk

            logits = self._forward_embeds(padded_embeds, padded_mask)  # (K, max_len, V)

            # Use logit at last real token position for each beam
            new_beams = []
            candidates = []
            for k, (score, ids, done) in enumerate(beams):
                last_pos = vis_len + len(ids) - 1
                if done:
                    candidates.append((score, ids, True, k, pad_id))
                    continue
                log_p = torch.log_softmax(logits[k, last_pos, :], dim=-1)  # (V,)
                # On first step only beam 0 is active; expand after
                if len(ids) == 0 and k > 0:
                    log_p = log_p.fill_(-1e9)
                topk_lp, topk_tok = log_p.topk(num_beams)
                for lp, tok in zip(topk_lp.tolist(), topk_tok.tolist()):
                    candidates.append((score + lp, ids + [tok], tok == eos_id, k, tok))

            # Pick top-K candidates
            candidates.sort(key=lambda x: -x[0])
            candidates = candidates[:num_beams]

            for score, ids, done, src_k, tok in candidates:
                new_beams.append((score, ids, done))

            beams = new_beams

            if all(done for _, _, done in beams):
                break

        # Return best beam (highest score)
        best_score, best_ids, _ = max(beams, key=lambda x: x[0])
        if not best_ids:
            return torch.zeros(B, 0, dtype=torch.long, device=device)
        return torch.tensor(best_ids, dtype=torch.long, device=device).unsqueeze(0)
