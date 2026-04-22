import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM
from tokenizer_v2 import LaTeXTokenizerV2


class CustomDecoder(nn.Module):
    def __init__(self, config: dict, tokenizer=None):
        super().__init__()
        repo = config.get("decoder_ckpt", "harryrobert/pretrain-decoder")
        self._model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
        self._model.config.tie_weights = False

        self.pad_token_id = self._model.config.pad_id
        self.eos_token_id = self._model.config.eos_id
        self._vocab_size  = self._model.config.vocab_size
        self._pad_id      = self._model.config.pad_id
        self.tokenizer    = tokenizer

        if config.get("qat", False):
            self._enable_qat()

    def _enable_qat(self):
        self._model.train()  
        self._model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
        torch.ao.quantization.prepare_qat(self._model, inplace=True)
        print("  [decoder] QAT enabled (fake-quantize on all Linear layers)")

    def get_input_embeddings(self) -> nn.Embedding:
        return self._model.embed_tokens

    def decoder_linear_weights(self):
        """Iterator over (name, weight) for all Linear layers in decoder — used for sparsity penalty."""
        for name, m in self._model.named_modules():
            if isinstance(m, nn.Linear) and m.weight.requires_grad:
                yield name, m.weight

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        logits = self._forward_embeds(inputs_embeds, attention_mask)

        if labels is None:
            return type("Out", (), {"logits": logits, "loss": None})()

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
        device  = inputs_embeds.device
        B       = inputs_embeds.shape[0]
        assert B == 1, "beam search only supports batch_size=1"

        vis_emb  = inputs_embeds[0]
        vis_len  = vis_emb.shape[0]
        vis_mask = attention_mask[0] if attention_mask is not None else None

        beams = [(0.0, [], False) for _ in range(num_beams)]

        for _ in range(max_new_tokens):
            all_embeds, all_masks = [], []

            for k, (score, ids, done) in enumerate(beams):
                if ids:
                    tok_emb = self._model.embed_tokens(
                        torch.tensor(ids, device=device, dtype=torch.long)
                    )
                    seq_emb = torch.cat([vis_emb, tok_emb], dim=0)
                else:
                    seq_emb = vis_emb
                all_embeds.append(seq_emb)
                if vis_mask is not None:
                    tok_mask = vis_mask.new_ones(len(ids)) if ids else vis_mask.new_zeros(0)
                    all_masks.append(torch.cat([vis_mask, tok_mask]) if ids else vis_mask)

            max_len = max(e.shape[0] for e in all_embeds)
            D = all_embeds[0].shape[-1]
            padded_embeds = vis_emb.new_zeros(num_beams, max_len, D)
            padded_mask = vis_mask.new_zeros(num_beams, max_len) if vis_mask is not None else None

            for k, emb in enumerate(all_embeds):
                L = emb.shape[0]
                padded_embeds[k, :L] = emb
                if padded_mask is not None:
                    padded_mask[k, :L] = all_masks[k]

            logits = self._forward_embeds(padded_embeds, padded_mask)

            candidates = []
            for k, (score, ids, done) in enumerate(beams):
                last_pos = vis_len + len(ids) - 1
                if done:
                    candidates.append((score, ids, True, k, pad_id))
                    continue
                log_p = torch.log_softmax(logits[k, last_pos, :], dim=-1)
                if len(ids) == 0 and k > 0:
                    log_p = log_p.fill_(-1e9)
                topk_lp, topk_tok = log_p.topk(num_beams)
                for lp, tok in zip(topk_lp.tolist(), topk_tok.tolist()):
                    candidates.append((score + lp, ids + [tok], tok == eos_id, k, tok))

            candidates.sort(key=lambda x: -x[0])
            beams = [(s, ids, done) for s, ids, done, *_ in candidates[:num_beams]]

            if all(done for _, _, done in beams):
                break

        best_score, best_ids, _ = max(beams, key=lambda x: x[0])
        if not best_ids:
            return torch.zeros(B, 0, dtype=torch.long, device=device)
        return torch.tensor(best_ids, dtype=torch.long, device=device).unsqueeze(0)
