import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pretrain_decoder.model import DecoderLM


class Nav2TexDecoder(nn.Module):
    """
    Thin wrapper around DecoderLM loaded from HuggingFace.
    Exposes the same interface (pad/eos token ids, get_input_embeddings,
    forward, generate) that LaTeXOCRModel expects.
    """

    def __init__(self, repo_id: str = "harryrobert/nav2tex-decoder", device: str = "cpu"):
        super().__init__()
        self._model = DecoderLM.from_pretrained(repo_id, device=device)
        cfg = self._model.config
        self.pad_token_id = cfg.pad_token_id
        self.eos_token_id = cfg.eos_token_id
        self.bos_token_id = cfg.bos_token_id
        self.vocab_size   = cfg.vocab_size

    def get_input_embeddings(self) -> nn.Embedding:
        return self._model.token_embed

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_output: torch.Tensor | None = None,
        labels:         torch.Tensor | None = None,
    ):
        """
        Returns (loss, lm_loss, len_loss) tuple when labels given,
        else logits tensor — matching DecoderLM.forward() exactly.
        """
        return self._model(
            input_ids,
            attention_mask=attention_mask,
            encoder_output=encoder_output,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,
        max_new_tokens:  int = 256,
        num_beams:       int = 1,
    ) -> list[list[int]]:
        """
        Greedy or beam-search decode given encoder_output.
        Returns list of token-id lists (one per batch item).
        """
        beams = max(1, int(num_beams))
        if beams == 1:
            return self._greedy_decode_batched(encoder_output, max_new_tokens)
        return self._beam_decode_batched(encoder_output, max_new_tokens, beams)

    def _greedy_decode_batched(self, encoder_output: torch.Tensor, max_new_tokens: int) -> list[list[int]]:
        bsz = encoder_output.size(0)
        input_ids = torch.full(
            (bsz, 1), self.bos_token_id, dtype=torch.long, device=encoder_output.device
        )
        finished = torch.zeros(bsz, dtype=torch.bool, device=encoder_output.device)
        generated: list[list[int]] = [[] for _ in range(bsz)]

        for _ in range(max_new_tokens):
            logits = self._model(input_ids, encoder_output=encoder_output)
            next_tok = logits[:, -1, :].argmax(dim=-1)
            for i in range(bsz):
                if finished[i]:
                    continue
                tok = int(next_tok[i].item())
                generated[i].append(tok)
                if tok == self.eos_token_id:
                    finished[i] = True
            if finished.all():
                break
            input_ids = torch.cat([input_ids, next_tok.unsqueeze(1)], dim=1)
        return generated

    def _beam_decode_batched(
        self, encoder_output: torch.Tensor, max_new_tokens: int, num_beams: int
    ) -> list[list[int]]:
        bsz, src_len, src_dim = encoder_output.shape
        device = encoder_output.device
        vocab_size = self.vocab_size
        eos = self.eos_token_id

        seqs = torch.full((bsz, num_beams, 1), self.bos_token_id, dtype=torch.long, device=device)
        scores = torch.full((bsz, num_beams), float("-inf"), device=device)
        scores[:, 0] = 0.0
        finished = torch.zeros((bsz, num_beams), dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            flat_seqs = seqs.view(bsz * num_beams, -1)
            flat_enc = (
                encoder_output.unsqueeze(1)
                .expand(bsz, num_beams, src_len, src_dim)
                .reshape(bsz * num_beams, src_len, src_dim)
            )
            logits = self._model(flat_seqs, encoder_output=flat_enc)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1).view(bsz, num_beams, vocab_size)

            stay_finished = torch.full_like(log_probs, float("-inf"))
            stay_finished[..., eos] = 0.0
            log_probs = torch.where(finished.unsqueeze(-1), stay_finished, log_probs)

            cand_scores = scores.unsqueeze(-1) + log_probs
            cand_scores = cand_scores.view(bsz, num_beams * vocab_size)
            top_scores, top_idx = torch.topk(cand_scores, k=num_beams, dim=-1)

            beam_idx = top_idx // vocab_size
            token_idx = top_idx % vocab_size

            gather_idx = beam_idx.unsqueeze(-1).expand(-1, -1, seqs.size(-1))
            selected_seqs = torch.gather(seqs, 1, gather_idx)
            seqs = torch.cat([selected_seqs, token_idx.unsqueeze(-1)], dim=-1)

            selected_finished = torch.gather(finished, 1, beam_idx)
            finished = selected_finished | token_idx.eq(eos)
            scores = top_scores

            if finished.all():
                break

        outputs: list[list[int]] = []
        for b in range(bsz):
            best_idx = 0
            best_score = float("-inf")
            for k in range(num_beams):
                seq = seqs[b, k].tolist()
                score = self._rerank_score(seq, float(scores[b, k].item()))
                if score > best_score:
                    best_score = score
                    best_idx = k
            outputs.append(seqs[b, best_idx, 1:].tolist())
        return outputs

    def _rerank_score(self, seq: list[int], logp_sum: float) -> float:
        token_count = max(1, len(seq) - 1)
        alpha = 0.7
        length_norm = ((5.0 + token_count) / 6.0) ** alpha
        rep_pen = self._repetition_penalty(seq[1:])
        return (logp_sum / length_norm) - rep_pen

    def _repetition_penalty(self, seq: list[int]) -> float:
        if len(seq) < 4:
            return 0.0
        seen = set()
        dup = 0
        for i in range(len(seq) - 2):
            tri = (seq[i], seq[i + 1], seq[i + 2])
            if tri in seen:
                dup += 1
            else:
                seen.add(tri)
        return 0.1 * dup
