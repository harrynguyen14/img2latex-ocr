import sys
from pathlib import Path

import torch
import torch.nn as nn

# Allow importing pretrain_decoder from the project root
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
        device = encoder_output.device
        B      = encoder_output.size(0)
        model  = self._model

        eos = self.eos_token_id
        bos = self.bos_token_id

        # Start every sequence with BOS
        input_ids = torch.full((B, 1), bos, dtype=torch.long, device=device)
        finished  = torch.zeros(B, dtype=torch.bool, device=device)
        generated = [[] for _ in range(B)]

        for _ in range(max_new_tokens):
            logits = model(
                input_ids,
                encoder_output=encoder_output,
            )                                           # (B, T, V)
            next_tok = logits[:, -1, :].argmax(dim=-1)  # (B,)

            for i in range(B):
                if not finished[i]:
                    tok = next_tok[i].item()
                    generated[i].append(tok)
                    if tok == eos:
                        finished[i] = True

            if finished.all():
                break

            input_ids = torch.cat(
                [input_ids, next_tok.unsqueeze(1)], dim=1
            )

        return generated
