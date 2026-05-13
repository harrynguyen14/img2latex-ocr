import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pretrain_decoder.model import DecoderLM


class Nav2TexDecoder(nn.Module):
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

    def forward(self, input_ids, attention_mask=None, encoder_output=None, labels=None, true_len=None):
        return self._model(input_ids, attention_mask=attention_mask, encoder_output=encoder_output, labels=labels, true_len=true_len)

    @torch.no_grad()
    def generate(self, encoder_output, max_new_tokens=256):
        bsz = encoder_output.size(0)
        input_ids = torch.full((bsz, 1), self.bos_token_id, dtype=torch.long, device=encoder_output.device)
        finished  = torch.zeros(bsz, dtype=torch.bool, device=encoder_output.device)
        generated: list[list[int]] = [[] for _ in range(bsz)]

        for _ in range(max_new_tokens):
            logits   = self._model(input_ids, encoder_output=encoder_output)
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
