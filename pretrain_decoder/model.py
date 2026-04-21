import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from pretrain_decoder.config import DecoderConfig
except ImportError:
    from config import DecoderConfig


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model
        self.dropout_p = cfg.dropout
        self.rope_theta = cfg.rope_theta

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj  = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self._rope_cache: dict = {}

    def _get_rope(self, seq_len: int, device, dtype):
        key = (seq_len, str(device), dtype)
        if key not in self._rope_cache:
            cos, sin = _build_rope_cache(seq_len, self.head_dim, self.rope_theta, device, dtype)
            self._rope_cache[key] = (cos, sin)
        return self._rope_cache[key]

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._get_rope(T, x.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)

        dropout_p = self.dropout_p if self.training else 0.0

        if attention_mask is not None:
            causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device, dtype=q.dtype), diagonal=1)
            pad    = (~attention_mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attn_bias = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T).clone()
            attn_bias = attn_bias.masked_fill(pad, float("-inf"))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj   = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout   = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn   = SwiGLUFFN(cfg)
        self.drop  = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), attention_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class LaTeXDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.embed_drop   = nn.Dropout(cfg.dropout)
        self.layers       = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_final   = RMSNorm(cfg.d_model)
        self.lm_head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

        residual_scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for block in self.layers:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=0.02 * residual_scale)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=0.02 * residual_scale)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed_drop(self.embed_tokens(input_ids))
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.lm_head(self.norm_final(x))

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_labels == self.cfg.pad_id, -100)
        return F.cross_entropy(
            shift_logits.view(-1, self.cfg.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        eos = eos_id if eos_id is not None else self.cfg.eos_id
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            ctx = generated[:, -self.cfg.max_seq_len:]
            next_logits = self.forward(ctx)[:, -1, :]

            if temperature == 0.0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                sorted_probs[cumsum - sorted_probs > top_p] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                next_id = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))

            generated = torch.cat([generated, next_id], dim=-1)
            if next_id.item() == eos:
                break

        return generated

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        cfg = self.cfg
        return (
            f"LaTeXDecoder(n_layers={cfg.n_layers}, d_model={cfg.d_model}, "
            f"n_heads={cfg.n_heads}, d_ff={cfg.d_ff}, "
            f"vocab={cfg.vocab_size}, params={self.num_parameters()/1e6:.1f}M)"
        )


if __name__ == "__main__":
    for n_layers in [6, 8]:
        cfg = DecoderConfig(n_layers=n_layers)
        model = LaTeXDecoder(cfg)
        print(model)
        B, T = 4, 64
        ids = torch.randint(4, cfg.vocab_size, (B, T))
        ids[:, 0] = cfg.bos_id
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[0, 50:] = False
        with torch.no_grad():
            loss = model.compute_loss(ids, mask)
        print(f"  loss={loss.item():.4f}\n")
