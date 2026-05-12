import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
from typing import List
from functools import partial
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy

_flash_attn_varlen_func = None


def _load_flash_attn():
    global _flash_attn_varlen_func
    if _flash_attn_varlen_func is not None:
        return True
    try:
        from flash_attn import flash_attn_varlen_func as _fn
        _flash_attn_varlen_func = _fn
        return True
    except ImportError:
        return False


def _build_block_diagonal_mask(cu_seqlens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    total = int(cu_seqlens[-1].item())
    mask = torch.full((total, total), float('-inf'), device=cu_seqlens.device, dtype=dtype)
    for i in range(len(cu_seqlens) - 1):
        s, e = int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())
        mask[s:e, s:e] = 0.0
    return mask


class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma.to(x.dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_2d_rope(q, k, h_idx, w_idx):
    B, H, N, D = q.shape
    device = q.device
    if D % 4 != 0:
        raise ValueError(f"apply_2d_rope expects dim_head divisible by 4, got D={D}")
    dim_half = D // 2

    q_h, q_w = q[..., :dim_half], q[..., dim_half:]
    k_h, k_w = k[..., :dim_half], k[..., dim_half:]
    dtype = q.dtype

    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_half, 2, device=device).float() / dim_half))

    h_theta = h_idx[..., None].float() * inv_freq
    w_theta = w_idx[..., None].float() * inv_freq

    sin_h = torch.cat([h_theta.sin(), h_theta.sin()], dim=-1).to(dtype)[:, None]
    cos_h = torch.cat([h_theta.cos(), h_theta.cos()], dim=-1).to(dtype)[:, None]
    sin_w = torch.cat([w_theta.sin(), w_theta.sin()], dim=-1).to(dtype)[:, None]
    cos_w = torch.cat([w_theta.cos(), w_theta.cos()], dim=-1).to(dtype)[:, None]

    def rope(x, sin, cos):
        return (x * cos) + (rotate_half(x) * sin)

    q = torch.cat([rope(q_h, sin_h, cos_h), rope(q_w, sin_w, cos_w)], dim=-1)
    k = torch.cat([rope(k_h, sin_h, cos_h), rope(k_w, sin_w, cos_w)], dim=-1)
    return q, k


class FineGrainedEmbedding(nn.Module):
    def __init__(self, channels: int, dim: int):
        super().__init__()
        mid = dim // 2
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """x: (B, C, H, W) — supports any batch size."""
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.conv2(x)
        H, W = x.shape[-2], x.shape[-1]
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.norm2(x)
        return x, (H, W)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2  = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(self.norm(x)))))


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim   = dim_head * heads
        self.heads  = heads
        self.norm   = nn.LayerNorm(dim)
        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)
        self.to_q   = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv  = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), nn.Dropout(dropout))
        self.dropout_p = dropout

    def forward(self, x, positions=None, cu_seqlens=None, max_seqlen=None, attn_bias=None):
        x = self.norm(x)
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        q = self.q_norm(q)
        k = self.k_norm(k)

        if positions is not None:
            h_idx, w_idx = positions
            q, k = apply_2d_rope(q, k, h_idx, w_idx)

        dropout_p = self.dropout_p if self.training else 0.0

        if cu_seqlens is not None and _load_flash_attn():
            fa_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
            q_ = rearrange(q, 'b h n d -> (b n) h d').contiguous().to(fa_dtype)
            k_ = rearrange(k, 'b h n d -> (b n) h d').contiguous().to(fa_dtype)
            v_ = rearrange(v, 'b h n d -> (b n) h d').contiguous().to(fa_dtype)
            out = _flash_attn_varlen_func(
                q_, k_, v_,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                dropout_p=dropout_p,
                causal=False,
            )
            out = rearrange(out, 'n h d -> () n (h d)').to(x.dtype)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p)
            out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ffn  = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, positions=None, cu_seqlens=None, max_seqlen=None, attn_bias=None):
        x = x + self.attn(x, positions=positions, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, attn_bias=attn_bias)
        x = x + self.ffn(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., grad_checkpoint=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x, positions=None, cu_seqlens=None, max_seqlen=None, attn_bias=None):
        for block in self.layers:
            if self.grad_checkpoint and self.training:
                x = checkpoint(
                    block, x, positions, cu_seqlens, max_seqlen, attn_bias,
                    use_reentrant=False,
                    context_fn=partial(
                        create_selective_checkpoint_contexts,
                        self._sac_policy,
                    ),
                )
            else:
                x = block(x, positions=positions, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, attn_bias=attn_bias)
        return self.norm(x)

    @staticmethod
    def _sac_policy(ctx, op, *args, **kwargs):
        _SAVE = {
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.addmm.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten._flash_attention_forward.default,
        }
        return CheckpointPolicy.MUST_SAVE if op in _SAVE else CheckpointPolicy.PREFER_RECOMPUTE


# ---------------------------------------------------------------------------
# Position grid cache — avoid meshgrid() allocation on every forward pass
# ---------------------------------------------------------------------------
_pos_cache: dict[tuple[int, int, torch.device], tuple[Tensor, Tensor]] = {}


def _get_pos_grid(ph: int, pw: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Cached flat (h_coords, w_coords) tensors of shape (ph*pw,)."""
    key = (ph, pw, device)
    if key not in _pos_cache:
        h_coords = torch.arange(ph, device=device).repeat_interleave(pw)  # raster order
        w_coords = torch.arange(pw, device=device).repeat(ph)
        _pos_cache[key] = (h_coords, w_coords)
    return _pos_cache[key]


class NaViT_Encoder(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.,
        emb_dropout: float = 0.,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.fge         = FineGrainedEmbedding(channels, dim)
        self.dropout     = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, grad_checkpoint=grad_checkpoint)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batched_images: List[List[Tensor]]):
        """
        Key changes vs original:
          1. Batched FGE — group same-size images, one kernel launch per size group
             instead of one kernel launch per image.
          2. Remove image.to(device) inside forward — move_batch() already placed
             all tensors on the correct device before this call.
          3. Cached position grids — _get_pos_grid() returns pre-built tensors,
             no meshgrid/stack/rearrange overhead per forward pass.
          4. seg_lens built entirely on CPU as a plain list → one torch.tensor()
             call at the end, avoiding repeated small GPU allocations.
        """
        device   = self.device
        pad_sequence = partial(orig_pad_sequence, batch_first=True)

        all_feats:       list[Tensor] = []   # (ph*pw, dim) per image
        all_h:           list[Tensor] = []   # (ph*pw,)     per image
        all_w:           list[Tensor] = []   # (ph*pw,)     per image
        seg_lens_cpu:    list[int]    = []   # flat, one int per image
        per_sample_lens: list[int]    = []   # total visual tokens per sample

        for images in batched_images:
            sample_tokens = 0

            # Group images within this sample by (H, W) so FGE can run batched.
            # In the common case (1 image per sample) this is a single group of 1.
            size_groups: dict[tuple[int, int], list[Tensor]] = {}
            for img in images:
                # img.shape == (C, H, W) — already on `device`
                sz = (img.shape[-2], img.shape[-1])
                size_groups.setdefault(sz, []).append(img)

            for (H, W), imgs in size_groups.items():
                stacked = torch.stack(imgs, dim=0)              # (G, C, H, W)
                feats, (ph, pw) = self.fge(stacked)             # (G, ph*pw, dim)
                h_coords, w_coords = _get_pos_grid(ph, pw, device)

                for feat in feats.unbind(0):                    # iterate G without copy
                    all_feats.append(feat)
                    all_h.append(h_coords)
                    all_w.append(w_coords)
                    seg_lens_cpu.append(ph * pw)
                    sample_tokens += ph * pw

            per_sample_lens.append(sample_tokens)

        flat_tokens = torch.cat(all_feats, dim=0)              # (total_tokens, dim)
        flat_h      = torch.cat(all_h,     dim=0)              # (total_tokens,)
        flat_w      = torch.cat(all_w,     dim=0)              # (total_tokens,)

        # Single device allocation for cu_seqlens
        seg_lens_t = torch.tensor(seg_lens_cpu, dtype=torch.int32, device=device)
        cu_seqlens = torch.zeros(len(seg_lens_cpu) + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = seg_lens_t.cumsum(0)
        max_seqlen     = int(seg_lens_t.max().item())

        h_idx = flat_h.unsqueeze(0)   # (1, total_tokens)
        w_idx = flat_w.unsqueeze(0)   # (1, total_tokens)

        x = self.dropout(flat_tokens.unsqueeze(0))
        x = self.transformer(
            x,
            positions=(h_idx, w_idx),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attn_bias=None,            # FA2 always used
        )
        x = x.squeeze(0)              # (total_tokens, dim)

        # Re-split by sample and pad to rectangular (B, max_len, dim)
        x_split  = x.split(per_sample_lens, dim=0)
        x        = pad_sequence(list(x_split))
        max_len  = x.shape[1]
        lengths  = torch.tensor(per_sample_lens, device=device)
        pad_mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]

        return x, pad_mask