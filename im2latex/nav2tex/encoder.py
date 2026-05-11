import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from typing import List
from functools import partial
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence

HAS_FLASH_ATTN = False  # flash-attn-3 package present but broken; use SDPA path


def exists(val):
    return val is not None


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
    """2D RoPE: split D into [0:D/2] for height, [D/2:D] for width."""
    B, H, N, D = q.shape
    device = q.device
    if D % 4 != 0:
        raise ValueError(f"apply_2d_rope expects dim_head divisible by 4, got D={D}")
    dim_half    = D // 2
    dim_quarter = D // 4

    q_h, q_w = q[..., :dim_half], q[..., dim_half:]
    k_h, k_w = k[..., :dim_half], k[..., dim_half:]
    dtype = q.dtype

    # Standard RoPE inv_freq: use even indices up to dim_half, divide by dim_half
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, dim_half, 2, device=device).float() / dim_half)
    )  # (dim_quarter,)

    h_theta = h_idx[..., None].float() * inv_freq   # (B, N, dim_quarter)
    w_theta = w_idx[..., None].float() * inv_freq

    # Expand to (B, N, dim_half) by interleaving sin and cos halves
    sin_h = torch.cat([h_theta.sin(), h_theta.sin()], dim=-1).to(dtype)[:, None]  # (B,1,N,D/2)
    cos_h = torch.cat([h_theta.cos(), h_theta.cos()], dim=-1).to(dtype)[:, None]
    sin_w = torch.cat([w_theta.sin(), w_theta.sin()], dim=-1).to(dtype)[:, None]
    cos_w = torch.cat([w_theta.cos(), w_theta.cos()], dim=-1).to(dtype)[:, None]

    def rope(x, sin, cos):
        return (x * cos) + (rotate_half(x) * sin)

    q = torch.cat([rope(q_h, sin_h, cos_h), rope(q_w, sin_w, cos_w)], dim=-1)
    k = torch.cat([rope(k_h, sin_h, cos_h), rope(k_w, sin_w, cos_w)], dim=-1)
    return q, k


# ── Fine-Grained Embedding (FGE) ──────────────────────────────────────────────
class FineGrainedEmbedding(nn.Module):
    def __init__(self, channels: int, dim: int):
        super().__init__()
        mid = dim // 2
        self.conv1 = nn.Conv2d(channels, mid,  kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(1, mid)
        self.conv2 = nn.Conv2d(mid,      dim,  kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.conv2(x)                           # (B, dim, H/4, W/4)
        H, W = x.shape[-2], x.shape[-1]
        x = rearrange(x, 'b d h w -> b (h w) d')   # (B, N, dim)
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., use_flash_attn: bool = False):
        super().__init__()
        inner_dim    = dim_head * heads
        self.heads   = heads
        self.use_flash_attn = use_flash_attn
        self.norm    = nn.LayerNorm(dim)
        self.q_norm  = RMSNorm(heads, dim_head)
        self.k_norm  = RMSNorm(heads, dim_head)
        self.to_q    = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv   = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out  = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, attn_mask=None, positions=None, cu_seqlens=None, max_seqlen=None):
        x = self.norm(x)
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
            (q, k, v)
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        if positions is not None:
            h_idx, w_idx = positions
            q, k = apply_2d_rope(q, k, h_idx, w_idx)

        dropout_p = self.dropout.p if self.training else 0.0

        if exists(cu_seqlens) and exists(max_seqlen):
            # Varlen SDPA: split flat sequence at image boundaries, run SDPA per-image
            # with no mask so PyTorch can select its fastest kernel (FA3 on Blackwell).
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            outs = []
            offset = 0
            for slen in seqlens:
                slen = int(slen)
                qi = q[:, :, offset:offset + slen, :]
                ki = k[:, :, offset:offset + slen, :]
                vi = v[:, :, offset:offset + slen, :]
                oi = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=dropout_p)
                outs.append(oi)
                offset += slen
            out = torch.cat(outs, dim=2)
            out = rearrange(out, 'b h n d -> b n (h d)')
        else:
            combined_mask = None
            if exists(mask):
                combined_mask = mask[:, None, None, :]
            if exists(attn_mask):
                combined_mask = attn_mask if combined_mask is None else (combined_mask & attn_mask)
            if combined_mask is not None:
                combined_mask = torch.zeros_like(combined_mask, dtype=q.dtype).masked_fill(
                    ~combined_mask, float('-inf')
                )
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=combined_mask, dropout_p=dropout_p)
            out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., use_flash_attn: bool = False):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout, use_flash_attn=use_flash_attn)
        self.ffn  = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, mask=None, attn_mask=None, positions=None, cu_seqlens=None, max_seqlen=None):
        x = x + self.attn(x, mask=mask, attn_mask=attn_mask, positions=positions,
                          cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = x + self.ffn(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., use_flash_attn: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, mlp_dim, dropout, use_flash_attn=use_flash_attn)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None, attn_mask=None, positions=None, cu_seqlens=None, max_seqlen=None):
        for block in self.layers:
            x = block(x, mask=mask, attn_mask=attn_mask, positions=positions,
                      cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.norm(x)


def _build_block_diagonal_mask(segment_lengths: List[List[int]], max_len: int, device) -> Tensor:
    """Build a block-diagonal boolean attention mask for Patch n' Pack.

    Returns (B, 1, max_len, max_len) where True = allowed to attend.
    Patches from different images within the same batch item cannot attend to each other.
    """
    B = len(segment_lengths)
    mask = torch.zeros(B, 1, max_len, max_len, dtype=torch.bool, device=device)
    for b, lengths in enumerate(segment_lengths):
        offset = 0
        for length in lengths:
            end = offset + length
            mask[b, 0, offset:end, offset:end] = True
            offset = end
    return mask


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
        flash_attn: bool = False,
        image_size=None,
        patch_size=None,
    ):
        super().__init__()
        self.fge     = FineGrainedEmbedding(channels, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, use_flash_attn=flash_attn
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batched_images: List[List[Tensor]]):
        device = self.device
        arange = partial(torch.arange, device=device)
        pad_sequence = partial(orig_pad_sequence, batch_first=True)

        batched_sequences  = []
        batched_positions  = []
        batched_seg_lens   = []   # per-batch-item list of per-image patch counts

        for images in batched_images:
            sequences = []
            positions = []
            seg_lens  = []

            for image in images:
                image = image.to(device)
                C, H, W = image.shape

                feat, (ph, pw) = self.fge(image.unsqueeze(0))
                feat = feat.squeeze(0)          # (N, dim)

                pos = torch.stack(torch.meshgrid(
                    arange(ph), arange(pw), indexing='ij'
                ), dim=-1)
                pos = rearrange(pos, 'h w c -> (h w) c')

                sequences.append(feat)
                positions.append(pos)
                seg_lens.append(ph * pw)

            batched_sequences.append(torch.cat(sequences, dim=0))
            batched_positions.append(torch.cat(positions, dim=0))
            batched_seg_lens.append(seg_lens)

        all_pos       = []
        seg_lens_flat = []
        for seqs, seglens, pos in zip(batched_sequences, batched_seg_lens, batched_positions):
            all_pos.append(pos)
            seg_lens_flat.extend(seglens)

        flat_tokens    = torch.cat(batched_sequences, dim=0)
        flat_positions = torch.cat(all_pos, dim=0)

        seg_lens_t = torch.tensor(seg_lens_flat, dtype=torch.int32, device=device)
        cu_seqlens = torch.zeros(len(seg_lens_flat) + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = seg_lens_t.cumsum(0)
        max_seqlen = int(seg_lens_t.max().item())

        h_idx = flat_positions[:, 0].unsqueeze(0)  # (1, total_tokens)
        w_idx = flat_positions[:, 1].unsqueeze(0)

        x = self.dropout(flat_tokens.unsqueeze(0))  # (1, total_tokens, dim)
        x = self.transformer(x, mask=None, attn_mask=None,
                             positions=(h_idx, w_idx),
                             cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = x.squeeze(0)  # (total_tokens, dim)

        split_sizes = [s.shape[0] for s in batched_sequences]
        x_split     = x.split(split_sizes, dim=0)
        x           = pad_sequence(list(x_split))
        max_len     = x.shape[1]
        lengths     = torch.tensor(split_sizes, device=device)
        pad_mask    = torch.arange(max_len, device=device)[None, :] < lengths[:, None]

        return x, pad_mask
