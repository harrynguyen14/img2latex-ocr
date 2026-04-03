import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
from typing import List
from functools import partial
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence

def exists(val):
    return val is not None

def divisible_by(numer, denom):
    return (numer % denom) == 0

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_2d_rope(q, k, h_idx, w_idx):
    B, H, N, D = q.shape
    device = q.device

    # With the current `rotate_half()` implementation we rotate over pairs,
    # so RoPE requires D to be divisible by 4 for the 2D split below.
    if D % 4 != 0:
        raise ValueError(f"apply_2d_rope expects dim_head divisible by 4, got D={D}")

    dim_half = D // 2
    dim_quarter = D // 4

    q_h, q_w = q[..., :dim_half], q[..., dim_half:]
    k_h, k_w = k[..., :dim_half], k[..., dim_half:]

    freq_seq = torch.arange(dim_quarter, device=device).float()
    inv_freq = 1.0 / (10000 ** (freq_seq / dim_quarter))

    h_theta = h_idx[..., None] * inv_freq
    w_theta = w_idx[..., None] * inv_freq

    sin_h, cos_h = h_theta.sin(), h_theta.cos()
    sin_w, cos_w = w_theta.sin(), w_theta.cos()

    # Expand (dim_quarter) -> (dim_half) to match x's last-dim.
    # `rope()` multiplies elementwise with `x`, so sin/cos must have the same
    # trailing dimension as `q_h` / `q_w` (= dim_half).
    sin_h = torch.cat([sin_h, sin_h], dim=-1)
    cos_h = torch.cat([cos_h, cos_h], dim=-1)
    sin_w = torch.cat([sin_w, sin_w], dim=-1)
    cos_w = torch.cat([cos_w, cos_w], dim=-1)

    sin_h = sin_h[:, None, :, :]
    cos_h = cos_h[:, None, :, :]
    sin_w = sin_w[:, None, :, :]
    cos_w = cos_w[:, None, :, :]

    def rope(x, sin, cos):
        return (x * cos) + (rotate_half(x) * sin)

    q_h = rope(q_h, sin_h, cos_h)
    k_h = rope(k_h, sin_h, cos_h)
    q_w = rope(q_w, sin_w, cos_w)
    k_w = rope(k_w, sin_w, cos_w)

    q = torch.cat([q_h, q_w], dim=-1)
    k = torch.cat([k_h, k_w], dim=-1)

    return q, k

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads

        self.norm = LayerNorm(dim)
        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None, attn_mask=None, positions=None):
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

        dots = torch.matmul(q, k.transpose(-1, -2))

        if exists(mask):
            mask_ = mask[:, None, None, :]
            dots = dots.masked_fill(~mask_, -torch.finfo(dots.dtype).max)

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]) for _ in range(depth)
        ])
        self.norm = LayerNorm(dim)

    def forward(self, x, mask=None, attn_mask=None, positions=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask, attn_mask=attn_mask, positions=positions) + x
            x = ff(x) + x
        return self.norm(x)

class NaViT_Encoder(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.
    ):
        super().__init__()

        image_height, image_width = image_size
        assert divisible_by(image_height, patch_size)
        assert divisible_by(image_width, patch_size)

        patch_dim = channels * (patch_size ** 2)
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim)
        )

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batched_images: List[List[Tensor]]):
        p, device = self.patch_size, self.device
        arange = partial(torch.arange, device=device)
        pad_sequence = partial(orig_pad_sequence, batch_first=True)

        batched_sequences = []
        batched_positions = []

        for images in batched_images:
            sequences = []
            positions = []

            for image in images:
                c, h, w = image.shape
                ph, pw = h // p, w // p

                seq = rearrange(
                    image, 'c (h p1) (w p2) -> (h w) (c p1 p2)', p1=p, p2=p
                )

                pos = torch.stack(torch.meshgrid(
                    arange(ph),
                    arange(pw),
                    indexing='ij'
                ), dim=-1)

                pos = rearrange(pos, 'h w c -> (h w) c')

                sequences.append(seq)
                positions.append(pos)

            batched_sequences.append(torch.cat(sequences, dim=0))
            batched_positions.append(torch.cat(positions, dim=0))

        patches = pad_sequence(batched_sequences)
        patch_positions = pad_sequence(batched_positions)

        lengths = torch.tensor([seq.shape[0] for seq in batched_sequences], device=device)
        max_len = patches.shape[1]
        mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]

        x = self.to_patch_embedding(patches)

        h_idx, w_idx = patch_positions.unbind(dim=-1)

        x = self.dropout(x)

        x = self.transformer(
            x,
            mask=mask,
            positions=(h_idx, w_idx)
        )

        return x, mask