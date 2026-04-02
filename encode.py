import torch
import torch.nn as nn

from constants import (
    PATCH_SIZE, IMAGE_HEIGHT, MAX_IMAGE_WIDTH,
    EMBED_DIM, BRIDGE_OUT_DIM, NUM_HEADS, NUM_LAYERS, MLP_RATIO, DROPOUT,
)


def patchify_single(img: torch.Tensor, patch_size: int):
    C, H, W = img.shape
    ph, pw = H // patch_size, W // patch_size
    x = img.reshape(C, ph, patch_size, pw, patch_size)
    x = x.permute(1, 3, 0, 2, 4).reshape(ph * pw, C * patch_size * patch_size)
    return x, ph, pw


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=PATCH_SIZE, in_channels=3, embed_dim=EMBED_DIM):
        super().__init__()
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, dropout=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        n = self.norm1(x)
        x = x + self.attn(n, n, n, attn_mask=attn_mask, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class NaViTEncoder(nn.Module):
    def __init__(
        self,
        patch_size=PATCH_SIZE, image_height=IMAGE_HEIGHT,
        max_image_width=MAX_IMAGE_WIDTH, embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        mlp_ratio=MLP_RATIO, dropout=DROPOUT,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim  = embed_dim
        self.patch_embed = PatchEmbedding(patch_size, 3, embed_dim)
        self.row_embed   = nn.Embedding(image_height // patch_size + 1, embed_dim // 2)
        self.col_embed   = nn.Embedding(max_image_width // patch_size + 1, embed_dim // 2)
        self.cls_token   = nn.Parameter(torch.zeros(1, embed_dim))
        self.dropout     = nn.Dropout(dropout)
        self.blocks      = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token,        std=0.02)
        nn.init.trunc_normal_(self.row_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.col_embed.weight, std=0.02)

    def _pos(self, ph, pw, device):
        rows = self.row_embed(torch.arange(ph, device=device))
        cols = self.col_embed(torch.arange(pw, device=device))
        pos  = torch.cat([
            rows.unsqueeze(1).expand(ph, pw, -1),
            cols.unsqueeze(0).expand(ph, pw, -1),
        ], dim=-1)
        return pos.reshape(ph * pw, self.embed_dim)

    def forward(self, packed: torch.Tensor, attn_mask: torch.Tensor):
        x = self.dropout(packed)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return self.norm(x)


class BridgeMLP(nn.Module):
    def __init__(self, in_dim=EMBED_DIM, out_dim=BRIDGE_OUT_DIM):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2), nn.GELU(), nn.Linear(in_dim * 2, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.net(x))


class VisualEncoder(nn.Module):
    def __init__(self, encoder: NaViTEncoder = None, bridge: BridgeMLP = None):
        super().__init__()
        self.encoder = encoder or NaViTEncoder()
        self.bridge  = bridge  or BridgeMLP()

    def forward(self, pixel_values: torch.Tensor, patch_mask: torch.Tensor = None):
        packed, attn_mask, lengths = _pack(pixel_values, patch_mask, self.encoder)
        encoded = self.encoder(packed.unsqueeze(0), attn_mask).squeeze(0)
        features, vis_mask = _unpack(encoded, lengths, self.encoder.embed_dim, pixel_values.device)
        return self.bridge(features), vis_mask


def _pack(pixel_values: torch.Tensor, patch_mask, encoder: NaViTEncoder):
    device = pixel_values.device
    seqs, lengths = [], []
    for i in range(pixel_values.shape[0]):
        patches, ph, pw = patchify_single(pixel_values[i], encoder.patch_size)
        emb = encoder.patch_embed(patches) + encoder._pos(ph, pw, device)
        if patch_mask is not None:
            emb = emb[patch_mask[i].bool()]
        seq = torch.cat([encoder.cls_token.to(emb.dtype), emb], dim=0)
        seqs.append(seq)
        lengths.append(seq.shape[0])

    packed = torch.cat(seqs, dim=0)
    total  = packed.shape[0]
    mask   = torch.full((total, total), float("-inf"), device=device, dtype=packed.dtype)
    off    = 0
    for l in lengths:
        mask[off:off + l, off:off + l] = 0.0
        off += l
    return packed, mask, lengths


def _unpack(encoded: torch.Tensor, lengths: list, embed_dim: int, device: torch.device):
    seqs   = []
    off    = 0
    for l in lengths:
        seqs.append(encoded[off:off + l])
        off += l
    max_len = max(s.shape[0] for s in seqs)
    B       = len(seqs)
    padded  = torch.zeros(B, max_len, embed_dim, device=device, dtype=encoded.dtype)
    vis_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0]]   = s
        vis_mask[i, :s.shape[0]] = 1
    return padded, vis_mask
