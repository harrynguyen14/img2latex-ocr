import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image


PATCH_SIZE = 16
IMAGE_HEIGHT = 64
MAX_IMAGE_WIDTH = 672
EMBED_DIM = 768
BRIDGE_OUT_DIM = 2048
NUM_HEADS = 12
NUM_LAYERS = 12
MLP_RATIO = 4
DROPOUT = 0.1


def patchify_single(img_tensor: torch.Tensor, patch_size: int):
    C, H, W = img_tensor.shape
    assert H % patch_size == 0 and W % patch_size == 0
    ph = H // patch_size
    pw = W // patch_size
    x = img_tensor.reshape(C, ph, patch_size, pw, patch_size)
    x = x.permute(1, 3, 0, 2, 4)
    x = x.reshape(ph * pw, C * patch_size * patch_size)
    return x, ph, pw


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=PATCH_SIZE, in_channels=3, embed_dim=EMBED_DIM):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, dropout=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        n = self.norm1(x)
        attn_out, _ = self.attn(n, n, n, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class NaViTEncoder(nn.Module):
    def __init__(
        self,
        patch_size=PATCH_SIZE,
        image_height=IMAGE_HEIGHT,
        max_image_width=MAX_IMAGE_WIDTH,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_embed = PatchEmbedding(patch_size, 3, embed_dim)
        self.row_embed = nn.Embedding(image_height // patch_size + 1, embed_dim // 2)
        self.col_embed = nn.Embedding(max_image_width // patch_size + 1, embed_dim // 2)
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.row_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.col_embed.weight, std=0.02)

    def _build_2d_pos_embed(self, ph, pw, device):
        rows = torch.arange(ph, device=device)
        cols = torch.arange(pw, device=device)
        row_emb = self.row_embed(rows)
        col_emb = self.col_embed(cols)
        row_emb = row_emb.unsqueeze(1).expand(ph, pw, -1)
        col_emb = col_emb.unsqueeze(0).expand(ph, pw, -1)
        pos = torch.cat([row_emb, col_emb], dim=-1)
        return pos.reshape(ph * pw, self.embed_dim)

    def forward(self, packed_tokens: torch.Tensor, attn_mask: torch.Tensor):
        x = self.dropout(packed_tokens)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return self.norm(x)


class BridgeMLP(nn.Module):
    def __init__(self, in_dim=EMBED_DIM, out_dim=BRIDGE_OUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.GELU(),
            nn.Linear(in_dim * 2, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.net(x))


class VisualEncoder(nn.Module):
    def __init__(self, encoder: NaViTEncoder = None, bridge: BridgeMLP = None):
        super().__init__()
        self.encoder = encoder or NaViTEncoder()
        self.bridge = bridge or BridgeMLP()

    def forward(self, pixel_values: torch.Tensor, patch_mask: torch.Tensor = None):
        packed, attn_mask, patch_lengths = _pack_images(
            pixel_values, patch_mask, self.encoder
        )
        encoded = self.encoder(packed.unsqueeze(0), attn_mask)
        encoded = encoded.squeeze(0)
        features, vis_mask = _unpack_to_batch(encoded, patch_lengths, self.encoder.embed_dim, pixel_values.device)
        return self.bridge(features), vis_mask


def _pack_images(pixel_values: torch.Tensor, patch_mask: torch.Tensor, encoder: NaViTEncoder):
    B, C, H, W = pixel_values.shape
    patch_size = encoder.patch_size
    device = pixel_values.device

    token_seqs = []
    patch_lengths = []

    for i in range(B):
        img = pixel_values[i]
        patches, ph, pw = patchify_single(img, patch_size)
        patch_emb = encoder.patch_embed(patches)
        pos = encoder._build_2d_pos_embed(ph, pw, device)
        patch_emb = patch_emb + pos

        if patch_mask is not None:
            valid = patch_mask[i].bool()
            patch_emb = patch_emb[valid]

        cls = encoder.cls_token.to(patch_emb.dtype)
        seq = torch.cat([cls, patch_emb], dim=0)
        token_seqs.append(seq)
        patch_lengths.append(seq.shape[0])

    packed = torch.cat(token_seqs, dim=0)

    total = packed.shape[0]
    attn_mask = torch.full((total, total), float("-inf"), device=device, dtype=packed.dtype)
    offset = 0
    for length in patch_lengths:
        attn_mask[offset:offset + length, offset:offset + length] = 0.0
        offset += length

    return packed, attn_mask, patch_lengths


def _unpack_to_batch(encoded: torch.Tensor, patch_lengths: list, embed_dim: int, device: torch.device):
    seqs = []
    offset = 0
    for length in patch_lengths:
        seqs.append(encoded[offset:offset + length])
        offset += length

    max_len = max(s.shape[0] for s in seqs)
    B = len(seqs)
    padded = torch.zeros(B, max_len, embed_dim, device=device, dtype=encoded.dtype)
    vis_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0]] = s
        vis_mask[i, :s.shape[0]] = 1
    return padded, vis_mask


def image_to_tensor(img: Image.Image, device="cpu") -> torch.Tensor:
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return tensor.unsqueeze(0).to(device)


def collate_images(images: list, device="cpu"):
    tensors = [TF.to_tensor(img) for img in images]
    tensors = [TF.normalize(t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) for t in tensors]
    max_w = max(t.shape[-1] for t in tensors)
    padded = []
    masks = []
    for t in tensors:
        w = t.shape[-1]
        pad_w = max_w - w
        t_pad = torch.nn.functional.pad(t, (0, pad_w), value=1.0)
        padded.append(t_pad)
        ph = IMAGE_HEIGHT // PATCH_SIZE
        pw_valid = w // PATCH_SIZE
        pw_total = max_w // PATCH_SIZE
        mask = torch.zeros(ph, pw_total, dtype=torch.bool)
        mask[:, :pw_valid] = True
        masks.append(mask.reshape(-1))
    return torch.stack(padded).to(device), torch.stack(masks).to(device)
