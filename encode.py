import torch
import torch.nn as nn

from constants import (
    PATCH_SIZE, IMAGE_HEIGHT, MAX_IMAGE_WIDTH,
    EMBED_DIM, BRIDGE_OUT_DIM, NUM_HEADS, NUM_LAYERS, MLP_RATIO, DROPOUT,
)


def patchify_single(img: torch.Tensor, patch_size: int):
    C, H, W = img.shape
    patch_height = H // patch_size
    patch_width  = W // patch_size
    x = img.reshape(C, patch_height, patch_size, patch_width, patch_size)
    x = x.permute(1, 3, 0, 2, 4).reshape(patch_height * patch_width, C * patch_size * patch_size)
    return x, patch_height, patch_width


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
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)[0]
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
        self.patch_size  = patch_size
        self.embed_dim   = embed_dim
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

    def _positional_embeddings(self, patch_height, patch_width, device):
        row_pos = self.row_embed(torch.arange(patch_height, device=device))
        col_pos = self.col_embed(torch.arange(patch_width,  device=device))
        pos_grid = torch.cat([
            row_pos.unsqueeze(1).expand(patch_height, patch_width, -1),
            col_pos.unsqueeze(0).expand(patch_height, patch_width, -1),
        ], dim=-1)
        return pos_grid.reshape(patch_height * patch_width, self.embed_dim)

    def forward(self, packed_sequences: torch.Tensor, sequence_attention_mask: torch.Tensor):
        x = self.dropout(packed_sequences)
        for block in self.blocks:
            x = block(x, attn_mask=sequence_attention_mask)
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
        packed, sequence_mask, sequence_lengths = _pack_images_into_sequence(
            pixel_values, patch_mask, self.encoder
        )
        encoded = self.encoder(packed.unsqueeze(0), sequence_mask).squeeze(0)
        visual_tokens, attention_mask = _unpack_sequence_to_padded_batch(
            encoded, sequence_lengths, self.encoder.embed_dim, pixel_values.device
        )
        return self.bridge(visual_tokens), attention_mask


def _pack_images_into_sequence(pixel_values: torch.Tensor, patch_mask, encoder: NaViTEncoder):
    """Pack a batch of variable-width images into a single flattened sequence for NaViT attention."""
    device = pixel_values.device
    image_embeddings = []
    sequence_lengths = []

    for i in range(pixel_values.shape[0]):
        patches, patch_height, patch_width = patchify_single(pixel_values[i], encoder.patch_size)
        patch_embeds = encoder.patch_embed(patches) + encoder._positional_embeddings(
            patch_height, patch_width, device
        )
        if patch_mask is not None:
            patch_embeds = patch_embeds[patch_mask[i].bool()]
        sequence = torch.cat([encoder.cls_token.to(patch_embeds.dtype), patch_embeds], dim=0)
        image_embeddings.append(sequence)
        sequence_lengths.append(sequence.shape[0])

    packed = torch.cat(image_embeddings, dim=0)
    total_len = packed.shape[0]

    # Block-diagonal attention mask: each image attends only to its own patches
    attention_mask = torch.full((total_len, total_len), float("-inf"),
                                device=device, dtype=packed.dtype)
    offset = 0
    for seq_len in sequence_lengths:
        attention_mask[offset:offset + seq_len, offset:offset + seq_len] = 0.0
        offset += seq_len

    return packed, attention_mask, sequence_lengths


def _unpack_sequence_to_padded_batch(
    encoded: torch.Tensor,
    sequence_lengths: list,
    embed_dim: int,
    device: torch.device,
):
    """Unpack a flat encoded sequence back into a padded batch tensor with an attention mask."""
    sequences = []
    offset = 0
    for seq_len in sequence_lengths:
        sequences.append(encoded[offset:offset + seq_len])
        offset += seq_len

    max_len  = max(s.shape[0] for s in sequences)
    B        = len(sequences)
    padded   = torch.zeros(B, max_len, embed_dim, device=device, dtype=encoded.dtype)
    attn_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        padded[i, :seq.shape[0]]    = seq
        attn_mask[i, :seq.shape[0]] = 1
    return padded, attn_mask
