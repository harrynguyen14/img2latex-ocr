import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
from torch import nn
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

try:
    from .configuration_latex_decoder import LaTeXDecoderConfig
    from .configuration_latex_ocr import Nav2TexConfig
    from .modeling_latex_decoder import LaTeXDecoderForCausalLM
except ImportError:
    from nav2tex.configuration_latex_decoder import LaTeXDecoderConfig
    from nav2tex.configuration_latex_ocr import Nav2TexConfig
    from nav2tex.modeling_latex_decoder import LaTeXDecoderForCausalLM

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def exists(val):
    return val is not None


def divisible_by(numer, denom):
    return (numer % denom) == 0


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(
            x.float(), self.normalized_shape,
            self.weight.float(), self.bias.float(), self.eps,
        ).to(x.dtype)


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
    _, _, _, d = q.shape
    if d % 4 != 0:
        raise ValueError(f"apply_2d_rope expects dim_head divisible by 4, got D={d}")
    dim_half = d // 2
    dim_quarter = d // 4
    inv_freq = 1.0 / (10000 ** (torch.arange(dim_quarter, device=q.device).float() / dim_quarter))
    h_theta = h_idx[..., None].float() * inv_freq
    w_theta = w_idx[..., None].float() * inv_freq
    sin_h = torch.cat([h_theta.sin(), h_theta.sin()], dim=-1).to(q.dtype)[:, None, :, :]
    cos_h = torch.cat([h_theta.cos(), h_theta.cos()], dim=-1).to(q.dtype)[:, None, :, :]
    sin_w = torch.cat([w_theta.sin(), w_theta.sin()], dim=-1).to(q.dtype)[:, None, :, :]
    cos_w = torch.cat([w_theta.cos(), w_theta.cos()], dim=-1).to(q.dtype)[:, None, :, :]

    def rope(x, sin, cos):
        return x * cos + rotate_half(x) * sin

    q = torch.cat([rope(q[..., :dim_half], sin_h, cos_h), rope(q[..., dim_half:], sin_w, cos_w)], dim=-1)
    k = torch.cat([rope(k[..., :dim_half], sin_h, cos_h), rope(k[..., dim_half:], sin_w, cos_w)], dim=-1)
    return q, k


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
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
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), nn.Dropout(dropout))

    def forward(self, x, mask=None, attn_mask=None, positions=None):
        x = self.norm(x)
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        q = self.q_norm(q)
        k = self.k_norm(k)

        if positions is not None:
            q, k = apply_2d_rope(q, k, positions[0], positions[1])

        if HAS_FLASH_ATTN and x.is_cuda and attn_mask is None:
            fa_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
            q_ = rearrange(q, "b h n d -> b n h d").contiguous().to(fa_dtype)
            k_ = rearrange(k, "b h n d -> b n h d").contiguous().to(fa_dtype)
            v_ = rearrange(v, "b h n d -> b n h d").contiguous().to(fa_dtype)
            if exists(mask):
                batch, seqlen = mask.shape
                q_unpad, indices, cu_q, max_q, *_ = unpad_input(q_, mask)
                k_unpad, _, cu_k, max_k, *_ = unpad_input(k_, mask)
                v_unpad, _, _, _, *_ = unpad_input(v_, mask)
                out_unpad = flash_attn_varlen_func(
                    q_unpad, k_unpad, v_unpad,
                    cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    max_seqlen_q=max_q, max_seqlen_k=max_k,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    causal=False,
                )
                out = pad_input(out_unpad, indices, batch, seqlen)
            else:
                out = flash_attn_func(
                    q_, k_, v_,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    causal=False,
                )
            out = rearrange(out, "b n h d -> b n (h d)").to(x.dtype)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2))
            if exists(mask):
                dots = dots.masked_fill(~mask[:, None, None, :], -torch.finfo(dots.dtype).max)
            if exists(attn_mask):
                dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)
            attn = self.dropout(self.attend(dots))
            out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([Attention(dim, heads, dim_head, dropout), FeedForward(dim, mlp_dim, dropout)])
            for _ in range(depth)
        ])
        self.norm = LayerNorm(dim)

    def forward(self, x, mask=None, attn_mask=None, positions=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask, attn_mask=attn_mask, positions=positions) + x
            x = ff(x) + x
        return self.norm(x)


class NaViT_Encoder(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim,
                 channels=3, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        image_height, image_width = image_size
        assert divisible_by(image_height, patch_size)
        assert divisible_by(image_width, patch_size)
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Sequential(
            LayerNorm(channels * patch_size ** 2),
            nn.Linear(channels * patch_size ** 2, dim),
            LayerNorm(dim),
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batched_images):
        p = self.patch_size
        device = self.device
        arange = partial(torch.arange, device=device)
        pad_sequence = partial(orig_pad_sequence, batch_first=True)
        batched_sequences, batched_positions = [], []

        for images in batched_images:
            sequences, positions = [], []
            for image in images:
                _, h, w = image.shape
                ph, pw = h // p, w // p
                seq = rearrange(image, "c (h p1) (w p2) -> (h w) (c p1 p2)", p1=p, p2=p)
                pos = torch.stack(torch.meshgrid(arange(ph), arange(pw), indexing="ij"), dim=-1)
                sequences.append(seq)
                positions.append(rearrange(pos, "h w c -> (h w) c"))
            batched_sequences.append(torch.cat(sequences, dim=0))
            batched_positions.append(torch.cat(positions, dim=0))

        patches = pad_sequence(batched_sequences)
        patch_positions = pad_sequence(batched_positions)
        lengths = torch.tensor([seq.shape[0] for seq in batched_sequences], device=device)
        mask = torch.arange(patches.shape[1], device=device)[None, :] < lengths[:, None]
        x = self.to_patch_embedding(patches.to(next(self.parameters()).dtype))
        h_idx, w_idx = patch_positions.unbind(dim=-1)
        x = self.dropout(x)
        x = self.transformer(x, mask=mask, positions=(h_idx, w_idx))
        return x, mask


class MLPProjector(nn.Module):
    def __init__(self, vision_hidden_size=1024, llm_hidden_size=512, intermediate_size=2048):
        super().__init__()
        self.norm = nn.LayerNorm(vision_hidden_size)
        self.gate_proj = nn.Linear(vision_hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(vision_hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, llm_hidden_size, bias=False)

    def forward(self, x):
        x = self.norm(x)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class VisualEncoder(nn.Module):
    def __init__(self, encoder, bridge, max_visual_tokens):
        super().__init__()
        self.navit = encoder
        self.projector = bridge
        self.max_visual_tokens = max_visual_tokens

    def forward(self, batched_images):
        x, mask = self.navit(batched_images)
        if x.shape[1] > self.max_visual_tokens:
            x = x[:, :self.max_visual_tokens]
            mask = mask[:, :self.max_visual_tokens]
        return self.projector(x), mask


class CustomDecoder(nn.Module):
    def __init__(self, config: Nav2TexConfig):
        super().__init__()
        dec = config.decoder_arch
        self._model = LaTeXDecoderForCausalLM(
            LaTeXDecoderConfig(
                vocab_size=dec["vocab_size"],
                pad_id=dec["pad_id"],
                bos_id=dec["bos_id"],
                eos_id=dec["eos_id"],
                d_model=dec["d_model"],
                n_heads=dec["n_heads"],
                n_layers=dec["n_layers"],
                d_ff=dec["d_ff"],
                dropout=dec.get("dropout", 0.1),
                max_seq_len=dec["max_seq_len"],
                rope_theta=dec.get("rope_theta", 10000.0),
                tie_weights=dec.get("tie_weights", True),
            )
        )
        self.pad_token_id = self._model.config.pad_id
        self.eos_token_id = self._model.config.eos_id
        self._vocab_size = self._model.config.vocab_size
        self._pad_id = self._model.config.pad_id
        if not config.decoder_weights_tied:
            self.untie_weights()

    def get_input_embeddings(self):
        return self._model.embed_tokens

    def tie_weights(self):
        self._model.lm_head.weight = self._model.embed_tokens.weight

    def untie_weights(self):
        if self.are_weights_tied():
            self._model.lm_head.weight = nn.Parameter(self._model.embed_tokens.weight.detach().clone())

    def are_weights_tied(self):
        return self._model.lm_head.weight.data_ptr() == self._model.embed_tokens.weight.data_ptr()

    def _forward_embeds(self, inputs_embeds, attention_mask=None):
        x = self._model.embed_drop(inputs_embeds)
        mask = attention_mask.bool() if attention_mask is not None else None
        for layer in self._model.layers:
            x = layer(x, mask)
        return self._model.lm_head(self._model.norm_final(x))

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        logits = self._forward_embeds(inputs_embeds, attention_mask)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous().masked_fill(
                labels[:, 1:].contiguous() == self._pad_id, -100
            )
            loss = F.cross_entropy(
                shift_logits.view(-1, self._vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return BaseModelOutput(last_hidden_state=logits, hidden_states=(loss,))

    @torch.no_grad()
    def generate(self, inputs_embeds, attention_mask, max_new_tokens, num_beams=1):
        device = inputs_embeds.device
        batch  = inputs_embeds.shape[0]

        if num_beams > 1:
            # beam search: only supports batch_size=1
            assert batch == 1, "beam search only supports batch_size=1"
            return self._beam_search(inputs_embeds, attention_mask, max_new_tokens, num_beams)

        return self._greedy_batch(inputs_embeds, attention_mask, max_new_tokens)

    @torch.no_grad()
    def _greedy_batch(self, inputs_embeds, attention_mask, max_new_tokens):
        """Greedy decoding with true batch support."""
        eos_id  = self.eos_token_id
        pad_id  = self._pad_id
        device  = inputs_embeds.device
        batch   = inputs_embeds.shape[0]
        d_model = inputs_embeds.shape[-1]

        # generated token ids per sample, and finished flags
        gen_ids  = [[] for _ in range(batch)]
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        cur_embeds = inputs_embeds                          # (B, vis_len, D)
        cur_mask   = attention_mask                         # (B, vis_len)

        for _ in range(max_new_tokens):
            logits   = self._forward_embeds(cur_embeds, cur_mask)  # (B, seq, vocab)
            next_tok = logits[:, -1, :].argmax(dim=-1)             # (B,)

            for i in range(batch):
                if not finished[i]:
                    gen_ids[i].append(next_tok[i].item())
            finished |= (next_tok == eos_id)
            if finished.all():
                break

            tok_emb  = self._model.embed_tokens(next_tok.unsqueeze(1))   # (B, 1, D)
            tok_mask = cur_mask.new_ones(batch, 1)
            cur_embeds = torch.cat([cur_embeds, tok_emb], dim=1)
            cur_mask   = torch.cat([cur_mask,   tok_mask], dim=1)

        # pad to same length and return (B, max_len)
        max_len = max((len(ids) for ids in gen_ids), default=0)
        if max_len == 0:
            return torch.zeros(batch, 0, dtype=torch.long, device=device)
        out = torch.full((batch, max_len), pad_id, dtype=torch.long, device=device)
        for i, ids in enumerate(gen_ids):
            if ids:
                out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        return out

    @torch.no_grad()
    def _beam_search(self, inputs_embeds, attention_mask, max_new_tokens, num_beams):
        """Original beam search (batch_size=1 only)."""
        eos_id   = self.eos_token_id
        device   = inputs_embeds.device
        vis_emb  = inputs_embeds[0]
        vis_len  = vis_emb.shape[0]
        vis_mask = attention_mask[0] if attention_mask is not None else None
        beams    = [(0.0, [], False) for _ in range(num_beams)]

        for _ in range(max_new_tokens):
            all_embeds, all_masks = [], []
            for score, ids, _ in beams:
                tok_emb = self._model.embed_tokens(torch.tensor(ids, device=device, dtype=torch.long)) if ids else None
                seq_emb = torch.cat([vis_emb, tok_emb], dim=0) if tok_emb is not None else vis_emb
                all_embeds.append(seq_emb)
                if vis_mask is not None:
                    tok_mask = vis_mask.new_ones(len(ids)) if ids else vis_mask.new_zeros(0)
                    all_masks.append(torch.cat([vis_mask, tok_mask]) if ids else vis_mask)

            max_len = max(e.shape[0] for e in all_embeds)
            d_model = all_embeds[0].shape[-1]
            padded_embeds = vis_emb.new_zeros(num_beams, max_len, d_model)
            padded_mask   = vis_mask.new_zeros(num_beams, max_len) if vis_mask is not None else None
            for idx, emb in enumerate(all_embeds):
                padded_embeds[idx, :emb.shape[0]] = emb
                if padded_mask is not None:
                    padded_mask[idx, :emb.shape[0]] = all_masks[idx]

            logits = self._forward_embeds(padded_embeds, padded_mask)
            candidates = []
            for beam_idx, (score, ids, done) in enumerate(beams):
                if done:
                    candidates.append((score, ids, True))
                    continue
                last_pos = vis_len + len(ids) - 1
                log_p = torch.log_softmax(logits[beam_idx, last_pos, :], dim=-1)
                if len(ids) == 0 and beam_idx > 0:
                    log_p = log_p.fill_(-1e9)
                for lp, tok in zip(*map(lambda t: t.tolist(), log_p.topk(num_beams))):
                    candidates.append((score + lp, ids + [tok], tok == eos_id))
            candidates.sort(key=lambda x: -x[0])
            beams = candidates[:num_beams]
            if all(done for _, _, done in beams):
                break

        best_ids = max(beams, key=lambda x: x[0])[1]
        if not best_ids:
            return torch.zeros(1, 0, dtype=torch.long, device=device)
        return torch.tensor(best_ids, dtype=torch.long, device=device).unsqueeze(0)


class Nav2TexModel(PreTrainedModel):
    config_class = Nav2TexConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def __init__(self, config: Nav2TexConfig):
        super().__init__(config)
        self.config = config
        self.visual_encoder = VisualEncoder(
            NaViT_Encoder(
                image_size=(config.image_height, config.max_image_width),
                patch_size=config.patch_size,
                dim=config.navit_dim,
                depth=config.navit_depth,
                heads=config.navit_heads,
                mlp_dim=config.navit_mlp_dim,
                dim_head=config.navit_dim_head,
                dropout=config.navit_dropout,
                emb_dropout=config.navit_emb_dropout,
            ),
            MLPProjector(
                vision_hidden_size=config.vision_hidden_size,
                llm_hidden_size=config.llm_hidden_size,
                intermediate_size=config.projector_intermediate_size,
            ),
            max_visual_tokens=config.max_visual_tokens,
        )
        self.decoder = CustomDecoder(config)
        self.post_init()

    def tie_weights(self):
        if self.config.decoder_weights_tied:
            self.decoder.tie_weights()
        else:
            self.decoder.untie_weights()

    def _init_weights(self, module):
        return

    @staticmethod
    def _to_batched_images(pixel_values):
        if isinstance(pixel_values, list):
            return pixel_values
        if isinstance(pixel_values, torch.Tensor):
            return [[img] for img in pixel_values]
        raise TypeError(f"Unsupported pixel_values type: {type(pixel_values)}")

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None, **kwargs):
        batched_images = self._to_batched_images(pixel_values)
        ve, vm = self.visual_encoder(batched_images)
        if input_ids is None:
            return BaseModelOutput(last_hidden_state=ve)
        te = self.decoder.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([ve, te], dim=1)
        am = torch.cat([vm.to(dtype=attention_mask.dtype), attention_mask], dim=1)
        lv = torch.full((labels.shape[0], ve.shape[1]), -100, dtype=labels.dtype, device=labels.device)
        out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=am,
            labels=torch.cat([lv, labels], dim=1),
        )
        return BaseModelOutput(last_hidden_state=out.last_hidden_state, hidden_states=(out.hidden_states[0],))

    @torch.no_grad()
    def generate(self, pixel_values, max_new_tokens=None, num_beams=None):
        batched_images = self._to_batched_images(pixel_values)
        ve, vm = self.visual_encoder(batched_images)
        batch = ve.shape[0]
        bos_id = self.config.decoder_arch["bos_id"]
        bos_emb = self.decoder.get_input_embeddings()(
            torch.full((batch, 1), bos_id, dtype=torch.long, device=ve.device)
        )
        inputs_embeds = torch.cat([ve, bos_emb], dim=1)
        attention_mask = torch.cat([
            vm.to(dtype=torch.long),
            torch.ones(batch, 1, dtype=torch.long, device=ve.device)
        ], dim=1)
        return self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            num_beams=num_beams or self.config.num_beams,
        )