import torch
import torch.nn as nn

from .decoder import CustomDecoder
from .mlp_projector import MLPProjector
from .encoder import NaViT_Encoder

def decode_ids(tokenizer, ids: list[int], skip_ids: set[int] | None = None) -> str:
    """Decode token ids bằng id_to_token join.
    
    tok.decode() của tokenizers library thêm spaces không mong muốn giữa
    các sub-word token (e.g. '\\frac' → '\\ f r a c'). id_to_token join
    là cách duy nhất cho kết quả đúng với BPE tokenizer này.
    """
    if skip_ids is None:
        skip_ids = set()
    return "".join(tokenizer.id_to_token(i) for i in ids if i not in skip_ids)

class VisualEncoder(nn.Module):
    def __init__(self, encoder: NaViT_Encoder, bridge: MLPProjector, max_visual_tokens: int):
        super().__init__()
        self.navit = encoder
        self.projector = bridge
        self.max_visual_tokens = max_visual_tokens

    def forward(self, batched_images):
        x, mask = self.navit(batched_images)
        if x.shape[1] > self.max_visual_tokens:
            x = x[:, : self.max_visual_tokens]
            mask = mask[:, : self.max_visual_tokens]
        x = self.projector(x)
        return x, mask


class LaTeXOCRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not isinstance(config, dict):
            config = vars(config)
        self.config = dict(config)
        ps = config["patch_size"]
        ih = config["image_height"]
        mw = config["max_image_width"]
        assert mw % ps == 0 and ih % ps == 0
        self.visual_encoder = VisualEncoder(
            NaViT_Encoder(
                image_size=(ih, mw),
                patch_size=ps,
                dim=config["navit_dim"],
                depth=config["navit_depth"],
                heads=config["navit_heads"],
                mlp_dim=config["navit_mlp_dim"],
                dim_head=config["navit_dim_head"],
                dropout=config["navit_dropout"],
                emb_dropout=config["navit_emb_dropout"],
            ),
            MLPProjector(
                vision_hidden_size=config["vision_hidden_size"],
                llm_hidden_size=config["llm_hidden_size"],
                intermediate_size=config["projector_intermediate_size"],
            ),
            max_visual_tokens=config["max_visual_tokens"],
        )
        self.decoder   = CustomDecoder(config)
        self.tokenizer = self.decoder.tokenizer
        if config["navit_dim"] != config["vision_hidden_size"]:
            raise ValueError("navit_dim must equal vision_hidden_size for the projector")

    def gradient_checkpointing_enable(self):
        pass

    def freeze_decoder(self):
        for p in self.decoder.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            if p.dtype.is_floating_point:
                p.requires_grad = True

    def set_train_stage(self, stage: int):
        self.unfreeze_all()

    def forward(
        self,
        batched_images,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        ve, vm = self.visual_encoder(batched_images)
        emb = self.decoder.get_input_embeddings()
        te = emb(input_ids)
        inputs_embeds = torch.cat([ve, te], dim=1)
        am = torch.cat([vm.to(dtype=attention_mask.dtype), attention_mask], dim=1)
        lv = torch.full(
            (labels.shape[0], ve.shape[1]),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        full_labels = torch.cat([lv, labels], dim=1)
        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=am,
            labels=full_labels,
        )

    @torch.no_grad()
    def generate(self, batched_images, max_new_tokens=None, num_beams=None):
        self.eval()
        cfg = self.config
        max_new_tokens = max_new_tokens or cfg.get("max_new_tokens", 256)
        
        ve, vm = self.visual_encoder(batched_images)
        B = ve.shape[0]

        bos_id = torch.full((B, 1), self.decoder.tokenizer.token_to_id("<bos>"),
                            dtype=torch.long, device=ve.device)
        bos_emb = self.decoder.get_input_embeddings()(bos_id)  # (B, 1, D)
        
        inputs_embeds = torch.cat([ve, bos_emb], dim=1)
        attention_mask = torch.cat([
            vm.to(dtype=torch.long),
            torch.ones(B, 1, dtype=torch.long, device=ve.device)
        ], dim=1)
        
        # _beam_search returns generated tokens only (no prompt prefix)
        gen_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams if num_beams is not None else cfg.get("num_beams", 4),
            pad_token_id=self.decoder.pad_token_id,
            eos_token_id=self.decoder.eos_token_id,
        )
        bos_id = self.tokenizer.token_to_id("<bos>")
        skip = {self.decoder.pad_token_id, self.decoder.eos_token_id}
        if bos_id is not None:
            skip.add(bos_id)
        return [
            decode_ids(self.tokenizer, ids.tolist(), skip_ids=skip)
            for ids in gen_ids
        ]

