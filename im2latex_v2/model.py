import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from transformers import AutoTokenizer

from im2latex_v2.decoder import QwenCausalDecoder
from im2latex_v2.latexOCR import VisualEncoder
from im2latex_v2.mlp_projector import MLPProjector
from im2latex_v2.encoder import NaViT_Encoder


class LaTeXOCRModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
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
        self.decoder = QwenCausalDecoder(config)
        self.tokenizer = self.decoder.tokenizer
        if self.decoder.model.config.hidden_size != config["llm_hidden_size"]:
            raise ValueError(
                "llm_hidden_size must equal decoder hidden_size "
                f"({self.decoder.model.config.hidden_size})"
            )
        if config["navit_dim"] != config["vision_hidden_size"]:
            raise ValueError("navit_dim must equal vision_hidden_size for the projector")

    def gradient_checkpointing_enable(self):
        self.decoder.model.gradient_checkpointing_enable()

    def set_train_stage(self, stage: int):
        if stage == 1:
            for p in self.visual_encoder.navit.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False
            for p in self.visual_encoder.projector.parameters():
                p.requires_grad = True
        else:
            for p in self.parameters():
                p.requires_grad = True

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
    def generate(
        self,
        batched_images,
        max_new_tokens: int | None = None,
        num_beams: int | None = None,
    ):
        self.eval()
        cfg = self.config
        max_new_tokens = max_new_tokens or cfg.get("max_new_tokens", 256)
        num_beams = num_beams or cfg.get("num_beams", 4)
        ve, vm = self.visual_encoder(batched_images)
        inputs_embeds = ve
        attention_mask = vm.to(dtype=torch.long)
        prompt_len = inputs_embeds.shape[1]
        out = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if out.shape[1] > prompt_len:
            gen = out[:, prompt_len:]
        else:
            gen = out
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    def save_checkpoint(self, save_dir: str, step: int = 0, metrics: dict | None = None):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        state = {}
        for k, v in self.visual_encoder.state_dict().items():
            state[f"visual_encoder.{k}"] = v.contiguous().cpu()
        for k, v in self.decoder.model.state_dict().items():
            state[f"decoder.model.{k}"] = v.contiguous().cpu()
        save_file(state, save_dir / "model.safetensors")
        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)
        self.tokenizer.save_pretrained(str(save_dir))
        meta = {"step": step, "metrics": metrics or {}}
        with open(save_dir / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, device: str = "cpu"):
        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir / "config.json", encoding="utf-8") as f:
            config = json.load(f)
        model = cls(config)
        raw = load_file(str(checkpoint_dir / "model.safetensors"), device=device)
        ve = {k.replace("visual_encoder.", ""): v for k, v in raw.items() if k.startswith("visual_encoder.")}
        dec = {k.replace("decoder.model.", ""): v for k, v in raw.items() if k.startswith("decoder.model.")}
        model.visual_encoder.load_state_dict(ve, strict=True)
        model.decoder.model.load_state_dict(dec, strict=True)
        if (checkpoint_dir / "tokenizer_config.json").exists():
            model.decoder.tokenizer = AutoTokenizer.from_pretrained(
                str(checkpoint_dir), trust_remote_code=True
            )
            model.tokenizer = model.decoder.tokenizer
        return model.to(device)


def alignment_loss(model: LaTeXOCRModel, batched_images, labels: torch.Tensor):
    ve, vm = model.visual_encoder(batched_images)
    mask = vm.unsqueeze(-1).float()
    denom = mask.sum(dim=1).clamp(min=1.0)
    vmean = (ve.float() * mask).sum(dim=1) / denom
    emb = model.decoder.get_input_embeddings()
    valid = labels != -100
    tgt = emb(labels.clamp(min=0)).float()
    tgt = tgt * valid.unsqueeze(-1).float()
    tmean = tgt.sum(dim=1) / valid.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
    y = torch.ones(vmean.shape[0], device=vmean.device)
    loss = torch.nn.functional.cosine_embedding_loss(vmean, tmean, y)
    return loss
