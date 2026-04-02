import json
import torch
import torch.nn as nn
from pathlib import Path
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from constants import EMBED_DIM, BRIDGE_OUT_DIM, TOKENIZER_NAME
from encode import NaViTEncoder, BridgeMLP, VisualEncoder


class LaTeXOCRModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        navit  = NaViTEncoder(
            patch_size=cfg["patch_size"],
            image_height=cfg["image_height"],
            max_image_width=cfg["max_image_width"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            mlp_ratio=cfg["mlp_ratio"],
            dropout=cfg["dropout"],
        )
        bridge = BridgeMLP(in_dim=cfg["embed_dim"], out_dim=cfg["bridge_out_dim"])
        self.visual_encoder = VisualEncoder(navit, bridge).to(torch.float16)

        self.decoder = AutoModelForCausalLM.from_pretrained(
            cfg["tokenizer_name"],
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        if cfg.get("use_lora", False):
            lora = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.get("lora_rank", 64),
                lora_alpha=cfg.get("lora_alpha", 128),
                lora_dropout=cfg.get("lora_dropout", 0.05),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                bias="none",
            )
            self.decoder = get_peft_model(self.decoder, lora)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["tokenizer_name"], trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def freeze_decoder(self):
        for p in self.decoder.parameters():
            p.requires_grad = False

    def unfreeze_lora(self):
        for name, p in self.decoder.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

    def forward(self, pixel_values, patch_mask=None, input_ids=None,
                attention_mask=None, labels=None):
        pixel_values = pixel_values.to(dtype=torch.float16)
        visual_tokens, vis_mask = self.visual_encoder(pixel_values, patch_mask)
        token_embeds   = self.decoder.get_input_embeddings()(input_ids)
        inputs_embeds  = torch.cat([visual_tokens, token_embeds], dim=1)

        B, vis_len, _ = visual_tokens.shape
        full_mask = torch.cat([
            vis_mask.to(device=attention_mask.device, dtype=attention_mask.dtype),
            attention_mask,
        ], dim=1)

        full_labels = None
        if labels is not None:
            ignore      = torch.full((B, vis_len), -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([ignore, labels], dim=1)

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )

    @torch.no_grad()
    def generate(self, pixel_values, patch_mask=None, max_new_tokens=None):
        max_new_tokens = max_new_tokens or self.cfg.get("max_new_tokens", 200)
        pixel_values   = pixel_values.to(dtype=torch.float16)
        visual_tokens, vis_mask = self.visual_encoder(pixel_values, patch_mask)

        B      = visual_tokens.shape[0]
        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id or 0
        bos    = torch.full((B, 1), bos_id, dtype=torch.long, device=pixel_values.device)
        inputs_embeds = torch.cat([visual_tokens, self.decoder.get_input_embeddings()(bos)], dim=1)
        attn_mask     = torch.cat([vis_mask, torch.ones(B, 1, dtype=torch.long, device=pixel_values.device)], dim=1)

        out = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    def save_checkpoint(self, save_dir: str, step: int, optimizer=None, metrics: dict = None):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        state = {}
        for k, v in self.visual_encoder.state_dict().items():
            state[f"visual_encoder.{k}"] = v.contiguous().cpu().float()
        for k, v in self.decoder.state_dict().items():
            if "lora_" in k:
                state[f"decoder.{k}"] = v.contiguous().cpu().float()

        save_file(state, save_dir / "model.safetensors")
        with open(save_dir / "config.json", "w") as f:
            json.dump(self.cfg, f, indent=2)
        self.tokenizer.save_pretrained(str(save_dir))

        trainer_state = {"step": step, "metrics": metrics or {}}
        if optimizer is not None:
            torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
            trainer_state["has_optimizer"] = True
        with open(save_dir / "trainer_state.json", "w") as f:
            json.dump(trainer_state, f, indent=2)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, device: str = "cpu"):
        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir / "config.json") as f:
            cfg = json.load(f)
        model = cls(cfg)

        state = load_file(str(checkpoint_dir / "model.safetensors"), device=device)
        enc_state  = {k.replace("visual_encoder.", ""): v
                      for k, v in state.items() if k.startswith("visual_encoder.")}
        lora_state = {k.replace("decoder.", ""): v
                      for k, v in state.items() if k.startswith("decoder.")}
        model.visual_encoder.load_state_dict(enc_state, strict=True)
        if lora_state:
            model.decoder.load_state_dict(lora_state, strict=False)

        trainer_state = {}
        p = checkpoint_dir / "trainer_state.json"
        if p.exists():
            with open(p) as f:
                trainer_state = json.load(f)

        return model.to(device), trainer_state
