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
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        navit = NaViTEncoder(
            patch_size=config["patch_size"],
            image_height=config["image_height"],
            max_image_width=config["max_image_width"],
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            mlp_ratio=config["mlp_ratio"],
            dropout=config["dropout"],
        )
        projection = BridgeMLP(in_dim=config["embed_dim"], out_dim=config["bridge_out_dim"])
        self.visual_encoder = VisualEncoder(navit, projection)

        self.decoder = AutoModelForCausalLM.from_pretrained(
            config["tokenizer_name"],
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        if config.get("use_lora", False):
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.get("lora_rank", 64),
                lora_alpha=config.get("lora_alpha", 128),
                lora_dropout=config.get("lora_dropout", 0.05),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                bias="none",
            )
            self.decoder = get_peft_model(self.decoder, lora_config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_name"], trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Recompute activations during backward to reduce peak VRAM ~40%
        if hasattr(self.decoder, "gradient_checkpointing_enable"):
            self.decoder.gradient_checkpointing_enable({"use_reentrant": False})

    def freeze_for_lora_finetuning(self):
        """Stage 2: freeze visual encoder + decoder base weights, keep only LoRA adapters trainable."""
        for p in self.visual_encoder.parameters():
            p.requires_grad = False
        for name, p in self.decoder.named_parameters():
            p.requires_grad = "lora_" in name

    def forward(self, pixel_values, patch_mask=None, input_ids=None,
                attention_mask=None, labels=None):
        visual_tokens, visual_attention_mask = self.visual_encoder(pixel_values, patch_mask)
        token_embeds  = self.decoder.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_tokens, token_embeds], dim=1)

        B, num_visual_tokens, _ = visual_tokens.shape
        combined_attention_mask = torch.cat([
            visual_attention_mask.to(device=attention_mask.device, dtype=attention_mask.dtype),
            attention_mask,
        ], dim=1)

        # Build full label sequence (ignore visual token positions)
        visual_ignore  = torch.full((B, num_visual_tokens), -100,
                                    dtype=labels.dtype, device=labels.device)
        decoder_labels = torch.cat([visual_ignore, labels], dim=1)  # (B, full_seq_len)

        if labels is None:
            out = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_attention_mask,
                return_dict=True,
            )
            import types
            return types.SimpleNamespace(loss=None, logits=out.logits)

        # Get final hidden states — do NOT pass labels to avoid HF computing loss internally
        # HF loss materializes full logits (vocab=151k) in float32 → OOM on T4
        decoder_out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            return_dict=True,
        )
        # lm_head projects hidden → logits; keep in computation graph (no detach)
        # but compute CE in float32 chunks to avoid peak memory spike
        logits       = decoder_out.logits                    # (B, S, V) — bf16, in graph
        shift_logits = logits[:, :-1].contiguous()           # (B, S-1, V)
        shift_labels = decoder_labels[:, 1:].contiguous()    # (B, S-1)
        num_tokens   = (shift_labels != -100).sum().clamp(min=1)

        # Chunked CE: slice seq_len dimension so only (B × chunk × V) is float32 at once
        # Crucially: we sum chunk losses before calling .backward() — only 1 backward pass
        chunk_size = 512
        loss = sum(
            torch.nn.functional.cross_entropy(
                shift_logits[:, start:start + chunk_size].reshape(-1, logits.size(-1)).float(),
                shift_labels[:, start:start + chunk_size].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            for start in range(0, shift_logits.size(1), chunk_size)
        ) / num_tokens

        import types
        return types.SimpleNamespace(loss=loss)

    @torch.no_grad()
    def generate(self, pixel_values, patch_mask=None, max_new_tokens=None):
        max_new_tokens = max_new_tokens or self.config.get("max_new_tokens", 200)
        visual_tokens, visual_attention_mask = self.visual_encoder(pixel_values, patch_mask)

        B      = visual_tokens.shape[0]
        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id or 0
        bos_tokens    = torch.full((B, 1), bos_id, dtype=torch.long, device=pixel_values.device)
        inputs_embeds = torch.cat([visual_tokens, self.decoder.get_input_embeddings()(bos_tokens)], dim=1)
        attention_mask = torch.cat([
            visual_attention_mask,
            torch.ones(B, 1, dtype=torch.long, device=pixel_values.device),
        ], dim=1)

        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def save_checkpoint(self, save_dir: str, step: int, optimizer=None, metrics: dict = None):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {}
        for k, v in self.visual_encoder.state_dict().items():
            checkpoint[f"visual_encoder.{k}"] = v.contiguous().cpu().float()
        for k, v in self.decoder.state_dict().items():
            if "lora_" in k:
                checkpoint[f"decoder.{k}"] = v.contiguous().cpu().float()

        save_file(checkpoint, save_dir / "model.safetensors")
        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        self.tokenizer.save_pretrained(str(save_dir))

        training_metadata = {"step": step, "metrics": metrics or {}}
        if optimizer is not None:
            torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
            training_metadata["has_optimizer"] = True
        with open(save_dir / "training_metadata.json", "w") as f:
            json.dump(training_metadata, f, indent=2)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, device: str = "cpu"):
        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir / "config.json") as f:
            config = json.load(f)
        model = cls(config)

        checkpoint = load_file(str(checkpoint_dir / "model.safetensors"), device=device)
        visual_encoder_state = {k.replace("visual_encoder.", ""): v
                                for k, v in checkpoint.items() if k.startswith("visual_encoder.")}
        lora_adapter_state   = {k.replace("decoder.", ""): v
                                for k, v in checkpoint.items() if k.startswith("decoder.")}
        model.visual_encoder.load_state_dict(visual_encoder_state, strict=True)
        if lora_adapter_state:
            model.decoder.load_state_dict(lora_adapter_state, strict=False)

        training_metadata = {}
        metadata_path = checkpoint_dir / "training_metadata.json"
        if not metadata_path.exists():
            metadata_path = checkpoint_dir / "trainer_state.json"  # backward compat
        if metadata_path.exists():
            with open(metadata_path) as f:
                training_metadata = json.load(f)

        return model.to(device), training_metadata
