import json
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, load_file
from pathlib import Path
from peft import LoraConfig, get_peft_model, TaskType

from encode import NaViTEncoder, BridgeMLP, VisualEncoder


class LaTeXOCRConfig(PretrainedConfig):
    model_type = "latex_ocr"

    def __init__(
        self,
        patch_size=16,
        image_height=64,
        max_image_width=672,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_ratio=4,
        encoder_dropout=0.1,
        max_seq_len=2048,
        bridge_out_dim=2048,
        decoder_name="Qwen/Qwen2.5-Coder-1.5B",
        max_new_tokens=200,
        tokenizer_name="Qwen/Qwen2.5-Coder-1.5B",
        use_lora=False,
        lora_rank=64,
        lora_alpha=128,
        lora_dropout=0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.image_height = image_height
        self.max_image_width = max_image_width
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.encoder_dropout = encoder_dropout
        self.max_seq_len = max_seq_len
        self.bridge_out_dim = bridge_out_dim
        self.decoder_name = decoder_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer_name = tokenizer_name
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout


class LaTeXOCRModel(PreTrainedModel):
    config_class = LaTeXOCRConfig
    base_model_prefix = "latex_ocr"
    supports_gradient_checkpointing = False
    main_input_name = "pixel_values"

    def __init__(self, config: LaTeXOCRConfig):
        super().__init__(config)
        navit = NaViTEncoder(
            patch_size=config.patch_size,
            image_height=config.image_height,
            max_image_width=config.max_image_width,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            mlp_ratio=config.mlp_ratio,
            dropout=config.encoder_dropout,
        )
        bridge = BridgeMLP(in_dim=config.embed_dim, out_dim=config.bridge_out_dim)
        self.visual_encoder = VisualEncoder(navit, bridge).to(torch.float16)

        self.decoder = AutoModelForCausalLM.from_pretrained(
            config.decoder_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        if config.use_lora:
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
            )
            self.decoder = get_peft_model(self.decoder, lora_cfg)

        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.post_init()

    def freeze_decoder(self):
        for p in self.decoder.parameters():
            p.requires_grad = False

    def unfreeze_lora(self):
        for name, p in self.decoder.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

    def stage1_params(self):
        return list(self.visual_encoder.parameters())

    def stage2_params(self):
        params = list(self.visual_encoder.parameters())
        for name, p in self.decoder.named_parameters():
            if "lora_" in name:
                params.append(p)
        return params

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def forward(self, pixel_values, patch_mask=None, input_ids=None, attention_mask=None, labels=None):
        pixel_values = pixel_values.to(dtype=torch.float16)
        visual_tokens, vis_mask = self.visual_encoder(pixel_values, patch_mask)
        token_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_tokens, token_embeds], dim=1)

        B, vis_len, _ = visual_tokens.shape
        full_mask = torch.cat([vis_mask.to(device=attention_mask.device, dtype=attention_mask.dtype), attention_mask], dim=1)

        if labels is not None:
            ignore = torch.full((B, vis_len), -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([ignore, labels], dim=1)
        else:
            full_labels = None

        return self.decoder(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=full_labels)

    @torch.no_grad()
    def generate(self, pixel_values, patch_mask=None, max_new_tokens=None):
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        pixel_values = pixel_values.to(dtype=torch.float16)
        visual_tokens, vis_mask = self.visual_encoder(pixel_values, patch_mask)

        B = visual_tokens.shape[0]
        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id or 0
        input_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=pixel_values.device)
        token_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_tokens, token_embeds], dim=1)

        bos_mask = torch.ones(B, 1, dtype=torch.long, device=pixel_values.device)
        attention_mask = torch.cat([vis_mask, bos_mask], dim=1)

        output = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)

    def save_checkpoint(self, save_dir: str, step: int, optimizer=None, metrics: dict = None, merge_lora: bool = False):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        decoder = self.decoder
        if merge_lora and hasattr(decoder, "merge_and_unload"):
            decoder = decoder.merge_and_unload()

        state_dict = {}
        for k, v in self.visual_encoder.state_dict().items():
            state_dict[f"visual_encoder.{k}"] = v.contiguous().cpu().float()

        if merge_lora:
            for k, v in decoder.state_dict().items():
                state_dict[f"decoder.{k}"] = v.contiguous().cpu().float()
        else:
            for k, v in decoder.state_dict().items():
                if "lora_" in k:
                    state_dict[f"decoder.{k}"] = v.contiguous().cpu().float()

        save_file(state_dict, save_dir / "model.safetensors")

        self.config.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))

        trainer_state = {
            "step": step,
            "metrics": metrics or {},
        }
        if optimizer is not None:
            torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
            trainer_state["has_optimizer"] = True

        with open(save_dir / "trainer_state.json", "w") as f:
            json.dump(trainer_state, f, indent=2)

        print(f"Checkpoint saved: {save_dir}")

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, device: str = "cpu"):
        checkpoint_dir = Path(checkpoint_dir)
        config = LaTeXOCRConfig.from_pretrained(str(checkpoint_dir))
        model = cls(config)

        state_dict = load_file(str(checkpoint_dir / "model.safetensors"), device=device)

        encoder_state = {k.replace("visual_encoder.", ""): v for k, v in state_dict.items() if k.startswith("visual_encoder.")}
        model.visual_encoder.load_state_dict(encoder_state, strict=True)

        lora_state = {k.replace("decoder.", ""): v for k, v in state_dict.items() if k.startswith("decoder.") and "lora_" in k}
        decoder_full = {k.replace("decoder.", ""): v for k, v in state_dict.items() if k.startswith("decoder.") and "lora_" not in k}

        if decoder_full:
            model.decoder.load_state_dict(decoder_full, strict=False)
        if lora_state:
            model.decoder.load_state_dict(lora_state, strict=False)

        trainer_state = {}
        state_file = checkpoint_dir / "trainer_state.json"
        if state_file.exists():
            with open(state_file) as f:
                trainer_state = json.load(f)

        model.eval()
        return model.to(device), trainer_state

    def can_generate(self) -> bool:
        return True

    def load_optimizer(self, checkpoint_dir: str, optimizer):
        opt_path = Path(checkpoint_dir) / "optimizer.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            print(f"Optimizer loaded from {opt_path}")


AutoConfig.register("latex_ocr", LaTeXOCRConfig)
AutoModel.register(LaTeXOCRConfig, LaTeXOCRModel)
