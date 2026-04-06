import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def _dtype_from_str(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


class QwenCausalDecoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        name = config["tokenizer_name"]
        dt = _dtype_from_str(config.get("dtype", config.get("torch_dtype", "bfloat16")))
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            dtype=dt,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def apply_lora(self):
        """Gọi sau khi load weights xong."""
        from peft import get_peft_model, LoraConfig, TaskType
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, inputs_embeds, attention_mask, max_new_tokens,
                 num_beams, pad_token_id, eos_token_id):
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )