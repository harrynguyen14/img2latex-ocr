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
        # Newer Transformers prefer `dtype=` over `torch_dtype=`.
        dt = _dtype_from_str(config.get("dtype", config.get("torch_dtype", "bfloat16")))
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            dtype=dt,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(
        self,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        num_beams: int,
        pad_token_id: int,
        eos_token_id: int,
    ):
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
