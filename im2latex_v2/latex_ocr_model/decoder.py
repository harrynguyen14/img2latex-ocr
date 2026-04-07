import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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
        stage = config.get("stage", 2)
        trainer = config.get("trainer", "ddp")

        if stage == 1:
            self.model = AutoModelForCausalLM.from_pretrained(
                name,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        elif trainer == "fsdp" or trainer == "ddp_multiGPU":
            # Multi-GPU stage 2: load bf16 không dùng device_map,
            # để trainer tự đặt lên đúng GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                name,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            # DDP single-GPU stage 2: dùng 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def apply_lora(self, use_qlora: bool = True):
        from peft import get_peft_model, LoraConfig, TaskType
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        if use_qlora:
            from peft import prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=False,
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
