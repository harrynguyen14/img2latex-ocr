import json

from transformers import AutoTokenizer
from PIL import Image
from torch.utils.data import Dataset

MAX_TOKEN_LEN = 200
TOKENIZER_NAME = "Qwen/Qwen2.5-Coder-1.5B"


class LaTeXDataset(Dataset):
    def __init__(self, manifest_path: str, tokenizer):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        label = sample["label"]

        encoding = self.tokenizer(
            label,
            max_length=MAX_TOKEN_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "image": img,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label,
        }


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
