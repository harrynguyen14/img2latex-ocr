from transformers import AutoTokenizer
from torch.utils.data import IterableDataset
from PIL import Image
import torch
import torchvision.transforms.functional as TF

MAX_TOKEN_LEN  = 200
TOKENIZER_NAME = "Qwen/Qwen2.5-Coder-1.5B"

TARGET_HEIGHT  = 64
MAX_WIDTH      = 672
PATCH_SIZE     = 16


def resize_image(img: Image.Image) -> Image.Image:
    """
    Resize về height=64px, scale width tỉ lệ, clamp về bội số PATCH_SIZE và tối đa MAX_WIDTH.
    Nếu ảnh đã đúng kích thước thì không làm gì.
    """
    w, h = img.size
    if h != TARGET_HEIGHT:
        scale = TARGET_HEIGHT / h
        w = int(w * scale)
        img = img.resize((w, TARGET_HEIGHT), Image.LANCZOS)

    w = img.size[0]
    w = min(w, MAX_WIDTH)
    w = (w // PATCH_SIZE) * PATCH_SIZE
    w = max(w, PATCH_SIZE)
    if img.size[0] != w:
        img = img.resize((w, TARGET_HEIGHT), Image.LANCZOS)

    return img


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    t = TF.to_tensor(img)
    return TF.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


class LaTeXDataset(IterableDataset):
    """
    Stream dataset từ HuggingFace Hub hoặc local Parquet folder.
    Dùng streaming=True để không load toàn bộ vào RAM.

    data_path: HF repo id (vd: "harryrobert/latex-ocr")
               hoặc local folder chứa Parquet (vd: "D:/data/augmented_v2/data")
    split:     "stage1" | "stage2" | "validation" | "test"
    """

    def __init__(self, data_path: str, split: str, tokenizer):
        from datasets import load_dataset
        from pathlib import Path

        self.tokenizer = tokenizer
        self.split = split

        p = Path(data_path)
        if p.exists():
            self.ds = load_dataset(
                "parquet",
                data_files={split: str(p / f"{split}-*.parquet")},
                split=split,
                streaming=True,
            )
        else:
            self.ds = load_dataset(data_path, split=split, streaming=True)

        print(f"LaTeXDataset: split={split}  source={data_path}  streaming=True")

    def __iter__(self):
        for sample in self.ds:
            img = resize_image(sample["image"].convert("RGB"))
            tensor = image_to_tensor(img)  # (3, H, W)

            label = sample["label"]
            encoding = self.tokenizer(
                label,
                max_length=MAX_TOKEN_LEN,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            yield {
                "pixel_values": tensor,          # (3, H, W) — float32
                "input_ids": input_ids,           # (L,)
                "attention_mask": attention_mask, # (L,)
                "labels": labels,                 # (L,) với -100 ở padding
                "label": label,                   # raw string cho compute_metrics
            }


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
