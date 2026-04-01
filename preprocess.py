import json
import argparse
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import latex_tokens, strip_align, is_valid_sample, resize_image, load_manifest

MAX_TOKEN_LEN = 200
TOKENIZER_NAME = "Qwen/Qwen2.5-Coder-1.5B"
DATA_DIR = Path("data")

LINXY_CONFIGS = ["full", "synthetic_handwrite", "human_handwrite"]
OLEEHYO_CONFIG = "cleaned_formulas"


def load_and_filter():
    all_train, all_val, all_test = [], [], []

    for cfg_name in tqdm(LINXY_CONFIGS, desc="Loading linxy configs"):
        ds = load_dataset("linxy/LaTeX_OCR", cfg_name, trust_remote_code=True)
        for split_name, split_data in ds.items():
            filtered = split_data.filter(
                lambda x: is_valid_sample(x["text"], MAX_TOKEN_LEN),
                desc=f"Filtering linxy/{cfg_name}/{split_name}",
            )
            renamed = filtered.select_columns(["image", "text"]).rename_column("text", "label")
            if split_name == "train":
                all_train.append(renamed)
            elif split_name == "validation":
                all_val.append(renamed)
            elif split_name == "test":
                all_test.append(renamed)

    print("Loading OleehyO/latex-formulas ...")
    ds_olee = load_dataset("OleehyO/latex-formulas", OLEEHYO_CONFIG, trust_remote_code=True)
    for split_name, split_data in ds_olee.items():
        filtered = split_data.map(
            lambda x: {"label": strip_align(x["latex_formula"])},
            desc=f"Processing OleehyO/{split_name}",
        )
        filtered = filtered.filter(
            lambda x: is_valid_sample(x["label"], MAX_TOKEN_LEN),
            desc=f"Filtering OleehyO/{split_name}",
        )
        filtered = filtered.select_columns(["image", "label"])
        if split_name == "train":
            all_train.append(filtered)

    print("Concatenating and shuffling ...")
    train_ds = concatenate_datasets(all_train).shuffle(seed=42)
    val_ds = concatenate_datasets(all_val) if all_val else None
    test_ds = concatenate_datasets(all_test) if all_test else None

    return train_ds, val_ds, test_ds


def save_splits(train_ds, val_ds, test_ds):
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Saving train ({len(train_ds)} samples) ...")
    train_ds.save_to_disk(str(DATA_DIR / "train"))

    if val_ds:
        print(f"Saving val ({len(val_ds)} samples) ...")
        val_ds.save_to_disk(str(DATA_DIR / "val"))

    if test_ds:
        print(f"Saving test ({len(test_ds)} samples) ...")
        test_ds.save_to_disk(str(DATA_DIR / "test"))

    print("Done.")


class LaTeXDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, augmented_manifest: str = None, **kwargs):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.augmented = None

        if augmented_manifest and Path(augmented_manifest).exists():
            self.augmented = load_manifest(augmented_manifest)
            assert len(self.augmented) == len(self.data), (
                f"Manifest size {len(self.augmented)} != dataset size {len(self.data)}"
            )
            print(f"Using pre-augmented images from {augmented_manifest}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.augmented is not None:
            img = Image.open(self.augmented[idx]["img_path"]).convert("RGB")
            label = self.augmented[idx]["label"]
        else:
            sample = self.data[idx]
            img = sample["image"].convert("RGB")
            label = sample["label"]

        img = resize_image(img)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Download, filter và lưu dataset về disk")
    parser.add_argument("--data_dir", type=str, default="data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DATA_DIR = Path(args.data_dir)
    print("Loading and filtering datasets...")
    train_ds, val_ds, test_ds = load_and_filter()
    save_splits(train_ds, val_ds, test_ds)
