import re
import json
import yaml
import torch
import torchvision.transforms.functional as TF
from PIL import Image


IMAGE_HEIGHT = 64
MAX_IMAGE_WIDTH = 672
PATCH_SIZE = 16


def latex_tokens(text: str) -> list:
    return re.findall(r"\\[a-zA-Z]+|[^\s]", text)


def strip_align(text: str) -> str:
    text = re.sub(r"\\begin\{align\*?\}", "", text)
    text = re.sub(r"\\end\{align\*?\}", "", text)
    return text.strip()


def is_valid_sample(text: str, max_token_len: int = 200) -> bool:
    toks = latex_tokens(text)
    return bool(text) and 2 <= len(toks) <= max_token_len


def resize_image(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = IMAGE_HEIGHT / h
    new_w = int(w * scale)
    new_w = min(new_w, MAX_IMAGE_WIDTH)
    new_w = (new_w // PATCH_SIZE) * PATCH_SIZE
    new_w = max(new_w, PATCH_SIZE)
    return img.resize((new_w, IMAGE_HEIGHT), Image.LANCZOS)


def collate_images(images: list, device: str = "cpu"):
    from encode import collate_images as _collate
    return _collate(images, device=device)


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_manifest(manifest_path: str) -> list:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def image_to_tensor(img: Image.Image, device: str = "cpu") -> torch.Tensor:
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return tensor.unsqueeze(0).to(device)
