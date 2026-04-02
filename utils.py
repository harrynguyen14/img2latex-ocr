import re
import yaml
from PIL import Image

from constants import IMAGE_HEIGHT, MAX_IMAGE_WIDTH, PATCH_SIZE, MAX_TOKEN_LEN


def resize_image(img: Image.Image) -> Image.Image:
    w, h = img.size
    if h != IMAGE_HEIGHT:
        w = int(w * IMAGE_HEIGHT / h)
        img = img.resize((w, IMAGE_HEIGHT), Image.LANCZOS)
    w = min(img.size[0], MAX_IMAGE_WIDTH)
    w = max((w // PATCH_SIZE) * PATCH_SIZE, PATCH_SIZE)
    if img.size[0] != w:
        img = img.resize((w, IMAGE_HEIGHT), Image.LANCZOS)
    return img


def strip_align(text: str) -> str:
    text = re.sub(r"\\begin\{align\*?\}|\\end\{align\*?\}", "", text)
    return text.strip()


def is_valid_sample(text: str, max_len: int = MAX_TOKEN_LEN) -> bool:
    tokens = re.findall(r"\\[a-zA-Z]+|[^\s]", text)
    return 2 <= len(tokens) <= max_len


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
