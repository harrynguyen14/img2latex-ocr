import io
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


def get_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _resize(img: Image.Image, image_height: int, max_image_width: int, patch_size: int) -> Image.Image:
    w, h = img.size
    new_w = int(round(w * image_height / max(h, 1)))
    new_w = min(new_w, max_image_width)
    new_w = max((new_w // patch_size) * patch_size, patch_size)
    if (w, h) != (new_w, image_height):
        img = img.resize((new_w, image_height), Image.BILINEAR)
    return img


def _pad_to_patch_grid(img: Image.Image, patch_size: int, max_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    w = min(w, max_w)
    h = min(h, max_h)
    if w < img.size[0] or h < img.size[1]:
        img = img.crop((0, 0, w, h))
    tw = min((w + patch_size - 1) // patch_size * patch_size, max_w)
    th = min((h + patch_size - 1) // patch_size * patch_size, max_h)
    tw = max(tw, patch_size)
    th = max(th, patch_size)
    if tw == w and th == h:
        return img
    out = Image.new("RGB", (tw, th), (255, 255, 255))
    out.paste(img, (0, 0))
    return out


def _to_tensor(img: Image.Image) -> torch.Tensor:
    t = TF.to_tensor(img)
    return TF.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def _decode_image(raw) -> Image.Image:
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, dict):
        if raw.get("bytes"):
            return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        if raw.get("path"):
            return Image.open(raw["path"]).convert("RGB")
    raise ValueError(f"Cannot decode image from {type(raw)}")


def _process(sample: dict, tokenizer, args) -> dict:
    pil = _decode_image(sample["image"])
    if getattr(args, "resize_in_dataset", True):
        img = _resize(pil, args.image_height, args.max_image_width, args.patch_size)
    else:
        img = _pad_to_patch_grid(
            pil, args.patch_size, args.max_image_width,
            getattr(args, "max_image_height", args.image_height),
        )
    tensor = _to_tensor(img)
    enc = tokenizer(
        sample["label"],
        max_length=args.max_token_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)
    lab = input_ids.clone()
    lab[attention_mask == 0] = -100
    return {
        "pixel_values":   tensor,
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         lab,
    }


class LaTeXOCRHFDataset(IterableDataset):
    def __init__(self, dataset_id: str, split: str, tokenizer, args, rank: int = 0, world_size: int = 1):
        from datasets import load_dataset
        self.tokenizer   = tokenizer
        self.args        = args
        self.num_samples = None
        ds = load_dataset(dataset_id, split=split, streaming=True)
        if world_size > 1:
            ds = ds.filter(lambda _, idx: idx % world_size == rank, with_indices=True)
        self.ds = ds

    def __iter__(self):
        for sample in self.ds:
            yield _process(sample, self.tokenizer, self.args)


class LaTeXOCRDiskDataset(IterableDataset):
    def __init__(self, cache_path: str, tokenizer, args, rank: int = 0, world_size: int = 1):
        from datasets import load_from_disk
        self.ds          = load_from_disk(cache_path)
        self.tokenizer   = tokenizer
        self.args        = args
        self.rank        = rank
        self.world_size  = world_size
        self.num_samples = len(self.ds) // world_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for i, sample in enumerate(self.ds):
            if i % self.world_size == self.rank:
                yield _process(sample, self.tokenizer, self.args)
