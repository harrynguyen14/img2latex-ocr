import io
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from transformers import AutoTokenizer

HF_DEFAULT_ID = "harryrobert/latex-ocr-v2"


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


def _process(sample: dict, tokenizer, cfg: dict) -> dict:
    pil = _decode_image(sample["image"])
    if cfg.get("resize_in_dataset", True):
        img = _resize(
            pil,
            cfg["image_height"],
            cfg["max_image_width"],
            cfg["patch_size"],
        )
    else:
        img = _pad_to_patch_grid(
            pil,
            cfg["patch_size"],
            cfg["max_image_width"],
            cfg.get("max_image_height", cfg["image_height"]),
        )
    tensor = _to_tensor(img)
    enc = tokenizer(
        sample["label"],
        max_length=cfg["max_token_len"],
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)
    lab = input_ids.clone()
    lab[attention_mask == 0] = -100
    return {
        "pixel_values": tensor,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": lab,
    }


def resolve_data_source(cfg: dict, data_path_override: str | None) -> str:
    raw = (data_path_override or cfg.get("data_path") or "").strip()
    if raw:
        return raw
    return cfg.get("dataset_id", HF_DEFAULT_ID)


def collect_parquet_files(root: Path, split: str) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        return []
    dash = split.replace("_", "-")
    files = sorted(root.glob(f"{dash}-*.parquet"))      
    if not files:
        files = sorted(root.glob(f"{split}-*.parquet")) 
    if not files:
        files = sorted(root.glob("*.parquet"))          
    return files


class LaTeXOCRParquetMapDataset(Dataset):
    def __init__(self, data_root: str, split: str, tokenizer, cfg: dict):
        from datasets import load_dataset

        safe_split = split.replace("-", "_")
        files = collect_parquet_files(Path(data_root), split)
        if not files:
            raise FileNotFoundError(f"No parquet found for split={split} under {data_root}")
        self.ds = load_dataset(
            "parquet",
            data_files={safe_split: [str(f) for f in files]},
            split=safe_split,
        )
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return _process(self.ds[idx], self.tokenizer, self.cfg)


class LaTeXOCRDataset(IterableDataset):
    def __init__(
        self,
        data_source: str,
        split: str,
        tokenizer,
        cfg: dict,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.tokenizer = tokenizer
        self.split = split
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.files = None
        self.streaming_ds = None
        self.hf_dataset_id: str | None = None
        self.num_samples = None
        self._row_filter = False

        p = Path(data_source)
        if p.exists() and p.is_dir():
            import pyarrow.parquet as pq

            all_files = collect_parquet_files(p, split)
            if all_files:
                self.num_samples = sum(pq.read_metadata(f).num_rows for f in all_files)
                self.files = [f for i, f in enumerate(all_files) if i % world_size == rank]
                if not self.files:
                    self.files = all_files
                    self._row_filter = True
        else:
            self.hf_dataset_id = data_source
            from datasets import load_dataset

            safe_split = split.replace("-", "_")
            ds = load_dataset(self.hf_dataset_id, split=safe_split, streaming=True)
            if world_size > 1:
                ds = ds.filter(lambda _, idx: idx % world_size == rank, with_indices=True)
            self.streaming_ds = ds
            self.num_samples = None

    def _iter_files(self, files):
        from datasets import load_dataset

        worker_info = get_worker_info()
        n_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        worker_files = [f for i, f in enumerate(files) if i % n_workers == worker_id]
        if not worker_files:
            return
        safe_split = self.split.replace("-", "_")
        ds = load_dataset(
            "parquet",
            data_files={safe_split: [str(f) for f in worker_files]},
            split=safe_split,
            streaming=True,
        )
        for sample in ds:
            yield _process(sample, self.tokenizer, self.cfg)

    def __iter__(self):
        if self.files is not None:
            yield from self._iter_files(self.files)
        else:
            for sample in self.streaming_ds:
                yield _process(sample, self.tokenizer, self.cfg)
