import yaml
from pathlib import Path


def _load() -> dict:
    p = Path(__file__).parent / "config.yaml"
    with open(p) as f:
        raw = yaml.safe_load(f)
    flat = {}
    for section in raw.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


_cfg = _load()

PATCH_SIZE: int       = _cfg["patch_size"]
IMAGE_HEIGHT: int     = _cfg["image_height"]
MAX_IMAGE_WIDTH: int  = _cfg["max_image_width"]
MAX_TOKEN_LEN: int    = _cfg["max_token_len"]
EMBED_DIM: int        = _cfg["embed_dim"]
BRIDGE_OUT_DIM: int   = _cfg["bridge_out_dim"]
NUM_HEADS: int        = _cfg["num_heads"]
NUM_LAYERS: int       = _cfg["num_layers"]
MLP_RATIO: int        = _cfg["mlp_ratio"]
DROPOUT: float        = _cfg["dropout"]
TOKENIZER_NAME: str   = _cfg["tokenizer_name"]
