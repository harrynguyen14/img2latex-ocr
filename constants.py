import yaml
from pathlib import Path


def _load_flat_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        nested_config = yaml.safe_load(f)
    flat_config = {}
    for section in nested_config.values():
        if isinstance(section, dict):
            flat_config.update(section)
    return flat_config


_config = _load_flat_config()

PATCH_SIZE: int       = _config["patch_size"]
IMAGE_HEIGHT: int     = _config["image_height"]
MAX_IMAGE_WIDTH: int  = _config["max_image_width"]
MAX_TOKEN_LEN: int    = _config["max_token_len"]
EMBED_DIM: int        = _config["embed_dim"]
BRIDGE_OUT_DIM: int   = _config["bridge_out_dim"]
NUM_HEADS: int        = _config["num_heads"]
NUM_LAYERS: int       = _config["num_layers"]
MLP_RATIO: int        = _config["mlp_ratio"]
DROPOUT: float        = _config["dropout"]
TOKENIZER_NAME: str   = _config["tokenizer_name"]
