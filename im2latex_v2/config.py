import yaml
from pathlib import Path


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parent / "config.yaml"
    with open(path, encoding="utf-8") as f:
        nested = yaml.safe_load(f)
    flat = {}
    for section in nested.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat
