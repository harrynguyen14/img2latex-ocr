import sys
import os
import warnings
from pathlib import Path
from PIL import Image
import torch
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", type=str, default="harryrobert/Nav2Tex")
    ap.add_argument("--image", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()

    from huggingface_hub import snapshot_download
    print(f"[load] Downloading {args.repo_id}...")
    bundle_path = Path(snapshot_download(args.repo_id))
    print(f"[load] Cached at: {bundle_path}")

    sys.path.insert(0, str(bundle_path))
    from nav2tex.pipeline_latex_ocr import Nav2TexPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    pipe = Nav2TexPipeline.from_pretrained(str(bundle_path), device=device)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[error] Image not found: {image_path}")
        return

    print(f"[test] Image: {image_path.name}")
    result = pipe(image_path)

    print("\n" + "=" * 30)
    print(f"PRED: {result}")
    print("=" * 30)
    print("[ok] Done!")


if __name__ == "__main__":
    main()