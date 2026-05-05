"""
Push latex_ocr/ to HuggingFace Hub.

Usage:
    python push_model_hf.py
    python push_model_hf.py --model_path D:/img2latex/checkpoints/best --repo_id harryrobert/latex-ocr
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=r"D:\img2latex\nav2tex")
    ap.add_argument("--repo_id",    type=str, default="harryrobert/Nav2Tex")
    return ap.parse_args()

def main():
    args  = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("Set HF_TOKEN environment variable first:\n  $env:HF_TOKEN='hf_...'")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, token=token, exist_ok=True)
    api.upload_folder(
        folder_path=str(Path(args.model_path)),
        repo_id=args.repo_id,
        token=token,
    )
    print(f"[ok] Pushed to https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
