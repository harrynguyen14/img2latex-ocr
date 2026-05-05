"""
Push dataset to HuggingFace Hub as a dataset repo.

Layout uploaded:
  train/heavy/*.parquet
  train/light/*.parquet
  train/raw/*.parquet
  train/screenshot/*.parquet
  validation/*.parquet
  test/*.parquet
  test/crohme/*.parquet

Usage:
    $env:HF_TOKEN = "hf_..."
    python push_dataset_hf.py
    python push_dataset_hf.py --data_path D:/other/path --repo_id harryrobert/other-repo
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str,
                    default=r"D:\dataset-ocr-builder\latex-ocr-dataset\ocr-data")
    ap.add_argument("--repo_id",   type=str, default="harryrobert/formular-img-latex")
    ap.add_argument("--private",   action="store_true", default=False)
    return ap.parse_args()


def main():
    args  = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("Set HF_TOKEN first:\n  $env:HF_TOKEN = 'hf_...'")

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"data_path not found: {data_path}")

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
        exist_ok=True,
        private=args.private,
    )
    print(f"Repo ready: https://huggingface.co/datasets/{args.repo_id}")
    print(f"Uploading {data_path} …\n")

    api.upload_folder(
        folder_path=str(data_path),
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
        allow_patterns=["train/**/*.parquet", "validation/*.parquet",
                        "test/*.parquet", "test/crohme/*.parquet"],
    )

    print(f"\n[ok] Dataset pushed → https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
