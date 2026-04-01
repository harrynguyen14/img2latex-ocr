import argparse
import torch
from pathlib import Path
from huggingface_hub import HfApi, login

from modeling_latex_ocr import LaTeXOCRConfig, LaTeXOCRModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repo: username/model-name")
    parser.add_argument("--save_dir", type=str, default="hf_export")
    parser.add_argument("--merge_lora", action="store_true", default=True)
    parser.add_argument("--private", action="store_true", default=False)
    parser.add_argument("--token", type=str, default=None, help="HF token (hoặc set HF_TOKEN env)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.token:
        login(token=args.token)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, _ = LaTeXOCRModel.from_checkpoint(args.checkpoint, device="cpu")

    print(f"Exporting to: {args.save_dir}")
    model.save_checkpoint(args.save_dir, step=0, merge_lora=args.merge_lora)

    print(f"Uploading to HuggingFace: {args.repo_id}")
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=args.save_dir,
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
