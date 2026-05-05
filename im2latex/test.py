import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from im2latex.utils import collate_fn, move_batch
from im2latex.preprocessor import LaTeXOCRHFDataset, LaTeXOCRFlatParquetDataset
from im2latex.evaluate import compute_metrics, print_metrics
from im2latex.latex_ocr_model import LaTeXOCRModel
from im2latex.latex_ocr_model.model import decode_ids
from tokenizer import LaTeXTokenizerV2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",    type=str, required=True)
    ap.add_argument("--dataset_id",    type=str, default="harryrobert/ocr-latex-filter")
    ap.add_argument("--data_path",     type=str, default="")
    ap.add_argument("--split",         type=str, default="test")
    ap.add_argument("--tokenizer_dir", type=str, default="")
    ap.add_argument("--batch_size",    type=int, default=1)
    ap.add_argument("--num_workers",   type=int, default=2)
    ap.add_argument("--max_token_len",     type=int,   default=200)
    ap.add_argument("--image_height",      type=int,   default=64)
    ap.add_argument("--max_image_width",   type=int,   default=672)
    ap.add_argument("--max_image_height",  type=int,   default=640)
    ap.add_argument("--patch_size",        type=int,   default=16)
    ap.add_argument("--resize_in_dataset", action="store_true", default=True)
    ap.add_argument("--max_samples",   type=int, default=500)
    ap.add_argument("--output",        type=str, default=None)
    return ap.parse_args()


def apply_checkpoint_preprocess_args(args, config: dict):
    """Sync preprocessing args từ checkpoint config để đảm bảo consistency."""
    for key in (
        "max_token_len",
        "image_height",
        "max_image_width",
        "max_image_height",
        "patch_size",
        "resize_in_dataset",
    ):
        if key in config:
            setattr(args, key, config[key])


def load_tokenizer(tokenizer_dir: str, checkpoint_dir: Path) -> LaTeXTokenizerV2:
    """
    Load tokenizer theo thứ tự ưu tiên:
      1. --tokenizer_dir nếu được chỉ định và tồn tại
      2. checkpoint_dir/tokenizer/ (subfolder được copy lúc save)
      3. checkpoint_dir trực tiếp (file tokenizer.json hoặc tokenizer_v2.json)
    """
    # 1. Explicit tokenizer_dir
    if tokenizer_dir:
        p = Path(tokenizer_dir)
        if p.exists():
            print(f"[tokenizer] Loading from --tokenizer_dir: {p}")
            return LaTeXTokenizerV2.load(p)
        print(f"[tokenizer] WARNING: --tokenizer_dir '{tokenizer_dir}' not found, falling back")

    # 2. Subfolder trong checkpoint
    tok_subdir = checkpoint_dir / "tokenizer"
    if tok_subdir.exists() and (
        (tok_subdir / "tokenizer_v2.json").exists() or
        (tok_subdir / "tokenizer.json").exists()
    ):
        print(f"[tokenizer] Loading from checkpoint subfolder: {tok_subdir}")
        return LaTeXTokenizerV2.load(tok_subdir)

    # 3. Checkpoint dir trực tiếp
    if (checkpoint_dir / "tokenizer_v2.json").exists() or \
       (checkpoint_dir / "tokenizer.json").exists():
        print(f"[tokenizer] Loading from checkpoint dir: {checkpoint_dir}")
        return LaTeXTokenizerV2.load(checkpoint_dir)

    raise FileNotFoundError(
        f"Tokenizer not found. Tried:\n"
        f"  1. --tokenizer_dir='{tokenizer_dir}'\n"
        f"  2. {tok_subdir}\n"
        f"  3. {checkpoint_dir}\n"
        f"Pass --tokenizer_dir trỏ tới thư mục chứa tokenizer_v2.json"
    )


def load_model(ckpt_dir: Path, config: dict, tokenizer, device: torch.device) -> LaTeXOCRModel:
    """
    Load model từ checkpoint và re-tie lm_head weights.

    Safetensors luôn save embed_tokens và lm_head thành 2 tensor riêng biệt
    → phải gọi decoder.tie_weights() sau khi load để restore tie.
    """
    from safetensors.torch import load_file

    def checkpoint_decoder_is_tied(state_dict: dict) -> bool | None:
        embed = state_dict.get("decoder._model.embed_tokens.weight")
        lm_head = state_dict.get("decoder._model.lm_head.weight")
        if embed is None or lm_head is None:
            return None
        return torch.equal(embed, lm_head)

    model = LaTeXOCRModel(config, tokenizer=tokenizer).to(device)

    sf_path = ckpt_dir / "model.safetensors"
    pt_path = ckpt_dir / "model.pt"

    if sf_path.exists():
        state = load_file(str(sf_path), device=str(device))
    elif pt_path.exists():
        state = torch.load(str(pt_path), map_location=device)
    else:
        raise FileNotFoundError(f"No model file found in {ckpt_dir}")

    # Load visual_encoder
    ve_state = {k[len("visual_encoder."):]: v
                for k, v in state.items() if k.startswith("visual_encoder.")}
    if ve_state:
        model.visual_encoder.load_state_dict(ve_state, strict=True)
        print(f"[load] visual_encoder: {len(ve_state)} tensors")

    # Load decoder
    dec_state = {k[len("decoder."):]: v
                 for k, v in state.items() if k.startswith("decoder.")}
    ckpt_tied = checkpoint_decoder_is_tied(state)
    if dec_state:
        if ckpt_tied is False:
            model.decoder.untie_weights()
            print("[load] decoder checkpoint is LEGACY untied; keeping separate lm_head/embed_tokens")
        model.decoder.load_state_dict(dec_state, strict=True)
        print(f"[load] decoder: {len(dec_state)} tensors")
    elif not ve_state:
        # Flat state dict không có prefix
        if ckpt_tied is False:
            model.decoder.untie_weights()
        model.load_state_dict(state, strict=True)
        print(f"[load] model (flat): {len(state)} tensors")

    if ckpt_tied is not False:
        # Tied checkpoints are duplicated as 2 tensors in safetensors, so re-tie after load.
        model.decoder.tie_weights()
    tied = model.decoder.are_weights_tied()
    if ckpt_tied is False:
        print("[load] tie_weights: LEGACY checkpoint left untied by design")
    else:
        print(f"[load] tie_weights: {'OK ✓' if tied else 'FAILED! ✗ — kiểm tra lại CustomDecoder'}")
    if ckpt_tied is not False and not tied:
        raise RuntimeError(
            "tie_weights() failed: lm_head.weight và embed_tokens.weight vẫn là 2 tensor khác nhau. "
            "Kiểm tra lại CustomDecoder.tie_weights()."
        )

    return model


def decode_labels(tokenizer: LaTeXTokenizerV2, labels: np.ndarray,
                  skip_ids: set | None = None) -> list[str]:
    """
    Decode label tensor thành string.
    Dùng decode_ids (join token strings) thay vì tokenizer.decode() để tránh
    thêm space không mong muốn giữa các subword tokens.
    """
    if skip_ids is None:
        pad_id = tokenizer.token_to_id("<pad>")
        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")
        skip_ids = {pad_id, bos_id, eos_id}

    texts = []
    for row in labels:
        # -100 là label padding, thay bằng pad_id để decode bình thường
        valid_ids = [int(x) if x >= 0 else tokenizer.token_to_id("<pad>") for x in row]
        texts.append(decode_ids(tokenizer, valid_ids, skip_ids=skip_ids))
    return texts


def make_dataset(data_source: str, split: str, tokenizer, args):
    if args.data_path:
        data_path = Path(args.data_path)
        if data_path.exists() and data_path.is_dir():
            # Thư mục trực tiếp chứa parquet
            if any(data_path.glob("*.parquet")):
                return LaTeXOCRFlatParquetDataset(str(data_path), tokenizer, args)
            # Thư mục có split subfolder
            split_dir = data_path / split
            if split_dir.exists() and split_dir.is_dir():
                return LaTeXOCRFlatParquetDataset(str(split_dir), tokenizer, args)
    return LaTeXOCRHFDataset(data_source, split, tokenizer, args)


def main():
    args = parse_args()

    if args.batch_size != 1:
        print("[warning] beam search chỉ hỗ trợ batch_size=1, override thành 1.")
        args.batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    ckpt_dir = Path(args.checkpoint)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir không tồn tại: {ckpt_dir}")

    # Load config từ checkpoint
    with open(ckpt_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    # Sync preprocessing args từ checkpoint config
    apply_checkpoint_preprocess_args(args, config)
    print(f"[config] image_height={args.image_height}, max_image_width={args.max_image_width}, "
          f"max_token_len={args.max_token_len}")

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_dir, ckpt_dir)
    print(f"[tokenizer] vocab_size={tokenizer.vocab_size}, merges={len(tokenizer.merges)}")

    # Load model với tie_weights fix
    model = load_model(ckpt_dir, config, tokenizer, device)
    model.eval()

    # Build skip_ids cho decode
    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    skip_ids = {pad_id, bos_id, eos_id}

    # Build dataset
    data_source = args.data_path.strip() or args.dataset_id
    ds = make_dataset(data_source, args.split, tokenizer, args)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # Generate
    preds, refs = [], []
    n_samples = 0
    pbar = tqdm(
        loader,
        desc="Generating",
        total=args.max_samples if args.max_samples > 0 else None,
        unit="sample",
    )

    for batch in pbar:
        batch = move_batch(batch, device)
        with torch.no_grad():
            pr = model.generate(batch["batched_images"])
        preds.extend(pr)
        refs.extend(decode_labels(tokenizer, batch["labels"].cpu().numpy(), skip_ids=skip_ids))
        n_samples += len(pr)
        pbar.set_postfix({"samples": n_samples})
        if args.max_samples > 0 and n_samples >= args.max_samples:
            preds = preds[:args.max_samples]
            refs  = refs[:args.max_samples]
            break

    pbar.close()

    # Metrics
    mets = compute_metrics(preds, refs)
    print_metrics(mets, prefix=args.split)

    # Debug: in vài sample để kiểm tra
    print("\n--- Sample predictions ---")
    for i in range(min(3, len(preds))):
        print(f"  REF : {refs[i][:120]}")
        print(f"  PRED: {preds[i][:120]}")
        print(f"  EXACT: {preds[i].strip() == refs[i].strip()}")
        print()

    # Save output nếu có
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for p, r in zip(preds, refs):
                f.write(f"REF: {r}\nPRED: {p}\nEXACT: {p.strip() == r.strip()}\n---\n")
        print(f"[output] Saved to {args.output}")


if __name__ == "__main__":
    main()
