import argparse
from pathlib import Path

_DEFAULT_TOKENIZER_DIR = str(Path(__file__).parent / "tokenizer")
_DEFAULT_DATA_GLOB     = "/workspace/data/*.parquet"
_DEFAULT_SAVE_DIR      = "/workspace/checkpoints"


def get_config():
    p = argparse.ArgumentParser(description="Pretrain LaTeX decoder")

    # tokenizer
    p.add_argument("--tokenizer-dir",  default=_DEFAULT_TOKENIZER_DIR)
    p.add_argument("--vocab-size",     type=int,   default=50000)
    p.add_argument("--pad-token-id",   type=int,   default=1)
    p.add_argument("--bos-token-id",   type=int,   default=0)
    p.add_argument("--eos-token-id",   type=int,   default=2)

    # data
    p.add_argument("--data-glob",            default=_DEFAULT_DATA_GLOB)
    p.add_argument("--max-seq-len",          type=int,   default=1024)
    p.add_argument("--cpe-score-threshold",  type=int,   default=400)
    p.add_argument("--cpe-ratio",            type=float, default=0.20)

    # model
    p.add_argument("--d-model",       type=int,   default=1024)
    p.add_argument("--n-heads",       type=int,   default=16)
    p.add_argument("--n-layers",      type=int,   default=8)
    p.add_argument("--d-ff",          type=int,   default=4096)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--squeeze-ratio", type=int,   default=4)

    # LAM
    p.add_argument("--lam-lambda",   type=float, default=0.1)

    # training
    p.add_argument("--batch-size",       type=int,   default=32)
    p.add_argument("--grad-accum",       type=int,   default=4)
    p.add_argument("--max-epochs",       type=int,   default=10)
    p.add_argument("--warmup-steps",     type=int,   default=1000)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight-decay",     type=float, default=0.01)
    p.add_argument("--max-grad-norm",    type=float, default=1.0)
    p.add_argument("--label-smoothing",  type=float, default=0.1)

    # checkpoint
    p.add_argument("--save-dir",           default=_DEFAULT_SAVE_DIR)
    p.add_argument("--save-every-n-steps", type=int, default=2000)
    p.add_argument("--log-every-n-steps",  type=int, default=100)

    # hardware
    p.add_argument("--num-workers",        type=int,            default=4)
    p.add_argument("--prefetch-factor",    type=int,            default=2)
    p.add_argument("--persistent-workers", action="store_true", default=False)
    p.add_argument("--cuda-benchmark",     action="store_true", default=False)
    p.add_argument("--bf16",               action="store_true", default=True)
    p.add_argument("--no-bf16",            action="store_false", dest="bf16")
    p.add_argument("--compile",            action="store_true", default=False)
    p.add_argument("--flash-attn",         action="store_true", default=False)

    args = p.parse_args()
    cfg = argparse.Namespace(**{k.replace("-", "_"): v for k, v in vars(args).items()})
    return cfg