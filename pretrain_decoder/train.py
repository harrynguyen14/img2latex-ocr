import argparse
from pathlib import Path

from config import DecoderConfig
from pretrain import train


def parse_args():
    ap = argparse.ArgumentParser(description="Pretrain LaTeX causal decoder")

    ap.add_argument("--config",          type=str,   default=None)
    ap.add_argument("--n-layers",        type=int,   default=6)
    ap.add_argument("--d-model",         type=int,   default=512)
    ap.add_argument("--n-heads",         type=int,   default=8)
    ap.add_argument("--d-ff",            type=int,   default=1408)
    ap.add_argument("--max-seq-len",     type=int,   default=200)
    ap.add_argument("--vocab-size",      type=int,   default=2046)
    ap.add_argument("--lr",              type=float, default=3e-4)
    ap.add_argument("--batch-size",      type=int,   default=128)
    ap.add_argument("--grad-accum",      type=int,   default=4)
    ap.add_argument("--max-steps",       type=int,   default=100_000)
    ap.add_argument("--warmup-steps",    type=int,   default=2000)
    ap.add_argument("--grad-clip",       type=float, default=1.0)
    ap.add_argument("--weight-decay",    type=float, default=0.1)
    ap.add_argument("--dropout",         type=float, default=0.1)
    ap.add_argument("--dtype",           type=str,   default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--save-every",      type=int,   default=2_000)
    ap.add_argument("--eval-every",      type=int,   default=1_000)
    ap.add_argument("--log-every",       type=int,   default=100)
    ap.add_argument("--keep-last-n",     type=int,   default=3)
    ap.add_argument("--patience",        type=int,   default=10)
    ap.add_argument("--num-workers",     type=int,   default=4)
    ap.add_argument("--compile",         action="store_true")
    ap.add_argument("--tokenizer-dir",   type=str,
                    default="/workspace/tokenizer")
    ap.add_argument("--out-dir",         type=str,
                    default="/workspace/checkpoints")
    ap.add_argument("--data-dir",        type=str,
                    default="/workspace/data")
    ap.add_argument("--raw-ratio",       type=float, default=0.70)
    ap.add_argument("--light-ratio",     type=float, default=0.70)
    ap.add_argument("--heavy-ratio",     type=float, default=0.30)
    ap.add_argument("--raw-weight",      type=float, default=2.0)
    ap.add_argument("--light-weight",    type=float, default=4.0)
    ap.add_argument("--heavy-weight",    type=float, default=1.0)
    ap.add_argument("--no-resume",        action="store_true")
    ap.add_argument("--resume-from",      type=str, default=None,
                    help="Path to specific checkpoint dir to resume from")
    ap.add_argument("--seed",            type=int,   default=42)

    return ap.parse_args()


def main():
    args = parse_args()

    if args.config:
        cfg = DecoderConfig.load(args.config)
    else:
        cfg = DecoderConfig(
            n_layers                 = args.n_layers,
            d_model                  = args.d_model,
            n_heads                  = args.n_heads,
            d_ff                     = args.d_ff,
            max_seq_len              = args.max_seq_len,
            vocab_size               = args.vocab_size,
            lr                       = args.lr,
            batch_size               = args.batch_size,
            grad_accum_steps         = args.grad_accum,
            max_steps                = args.max_steps,
            warmup_steps             = args.warmup_steps,
            grad_clip                = args.grad_clip,
            weight_decay             = args.weight_decay,
            dropout                  = args.dropout,
            dtype                    = args.dtype,
            save_every_steps         = args.save_every,
            eval_every_steps         = args.eval_every,
            log_every_steps          = args.log_every,
            keep_last_n_ckpt         = args.keep_last_n,
            early_stopping_patience  = args.patience,
            num_workers              = args.num_workers,
            compile                  = args.compile,
            tokenizer_dir            = args.tokenizer_dir,
            out_dir                  = args.out_dir,
            data_dir                 = args.data_dir,
            raw_ratio                = args.raw_ratio,
            light_ratio              = args.light_ratio,
            heavy_ratio              = args.heavy_ratio,
            raw_weight               = args.raw_weight,
            light_weight             = args.light_weight,
            heavy_weight             = args.heavy_weight,
        )

    print(cfg)
    resume = args.resume_from if args.resume_from else (not args.no_resume)
    train(cfg, resume=resume)


if __name__ == "__main__":
    main()
