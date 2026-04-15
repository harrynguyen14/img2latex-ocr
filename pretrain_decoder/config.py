import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class DecoderConfig:
    vocab_size:       int   = 8192
    pad_id:           int   = 0
    bos_id:           int   = 2
    eos_id:           int   = 3

    d_model:          int   = 512
    n_heads:          int   = 8
    n_layers:         int   = 6
    d_ff:             int   = 1408
    dropout:          float = 0.1
    max_seq_len:      int   = 200
    rope_theta:       float = 10000.0
    tie_weights:      bool  = True

    pack_sequences:   bool  = True
    batch_size:       int   = 64
    grad_accum_steps: int   = 8

    lr:               float = 3e-4
    weight_decay:     float = 0.1
    beta1:            float = 0.9
    beta2:            float = 0.95
    eps:              float = 1e-8
    grad_clip:        float = 1.0

    warmup_steps:     int   = 2000
    max_steps:        int   = 100_000

    dtype:            str   = "bfloat16"

    save_every_steps:        int   = 2_000
    eval_every_steps:        int   = 1_000
    keep_last_n_ckpt:        int   = 3
    log_every_steps:         int   = 100
    early_stopping_patience: int   = 5
    num_workers:             int   = 2
    compile:                 bool  = False

    tokenizer_dir:    str   = "D:/img2latex/tokenizer"
    out_dir:          str   = "D:/img2latex/pretrain_decoder/checkpoints"
    data_dir:         str   = "D:/dataset-ocr-builder/latex-ocr-dataset"

    raw_ratio:        float = 0.70
    light_ratio:      float = 0.70
    heavy_ratio:      float = 0.30

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "DecoderConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(**d)

    def __repr__(self) -> str:
        return (
            f"DecoderConfig("
            f"n_layers={self.n_layers}, d_model={self.d_model}, "
            f"n_heads={self.n_heads}, d_ff={self.d_ff}, "
            f"max_seq_len={self.max_seq_len}, vocab={self.vocab_size})"
        )


if __name__ == "__main__":
    cfg = DecoderConfig()
    print(cfg)
    cfg.save("/tmp/test_cfg.json")
    loaded = DecoderConfig.load("/tmp/test_cfg.json")
    assert asdict(loaded) == asdict(cfg)
    print("save/load OK")
