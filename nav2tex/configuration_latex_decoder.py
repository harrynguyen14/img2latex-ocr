from transformers import PretrainedConfig


class LaTeXDecoderConfig(PretrainedConfig):
    model_type = "latex_decoder"

    def __init__(
        self,
        vocab_size: int = 8192,
        pad_id: int = 0,
        bos_id: int = 2,
        eos_id: int = 3,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1408,
        dropout: float = 0.1,
        max_seq_len: int = 200,
        rope_theta: float = 10000.0,
        tie_weights: bool = False,
        **kwargs,
    ):
        kwargs.pop("pad_token_id", None)
        kwargs.pop("bos_token_id", None)
        kwargs.pop("eos_token_id", None)
        super().__init__(
            pad_token_id=pad_id,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.tie_weights = tie_weights

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads
