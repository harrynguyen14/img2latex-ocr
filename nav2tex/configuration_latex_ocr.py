from transformers import PretrainedConfig


class Nav2TexConfig(PretrainedConfig):
    model_type = "nav2tex"

    def __init__(
        self,
        patch_size: int = 16,
        image_height: int = 64,
        max_image_width: int = 1024,
        max_image_height: int = 640,
        resize_in_dataset: bool = True,
        max_token_len: int = 200,
        navit_dim: int = 512,
        navit_depth: int = 8,
        navit_heads: int = 8,
        navit_dim_head: int = 64,
        navit_mlp_dim: int = 2048,
        navit_dropout: float = 0.0,
        navit_emb_dropout: float = 0.0,
        vision_hidden_size: int = 512,
        llm_hidden_size: int = 512,
        projector_intermediate_size: int = 1024,
        max_visual_tokens: int = 256,
        max_new_tokens: int = 200,
        num_beams: int = 4,
        decoder_arch: dict | None = None,
        decoder_weights_tied: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.image_height = image_height
        self.max_image_width = max_image_width
        self.max_image_height = max_image_height
        self.resize_in_dataset = resize_in_dataset
        self.max_token_len = max_token_len
        self.navit_dim = navit_dim
        self.navit_depth = navit_depth
        self.navit_heads = navit_heads
        self.navit_dim_head = navit_dim_head
        self.navit_mlp_dim = navit_mlp_dim
        self.navit_dropout = navit_dropout
        self.navit_emb_dropout = navit_emb_dropout
        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.projector_intermediate_size = projector_intermediate_size
        self.max_visual_tokens = max_visual_tokens
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.decoder_arch = decoder_arch or {
            "vocab_size": 2046,
            "pad_id": 0,
            "bos_id": 2,
            "eos_id": 3,
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 1408,
            "dropout": 0.1,
            "max_seq_len": 200,
            "rope_theta": 10000.0,
            "tie_weights": True,
        }
        self.decoder_weights_tied = decoder_weights_tied
