import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as st_load_file
from transformers import AutoModelForCausalLM

from hf_upload.configuration_latex_decoder import LaTeXDecoderConfig
from hf_upload.modeling_latex_decoder import LaTeXDecoderForCausalLM
from pretrain_decoder.config import DecoderConfig
from tokenizer import LaTeXTokenizerV2


class CustomDecoder(nn.Module):
    def __init__(self, config: dict, tokenizer=None):
        super().__init__()
        repo = config.get("decoder_ckpt", "harryrobert/pretrain-decoder")
        self._model = self._load_decoder_model(repo, config)

        # KHÔNG set tie_weights=False ở đây vì sẽ break tie ngay từ đầu.
        # tie_weights được quản lý thủ công qua tie_weights() method bên dưới.
        # self._model.config.tie_weights = False  ← ĐÃ XÓA

        self.pad_token_id = self._model.config.pad_id
        self.eos_token_id = self._model.config.eos_id
        self._vocab_size  = self._model.config.vocab_size
        self._pad_id      = self._model.config.pad_id
        self.tokenizer    = tokenizer

        if config.get("qat", False):
            self._enable_qat()

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _build_model_from_arch_config(arch_cfg: dict) -> LaTeXDecoderForCausalLM:
        hf_cfg = LaTeXDecoderConfig(
            vocab_size=arch_cfg["vocab_size"],
            pad_id=arch_cfg["pad_id"],
            bos_id=arch_cfg["bos_id"],
            eos_id=arch_cfg["eos_id"],
            d_model=arch_cfg["d_model"],
            n_heads=arch_cfg["n_heads"],
            n_layers=arch_cfg["n_layers"],
            d_ff=arch_cfg["d_ff"],
            dropout=arch_cfg.get("dropout", 0.1),
            max_seq_len=arch_cfg["max_seq_len"],
            rope_theta=arch_cfg.get("rope_theta", 10000.0),
            tie_weights=arch_cfg.get("tie_weights", True),
        )
        return LaTeXDecoderForCausalLM(hf_cfg)

    @classmethod
    def _original_local_pretrain_ckpt(cls) -> Path | None:
        candidates = [
            Path(r"D:\checkpoints\checkpoints\step_0056000"),
            cls._repo_root() / "checkpoints" / "checkpoints" / "step_0056000",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    @classmethod
    def _resolve_local_decoder_dir(cls, repo: str, config: dict) -> Path | None:
        candidates: list[Path] = []

        local_override = config.get("decoder_ckpt_local")
        if local_override:
            candidates.append(Path(local_override))

        if repo:
            repo_path = Path(repo)
            candidates.append(repo_path)
            if not repo_path.is_absolute():
                candidates.append(cls._repo_root() / repo_path)

        if repo == "harryrobert/pretrain-decoder":
            original_ckpt = cls._original_local_pretrain_ckpt()
            if original_ckpt is not None:
                candidates.append(original_ckpt)

        seen: set[Path] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.exists():
                return resolved
        return None

    @staticmethod
    def _build_local_hf_config(train_info: dict | None = None, state: dict | None = None) -> LaTeXDecoderConfig:
        cfg = DecoderConfig()
        embed = (train_info or {}).get("embed", {})
        arch = (train_info or {}).get("arch", {})

        vocab_size = embed.get("vocab_size", cfg.vocab_size)
        d_model = arch.get("d_model", cfg.d_model)
        d_ff = arch.get("d_ff", cfg.d_ff)
        n_layers = arch.get("n_layers", cfg.n_layers)
        n_heads = arch.get("n_heads", cfg.n_heads)

        if state is not None:
            embed_w = state.get("embed_tokens.weight")
            if embed_w is not None:
                vocab_size, d_model = embed_w.shape
            layer_ids = {
                int(k.split(".")[1])
                for k in state.keys()
                if k.startswith("layers.") and k.split(".")[1].isdigit()
            }
            if layer_ids:
                n_layers = max(layer_ids) + 1
            gate_w = state.get("layers.0.ffn.gate_proj.weight")
            if gate_w is not None:
                d_ff = gate_w.shape[0]

        return LaTeXDecoderConfig(
            vocab_size=vocab_size,
            pad_id=embed.get("pad_id", cfg.pad_id),
            bos_id=embed.get("bos_id", cfg.bos_id),
            eos_id=embed.get("eos_id", cfg.eos_id),
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=arch.get("dropout", cfg.dropout),
            max_seq_len=arch.get("max_seq_len", cfg.max_seq_len),
            rope_theta=arch.get("rope_theta", cfg.rope_theta),
            tie_weights=embed.get("tie_weights", cfg.tie_weights),
        )

    @classmethod
    def _load_raw_local_checkpoint(cls, ckpt_dir: Path) -> LaTeXDecoderForCausalLM:
        model_path = ckpt_dir / "model.safetensors"
        if not model_path.exists():
            raise FileNotFoundError(f"Raw decoder checkpoint not found: {model_path}")

        state = st_load_file(str(model_path))
        train_info_path = ckpt_dir / "train_info.json"
        train_info = None
        if train_info_path.exists():
            train_info = json.loads(train_info_path.read_text(encoding="utf-8"))

        hf_cfg = cls._build_local_hf_config(train_info=train_info, state=state)
        model = LaTeXDecoderForCausalLM(hf_cfg)
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_missing = {"lm_head.weight"}
        if set(missing) - allowed_missing or unexpected:
            raise RuntimeError(
                f"Failed to load raw decoder checkpoint from {ckpt_dir}. "
                f"missing={missing}, unexpected={unexpected}"
            )
        return model

    @classmethod
    def _load_decoder_model(cls, repo: str, config: dict):
        if config.get("decoder_init") == "from_config":
            arch_cfg = config.get("decoder_arch")
            if not arch_cfg:
                raise RuntimeError("decoder_init='from_config' nhưng thiếu decoder_arch trong config.")
            print("[decoder] Building decoder from embedded arch config")
            return cls._build_model_from_arch_config(arch_cfg)

        local_dir = cls._resolve_local_decoder_dir(repo, config)
        if local_dir is not None:
            if (local_dir / "config.json").exists():
                print(f"[decoder] Loading local HF decoder from: {local_dir}")
                return AutoModelForCausalLM.from_pretrained(
                    str(local_dir),
                    trust_remote_code=True,
                    local_files_only=True,
                )
            if (local_dir / "model.safetensors").exists():
                print(f"[decoder] Loading raw local decoder checkpoint from: {local_dir}")
                return cls._load_raw_local_checkpoint(local_dir)

        print(f"[decoder] Loading decoder from local cache only: {repo}")
        try:
            return AutoModelForCausalLM.from_pretrained(
                repo,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as e:
            raise RuntimeError(
                "Decoder local-only load failed. "
                f"Tried local path='{local_dir}' and local cache for '{repo}'. "
                "Set config['decoder_ckpt_local'] to a local decoder checkpoint directory if needed."
            ) from e

    # ── Tie weights ───────────────────────────────────────────────────────────

    def tie_weights(self) -> None:
        """
        Đảm bảo lm_head.weight và embed_tokens.weight là CÙNG tensor object.

        Phải gọi sau mọi load_state_dict() vì safetensors luôn save/load 2
        tensor riêng biệt, phá vỡ tie dù model ban đầu có tie hay không.
        """
        self._model.lm_head.weight = self._model.embed_tokens.weight

    def untie_weights(self) -> None:
        """
        Tách lm_head khỏi embed_tokens để load legacy checkpoint untied.
        """
        if self.are_weights_tied():
            cloned = self._model.embed_tokens.weight.detach().clone()
            self._model.lm_head.weight = nn.Parameter(cloned)

    def are_weights_tied(self) -> bool:
        """Kiểm tra nhanh xem tie có đang được giữ không."""
        return self._model.lm_head.weight.data_ptr() == \
               self._model.embed_tokens.weight.data_ptr()

    # ── QAT ──────────────────────────────────────────────────────────────────

    def _enable_qat(self):
        self._model.train()
        self._model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
        torch.ao.quantization.prepare_qat(self._model, inplace=True)
        print("  [decoder] QAT enabled (fake-quantize on all Linear layers)")

    # ── Getters ───────────────────────────────────────────────────────────────

    def get_input_embeddings(self) -> nn.Embedding:
        return self._model.embed_tokens

    def decoder_linear_weights(self):
        """Iterator over (name, weight) cho L1 sparsity penalty."""
        for name, m in self._model.named_modules():
            if isinstance(m, nn.Linear) and m.weight.requires_grad:
                yield name, m.weight

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        logits = self._forward_embeds(inputs_embeds, attention_mask)

        if labels is None:
            return type("Out", (), {"logits": logits, "loss": None})()

        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_labels == self._pad_id, -100)
        loss = F.cross_entropy(
            shift_logits.view(-1, self._vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return type("Out", (), {"logits": logits, "loss": loss})()

    def _forward_embeds(self, inputs_embeds: torch.Tensor, attention_mask=None) -> torch.Tensor:
        m = self._model
        x = m.embed_drop(inputs_embeds)
        mask = attention_mask.bool() if attention_mask is not None else None
        for layer in m.layers:
            x = layer(x, mask)
        return m.lm_head(m.norm_final(x))

    # ── Generate ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, inputs_embeds, attention_mask, max_new_tokens, num_beams=1,
                 pad_token_id=None, eos_token_id=None):
        eos = eos_token_id if eos_token_id is not None else self.eos_token_id
        pad = pad_token_id if pad_token_id is not None else self.pad_token_id
        return self._beam_search(inputs_embeds, attention_mask, max_new_tokens, num_beams, eos, pad)

    @torch.no_grad()
    def _beam_search(self, inputs_embeds, attention_mask, max_new_tokens, num_beams, eos_id, pad_id):
        """Beam search decode (num_beams=1 = greedy), batch_size=1 only."""
        device  = inputs_embeds.device
        B       = inputs_embeds.shape[0]
        assert B == 1, "beam search only supports batch_size=1"

        vis_emb  = inputs_embeds[0]
        vis_len  = vis_emb.shape[0]
        vis_mask = attention_mask[0] if attention_mask is not None else None

        beams = [(0.0, [], False) for _ in range(num_beams)]

        for _ in range(max_new_tokens):
            all_embeds, all_masks = [], []

            for k, (score, ids, done) in enumerate(beams):
                if ids:
                    tok_emb = self._model.embed_tokens(
                        torch.tensor(ids, device=device, dtype=torch.long)
                    )
                    seq_emb = torch.cat([vis_emb, tok_emb], dim=0)
                else:
                    seq_emb = vis_emb
                all_embeds.append(seq_emb)
                if vis_mask is not None:
                    tok_mask = vis_mask.new_ones(len(ids)) if ids else vis_mask.new_zeros(0)
                    all_masks.append(torch.cat([vis_mask, tok_mask]) if ids else vis_mask)

            max_len = max(e.shape[0] for e in all_embeds)
            D = all_embeds[0].shape[-1]
            padded_embeds = vis_emb.new_zeros(num_beams, max_len, D)
            padded_mask = vis_mask.new_zeros(num_beams, max_len) if vis_mask is not None else None

            for k, emb in enumerate(all_embeds):
                L = emb.shape[0]
                padded_embeds[k, :L] = emb
                if padded_mask is not None:
                    padded_mask[k, :L] = all_masks[k]

            logits = self._forward_embeds(padded_embeds, padded_mask)

            candidates = []
            for k, (score, ids, done) in enumerate(beams):
                last_pos = vis_len + len(ids) - 1
                if done:
                    candidates.append((score, ids, True, k, pad_id))
                    continue
                log_p = torch.log_softmax(logits[k, last_pos, :], dim=-1)
                if len(ids) == 0 and k > 0:
                    log_p = log_p.fill_(-1e9)
                topk_lp, topk_tok = log_p.topk(num_beams)
                for lp, tok in zip(topk_lp.tolist(), topk_tok.tolist()):
                    candidates.append((score + lp, ids + [tok], tok == eos_id, k, tok))

            candidates.sort(key=lambda x: -x[0])
            beams = [(s, ids, done) for s, ids, done, *_ in candidates[:num_beams]]

            if all(done for _, _, done in beams):
                break

        best_score, best_ids, _ = max(beams, key=lambda x: x[0])
        if not best_ids:
            return torch.zeros(B, 0, dtype=torch.long, device=device)
        return torch.tensor(best_ids, dtype=torch.long, device=device).unsqueeze(0)
