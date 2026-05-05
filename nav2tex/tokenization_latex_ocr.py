import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizer


class LaTeXTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        **kwargs,
    ):
        with open(Path(vocab_file), encoding="utf-8") as f:
            data = json.load(f)

        if "model" in data:
            self.token2id: Dict[str, int] = data["model"]["vocab"]
            self.id2token: Dict[int, str] = {int(v): k for k, v in self.token2id.items()}
            self.merges = []
            cfg = {}
        else:
            self.token2id = data["token2id"]
            self.id2token = {int(k): v for k, v in data["id2token"].items()}
            self.merges = data.get("merges", [])
            cfg = data.get("config", {})

        kwargs.setdefault("model_max_length", cfg.get("model_max_length", 256))
        kwargs.setdefault("padding_side", cfg.get("padding_side", "right"))
        kwargs.setdefault("truncation_side", cfg.get("truncation_side", "right"))

        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.token2id)

    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for length in range(min(20, len(text) - i), 0, -1):
                substr = text[i:i + length]
                if substr in self.token2id:
                    tokens.append(substr)
                    i += length
                    matched = True
                    break
            if not matched:
                tokens.append(text[i])
                i += 1
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.token2id.get(token, self.token2id.get("<unk>", 1))

    def _convert_id_to_token(self, index: int) -> str:
        return self.id2token.get(index, "<unk>")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        vocab_file = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + "tokenizer.json"
        )
        data = {
            "token2id": self.token2id,
            "id2token": {str(k): v for k, v in self.id2token.items()},
            "merges": [list(p) for p in self.merges],
            "config": {
                "vocab_size": self.vocab_size,
                "pad_token": str(self.pad_token),
                "unk_token": str(self.unk_token),
                "bos_token": str(self.bos_token),
                "eos_token": str(self.eos_token),
                "pad_id": self.pad_token_id,
                "unk_id": self.unk_token_id,
                "bos_id": self.bos_token_id,
                "eos_id": self.eos_token_id,
                "model_max_length": self.model_max_length,
                "padding_side": self.padding_side,
                "truncation_side": self.truncation_side,
            },
        }
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return (str(vocab_file),)