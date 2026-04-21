"""
LaTeX Lexer v2.

Token types (theo thứ tự ưu tiên trong regex):
  LINEBREAK   — \\\\ (matrix/tabular row separator)
  COMMAND     — \\cmd hoặc \\cmd* (e.g. \\frac, \\begin*)
  ESCAPED     — \\<single non-alpha non-space> (e.g. \\{, \\,, \\.)
  LBRACE/RBRACE/LPAREN/RPAREN/LBRACKET/RBRACKET
  PIPE        — |
  CARET / UNDERSCORE / TILDE / AMPERSAND / HASH
  LINEBREAK is handled before COMMAND to avoid \\[a-zA-Z] matching \\\\
  EQUALS / PLUS / MINUS / STAR / SLASH / LT / GT
  NUMBER      — integer hoặc decimal liên tục: 3, 3.14, 1000  (atomic)
  GREEK_UC    — Greek uppercase unicode Α-Ω
  GREEK_LC    — Greek lowercase unicode α-ω
  LETTER      — single a-z A-Z (BPE candidate)
  SPACE       — \\s+ → chuẩn hóa thành ' '
  OTHER       — ký tự còn lại

Đặc biệt:
  ENV_NAME    — không phải regex token; được inject bởi post_process()
                khi context là \\begin{...} hoặc \\end{...}

Frozen types: COMMAND, ESCAPED, LINEBREAK, tất cả brackets/symbols, SPACE, ENV_NAME
BPE types:    NUMBER, LETTER, GREEK_UC, GREEK_LC, OTHER
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum, auto

import sys
from pathlib import Path
_here = str(Path(__file__).parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from vocab_v2 import FROZEN_ENV_NAMES as _FROZEN_ENV_SET

_ENV_SET: frozenset[str] = frozenset(_FROZEN_ENV_SET)


class TT(Enum):
    COMMAND    = auto()
    ESCAPED    = auto()
    LINEBREAK  = auto()
    LBRACE     = auto()
    RBRACE     = auto()
    LPAREN     = auto()
    RPAREN     = auto()
    LBRACKET   = auto()
    RBRACKET   = auto()
    PIPE       = auto()
    CARET      = auto()
    UNDERSCORE = auto()
    TILDE      = auto()
    AMPERSAND  = auto()
    HASH       = auto()
    EQUALS     = auto()
    PLUS       = auto()
    MINUS      = auto()
    STAR       = auto()
    SLASH      = auto()
    LT         = auto()
    GT         = auto()
    NUMBER     = auto()   # numeric literal: 3, 3.14, .5
    GREEK_UC   = auto()
    GREEK_LC   = auto()
    LETTER     = auto()
    ENV_NAME   = auto()   # injected post-lex for \begin{env}/\end{env}
    SPACE      = auto()
    OTHER      = auto()


FROZEN_TYPES: frozenset[TT] = frozenset({
    TT.COMMAND, TT.ESCAPED, TT.LINEBREAK,
    TT.LBRACE, TT.RBRACE, TT.LPAREN, TT.RPAREN,
    TT.LBRACKET, TT.RBRACKET, TT.PIPE,
    TT.CARET, TT.UNDERSCORE, TT.TILDE,
    TT.AMPERSAND, TT.HASH,
    TT.EQUALS, TT.PLUS, TT.MINUS, TT.STAR, TT.SLASH,
    TT.LT, TT.GT,
    TT.NUMBER,     # numeric literal là atomic, không BPE
    TT.SPACE,
    TT.ENV_NAME,
})

BPE_TYPES: frozenset[TT] = frozenset({
    TT.LETTER, TT.GREEK_UC, TT.GREEK_LC, TT.OTHER,
})


@dataclass(slots=True)
class Token:
    text:  str
    ttype: TT

    @property
    def is_frozen(self) -> bool:
        return self.ttype in FROZEN_TYPES

    @property
    def is_bpe_eligible(self) -> bool:
        return self.ttype in BPE_TYPES


# ── Master regex ──────────────────────────────────────────────────────────────
# Thứ tự quan trọng: LINEBREAK trước COMMAND, NUMBER trước LETTER
_PATTERN = re.compile(
    r"(?P<LINEBREAK>\\\\)"
    r"|(?P<COMMAND>\\[a-zA-Z]+\*?)"
    r"|(?P<ESCAPED>\\[^a-zA-Z\s])"
    r"|(?P<LBRACE>\{)"
    r"|(?P<RBRACE>\})"
    r"|(?P<LPAREN>\()"
    r"|(?P<RPAREN>\))"
    r"|(?P<LBRACKET>\[)"
    r"|(?P<RBRACKET>\])"
    r"|(?P<PIPE>\|)"
    r"|(?P<CARET>\^)"
    r"|(?P<UNDERSCORE>_)"
    r"|(?P<TILDE>~)"
    r"|(?P<AMPERSAND>&)"
    r"|(?P<HASH>\#)"
    r"|(?P<EQUALS>=)"
    r"|(?P<PLUS>\+)"
    r"|(?P<MINUS>-)"
    r"|(?P<STAR>\*)"
    r"|(?P<SLASH>/)"
    r"|(?P<LT><)"
    r"|(?P<GT>>)"
    r"|(?P<NUMBER>\d+\.?\d*|\.\d+)"   # 42, 3.14, .5
    r"|(?P<GREEK_UC>[\u0391-\u03A9])"
    r"|(?P<GREEK_LC>[\u03B1-\u03C9])"
    r"|(?P<LETTER>[a-zA-Z])"
    r"|(?P<SPACE>\s+)"
    r"|(?P<OTHER>[^\s])",
    re.UNICODE,
)

_TT_MAP: dict[str, TT] = {name: TT[name] for name in TT.__members__}


def _raw_tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    for m in _PATTERN.finditer(text):
        ttype_name = m.lastgroup
        raw = m.group()
        tt = _TT_MAP[ttype_name]
        text_val = " " if tt == TT.SPACE else raw
        tokens.append(Token(text=text_val, ttype=tt))
    return tokens


def _inject_env_names(tokens: list[Token]) -> list[Token]:
    """
    Post-process: deteksi pattern CMD({begin|end}) LBRACE env_name RBRACE
    roi gop cac LETTER token giua {} thanh mot ENV_NAME token neu no nam
    trong FROZEN_ENV_NAMES. Nếu không nằm trong danh sách → giữ nguyên (BPE).

    Pattern duoc nhan dang:
      COMMAND(begin|end)  LBRACE  [SPACE?] LETTER+ [STAR?] [SPACE?]  RBRACE
    """
    out: list[Token] = []
    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]

        # Nhận dạng \begin hoặc \end
        if tok.ttype == TT.COMMAND and tok.text in (r"\begin", r"\end"):
            out.append(tok)
            i += 1

            # Bỏ qua space tùy chọn
            if i < n and tokens[i].ttype == TT.SPACE:
                out.append(tokens[i]); i += 1

            # Phải là LBRACE
            if i < n and tokens[i].ttype == TT.LBRACE:
                out.append(tokens[i]); i += 1

                # Thu thập body bên trong {}
                body_tokens: list[Token] = []
                j = i
                while j < n and tokens[j].ttype not in (TT.RBRACE, TT.LBRACE):
                    body_tokens.append(tokens[j])
                    j += 1

                # Reconstruct env name text (bỏ space 2 đầu)
                env_text = "".join(t.text for t in body_tokens).strip()

                if env_text in _ENV_SET:
                    # Frozen env name — emit một ENV_NAME token duy nhất
                    # Giữ lại space prefix/suffix nếu có
                    prefix = [t for t in body_tokens if t.ttype == TT.SPACE
                               and body_tokens.index(t) == 0]
                    suffix = [t for t in body_tokens if t.ttype == TT.SPACE
                               and body_tokens.index(t) == len(body_tokens) - 1]
                    for t in prefix:
                        out.append(t)
                    out.append(Token(text=env_text, ttype=TT.ENV_NAME))
                    for t in suffix:
                        out.append(t)
                else:
                    # Unknown env → giữ nguyên từng token (BPE sẽ xử lý)
                    out.extend(body_tokens)

                i = j
                # RBRACE
                if i < n and tokens[i].ttype == TT.RBRACE:
                    out.append(tokens[i]); i += 1
            # Không có LBRACE → tiếp tục bình thường
        else:
            out.append(tok)
            i += 1

    return out


def tokenize(text: str) -> list[Token]:
    """Lexe LaTeX string → list[Token] với ENV_NAME injection."""
    raw = _raw_tokenize(text)
    return _inject_env_names(raw)


def tokenize_to_strings(text: str) -> list[str]:
    return [t.text for t in tokenize(text)]


def split_bpe_zones(tokens: list[Token]) -> list[list[str]]:
    """
    Tách stream thành các zone BPE-eligible liên tiếp.
    Frozen token kết thúc zone hiện tại — đảm bảo BPE không cross boundary.
    """
    zones: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if tok.is_bpe_eligible:
            current.append(tok.text)
        else:
            if current:
                zones.append(current)
                current = []
    if current:
        zones.append(current)
    return zones


# ── Debug helper ──────────────────────────────────────────────────────────────
def print_tokens(text: str) -> None:
    toks = tokenize(text)
    print(f"Input: {text!r}")
    for t in toks:
        mark = "F" if t.is_frozen else "B"
        print(f"  [{mark}] {t.ttype.name:<12}  {t.text!r}")
    print(f"  → {len(toks)} tokens")


if __name__ == "__main__":
    tests = [
        r"\frac{dy}{dt}",
        r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",
        r"\begin{align*} x &= 1 \\ y &= 2 \end{align*}",
        r"\int_0^\infty e^{-x^2}\,dx",
        r"\sqrt[3]{x^2 + 1}",
        r"3.14 \times 10^{-9}",
        r"\frac{1}{2} + 0.5 = 1.0",
        r"\begin{unknown_env} x \end{unknown_env}",
    ]
    for ex in tests:
        print()
        print_tokens(ex)
