r"""
latex_text_aug.py
-----------------
Text-level LaTeX augmentation (Group 1 + Group 2).

Group 1 — Regex/string (fast, no parsing):
  - display mode switching
  - bracket size variation
  - whitespace noise
  - redundant braces
  - \frac ↔ \over
  - symbol equivalences

Group 2 — AST via sympy + latex2sympy2 (semantic):
  - variable renaming
  - environment switching
  - commutativity (simple additive terms)

Usage (as module):
    from latex_text_aug import augment_latex
    new_latex = augment_latex(latex_string, level="light")  # or "heavy"

Usage (standalone test):
    python latex_text_aug.py
"""

import random
import re
from typing import Optional

# ── Optional imports (Group 2) ────────────────────────────────────────────────
try:
    from sympy import latex as sympy_latex
    from sympy.parsing.latex import parse_latex as sympy_parse
    SYMPY_OK = True
except ImportError:
    SYMPY_OK = False

try:
    from latex2sympy2 import latex2sympy
    L2S_OK = True
except ImportError:
    L2S_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 1 — Regex / String
# ══════════════════════════════════════════════════════════════════════════════

# ── Display mode switching ────────────────────────────────────────────────────

_DISPLAY_FORMS = [
    r"\[ {content} \]",
    r"\begin{{equation}} {content} \end{{equation}}",
    r"\begin{{equation*}} {content} \end{{equation*}}",
]

def toggle_display_mode(latex: str, p: float = 0.3) -> str:
    r"""$...$ ↔ \[...\] ↔ \begin{equation}...\end{equation}"""
    if random.random() > p:
        return latex
    # strip existing wrappers
    stripped = latex.strip()
    for pat in [
        r"^\\\[(.+?)\\\]$",
        r"^\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}$",
        r"^\$(.+?)\$$",
    ]:
        m = re.match(pat, stripped, re.DOTALL)
        if m:
            content = m.group(1).strip()
            fmt = random.choice(_DISPLAY_FORMS)
            return fmt.format(content=content)
    return latex


# ── Bracket size variation ────────────────────────────────────────────────────

_BRACKET_SIZES = ["", r"\big", r"\Big", r"\bigg", r"\Bigg"]
_BRACKET_PAIRS = [("(", ")"), ("[", "]"), (r"\{", r"\}")]

_LEFT_RIGHT_RE = re.compile(
    r"\\left\s*([(\[{|])(.*?)\\right\s*([)\]|}])", re.DOTALL
)

def vary_bracket_size(latex: str, p: float = 0.3) -> str:
    if random.random() > p:
        return latex

    # \left( ... \right) → \big( ... \big) etc.
    def replace_bracket(m):
        open_b  = m.group(1)
        content = m.group(2)
        close_b = m.group(3)
        size = random.choice(_BRACKET_SIZES)
        if size == "":
            return f"{open_b}{content}{close_b}"
        return f"{size}l{open_b}{content}{size}r{close_b}"

    return _LEFT_RIGHT_RE.sub(replace_bracket, latex)


def add_left_right(latex: str, p: float = 0.2) -> str:
    r"""Bọc ( ) đơn thuần bằng \left \right"""
    if random.random() > p:
        return latex
    return re.sub(r"(?<!\\)\((.{1,30}?)\)", r"\\left(\1\\right)", latex)


# ── Whitespace noise ──────────────────────────────────────────────────────────

def whitespace_noise(latex: str, p: float = 0.3) -> str:
    """Thêm/bớt spaces vô nghĩa xung quanh operators."""
    if random.random() > p:
        return latex
    ops = ["+", "-", "=", "<", ">"]
    result = latex
    for op in ops:
        if random.random() < 0.4:
            spaces = " " * random.randint(0, 2)
            result = result.replace(op, f"{spaces}{op}{spaces}")
    # collapse multiple spaces
    result = re.sub(r" {3,}", "  ", result)
    return result


# ── Redundant braces ──────────────────────────────────────────────────────────

def add_redundant_braces(latex: str, p: float = 0.25) -> str:
    """{a}+{b} style — bọc single char/digit bằng {}"""
    if random.random() > p:
        return latex

    def wrap_char(m):
        if random.random() < 0.4:
            return "{" + m.group(0) + "}"
        return m.group(0)

    # chỉ wrap char đơn không phải trong command
    return re.sub(r"(?<!\\)(?<!\{)([a-zA-Z0-9])(?!\})", wrap_char, latex)


def add_double_braces(latex: str, p: float = 0.15) -> str:
    """{{expr}} thay vì {expr} ở một số vị trí"""
    if random.random() > p:
        return latex
    return re.sub(r"\{([^{}]{1,10})\}", lambda m: "{{" + m.group(1) + "}}" if random.random() < 0.3 else m.group(0), latex)


# ── \frac ↔ \over ────────────────────────────────────────────────────────────

_FRAC_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")

def frac_to_over(latex: str, p: float = 0.2) -> str:
    if random.random() > p:
        return latex
    return _FRAC_RE.sub(lambda m: "{" + m.group(1) + r" \over " + m.group(2) + "}", latex)


_OVER_RE = re.compile(r"\{([^{}]+)\\over\s*([^{}]+)\}")

def over_to_frac(latex: str, p: float = 0.2) -> str:
    if random.random() > p:
        return latex
    return _OVER_RE.sub(lambda m: r"\frac{" + m.group(1).strip() + "}{" + m.group(2).strip() + "}", latex)


# ── Symbol equivalences ───────────────────────────────────────────────────────

_SYMBOL_MAP = [
    # multiplication
    (r"\\cdot",  [r"\\times", r"\\cdot", r"\*"]),
    (r"\\times", [r"\\times", r"\\cdot"]),
    # comparison
    (r"\\leq",   [r"\\leq", r"\\le"]),
    (r"\\le\b",  [r"\\leq", r"\\le"]),
    (r"\\geq",   [r"\\geq", r"\\ge"]),
    (r"\\ge\b",  [r"\\geq", r"\\ge"]),
    (r"\\neq",   [r"\\neq", r"\\ne", r"\\not="]),
    # bold
    (r"\\mathbf\{([^}]+)\}", None),   # handled separately
    (r"\\boldsymbol\{([^}]+)\}", None),
    # text
    (r"\\text\{([^}]+)\}",    None),
    (r"\\mathrm\{([^}]+)\}",  None),
    # displaystyle
    (r"\\displaystyle\s*", [r"\\displaystyle ", ""]),
    # subset/superset
    (r"\\subset",  [r"\\subset", r"\\subseteq"]),
    (r"\\supset",  [r"\\supset", r"\\supseteq"]),
    # arrows
    (r"\\to\b",      [r"\\to", r"\\rightarrow"]),
    (r"\\rightarrow",[r"\\to", r"\\rightarrow"]),
    (r"\\gets\b",    [r"\\gets", r"\\leftarrow"]),
    (r"\\leftarrow", [r"\\gets", r"\\leftarrow"]),
]


def symbol_equiv(latex: str, p: float = 0.3) -> str:
    if random.random() > p:
        return latex
    result = latex
    for pattern, alts in _SYMBOL_MAP:
        if alts is None:
            continue
        if re.search(pattern, result):
            if random.random() < 0.4:
                replacement = random.choice(alts)
                result = re.sub(pattern, replacement, result, count=1)

    # mathbf ↔ boldsymbol
    if random.random() < 0.3:
        result = re.sub(r"\\mathbf\{([^}]+)\}", r"\\boldsymbol{\1}", result)
    elif random.random() < 0.3:
        result = re.sub(r"\\boldsymbol\{([^}]+)\}", r"\\mathbf{\1}", result)

    # text ↔ mathrm
    if random.random() < 0.3:
        result = re.sub(r"\\text\{([^}]+)\}", r"\\mathrm{\1}", result)
    elif random.random() < 0.3:
        result = re.sub(r"\\mathrm\{([^}]+)\}", r"\\text{\1}", result)

    return result


# ── Subscript braces noise ────────────────────────────────────────────────────

def subscript_brace_noise(latex: str, p: float = 0.2) -> str:
    """x_i ↔ x_{i} (single char subscript/superscript)"""
    if random.random() > p:
        return latex
    # x_i → x_{i}
    if random.random() < 0.5:
        return re.sub(r"([_^])([a-zA-Z0-9])(?!\})", r"\1{\2}", latex)
    # x_{i} → x_i
    return re.sub(r"([_^])\{([a-zA-Z0-9])\}", r"\1\2", latex)


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 2 — AST-based (pylatexenc + sympy)
# ══════════════════════════════════════════════════════════════════════════════

# ── Variable renaming ─────────────────────────────────────────────────────────

_LATIN_VARS   = list("abcdefghijklmnopqrstuvwxyz")
_GREEK_VARS   = [r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon",
                 r"\zeta", r"\eta", r"\theta", r"\lambda", r"\mu",
                 r"\nu", r"\xi", r"\rho", r"\sigma", r"\tau",
                 r"\phi", r"\psi", r"\omega"]
_INDEXED_VARS = [f"x_{i}" for i in range(1, 6)] + \
                [f"a_{{n}}", f"b_{{k}}", f"c_{{m}}"]

_ALL_VARS = _LATIN_VARS + _GREEK_VARS + _INDEXED_VARS


def rename_variables(latex: str, p: float = 0.25) -> str:
    """Thay thế 1-2 biến đơn lẻ bằng biến khác."""
    if random.random() > p:
        return latex

    # Tìm các biến đơn lẻ (chữ cái không nằm trong command)
    single_vars = re.findall(r"(?<!\\)\b([a-z])\b", latex)
    if not single_vars:
        return latex

    unique_vars = list(set(single_vars))
    n_replace   = random.randint(1, min(2, len(unique_vars)))
    to_replace  = random.sample(unique_vars, n_replace)

    result = latex
    for var in to_replace:
        new_var = random.choice(_ALL_VARS)
        if new_var == var:
            continue
        # chỉ replace word boundary để tránh replace trong command
        # dùng lambda để tránh re interpret backslash trong new_var (vd \alpha)
        result = re.sub(rf"(?<!\\)\b{var}\b", lambda _, r=new_var: r, result)

    return result


# ── Environment switching ─────────────────────────────────────────────────────

_ENV_EQUIV = {
    "equation":  ["equation", "equation*"],
    "equation*": ["equation", "equation*"],
    "align":     ["align", "align*", "gather", "gather*"],
    "align*":    ["align", "align*", "gather", "gather*"],
    "gather":    ["align", "align*", "gather", "gather*"],
    "gather*":   ["align", "align*", "gather", "gather*"],
}

_ENV_RE = re.compile(r"\\begin\{(\w+\*?)\}(.*?)\\end\{\1\}", re.DOTALL)


def switch_environment(latex: str, p: float = 0.2) -> str:
    if random.random() > p:
        return latex

    def replace_env(m):
        env     = m.group(1)
        content = m.group(2)
        alts    = _ENV_EQUIV.get(env)
        if not alts:
            return m.group(0)
        new_env = random.choice(alts)
        return f"\\begin{{{new_env}}}{content}\\end{{{new_env}}}"

    return _ENV_RE.sub(replace_env, latex)


# ── Commutativity (simple) ────────────────────────────────────────────────────

def commute_addition(latex: str, p: float = 0.2) -> str:
    """Đảo thứ tự các terms trong phép cộng đơn giản: a+b → b+a"""
    if random.random() > p:
        return latex

    # Chỉ xử lý pattern đơn giản: token + token
    # token = chữ cái/số/command đơn giản
    _TOKEN = r"(?:\\[a-zA-Z]+(?:\{[^{}]*\})*|[a-zA-Z0-9]+)"
    pattern = rf"({_TOKEN})\s*\+\s*({_TOKEN})"

    def swap(m):
        if random.random() < 0.5:
            return f"{m.group(2)} + {m.group(1)}"
        return m.group(0)

    return re.sub(pattern, swap, latex)


def commute_multiplication(latex: str, p: float = 0.15) -> str:
    r"""a \cdot b → b \cdot a"""
    if random.random() > p:
        return latex

    _TOKEN = r"(?:\\[a-zA-Z]+(?:\{[^{}]*\})*|[a-zA-Z0-9]+)"
    pattern = rf"({_TOKEN})\s*\\cdot\s*({_TOKEN})"

    def swap(m):
        if random.random() < 0.5:
            return f"{m.group(2)} \\cdot {m.group(1)}"
        return m.group(0)

    return re.sub(pattern, swap, latex)


# ── Sympy-based roundtrip (nếu có thư viện) ──────────────────────────────────

def sympy_roundtrip(latex: str, p: float = 0.15) -> str:
    """Parse LaTeX → SymPy → back to LaTeX (normalize + vary formatting)"""
    if not SYMPY_OK or random.random() > p:
        return latex
    try:
        expr   = sympy_parse(latex)
        result = sympy_latex(expr)
        return result if result.strip() else latex
    except Exception:
        return latex


def l2s_roundtrip(latex: str, p: float = 0.15) -> str:
    """latex2sympy2 roundtrip"""
    if not L2S_OK or random.random() > p:
        return latex
    try:
        expr   = latex2sympy(latex)
        result = sympy_latex(expr)
        return result if result.strip() else latex
    except Exception:
        return latex


# ══════════════════════════════════════════════════════════════════════════════
# Combined pipelines
# ══════════════════════════════════════════════════════════════════════════════

_LIGHT_FNS = [
    whitespace_noise,
    subscript_brace_noise,
    symbol_equiv,
    over_to_frac,
    frac_to_over,
]

_HEAVY_FNS = [
    whitespace_noise,
    subscript_brace_noise,
    symbol_equiv,
    over_to_frac,
    frac_to_over,
    add_redundant_braces,
    add_double_braces,
    vary_bracket_size,
    add_left_right,
    toggle_display_mode,
    rename_variables,
    switch_environment,
    commute_addition,
    commute_multiplication,
    sympy_roundtrip,
]


def augment_latex(latex: str, level: str = "light") -> str:
    """
    Apply random text-level aug lên LaTeX string.
    level: "light" | "heavy"
    """
    if not latex or not latex.strip():
        return latex

    fns = _LIGHT_FNS if level == "light" else _HEAVY_FNS
    result = latex

    # shuffle để không luôn apply cùng thứ tự
    for fn in random.sample(fns, len(fns)):
        result = fn(result)
        if not result.strip():
            return latex  # fallback nếu aug làm rỗng string

    return result


# ── Pipeline: read raw_train → aug text → write shards ───────────────────────

def _run_pipeline():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pathlib import Path
    from tqdm import tqdm

    OUT_DIR   = Path("D:/dataset-ocr-builder/latex-ocr-dataset")
    TRAIN_DIR = OUT_DIR / "train"
    out_dir   = TRAIN_DIR / "light_text"
    prefix    = "light_text_train"

    raw_files = sorted((TRAIN_DIR / "raw").glob("raw_train-*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"No raw_train shards in {TRAIN_DIR / 'raw'}. Run run_split.py first.")

    n_shards = len(raw_files)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n== {prefix} ({n_shards} shards) ==")
    print(f"  output -> {out_dir}\n")
    total = 0

    for i, pfile in enumerate(raw_files):
        table   = pq.read_table(str(pfile))
        latexes = table["latex"].to_pylist()
        images  = table["image"].to_pylist()
        sources = table["source"].to_pylist()
        idxs    = table["idx"].to_pylist()
        del table

        aug_latexes = []
        for lat in tqdm(latexes, desc=f"  shard {i+1}/{n_shards}", ncols=80, leave=False):
            aug_latexes.append(augment_latex(lat, level="light"))

        fname = f"{prefix}-{str(i).zfill(5)}-of-{str(n_shards).zfill(5)}.parquet"
        out_table = pa.table({
            "idx":    pa.array(idxs,        type=pa.int64()),
            "image":  pa.array(images,       type=pa.binary()),
            "latex":  pa.array(aug_latexes,  type=pa.string()),
            "source": pa.array(sources,      type=pa.string()),
        })
        pq.write_table(out_table, str(out_dir / fname), compression="snappy")
        total += len(aug_latexes)
        print(f"  shard {i+1}/{n_shards}: {len(aug_latexes):,} rows -> {out_dir.name}/")

        del images, latexes, sources, idxs, aug_latexes, out_table

    print(f"\n  total {prefix}: {total:,}")


# ── Standalone test / pipeline runner ────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LaTeX text augmentation")
    ap.add_argument("--run", action="store_true",
                    help="Run pipeline: raw_train -> light_text/")
    args = ap.parse_args()

    if args.run:
        _run_pipeline()
    else:
        # Standalone test
        samples = [
            r"\frac{dy}{dt} = y^2 + x",
            r"\sum_{i=0}^{n} x_i^2 + a + b",
            r"\int_0^\infty e^{-x} dx",
            r"E = mc^2",
            r"\left( \frac{a+b}{c} \right)^2",
            r"\begin{align} x &= a + b \\ y &= c + d \end{align}",
            r"\alpha \cdot \beta + \gamma",
        ]

        print("=" * 60)
        print("LIGHT AUG")
        print("=" * 60)
        for s in samples:
            out = augment_latex(s, level="light")
            print(f"  IN : {s}")
            print(f"  OUT: {out}")
            print()

        print("=" * 60)
        print("HEAVY AUG")
        print("=" * 60)
        for s in samples:
            out = augment_latex(s, level="heavy")
            print(f"  IN : {s}")
            print(f"  OUT: {out}")
            print()

        print(f"sympy  available: {SYMPY_OK}")
        print(f"latex2sympy2 available: {L2S_OK}")
