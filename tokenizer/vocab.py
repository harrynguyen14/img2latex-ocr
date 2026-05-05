"""
Vocabulary definition cho tokenizer v2.

Layout ID:
  0        : <pad>
  1        : <unk>
  2        : <bos>
  3        : <eos>
  4 ..     : FROZEN_COMMANDS  (LaTeX commands)
  ..       : FROZEN_SYMBOLS   (brackets, operators, punctuation)
  ..       : FROZEN_CHARS     (digits, a-z, A-Z, Greek unicode)
  ..  8191 : BPE learned merges (subword cho variable/text)
"""

# ── Special tokens ────────────────────────────────────────────────────────────
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

VOCAB_SIZE = 2048

# ── Frozen LaTeX commands ─────────────────────────────────────────────────────
# Mỗi token này sẽ được gán ID cố định và KHÔNG BAO GIỜ bị BPE merge/split.
FROZEN_COMMANDS: list[str] = [
    # Fraction / root
    r"\frac", r"\dfrac", r"\tfrac", r"\cfrac", r"\over",
    r"\sqrt", r"\root",

    # Binary operators
    r"\cdot", r"\times", r"\div", r"\pm", r"\mp",
    r"\oplus", r"\ominus", r"\otimes", r"\oslash", r"\odot",
    r"\circ", r"\ast", r"\star", r"\bullet",
    r"\cup", r"\cap", r"\sqcup", r"\sqcap",
    r"\wedge", r"\vee", r"\neg",
    r"\setminus", r"\smallsetminus",

    # Relations
    r"\leq", r"\geq", r"\le", r"\ge",
    r"\neq", r"\ne", r"\approx", r"\simeq", r"\sim",
    r"\equiv", r"\cong", r"\propto", r"\doteq",
    r"\prec", r"\succ", r"\preceq", r"\succeq",
    r"\ll", r"\gg", r"\lll", r"\ggg",
    r"\subset", r"\supset", r"\subseteq", r"\supseteq",
    r"\subsetneq", r"\supsetneq",
    r"\in", r"\notin", r"\ni",
    r"\perp", r"\parallel", r"\mid", r"\nmid",

    # Logic / quantifiers
    r"\forall", r"\exists", r"\nexists",
    r"\therefore", r"\because",

    # Calculus / analysis
    r"\nabla", r"\partial", r"\infty",
    r"\sum", r"\prod", r"\coprod",
    r"\int", r"\oint", r"\iint", r"\iiint", r"\idotsint",
    r"\lim", r"\sup", r"\inf", r"\max", r"\min",
    r"\limsup", r"\liminf",
    r"\limits", r"\nolimits",

    # Greek lowercase
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon",
    r"\varepsilon", r"\zeta", r"\eta", r"\theta", r"\vartheta",
    r"\iota", r"\kappa", r"\lambda", r"\mu", r"\nu", r"\xi",
    r"\pi", r"\varpi", r"\rho", r"\varrho",
    r"\sigma", r"\varsigma", r"\tau", r"\upsilon",
    r"\phi", r"\varphi", r"\chi", r"\psi", r"\omega",

    # Greek uppercase
    r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Xi",
    r"\Pi", r"\Sigma", r"\Upsilon", r"\Phi", r"\Psi", r"\Omega",

    # Arrows
    r"\to", r"\gets",
    r"\rightarrow", r"\leftarrow", r"\leftrightarrow",
    r"\Rightarrow", r"\Leftarrow", r"\Leftrightarrow",
    r"\longrightarrow", r"\longleftarrow", r"\longleftrightarrow",
    r"\Longrightarrow", r"\Longleftarrow", r"\Longleftrightarrow",
    r"\mapsto", r"\longmapsto",
    r"\uparrow", r"\downarrow", r"\updownarrow",
    r"\Uparrow", r"\Downarrow", r"\Updownarrow",
    r"\nearrow", r"\searrow", r"\nwarrow", r"\swarrow",
    r"\hookrightarrow", r"\hookleftarrow",
    r"\rightharpoonup", r"\leftharpoonup",
    r"\rightleftharpoons",

    # Delimiters (commands)
    r"\left", r"\right",
    r"\big", r"\Big", r"\bigg", r"\Bigg",
    r"\bigl", r"\bigr", r"\Bigl", r"\Bigr",
    r"\biggl", r"\biggr", r"\Biggl", r"\Biggr",
    r"\langle", r"\rangle",
    r"\lfloor", r"\rfloor", r"\lceil", r"\rceil",
    r"\lvert", r"\rvert", r"\lVert", r"\rVert",
    r"\vert", r"\Vert",

    # Accents / decorators
    r"\hat", r"\tilde", r"\bar", r"\vec",
    r"\dot", r"\ddot", r"\dddot",
    r"\acute", r"\grave", r"\breve", r"\check",
    r"\overline", r"\underline", r"\widehat", r"\widetilde",
    r"\overrightarrow", r"\overleftarrow",
    r"\overbrace", r"\underbrace",
    r"\overset", r"\underset",
    r"\stackrel",

    # Font / style
    r"\mathbb", r"\mathcal", r"\mathbf", r"\mathrm", r"\mathit",
    r"\mathsf", r"\mathtt", r"\mathfrak", r"\mathscr",
    r"\boldsymbol", r"\bm",
    r"\bf", r"\rm", r"\it", r"\cal", r"\sf", r"\tt",
    r"\text", r"\mbox", r"\hbox",
    r"\mathnormal",

    # Environments
    r"\begin", r"\end",

    # Spacing
    r"\quad", r"\qquad",
    r"\,", r"\:", r"\;", r"\!", r"\ ",
    r"\hspace", r"\vspace",

    # Trig / log functions
    r"\sin", r"\cos", r"\tan", r"\cot", r"\sec", r"\csc",
    r"\arcsin", r"\arccos", r"\arctan",
    r"\sinh", r"\cosh", r"\tanh", r"\coth",
    r"\log", r"\ln", r"\lg", r"\exp",
    r"\det", r"\dim", r"\ker", r"\deg",
    r"\gcd", r"\lcm", r"\hom",
    r"\arg", r"\Re", r"\Im",

    # Misc symbols
    r"\hbar", r"\ell", r"\wp", r"\aleph", r"\beth",
    r"\prime", r"\emptyset", r"\varnothing",
    r"\angle", r"\triangle", r"\square", r"\diamond",
    r"\dagger", r"\ddagger",
    r"\clubsuit", r"\diamondsuit", r"\heartsuit", r"\spadesuit",
    r"\cdots", r"\ldots", r"\dots", r"\vdots", r"\ddots",
    r"\hdots",

    # Matrix / array helpers
    r"\hline", r"\cline",
    r"\multicolumn", r"\multirow",

    # Misc formatting
    r"\mathop", r"\operatorname",
    r"\not",
    r"\binom", r"\dbinom", r"\tbinom",
    r"\pmod", r"\pod", r"\mod",
    r"\choose",
    r"\displaystyle", r"\textstyle", r"\scriptstyle", r"\scriptscriptstyle",
    r"\color", r"\textcolor",
    r"\boxed",
    r"\label", r"\tag", r"\ref",
    r"\nonumber", r"\notag",
    r"\usepackage", r"\newcommand", r"\renewcommand",
    r"\DeclareMathOperator",

    # Text font commands
    r"\textrm", r"\textbf", r"\textit", r"\texttt", r"\textsf",
    r"\textup", r"\textsc", r"\textsl",
    r"\boldmath", r"\unboldmath",

    # Big operators
    r"\bigcup", r"\bigcap", r"\bigsqcup",
    r"\bigvee", r"\bigwedge",
    r"\bigoplus", r"\bigotimes", r"\bigodot",
    r"\biguplus",

    # Extra relations / operators
    r"\top", r"\bot",
    r"\lesssim", r"\gtrsim", r"\lessgtr", r"\gtrless",
    r"\leqslant", r"\geqslant",
    r"\preccurlyeq", r"\succcurlyeq",
    r"\lnsim", r"\gnsim",
    r"\vdash", r"\dashv", r"\models",
    r"\backslash",
    r"\lbrack", r"\rbrack",

    # Misc
    r"\substack", r"\phantom", r"\vphantom", r"\hphantom",
    r"\smash", r"\mathstrut",
    r"\dag", r"\ddag",
    r"\Pr", r"\tr",
    r"\imath", r"\jmath",
    r"\complement", r"\eth",
    r"\hslash", r"\mho",

    # Single-letter accent/special commands (LaTeX chuẩn)
    r"\l", r"\L",   # ł Ł
    r"\d",          # dot-under accent
    r"\c",          # cedilla accent
    r"\t",          # tie-after accent
    r"\i",          # dotless i
    r"\j",          # dotless j
    r"\P",          # pilcrow
    r"\S",          # section sign
    r"\o", r"\O",   # ø Ø
    r"\ae", r"\AE", # æ Æ
    r"\oe", r"\OE", # œ Œ
    r"\aa", r"\AA", # å Å
    r"\ss",         # ß
    # Single-letter commands từ dataset
    r"\a", r"\b", r"\e", r"\g", r"\h", r"\k",
    r"\n", r"\q", r"\r", r"\s", r"\u", r"\v",
    r"\x", r"\y", r"\z",
    r"\B", r"\D", r"\T", r"\X",

    # Escaped punctuation
    r"\#", r"\%", r"\_",

    # Extra math symbols
    r"\atop",
    r"\colon",
    r"\triangleq",
    r"\operatorname*",
    r"\lbrace", r"\rbrace",
    r"\sharp", r"\flat", r"\natural",
    r"\pmb",
    r"\intercal",
    r"\iff", r"\implies",
    r"\Box", r"\blacksquare",
    r"\boxtimes", r"\boxplus",
    r"\varPsi", r"\varPhi", r"\varSigma", r"\varOmega",
    r"\varGamma", r"\varDelta", r"\varTheta", r"\varLambda",
    r"\varXi", r"\varPi", r"\varUpsilon",
    r"\middle",
    r"\eqref",
    r"\dotsc", r"\dotsb", r"\dotsi", r"\dotsm",

    # Size/font commands thường gặp trong OCR dataset
    r"\scriptsize", r"\small", r"\normalsize",
    r"\large", r"\Large", r"\LARGE", r"\huge", r"\Huge",
    r"\tiny", r"\footnotesize",
    r"\textnormal",
    r"\cr",

    # Spacing
    r"\thinspace", r"\enspace", r"\medskip",
    r"\kern", r"\hskip", r"\mskip",
    r"\intertext",

    # Misc
    r"\dx",         # custom differential (phổ biến trong dataset OCR)
    r"\pounds",
]

# ── Frozen environment names ──────────────────────────────────────────────────
# Dùng trong \begin{env} và \end{env} — phải là atomic token, không BPE.
FROZEN_ENV_NAMES: list[str] = [
    # Math environments
    "equation", "equation*",
    "align", "align*",
    "aligned", "alignat", "alignat*",
    "gather", "gather*", "gathered",
    "multline", "multline*",
    "split",
    "flalign", "flalign*",

    # Matrix environments
    "matrix", "pmatrix", "bmatrix", "Bmatrix",
    "vmatrix", "Vmatrix", "smallmatrix",

    # Array / tabular
    "array", "tabular",

    # Cases
    "cases", "rcases", "dcases",

    # Misc math
    "subequations",
    "math", "displaymath",

    # Document structure (xuất hiện trong dataset OCR)
    "document",
    "figure", "table",
    "center", "flushleft", "flushright",
    "itemize", "enumerate", "description",
    "theorem", "proof", "lemma", "corollary",
    "abstract",
]

# ── Frozen symbols / punctuation ──────────────────────────────────────────────
FROZEN_SYMBOLS: list[str] = [
    # Brackets
    "{", "}", "(", ")", "[", "]",
    r"\{", r"\}",
    "|",

    # Math operators (single char)
    "+", "-", "*", "/", "=",
    "<", ">",
    "^", "_",
    "~",

    # Punctuation
    ".", ",", ";", ":", "!",
    "'", "`",
    "@", "#", "%", "&", "$",
    "\\",     # single backslash (noise trong dataset)
    "\\\\",   # line break in tabular/matrix
    r"\|",    # double vert

    # Whitespace token
    " ",
]

# ── Frozen base characters (BPE sẽ merge từ đây) ─────────────────────────────
FROZEN_BASE_CHARS: list[str] = (
    [str(d) for d in range(10)]                        # 0-9
    + [chr(c) for c in range(ord("a"), ord("z") + 1)] # a-z
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)] # A-Z
    # Greek unicode
    + [chr(c) for c in range(0x0391, 0x03AA)]          # Α-Ω
    + [chr(c) for c in range(0x03B1, 0x03CA)]          # α-ω
    # Unicode punctuation / math thường xuất hiện trong dataset OCR
    + [
        "\u2212",  # − minus sign
        "\u2013",  # – en dash
        "\u2014",  # — em dash
        "\u2010",  # ‐ hyphen
        "\u00b7",  # · middle dot
        "\u00d7",  # × multiplication sign
        "\u00f7",  # ÷ division sign
        "\u00b1",  # ± plus-minus
        "\u2032",  # ′ prime
        "\u2033",  # ″ double prime
        "\u2026",  # … ellipsis
        "\u221e",  # ∞ infinity
        "\u2202",  # ∂ partial
        "\u2207",  # ∇ nabla
        "\u2211",  # ∑ summation
        "\u222b",  # ∫ integral
        "\u2248",  # ≈ approx
        "\u2260",  # ≠ not equal
        "\u2264",  # ≤
        "\u2265",  # ≥
        "\u2208",  # ∈
        "\u2209",  # ∉
        "\u2282",  # ⊂
        "\u2283",  # ⊃
        "\u222a",  # ∪
        "\u2229",  # ∩
        "\u2200",  # ∀
        "\u2203",  # ∃
        "\u2192",  # →
        "\u21d2",  # ⇒
        "\u21d4",  # ⇔
        "\u00b0",  # ° degree
        "\u2061",  # function application (invisible)
    ]
)

# ── Toàn bộ frozen tokens theo thứ tự (thứ tự = ID) ──────────────────────────
ALL_FROZEN_TOKENS: list[str] = (
    SPECIAL_TOKENS
    + FROZEN_COMMANDS
    + FROZEN_ENV_NAMES
    + FROZEN_SYMBOLS
    + FROZEN_BASE_CHARS
)

# Số ID bắt đầu của phần BPE learned
N_FROZEN = len(ALL_FROZEN_TOKENS)
N_BPE_SLOTS = VOCAB_SIZE - N_FROZEN   # slot còn lại cho BPE merges

assert N_FROZEN < VOCAB_SIZE, (
    f"Frozen tokens ({N_FROZEN}) vượt quá vocab_size ({VOCAB_SIZE}). "
    f"Tăng VOCAB_SIZE hoặc giảm frozen list."
)

# ── Token config cho save/load ────────────────────────────────────────────────
TOKENIZER_CONFIG = {
    "vocab_size":       VOCAB_SIZE,
    "n_frozen":         N_FROZEN,
    "special_tokens":   SPECIAL_TOKENS,
    "pad_token":        "<pad>",
    "unk_token":        "<unk>",
    "bos_token":        "<bos>",
    "eos_token":        "<eos>",
    "pad_id":           PAD_ID,
    "unk_id":           UNK_ID,
    "bos_id":           BOS_ID,
    "eos_id":           EOS_ID,
    "model_max_length": 256,
    "padding_side":     "right",
    "truncation_side":  "right",
    "tokenizer_version": 2,
}

# ── Dataset paths ─────────────────────────────────────────────────────────────
from pathlib import Path

DATASET_CONFIGS = {
    "raw": {
        "path": Path("D:/dataset-ocr-builder/latex-ocr-dataset/train_filtered/raw"),
        "col":  "latex",
        "ratio": 1.0,
    },
    "light_text": {
        "path": Path("D:/dataset-ocr-builder/latex-ocr-dataset/train_filtered/light_text"),
        "col":  "latex",
        "ratio": 1.0,
    },
    "heavy_text": {
        "path": Path("D:/dataset-ocr-builder/latex-ocr-dataset/train_filtered/heavy_text"),
        "col":  "latex",
        "ratio": 1.0,
    },
}
