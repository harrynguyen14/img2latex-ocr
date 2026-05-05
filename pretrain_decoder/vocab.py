import re

VOCAB_SIZE     = 8192
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

LATEX_TOKEN_RE = re.compile(
    r"\\[a-zA-Z]+"
    r"|\\."
    r"|\d"
    r"|[a-zA-Z]"
    r"|[^\s]"
)


def pretokenize(text: str) -> list[str]:
    return LATEX_TOKEN_RE.findall(text)


TOP_LATEX_COMMANDS = [
    r"\frac", r"\over", r"\sqrt", r"\cdot", r"\times", r"\div",
    r"\pm", r"\mp", r"\oplus", r"\otimes", r"\circ", r"\ast",
    r"\leq", r"\geq", r"\le", r"\ge", r"\neq", r"\approx",
    r"\equiv", r"\sim", r"\simeq", r"\propto", r"\in", r"\notin",
    r"\subset", r"\supset", r"\subseteq", r"\supseteq",
    r"\cup", r"\cap", r"\wedge", r"\vee", r"\neg",
    r"\forall", r"\exists", r"\nabla", r"\partial", r"\infty",

    r"\sum", r"\prod", r"\int", r"\oint", r"\iint", r"\iiint",
    r"\lim", r"\sup", r"\inf", r"\max", r"\min",
    r"\limits", r"\nolimits",

    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon",
    r"\varepsilon", r"\zeta", r"\eta", r"\theta", r"\vartheta",
    r"\iota", r"\kappa", r"\lambda", r"\mu", r"\nu", r"\xi",
    r"\pi", r"\varpi", r"\rho", r"\varrho", r"\sigma", r"\varsigma",
    r"\tau", r"\upsilon", r"\phi", r"\varphi", r"\chi", r"\psi", r"\omega",

    r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Xi",
    r"\Pi", r"\Sigma", r"\Upsilon", r"\Phi", r"\Psi", r"\Omega",

    r"\to", r"\rightarrow", r"\leftarrow", r"\leftrightarrow",
    r"\Rightarrow", r"\Leftarrow", r"\Leftrightarrow",
    r"\mapsto", r"\longrightarrow", r"\longleftarrow",
    r"\uparrow", r"\downarrow", r"\updownarrow",

    r"\left", r"\right", r"\big", r"\Big", r"\bigg", r"\Bigg",
    r"\bigl", r"\bigr", r"\langle", r"\rangle",
    r"\lfloor", r"\rfloor", r"\lceil", r"\rceil",
    r"\vert", r"\Vert",

    r"\hat", r"\tilde", r"\bar", r"\vec", r"\dot", r"\ddot",
    r"\overline", r"\underline", r"\widehat", r"\widetilde",
    r"\overrightarrow", r"\overleftarrow",
    r"\prime", r"\ell",

    r"\mathbb", r"\mathcal", r"\mathbf", r"\mathrm", r"\mathit",
    r"\mathsf", r"\mathtt", r"\mathfrak",
    r"\bf", r"\rm", r"\cal",
    r"\text", r"\mbox",

    r"\begin", r"\end",

    r"\quad", r"\qquad", r"\,", r"\:", r"\;", r"\!",

    r"\log", r"\ln", r"\exp", r"\sin", r"\cos", r"\tan",
    r"\arcsin", r"\arccos", r"\arctan", r"\sinh", r"\cosh", r"\tanh",
    r"\det", r"\dim", r"\ker", r"\deg", r"\gcd", r"\lcm",
    r"\Re", r"\Im", r"\arg", r"\hbar",
    r"\cdots", r"\ldots", r"\dots", r"\vdots", r"\ddots",
    r"\dagger", r"\ddagger",
    r"\perp", r"\parallel", r"\angle", r"\triangle",
    r"\emptyset", r"\varnothing",
    r"\mathop",
]

TOKENIZER_CONFIG = {
    "vocab_size":       VOCAB_SIZE,
    "special_tokens":   SPECIAL_TOKENS,
    "pad_token":        "<pad>",
    "unk_token":        "<unk>",
    "bos_token":        "<bos>",
    "eos_token":        "<eos>",
    "pad_id":           PAD_ID,
    "unk_id":           UNK_ID,
    "bos_id":           BOS_ID,
    "eos_id":           EOS_ID,
    "min_frequency":    2,
    "model_max_length": 128,
    "padding_side":     "right",
    "truncation_side":  "right",
}

TEACHER_MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
