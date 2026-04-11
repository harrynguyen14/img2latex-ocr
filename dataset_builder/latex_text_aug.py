import random
import re

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

_SHORT_THRESH = 40
_LONG_THRESH  = 150

_LATEX_SPACES = [r"\,", r"\!", r"\;", r"\quad", r"\ "]

_GREEK_PAIRS = [
    (r"\epsilon",  r"\varepsilon"),
    (r"\phi",      r"\varphi"),
    (r"\theta",    r"\vartheta"),
    (r"\rho",      r"\varrho"),
]

_ALIAS_PAIRS = [
    (r"\leq",        r"\le"),
    (r"\geq",        r"\ge"),
    (r"\rightarrow", r"\to"),
    (r"\leftarrow",  r"\gets"),
    (r"\neq",        r"\ne"),
]

_FRAC_RE       = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
_OVER_RE       = re.compile(r"\{([^{}]+)\\over\s*([^{}]+)\}")
_LEFT_RIGHT_RE = re.compile(r"\\left\s*([(\[{])(.*?)\\right\s*([)\]}])", re.DOTALL)


def _make_cmd_pattern(cmd: str) -> str:
    return re.escape(cmd) + r"(?![a-zA-Z])"


def whitespace_noise(latex: str, p: float = 0.8) -> str:
    if random.random() > p:
        return latex
    result = latex
    candidates = [op for op in ["+", "-", "=", "<", ">"] if op in result]
    if candidates:
        picked_ops = random.sample(candidates, k=min(len(candidates), random.choice([1, 1, 2])))
        for op in picked_ops:
            spaces = " " * random.randint(0, 1)
            result = result.replace(op, f"{spaces}{op}{spaces}", 1)
    if random.random() < 0.35:
        sp = random.choice(_LATEX_SPACES)
        result = re.sub(
            r"(\\(?:frac|sum|int|prod|lim|sqrt))",
            lambda m: sp + m.group(1),
            result,
            count=1,
        )
    result = re.sub(r" {2,}", " ", result)
    return result


def subscript_brace_noise(latex: str, p: float = 0.7) -> str:
    if random.random() > p:
        return latex
    bare = list(re.finditer(r"([_^])([a-zA-Z0-9])(?![^{]*\})", latex))
    braced = list(re.finditer(r"([_^])\{([a-zA-Z0-9])\}", latex))
    if bare and (not braced or random.random() < 0.5):
        m = random.choice(bare)
        return latex[:m.start()] + f"{m.group(1)}{{{m.group(2)}}}" + latex[m.end():]
    if braced:
        m = random.choice(braced)
        return latex[:m.start()] + f"{m.group(1)}{m.group(2)}" + latex[m.end():]
    return latex


def command_alias(latex: str, p: float = 0.6) -> str:
    if random.random() > p:
        return latex
    result = latex
    for a, b in _ALIAS_PAIRS:
        if random.random() < 0.5:
            pat_a = _make_cmd_pattern(a)
            pat_b = _make_cmd_pattern(b)
            if re.search(pat_a, result):
                result = re.sub(pat_a, lambda _, repl=b: repl, result)
            elif re.search(pat_b, result):
                result = re.sub(pat_b, lambda _, repl=a: repl, result)
    return result


def greek_variant(latex: str, p: float = 0.7) -> str:
    if random.random() > p:
        return latex
    result = latex
    for std, var in _GREEK_PAIRS:
        if random.random() < 0.5:
            pat_std = _make_cmd_pattern(std)
            pat_var = _make_cmd_pattern(var)
            if re.search(pat_std, result):
                result = re.sub(pat_std, lambda _, repl=var: repl, result)
            elif re.search(pat_var, result):
                result = re.sub(pat_var, lambda _, repl=std: repl, result)
    return result


def frac_to_over(latex: str, p: float = 0.6) -> str:
    if random.random() > p:
        return latex
    return _FRAC_RE.sub(lambda m: "{" + m.group(1) + r" \over " + m.group(2) + "}", latex)


def over_to_frac(latex: str, p: float = 0.6) -> str:
    if random.random() > p:
        return latex
    return _OVER_RE.sub(lambda m: r"\frac{" + m.group(1).strip() + "}{" + m.group(2).strip() + "}", latex)


def vary_bracket_size(latex: str, p: float = 0.6) -> str:
    _SIZES = ["", r"\big", r"\Big", r"\bigg", r"\Bigg"]
    _ESCAPE = {"{": r"\{", "}": r"\}"}
    if random.random() > p:
        return latex
    def repl(m):
        l, content, r = m.groups()
        if len(content.strip()) < 8:
            return m.group(0)
        size = random.choice(_SIZES)
        l_esc = _ESCAPE.get(l, l)
        r_esc = _ESCAPE.get(r, r)
        if size == "":
            return f"{l_esc}{content}{r_esc}"
        return f"{size}l{l_esc}{content}{size}r{r_esc}"
    return _LEFT_RIGHT_RE.sub(repl, latex)


def add_left_right(latex: str, p: float = 0.6) -> str:
    if random.random() > p:
        return latex
    return re.sub(
        r"(?<!\\left)(?<!\\bigl)(?<!\\Bigl)(?<!\\)\(([^()]{1,40})\)(?!\\right)",
        r"\\left(\1\\right)",
        latex,
    )


def tiny_space_tweak(latex: str, p: float = 0.95) -> str:
    if random.random() > p:
        return latex
    candidates = [r"\frac", r"\sum", r"\int", r"\prod", r"\lim", r"\sqrt",
                  r"\partial", r"\nabla", r"\infty", r"\cdot", r"\times"]
    targets = [c for c in candidates if c in latex]
    if targets:
        sp = random.choice(_LATEX_SPACES)
        t = random.choice(targets)
        return latex.replace(t, sp + t, 1)
    return latex


def redundant_braces(latex: str, p: float = 0.6) -> str:
    if random.random() > p:
        return latex
    return subscript_brace_noise(latex, p=1.0)


_LIGHT_FAMILIES = {
    "spacing": [whitespace_noise, tiny_space_tweak],
    "subsup_braces": [subscript_brace_noise, redundant_braces],
    "command_alias": [command_alias],
    "greek_variant": [greek_variant],
    "frac_style": [frac_to_over, over_to_frac],
    "delimiter_style": [add_left_right, vary_bracket_size],
}

_LIGHT_STYLE_PLANS = {
    0: ("spacing", "subsup_braces", "delimiter_style", "command_alias"),
    1: ("spacing", "subsup_braces", "frac_style", "command_alias", "greek_variant"),
}

_FALLBACK_FNS = [tiny_space_tweak, whitespace_noise, subscript_brace_noise, redundant_braces]


def _pick_num_light_families(latex: str) -> int:
    n = len(latex)
    if n < _SHORT_THRESH:
        return 1
    if n < _LONG_THRESH:
        return 1 if random.random() < 0.65 else 2
    if random.random() < 0.80:
        return 2
    return 1


def _ordered_family_names(aug_id: int | None) -> list[str]:
    if aug_id in _LIGHT_STYLE_PLANS:
        preferred = list(_LIGHT_STYLE_PLANS[aug_id])
        remaining = [name for name in _LIGHT_FAMILIES if name not in preferred]
        return preferred + remaining
    names = list(_LIGHT_FAMILIES)
    random.shuffle(names)
    return names


def _apply_light_family(latex: str, family_name: str) -> str:
    result = latex
    for fn in _LIGHT_FAMILIES[family_name]:
        new = fn(result, p=1.0)
        if new != result and new.strip():
            return new
    return latex


def _apply_spacing_style(latex: str, aug_id: int | None) -> str:
    if aug_id == 0:
        return whitespace_noise(latex, p=1.0)

    command_targets = [c for c in [r"\frac", r"\sum", r"\int", r"\prod", r"\lim", r"\sqrt"] if c in latex]
    if command_targets:
        sp = random.choice(_LATEX_SPACES)
        target = random.choice(command_targets)
        return latex.replace(target, sp + target, 1)
    return tiny_space_tweak(latex, p=1.0)


def _apply_aug_id_specific_family(latex: str, aug_id: int | None) -> str:
    if aug_id == 0:
        for family_name in ("subsup_braces", "delimiter_style", "command_alias"):
            if _should_apply_family(latex, family_name, aug_id):
                new = _apply_light_family(latex, family_name)
                if new != latex and new.strip():
                    return new
    elif aug_id == 1:
        for family_name in ("frac_style", "command_alias", "greek_variant", "delimiter_style"):
            if _should_apply_family(latex, family_name, aug_id):
                new = _apply_light_family(latex, family_name)
                if new != latex and new.strip():
                    return new
    return latex


def _should_apply_family(latex: str, family_name: str, aug_id: int | None) -> bool:
    if family_name == "spacing":
        return True
    if family_name == "subsup_braces":
        return bool(re.search(r"[_^](?:\{|[A-Za-z0-9])", latex))
    if family_name == "delimiter_style":
        return "(" in latex and ")" in latex
    if family_name == "frac_style":
        return r"\frac" in latex or r"\over" in latex
    if family_name == "command_alias":
        return any(a in latex or b in latex for a, b in _ALIAS_PAIRS)
    if family_name == "greek_variant":
        return aug_id == 1 and any(std in latex or var in latex for std, var in _GREEK_PAIRS)
    return True


def augment_latex(latex: str, level: str = "light", aug_id: int | None = None) -> str:
    if not latex or not latex.strip():
        return latex

    if level != "light":
        raise ValueError(f"Unsupported augmentation level: {level}")

    result = latex
    changed = False
    chosen_families = []
    family_names = _ordered_family_names(aug_id)
    n_families = min(_pick_num_light_families(latex), len(family_names))

    # Keep light aug close to raw: spacing is the anchor, then add at most one targeted family.
    for family_name in family_names:
        if len(chosen_families) >= n_families:
            break
        if not _should_apply_family(latex, family_name, aug_id):
            continue
        if family_name == "spacing":
            chosen_families.append(family_name)
            continue
        if random.random() < 0.45:
            chosen_families.append(family_name)

    if not chosen_families:
        chosen_families = [family_names[0]]

    if "spacing" not in chosen_families and n_families > 1:
        chosen_families = ["spacing"] + chosen_families[: n_families - 1]
    else:
        chosen_families = chosen_families[:n_families]

    # Encourage aug_id=0 and aug_id=1 to differ in style without drifting too far from raw.
    if aug_id in (0, 1):
        specific = _apply_aug_id_specific_family(latex, aug_id)
        if specific != latex:
            result = specific
            changed = True
            chosen_families = [name for name in chosen_families if name != "subsup_braces"]
            if aug_id == 1:
                chosen_families = [name for name in chosen_families if name != "frac_style"]

    for family_name in chosen_families:
        if family_name == "spacing":
            new = _apply_spacing_style(result, aug_id)
        else:
            new = _apply_light_family(result, family_name)
        if new != result and new.strip():
            result = new
            changed = True

    if not changed:
        for fn in _FALLBACK_FNS:
            new = fn(latex, p=1.0)
            if new != latex and new.strip():
                result = new
                changed = True
                break
    if not changed:
        result = whitespace_noise(latex, p=1.0)
    if result == latex:
        sp = random.choice(_LATEX_SPACES)
        result = sp + latex
    return result


def _run_pipeline():
    import os
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pathlib import Path
    from tqdm import tqdm

    AUG_RATIO = 2
    OUT_DIR   = Path("D:/dataset-ocr-builder/latex-ocr-dataset")
    TRAIN_DIR = OUT_DIR / "train"
    out_dir   = TRAIN_DIR / "light_text_v5"
    prefix    = "light_text_train"

    raw_files = sorted((TRAIN_DIR / "raw").glob("raw_train-*.parquet"))
    if not raw_files:
        raise FileNotFoundError("No raw_train shards found.")

    out_dir.mkdir(parents=True, exist_ok=True)
    n_out_shards = len(raw_files) * AUG_RATIO
    out_idx = 0
    devnull = open(os.devnull, "w")

    for i, pfile in enumerate(raw_files):
        table = pq.read_table(str(pfile))
        latexes = table["latex"].to_pylist()
        sources = table["source"].to_pylist()
        idxs    = table["idx"].to_pylist()

        for v in range(AUG_RATIO):
            random.seed(i * AUG_RATIO + v)
            aug_latexes = [augment_latex(lat, aug_id=v) for lat in tqdm(latexes, desc=f"Shard {i+1} Aug {v+1}", leave=False, file=devnull)]
            fname = f"{prefix}-{str(out_idx).zfill(5)}-of-{str(n_out_shards).zfill(5)}.parquet"
            pq.write_table(
                pa.table({"idx": idxs, "aug_id": [v] * len(idxs), "latex": aug_latexes, "source": sources}),
                str(out_dir / fname),
            )
            out_idx += 1

    devnull.close()


def _run_test(num_samples: int | None, output: str | None):
    import pyarrow.parquet as pq
    from pathlib import Path

    OUT_DIR   = Path("D:/dataset-ocr-builder/latex-ocr-dataset")
    TRAIN_DIR = OUT_DIR / "train"
    raw_files = sorted((TRAIN_DIR / "raw").glob("raw_train-*.parquet"))
    if not raw_files:
        raise FileNotFoundError("No raw_train shards found.")

    all_latexes = []
    for pfile in raw_files:
        all_latexes.extend(pq.read_table(str(pfile))["latex"].to_pylist())

    samples = all_latexes if num_samples is None else random.sample(all_latexes, min(num_samples, len(all_latexes)))

    results = []
    for s in samples:
        out = augment_latex(s)
        results.append(f"IN : {s}\nOUT: {out}\n")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write("".join(results))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--output", type=str)
    args = ap.parse_args()

    if args.run:
        _run_pipeline()
    else:
        _run_test(num_samples=args.num_samples, output=args.output)
