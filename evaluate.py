import re
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def tokenize_latex(text: str) -> list:
    """Split a LaTeX string into tokens: commands (e.g. \\frac) and individual characters."""
    return re.findall(r"\\[a-zA-Z]+|[^\s]", text)


def edit_distance(seq1: list, seq2: list) -> int:
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if seq1[i-1] == seq2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def compute_metrics(predictions: list, references: list) -> dict:
    hypotheses     = [tokenize_latex(p) for p in predictions]
    reference_list = [[tokenize_latex(r)] for r in references]

    bleu        = corpus_bleu(reference_list, hypotheses,
                              smoothing_function=SmoothingFunction().method1)
    exact_match = sum(p.strip() == r.strip() for p, r in zip(predictions, references)) / max(len(references), 1)
    normalized_edit_distances = [
        edit_distance(tokenize_latex(p), tokenize_latex(r))
        / max(len(tokenize_latex(p)), len(tokenize_latex(r)), 1)
        for p, r in zip(predictions, references)
    ]
    return {
        "bleu4":         round(float(bleu),                              4),
        "exact_match":   round(float(exact_match),                       4),
        "edit_distance": round(float(np.mean(normalized_edit_distances)), 4),
        "n_samples":     len(predictions),
    }


def print_metrics(metrics: dict, prefix: str = ""):
    log_prefix = f"[{prefix}] " if prefix else ""
    print(f"{log_prefix}BLEU-4={metrics['bleu4']:.4f}  "
          f"Exact={metrics['exact_match']:.4f}  "
          f"EditDist={metrics['edit_distance']:.4f}  "
          f"n={metrics['n_samples']}")
