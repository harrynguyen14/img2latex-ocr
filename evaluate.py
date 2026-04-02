import re
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def latex_tokens(text: str) -> list:
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


def compute_metrics(preds: list, refs: list) -> dict:
    hypotheses = [latex_tokens(p) for p in preds]
    references = [[latex_tokens(r)] for r in refs]

    bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
    exact = sum(p.strip() == r.strip() for p, r in zip(preds, refs)) / max(len(refs), 1)
    eds = [
        edit_distance(latex_tokens(p), latex_tokens(r)) / max(len(latex_tokens(p)), len(latex_tokens(r)), 1)
        for p, r in zip(preds, refs)
    ]
    return {
        "bleu4":         round(float(bleu),         4),
        "exact_match":   round(float(exact),         4),
        "edit_distance": round(float(np.mean(eds)),  4),
        "n_samples":     len(preds),
    }


def print_metrics(metrics: dict, prefix: str = ""):
    tag = f"[{prefix}] " if prefix else ""
    print(f"{tag}BLEU-4={metrics['bleu4']:.4f}  "
          f"Exact={metrics['exact_match']:.4f}  "
          f"EditDist={metrics['edit_distance']:.4f}  "
          f"n={metrics['n_samples']}")
