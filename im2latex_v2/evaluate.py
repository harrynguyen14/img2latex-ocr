import re

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu


def tokenize_latex(text: str) -> list:
    return re.findall(r"\\[a-zA-Z]+|[^\s]", text)


def edit_distance(seq1: list, seq2: list) -> int:
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if seq1[i - 1] == seq2[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_metrics(predictions: list, references: list) -> dict:
    hyp = [tokenize_latex(x) for x in predictions]
    ref = [[tokenize_latex(x)] for x in references]
    bleu = corpus_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1)
    exact = sum(p.strip() == r.strip() for p, r in zip(predictions, references)) / max(len(references), 1)
    eds = [
        edit_distance(tokenize_latex(p), tokenize_latex(r))
        / max(len(tokenize_latex(p)), len(tokenize_latex(r)), 1)
        for p, r in zip(predictions, references)
    ]
    return {
        "bleu4": round(float(bleu), 4),
        "exact_match": round(float(exact), 4),
        "edit_distance": round(float(np.mean(eds)), 4),
        "n_samples": len(predictions),
    }


def print_metrics(metrics: dict, prefix: str = ""):
    pre = f"[{prefix}] " if prefix else ""
    print(
        f"{pre}BLEU-4={metrics['bleu4']:.4f} Exact={metrics['exact_match']:.4f} "
        f"EditDist={metrics['edit_distance']:.4f} n={metrics['n_samples']}"
    )


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", type=str, required=True)
    ap.add_argument("--ref_file", type=str, required=True)
    args = ap.parse_args()
    with open(args.pred_file, encoding="utf-8") as f:
        preds = [x.rstrip("\n") for x in f]
    with open(args.ref_file, encoding="utf-8") as f:
        refs = [x.rstrip("\n") for x in f]
    n = min(len(preds), len(refs))
    m = compute_metrics(preds[:n], refs[:n])
    print_metrics(m)


if __name__ == "__main__":
    main()
