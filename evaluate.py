import re
import numpy as np
from collections import Counter
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
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def normalized_edit_distance(pred: str, ref: str) -> float:
    t_pred = latex_tokens(pred)
    t_ref = latex_tokens(ref)
    dist = edit_distance(t_pred, t_ref)
    max_len = max(len(t_pred), len(t_ref), 1)
    return dist / max_len


def bleu4(preds: list, refs: list) -> float:
    hypotheses = [latex_tokens(p) for p in preds]
    references = [[latex_tokens(r)] for r in refs]
    smoothie = SmoothingFunction().method1
    return corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)


def exact_match(preds: list, refs: list) -> float:
    matches = sum(p.strip() == r.strip() for p, r in zip(preds, refs))
    return matches / len(refs) if refs else 0.0


def token_edit_distance_mean(preds: list, refs: list) -> float:
    dists = [normalized_edit_distance(p, r) for p, r in zip(preds, refs)]
    return float(np.mean(dists))


def compute_metrics(preds: list, refs: list) -> dict:
    assert len(preds) == len(refs), "preds and refs must have same length"
    return {
        "bleu4": bleu4(preds, refs),
        "exact_match": exact_match(preds, refs),
        "edit_distance": token_edit_distance_mean(preds, refs),
        "n_samples": len(preds),
    }


def print_metrics(metrics: dict, prefix: str = ""):
    tag = f"[{prefix}] " if prefix else ""
    print(f"{tag}BLEU-4:      {metrics['bleu4']:.4f}")
    print(f"{tag}Exact Match: {metrics['exact_match']:.4f}  ({int(metrics['exact_match'] * metrics['n_samples'])}/{metrics['n_samples']})")
    print(f"{tag}Edit Dist:   {metrics['edit_distance']:.4f}  (normalized, token-level)")
