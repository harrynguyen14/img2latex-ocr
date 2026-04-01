"""
Thống kê dataset OleehyO/latex-formulas từ HuggingFace.

Configs:
  - cleaned_formulas: có ảnh + label, ~552k mẫu
  - raw_formulas:     chỉ có label (text), ~1M mẫu

Chạy toàn bộ:              python analyze_oleehyo.py
Chạy 1 config cụ thể:     python analyze_oleehyo.py cleaned_formulas
                           python analyze_oleehyo.py raw_formulas
"""

import re
import sys
import os
from collections import Counter
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

OUTPUT_DIR = "stats_oleehyo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_COL = "latex_formula"

CONFIGS = [
    {"name": "cleaned_formulas", "has_image": True},
    {"name": "raw_formulas",     "has_image": False},
]


# ─── Helper ───────────────────────────────────────────────────────────────────
def latex_tokens(text: str) -> list[str]:
    """Tách LaTeX thành tokens: lệnh (\\frac, ...) và ký tự đơn."""
    return re.findall(r"\\[a-zA-Z]+|[^\s]", text)


def strip_align(text: str) -> str:
    """OleehyO bọc label trong \\begin{align*}...\\end{align*} — bỏ wrapper."""
    text = re.sub(r"\\begin\{align\*?\}", "", text)
    text = re.sub(r"\\end\{align\*?\}", "", text)
    return text.strip()


def is_blank_image(img) -> bool:
    arr = np.array(img.convert("L"))
    return arr.std() < 5


def plot_hist(data, title, xlabel, filename, bins=50, log_y=False):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=bins, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if log_y:
        ax.set_yscale("log")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"  [saved] {path}")


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def describe(arr, name=""):
    arr = np.array(arr)
    print(f"  {name}: min={arr.min():.0f}  max={arr.max():.0f}  "
          f"mean={arr.mean():.1f}  median={np.median(arr):.0f}  std={arr.std():.1f}")


# ─── Phân tích ───────────────────────────────────────────────────────────────
def analyze_config(config_name: str, has_image: bool):
    print_section(f"OleehyO/latex-formulas  |  config: {config_name}")

    try:
        ds = load_dataset("OleehyO/latex-formulas", config_name, trust_remote_code=True)
    except Exception as e:
        print(f"  [ERROR] Không load được: {e}")
        return None

    # ── 1. Tổng số mẫu ──────────────────────────────────────────────────────
    print("\n[1] Số mẫu theo split:")
    all_samples = []
    for split_name, split_data in ds.items():
        n = len(split_data)
        print(f"  {split_name:10s}: {n:,}")
        all_samples.extend(split_data)
    total = len(all_samples)
    print(f"  {'TOTAL':10s}: {total:,}")
    if total == 0:
        return None

    tag = f"oleehyo_{config_name}"

    # ── 2. Thống kê ảnh ─────────────────────────────────────────────────────
    if has_image:
        print("\n[2] Thống kê ảnh:")
        widths, heights, aspect_ratios, blank_count = [], [], [], 0

        for sample in all_samples:
            img = sample["image"]
            w, h = img.size
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            if is_blank_image(img):
                blank_count += 1

        describe(widths, "Width ")
        describe(heights, "Height")
        describe(aspect_ratios, "Aspect ratio (W/H)")
        print(f"  Ảnh trắng/đen (blank): {blank_count} / {total} ({100*blank_count/total:.2f}%)")

        plot_hist(widths,        f"[{tag}] Width ảnh",        "Width (px)",  f"{tag}_img_width.png")
        plot_hist(heights,       f"[{tag}] Height ảnh",       "Height (px)", f"{tag}_img_height.png")
        plot_hist(aspect_ratios, f"[{tag}] Aspect Ratio W/H", "W/H",         f"{tag}_img_aspect.png")
    else:
        print("\n[2] Thống kê ảnh: (config này không có ảnh)")

    # ── 3. Thống kê label ───────────────────────────────────────────────────
    print("\n[3] Thống kê label LaTeX:")
    raw_texts = [s[LABEL_COL] for s in all_samples]
    texts = [strip_align(t) for t in raw_texts]

    has_align = sum(1 for t in raw_texts if r"\begin{align" in t)
    print(f"  Có wrapper \\begin{{align*}}: {has_align:,} / {total:,} ({100*has_align/total:.1f}%)")

    char_lens  = [len(t) for t in texts]
    token_lens = [len(latex_tokens(t)) for t in texts]

    empty_count = sum(1 for t in texts if t.strip() == "")
    print(f"  Label rỗng: {empty_count} / {total} ({100*empty_count/total:.2f}%)")
    describe(char_lens,  "Độ dài ký tự ")
    describe(token_lens, "Độ dài token ")

    plot_hist(char_lens,  f"[{tag}] Độ dài label (ký tự)", "Số ký tự", f"{tag}_label_char_len.png",  log_y=True)
    plot_hist(token_lens, f"[{tag}] Độ dài label (token)", "Số token",  f"{tag}_label_token_len.png", log_y=True)

    # ── 4. Vocabulary & top tokens ──────────────────────────────────────────
    print("\n[4] Vocabulary & token phổ biến:")
    all_tokens = []
    for t in texts:
        all_tokens.extend(latex_tokens(t))

    token_counter = Counter(all_tokens)
    print(f"  Vocab size (unique tokens): {len(token_counter):,}")
    print(f"  Tổng tokens: {len(all_tokens):,}")

    print("\n  Top 30 tokens phổ biến nhất:")
    top1_count = token_counter.most_common(1)[0][1]
    for token, count in token_counter.most_common(30):
        bar = "█" * min(40, int(40 * count / top1_count))
        print(f"    {token:20s} {count:8,}  {bar}")

    cmd_counter = Counter({k: v for k, v in token_counter.items() if k.startswith("\\")})
    print("\n  Top 20 lệnh LaTeX (\\command):")
    for cmd, count in cmd_counter.most_common(20):
        print(f"    {cmd:20s} {count:8,}")

    top_cmds = cmd_counter.most_common(20)
    if top_cmds:
        labels, counts = zip(*top_cmds)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(labels[::-1], counts[::-1])
        ax.set_title(f"[{tag}] Top 20 lệnh LaTeX")
        ax.set_xlabel("Frequency")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"{tag}_top_commands.png")
        plt.savefig(path)
        plt.close()
        print(f"  [saved] {path}")

    # ── 5. Kiểm tra duplicate ───────────────────────────────────────────────
    print("\n[5] Kiểm tra duplicate label:")
    label_counter = Counter(texts)
    dup_count  = sum(1 for c in label_counter.values() if c > 1)
    dup_samples = sum(c - 1 for c in label_counter.values() if c > 1)
    print(f"  Label bị trùng (unique duplicated): {dup_count:,}")
    print(f"  Tổng mẫu dư thừa: {dup_samples:,}")

    # ── 6. Ngưỡng max_seq_len đề xuất ──────────────────────────────────────
    print("\n[6] Đề xuất max_seq_len:")
    for pct in [90, 95, 99, 100]:
        val = int(np.percentile(token_lens, pct))
        print(f"  Percentile {pct:3d}%: {val} tokens")
    print(f"\n  => Đề xuất max_seq_len = {int(np.percentile(token_lens, 95))} "
          f"(covers 95% samples)")

    return {
        "config": config_name,
        "total": total,
        "mean_tokens": np.mean(token_lens),
        "median_tokens": np.median(token_lens),
        "max_tokens": max(token_lens),
    }


# ─── So sánh giữa các config ──────────────────────────────────────────────────
def compare_configs(results: list[dict]):
    print_section("SO SÁNH GIỮA CÁC CONFIG")
    print(f"\n  {'Config':25s} {'Total':>9} {'Mean tok':>10} {'Median tok':>12} {'Max tok':>9}")
    print(f"  {'-'*70}")
    for r in results:
        print(f"  {r['config']:25s} {r['total']:>9,} {r['mean_tokens']:>10.1f} "
              f"{r['median_tokens']:>12.0f} {r['max_tokens']:>9}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        configs_to_run = [c for c in CONFIGS if c["name"] in args]
        if not configs_to_run:
            print(f"Config không hợp lệ. Chọn trong: {[c['name'] for c in CONFIGS]}")
            sys.exit(1)
    else:
        configs_to_run = CONFIGS

    results = []
    for cfg in configs_to_run:
        result = analyze_config(cfg["name"], cfg["has_image"])
        if result:
            results.append(result)

    if len(results) > 1:
        compare_configs(results)

    print(f"\nHoàn thành! Charts lưu tại: {OUTPUT_DIR}/")
