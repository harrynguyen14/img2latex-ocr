"""
Thống kê dataset linxy/LaTeX_OCR từ HuggingFace.
Chạy: python analyze.py
Chạy 1 config: python analyze.py small
"""

import re
from collections import Counter
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─── Config ───────────────────────────────────────────────────────────────────
CONFIGS = ["small", "full", "synthetic_handwrite", "human_handwrite", "human_handwrite_print"]
OUTPUT_DIR = "stats"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Helper ───────────────────────────────────────────────────────────────────
def latex_tokens(text: str) -> list[str]:
    """Tách LaTeX thành tokens: lệnh (\frac, \sum, ...) và ký tự đơn."""
    return re.findall(r"\\[a-zA-Z]+|[^\s]", text)


def is_blank_image(img) -> bool:
    """Ảnh trắng hoặc đen hoàn toàn."""
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
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  [saved] {OUTPUT_DIR}/{filename}")


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def describe(arr, name=""):
    arr = np.array(arr)
    print(f"  {name}: min={arr.min():.0f}  max={arr.max():.0f}  "
          f"mean={arr.mean():.1f}  median={np.median(arr):.0f}  std={arr.std():.1f}")


# ─── Phân tích 1 config ───────────────────────────────────────────────────────
def analyze_config(config_name: str):
    print_section(f"CONFIG: {config_name}")

    try:
        ds = load_dataset("linxy/LaTeX_OCR", config_name, trust_remote_code=True)
    except Exception as e:
        print(f"  [ERROR] Không load được: {e}")
        return

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
        return

    # ── 2. Thống kê ảnh ─────────────────────────────────────────────────────
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

    plot_hist(widths, f"[{config_name}] Phân phối Width ảnh", "Width (px)",
              f"{config_name}_img_width.png")
    plot_hist(heights, f"[{config_name}] Phân phối Height ảnh", "Height (px)",
              f"{config_name}_img_height.png")
    plot_hist(aspect_ratios, f"[{config_name}] Aspect Ratio (W/H)", "W/H",
              f"{config_name}_img_aspect.png")

    # ── 3. Thống kê label ───────────────────────────────────────────────────
    print("\n[3] Thống kê label LaTeX:")
    texts = [s["text"] for s in all_samples]
    char_lens = [len(t) for t in texts]
    token_lens = [len(latex_tokens(t)) for t in texts]

    empty_count = sum(1 for t in texts if t.strip() == "")
    print(f"  Label rỗng: {empty_count} / {total} ({100*empty_count/total:.2f}%)")
    describe(char_lens, "Độ dài ký tự ")
    describe(token_lens, "Độ dài token ")

    plot_hist(char_lens, f"[{config_name}] Độ dài label (ký tự)", "Số ký tự",
              f"{config_name}_label_char_len.png", log_y=True)
    plot_hist(token_lens, f"[{config_name}] Độ dài label (token)", "Số token",
              f"{config_name}_label_token_len.png", log_y=True)

    # ── 4. Vocabulary & top tokens ──────────────────────────────────────────
    print("\n[4] Vocabulary & token phổ biến:")
    all_tokens = []
    for t in texts:
        all_tokens.extend(latex_tokens(t))

    token_counter = Counter(all_tokens)
    vocab_size = len(token_counter)
    print(f"  Vocab size (unique tokens): {vocab_size:,}")
    print(f"  Tổng tokens: {len(all_tokens):,}")

    print("\n  Top 30 tokens phổ biến nhất:")
    for token, count in token_counter.most_common(30):
        bar = "█" * min(40, int(40 * count / token_counter.most_common(1)[0][1]))
        print(f"    {token:20s} {count:8,}  {bar}")

    # Top LaTeX commands only
    cmd_counter = Counter({k: v for k, v in token_counter.items() if k.startswith("\\")})
    print("\n  Top 20 lệnh LaTeX (\\command):")
    for cmd, count in cmd_counter.most_common(20):
        print(f"    {cmd:20s} {count:8,}")

    # Bar chart top 20 commands
    top_cmds = cmd_counter.most_common(20)
    if top_cmds:
        labels, counts = zip(*top_cmds)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(labels[::-1], counts[::-1])
        ax.set_title(f"[{config_name}] Top 20 lệnh LaTeX")
        ax.set_xlabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{config_name}_top_commands.png"))
        plt.close()
        print(f"  [saved] {OUTPUT_DIR}/{config_name}_top_commands.png")

    # ── 5. Kiểm tra duplicate ───────────────────────────────────────────────
    print("\n[5] Kiểm tra duplicate label:")
    label_counter = Counter(texts)
    dup_count = sum(1 for c in label_counter.values() if c > 1)
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


# ─── So sánh giữa các config ──────────────────────────────────────────────────
def compare_configs():
    print_section("SO SÁNH GIỮA CÁC CONFIG")
    summary = {}

    for config_name in CONFIGS:
        try:
            ds = load_dataset("linxy/LaTeX_OCR", config_name, trust_remote_code=True)
        except Exception:
            continue

        all_texts = []
        for split_data in ds.values():
            all_texts.extend([s["text"] for s in split_data])

        if not all_texts:
            continue

        token_lens = [len(latex_tokens(t)) for t in all_texts]
        summary[config_name] = {
            "total": len(all_texts),
            "mean_tokens": np.mean(token_lens),
            "median_tokens": np.median(token_lens),
            "max_tokens": max(token_lens),
        }

    print(f"\n  {'Config':25s} {'Total':>8} {'Mean tok':>10} {'Median tok':>12} {'Max tok':>9}")
    print(f"  {'-'*70}")
    for cfg, s in summary.items():
        print(f"  {cfg:25s} {s['total']:>8,} {s['mean_tokens']:>10.1f} "
              f"{s['median_tokens']:>12.0f} {s['max_tokens']:>9}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Cho phép chọn config qua CLI: python analyze.py small
    # Mặc định chạy tất cả
    configs_to_run = sys.argv[1:] if len(sys.argv) > 1 else CONFIGS

    for cfg in configs_to_run:
        analyze_config(cfg)

    if len(configs_to_run) > 1:
        compare_configs()

    print(f"\nHoàn thành! Charts lưu tại: {OUTPUT_DIR}/")
