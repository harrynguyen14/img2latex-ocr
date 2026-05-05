r"""
Dataset quality analysis for D:\dataset-ocr-builder\latex-ocr-dataset\ocr-data\train

Analyzes 4 subsets: heavy, light, raw, screenshot
Covers: distribution, image quality, latex quality, duplicates, outliers
"""

import os
import io
import re
import json
import time
import hashlib
import pickle
import warnings
from pathlib import Path
from collections import Counter, defaultdict


import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = Path(r"D:\dataset-ocr-builder\latex-ocr-dataset\ocr-data\train")
OUT_DIR   = Path(r"D:\img2latex\dataset_builder\analysis_output")

# Sample size per shard for heavy analysis (set to None for full scan)
SAMPLE_PER_SHARD = None
RANDOM_SEED = 42

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_sample(paths: list[Path], n_per_shard: int | None = None) -> pd.DataFrame:
    frames = []
    for p in tqdm(paths, desc=f"  loading {paths[0].parent.name}", leave=False):
        df = pd.read_parquet(p, columns=["idx", "image", "latex", "source"])
        if n_per_shard and len(df) > n_per_shard:
            df = df.sample(n_per_shard, random_state=RANDOM_SEED)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def decode_image(raw: bytes) -> Image.Image | None:
    try:
        return Image.open(io.BytesIO(raw))
    except Exception:
        return None


def image_stats(raw: bytes) -> dict:
    img = decode_image(raw)
    if img is None:
        return {"ok": False}
    arr = np.array(img.convert("L"))
    return {
        "ok":           True,
        "width":        img.width,
        "height":       img.height,
        "aspect":       round(img.width / max(img.height, 1), 3),
        "mode":         img.mode,
        "file_bytes":   len(raw),
        "mean_px":      float(arr.mean()),
        "std_px":       float(arr.std()),
        "is_blank":     bool(arr.std() < 2.0),        # nearly uniform
        "mostly_white": bool(arr.mean() > 245),
        "mostly_dark":  bool(arr.mean() < 20),
    }


# ── LaTeX analysis ────────────────────────────────────────────────────────────
LATEX_CMD_RE  = re.compile(r"\\[a-zA-Z]+")
BRACE_OPEN    = re.compile(r"\{")
BRACE_CLOSE   = re.compile(r"\}")
ENV_RE        = re.compile(r"\\begin\{([^}]+)\}")
FRAC_RE       = re.compile(r"\\frac")
SUP_SUB_RE    = re.compile(r"[\^_]")
NUMBER_RE     = re.compile(r"\d+")

def latex_stats(s: str) -> dict:
    if not isinstance(s, str):
        return {"ok": False}
    cmds      = LATEX_CMD_RE.findall(s)
    envs      = ENV_RE.findall(s)
    opens     = len(BRACE_OPEN.findall(s))
    closes    = len(BRACE_CLOSE.findall(s))
    return {
        "ok":              True,
        "length":          len(s),
        "n_commands":      len(cmds),
        "n_unique_cmds":   len(set(cmds)),
        "n_envs":          len(envs),
        "n_fracs":         len(FRAC_RE.findall(s)),
        "n_sup_sub":       len(SUP_SUB_RE.findall(s)),
        "has_numbers":     bool(NUMBER_RE.search(s)),
        "brace_balanced":  opens == closes,
        "top_cmds":        Counter(cmds).most_common(5),
        "top_envs":        Counter(envs).most_common(3),
        "is_empty":        len(s.strip()) == 0,
        "has_newline":     "\n" in s or r"\\" in s,
    }


# ── Per-subset analysis ───────────────────────────────────────────────────────







def analyse_subset(name: str, paths: list[Path]) -> dict:
    print(f"\n{'='*60}")
    print(f"  Subset: {name.upper()}  ({len(paths)} shards)")
    print(f"{'='*60}")

    df = load_sample(paths, SAMPLE_PER_SHARD)
    print(f"  Sampled {len(df):,} rows")

    print("  Analysing images …")
    img_rows = [image_stats(r) for r in tqdm(df["image"], leave=False)]
    idf = pd.DataFrame(img_rows)

    print("  Analysing LaTeX …")
    lat_rows = [latex_stats(s) for s in tqdm(df["latex"], leave=False)]
    ldf = pd.DataFrame(lat_rows)

    print("  Detecting duplicates …")
    hashes = [hashlib.md5(b).hexdigest() for b in df["image"]]
    dup_count = len(hashes) - len(set(hashes))

    # ── Latex duplicate ──
    lat_dup = df["latex"].duplicated().sum()

    # ── Source distribution ──
    sources = df["source"].value_counts().to_dict()

    result = {
        "name":       name,
        "n_shards":   len(paths),
        "n_sampled":  len(df),
        "sources":    sources,
        # image
        "img_decode_fail": int((~idf["ok"]).sum()),
        "img_blank":       int(idf[idf["ok"]]["is_blank"].sum()),
        "img_mostly_white":int(idf[idf["ok"]]["mostly_white"].sum()),
        "img_mostly_dark": int(idf[idf["ok"]]["mostly_dark"].sum()),
        "img_dup_count":   dup_count,
        "img_width_p":   idf[idf["ok"]]["width"].describe().to_dict(),
        "img_height_p":  idf[idf["ok"]]["height"].describe().to_dict(),
        "img_aspect_p":  idf[idf["ok"]]["aspect"].describe().to_dict(),
        "img_bytes_p":   idf[idf["ok"]]["file_bytes"].describe().to_dict(),
        "img_mean_px":   idf[idf["ok"]]["mean_px"].describe().to_dict(),
        "img_std_px":    idf[idf["ok"]]["std_px"].describe().to_dict(),
        "img_modes":     idf[idf["ok"]]["mode"].value_counts().to_dict(),
        # latex
        "lat_decode_fail":   int((~ldf["ok"]).sum()),
        "lat_empty":         int(ldf[ldf["ok"]]["is_empty"].sum()),
        "lat_brace_unbal":   int((~ldf[ldf["ok"]]["brace_balanced"]).sum()),
        "lat_dup_count":     int(lat_dup),
        "lat_length_p":      ldf[ldf["ok"]]["length"].describe().to_dict(),
        "lat_n_cmds_p":      ldf[ldf["ok"]]["n_commands"].describe().to_dict(),
        "lat_n_fracs_p":     ldf[ldf["ok"]]["n_fracs"].describe().to_dict(),
        "lat_n_sup_sub_p":   ldf[ldf["ok"]]["n_sup_sub"].describe().to_dict(),
        "lat_has_newline":   int(ldf[ldf["ok"]]["has_newline"].sum()),
        # raw dataframes for plotting
        "_idf": idf,
        "_ldf": ldf,
    }
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {"heavy": "#e06c75", "light": "#61afef", "raw": "#98c379", "screenshot": "#e5c07b"}

def plot_all(results: list[dict]):
    names  = [r["name"] for r in results]
    colors = [COLORS.get(n, "#abb2bf") for n in names]

    # ── Figure 1: image distributions ──────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("Image Quality – Distribution per Subset", fontsize=15, fontweight="bold")

    metrics = [
        ("width",      "Image Width (px)",       "img_width_p"),
        ("height",     "Image Height (px)",       "img_height_p"),
        ("aspect",     "Aspect Ratio (W/H)",      "img_aspect_p"),
        ("file_bytes", "File Size (bytes)",        "img_bytes_p"),
        ("mean_px",    "Mean Pixel (0-255)",       "img_mean_px"),
        ("std_px",     "Pixel Std-Dev",            "img_std_px"),
    ]

    for ax_row, (col_key, title, _) in zip(axes.flat, metrics):
        for r, c in zip(results, colors):
            idf = r["_idf"]
            ok  = idf[idf["ok"]]
            if col_key in ok.columns:
                ax_row.hist(ok[col_key].clip(ok[col_key].quantile(0.01),
                                              ok[col_key].quantile(0.99)),
                            bins=60, alpha=0.55, color=c, label=r["name"], density=True)
        ax_row.set_title(title, fontsize=10)
        ax_row.legend(fontsize=7)
        ax_row.set_ylabel("Density")

    # Blank / dark / white counts
    ax = axes[1][2]
    x = np.arange(len(names))
    blanks = [r["img_blank"]       / r["n_sampled"] * 100 for r in results]
    whites = [r["img_mostly_white"]/ r["n_sampled"] * 100 for r in results]
    darks  = [r["img_mostly_dark"] / r["n_sampled"] * 100 for r in results]
    w = 0.25
    ax.bar(x - w,  blanks, w, label="Blank (std<2)",  color="#e06c75")
    ax.bar(x,      whites, w, label="Mostly white",   color="#e5c07b")
    ax.bar(x + w,  darks,  w, label="Mostly dark",    color="#282c34")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("Problematic Images (%)", fontsize=10)
    ax.legend(fontsize=7); ax.set_ylabel("%")

    # Color modes
    ax = axes[1][3]
    all_modes = sorted({m for r in results for m in r["img_modes"]})
    x = np.arange(len(names))
    for i, mode in enumerate(all_modes):
        vals = [r["img_modes"].get(mode, 0) / r["n_sampled"] * 100 for r in results]
        ax.bar(x + i * 0.2, vals, 0.2, label=mode)
    ax.set_xticks(x + 0.1 * len(all_modes))
    ax.set_xticklabels(names)
    ax.set_title("Color Mode Distribution (%)", fontsize=10)
    ax.legend(fontsize=7); ax.set_ylabel("%")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "01_image_quality.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 01_image_quality.png")

    # ── Figure 2: latex distributions ─────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("LaTeX Quality – Distribution per Subset", fontsize=15, fontweight="bold")

    lat_metrics = [
        ("length",    "Token Length (chars)"),
        ("n_commands","# LaTeX Commands"),
        ("n_fracs",   "# \\frac"),
        ("n_sup_sub", "# ^ and _"),
    ]

    for ax, (col, title) in zip(axes.flat, lat_metrics):
        for r, c in zip(results, colors):
            ldf = r["_ldf"]
            ok  = ldf[ldf["ok"]]
            if col in ok.columns:
                ax.hist(ok[col].clip(0, ok[col].quantile(0.99)),
                        bins=60, alpha=0.55, color=c, label=r["name"], density=True)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)

    # Issues bar
    ax = axes.flat[4]
    issues = {
        "Empty LaTeX":       [r["lat_empty"]       / r["n_sampled"] * 100 for r in results],
        "Unbal. Braces":     [r["lat_brace_unbal"]  / r["n_sampled"] * 100 for r in results],
        "Dup LaTeX":         [r["lat_dup_count"]    / r["n_sampled"] * 100 for r in results],
        "Dup Image":         [r["img_dup_count"]    / r["n_sampled"] * 100 for r in results],
    }
    x = np.arange(len(names))
    for i, (label, vals) in enumerate(issues.items()):
        ax.bar(x + i * 0.2, vals, 0.2, label=label)
    ax.set_xticks(x + 0.3); ax.set_xticklabels(names)
    ax.set_title("Quality Issues (%)", fontsize=10)
    ax.legend(fontsize=7); ax.set_ylabel("%")

    # Multiline
    ax = axes.flat[5]
    ml = [r["lat_has_newline"] / r["n_sampled"] * 100 for r in results]
    ax.bar(names, ml, color=colors)
    ax.set_title("Multi-line LaTeX (%)", fontsize=10)
    ax.set_ylabel("%")

    # Unique commands per subset (top-15 combined)
    ax = axes.flat[6]
    all_cmd_counts: Counter = Counter()
    cmd_by_subset = {}
    for r in results:
        ldf = r["_ldf"]
        ok  = ldf[ldf["ok"]]
        cmds: Counter = Counter()
        for row in ok["top_cmds"]:
            if isinstance(row, (list, tuple)) and row:
                cmds.update({c: n for c, n in row})
        cmd_by_subset[r["name"]] = cmds
        all_cmd_counts.update(cmds)
    top15 = [c for c, _ in all_cmd_counts.most_common(15)]
    y = np.arange(len(top15))
    for i, (r, c) in enumerate(zip(results, colors)):
        vals = [cmd_by_subset[r["name"]].get(cmd, 0) for cmd in top15]
        ax.barh(y + i * 0.2, vals, 0.2, label=r["name"], color=c)
    ax.set_yticks(y + 0.3); ax.set_yticklabels(top15, fontsize=7)
    ax.set_title("Top-15 LaTeX Commands", fontsize=10)
    ax.legend(fontsize=7); ax.set_xlabel("Count")

    # Dataset sizes
    ax = axes.flat[7]
    n_shards = [r["n_shards"] for r in results]
    approx_total = [r["n_shards"] * 50_000 for r in results]
    ax.bar(names, approx_total, color=colors)
    ax.set_title("Approx. Total Rows per Subset\n(50k × shards)", fontsize=10)
    ax.set_ylabel("Rows")
    for i, v in enumerate(approx_total):
        ax.text(i, v + 5000, f"{v/1e6:.2f}M", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "02_latex_quality.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 02_latex_quality.png")

    # ── Figure 3: cross-subset comparison ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Cross-Subset Comparison", fontsize=14, fontweight="bold")

    stats_to_compare = [
        ("img_width_p",   "Median Image Width (px)"),
        ("img_bytes_p",   "Median File Size (bytes)"),
        ("lat_length_p",  "Median LaTeX Length (chars)"),
    ]
    for ax, (key, title) in zip(axes, stats_to_compare):
        def _safe(v):
            try:
                f = float(v)
                return 0.0 if f != f else f  # NaN check
            except (TypeError, ValueError):
                return 0.0

        medians = [_safe(r[key].get("50%")) for r in results]
        q25     = [_safe(r[key].get("25%")) for r in results]
        q75     = [_safe(r[key].get("75%")) for r in results]
        yerr_lo = [max(0.0, m - q) for m, q in zip(medians, q25)]
        yerr_hi = [max(0.0, q - m) for m, q in zip(q75, medians)]
        ax.bar(names, medians, color=colors, alpha=0.8)
        ax.errorbar(names, medians, yerr=[yerr_lo, yerr_hi],
                    fmt="none", color="black", capsize=5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Value (median ± IQR)")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "03_cross_subset.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 03_cross_subset.png")


# ── Text report ───────────────────────────────────────────────────────────────

def fmt_pct(n, total):
    return f"{n:,}  ({100*n/total:.2f}%)"

def print_report(results: list[dict]):
    lines = []
    sep   = "=" * 70

    def p(s=""):
        lines.append(s)
        print(s)

    p(sep)
    p("  DATASET QUALITY ANALYSIS REPORT")
    p(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    p(sep)

    overall_sampled = sum(r["n_sampled"] for r in results)
    p(f"\n  Total subsets   : {len(results)}")
    p(f"  Total sampled   : {overall_sampled:,}")
    p(f"  Sample per shard: {SAMPLE_PER_SHARD}")

    for r in results:
        n = r["n_sampled"]
        p(f"\n{sep}")
        p(f"  SUBSET: {r['name'].upper()}")
        p(sep)
        p(f"  Shards          : {r['n_shards']}")
        p(f"  Approx. total   : ~{r['n_shards']*50_000:,} rows (50k × shards)")
        p(f"  Sampled rows    : {n:,}")
        p(f"  Sources         : {r['sources']}")
        p()
        p("  ── IMAGE ──")
        p(f"  Decode failures : {fmt_pct(r['img_decode_fail'], n)}")
        p(f"  Blank images    : {fmt_pct(r['img_blank'], n)}")
        p(f"  Mostly white    : {fmt_pct(r['img_mostly_white'], n)}")
        p(f"  Mostly dark     : {fmt_pct(r['img_mostly_dark'], n)}")
        p(f"  Duplicate imgs  : {fmt_pct(r['img_dup_count'], n)}")
        p(f"  Color modes     : {r['img_modes']}")
        wp = r["img_width_p"]
        hp = r["img_height_p"]
        ap = r["img_aspect_p"]
        bp = r["img_bytes_p"]
        p(f"  Width  px  mean={wp['mean']:.0f}  std={wp['std']:.0f}  "
          f"min={wp['min']:.0f}  p25={wp['25%']:.0f}  p50={wp['50%']:.0f}  "
          f"p75={wp['75%']:.0f}  max={wp['max']:.0f}")
        p(f"  Height px  mean={hp['mean']:.0f}  std={hp['std']:.0f}  "
          f"min={hp['min']:.0f}  p25={hp['25%']:.0f}  p50={hp['50%']:.0f}  "
          f"p75={hp['75%']:.0f}  max={hp['max']:.0f}")
        p(f"  Aspect     mean={ap['mean']:.2f}  p50={ap['50%']:.2f}  "
          f"min={ap['min']:.2f}  max={ap['max']:.2f}")
        p(f"  Bytes      mean={bp['mean']:.0f}  p50={bp['50%']:.0f}  "
          f"min={bp['min']:.0f}  max={bp['max']:.0f}")
        p()
        p("  ── LATEX ──")
        lp  = r["lat_length_p"]
        cp  = r["lat_n_cmds_p"]
        fp  = r["lat_n_fracs_p"]
        ssp = r["lat_n_sup_sub_p"]
        p(f"  Decode failures : {fmt_pct(r['lat_decode_fail'], n)}")
        p(f"  Empty strings   : {fmt_pct(r['lat_empty'], n)}")
        p(f"  Unbal. braces   : {fmt_pct(r['lat_brace_unbal'], n)}")
        p(f"  Duplicate latex : {fmt_pct(r['lat_dup_count'], n)}")
        p(f"  Multi-line      : {fmt_pct(r['lat_has_newline'], n)}")
        p(f"  Length chars   mean={lp['mean']:.0f}  p50={lp['50%']:.0f}  "
          f"min={lp['min']:.0f}  max={lp['max']:.0f}")
        p(f"  # Commands     mean={cp['mean']:.1f}  p50={cp['50%']:.0f}  "
          f"min={cp['min']:.0f}  max={cp['max']:.0f}")
        p(f"  # \\frac       mean={fp['mean']:.2f}  p50={fp['50%']:.0f}  max={fp['max']:.0f}")
        p(f"  # ^/_          mean={ssp['mean']:.2f}  p50={ssp['50%']:.0f}  max={ssp['max']:.0f}")

    p(f"\n{sep}")
    p("  OVERALL QUALITY SUMMARY")
    p(sep)
    for r in results:
        n = r["n_sampled"]
        issues = (r["img_blank"] + r["img_mostly_white"] +
                  r["img_mostly_dark"] + r["lat_empty"] + r["lat_brace_unbal"])
        pct = 100 * issues / n
        grade = "GOOD" if pct < 1 else ("WARN" if pct < 5 else "BAD ")
        p(f"  [{grade}] {r['name']:12s}  issue_rate={pct:.2f}%  "
          f"img_dups={r['img_dup_count']:,}  lat_dups={r['lat_dup_count']:,}")

    report_path = OUT_DIR / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report saved → {report_path}")


# ── JSON summary ──────────────────────────────────────────────────────────────

def save_json(results: list[dict]):
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if not k.startswith("_")}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, list):
            return [clean(i) for i in obj]
        return obj

    out = {r["name"]: clean(r) for r in results}
    path = OUT_DIR / "stats.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON stats saved → {path}")


# ── Cache / Resume ───────────────────────────────────────────────────────────

CACHE_DIR = OUT_DIR / "_cache"


def save_subset_cache(name: str, result: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(result, f)


def load_subset_cache(name: str) -> dict | None:
    path = CACHE_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Dataset root : {DATA_ROOT}")
    print(f"Output dir   : {OUT_DIR}")
    print(f"Sample/shard : {SAMPLE_PER_SHARD}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subsets = {
        "heavy":      sorted((DATA_ROOT / "heavy").glob("*.parquet")),
        "light":      sorted((DATA_ROOT / "light").glob("*.parquet")),
        "raw":        sorted((DATA_ROOT / "raw").glob("*.parquet")),
        "screenshot": sorted((DATA_ROOT / "screenshot").glob("*.parquet")),
    }

    results = []
    for name, paths in subsets.items():
        if not paths:
            print(f"  [SKIP] {name} – no files found")
            continue
        cached = load_subset_cache(name)
        if cached is not None:
            print(f"\n  [RESUME] {name.upper()} – loaded from cache")
            results.append(cached)
        else:
            r = analyse_subset(name, paths)
            save_subset_cache(name, r)
            results.append(r)

    print("\nGenerating plots …")
    plot_all(results)

    print("\nGenerating report …")
    print_report(results)
    save_json(results)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  →  {OUT_DIR}")


if __name__ == "__main__":
    main()