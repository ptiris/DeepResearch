"""
Visualize grouped Rel statistics (correct vs incorrect).

Usage:
    python3 scripts/visualize_rel_grouped.py \
        --results output/deepseek-v4-pro/bc-zn10-control/rel_results_all.json \
        --scored output/deepseek-v4-pro/bc-zn10-control/iter1_scored.jsonl
"""

import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def find_chinese_font():
    candidates = [
        "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
        "Source Han Sans SC", "PingFang SC", "Microsoft YaHei",
        "AR PL UMing CN", "WenQuanYi Zen Hei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    for f in fm.fontManager.ttflist:
        if "CJK" in f.name or "Hei" in f.name or "Song" in f.name or "Ming" in f.name:
            return f.name
    return None


CHINESE_FONT = find_chinese_font()
if CHINESE_FONT:
    plt.rcParams["font.sans-serif"] = [CHINESE_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

REL_BINS = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.01]
REL_LABELS = ["[0,0.3)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,1.0]"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--scored", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        results_data = json.load(f)

    correctness = {}
    with open(args.scored, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            d = json.loads(line)
            correctness[i] = d.get("is_correct", None)

    results = results_data["results"]

    correct_rels_by_q = []
    incorrect_rels_by_q = []
    all_per_q = []

    for r in results:
        line_idx = r["line_idx"]
        is_correct = correctness.get(line_idx, None)
        rels = r["all_rels"]
        q_short = r["question"][:35]
        all_per_q.append({"rels": rels, "label": f"L{line_idx}({q_short})", "is_correct": is_correct})
        if is_correct is True:
            correct_rels_by_q.append({"rels": rels, "label": f"L{line_idx}"})
        elif is_correct is False:
            incorrect_rels_by_q.append({"rels": rels, "label": f"L{line_idx}"})

    all_correct = [r for item in correct_rels_by_q for r in item["rels"]]
    all_incorrect = [r for item in incorrect_rels_by_q for r in item["rels"]]

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    # 1. Overlaid histogram: correct vs incorrect
    ax = axes[0, 0]
    if all_correct or all_incorrect:
        data_list = []
        labels_list = []
        colors_list = []
        if all_correct:
            data_list.append(all_correct)
            labels_list.append(f"Correct (n={len(all_correct)})")
            colors_list.append("#2ecc71")
        if all_incorrect:
            data_list.append(all_incorrect)
            labels_list.append(f"Incorrect (n={len(all_incorrect)})")
            colors_list.append("#e74c3c")
        ax.hist(data_list, bins=REL_BINS, label=labels_list, alpha=0.7, edgecolor="white", color=colors_list)
        ax.legend(fontsize=9)
        ax.set_xlabel("Rel Score")
        ax.set_ylabel("Count")
        ax.set_title("Rel Distribution: Correct vs Incorrect", fontweight="bold")
        ax.set_xlim(0, 1.05)

    # 2. Per-question mean Rel, colored by correctness
    ax = axes[0, 1]
    q_labels = [q["label"] for q in all_per_q]
    q_means = [np.mean(q["rels"]) if q["rels"] else 0 for q in all_per_q]
    q_colors = ["#2ecc71" if q["is_correct"] else "#e74c3c" for q in all_per_q]
    bars = ax.barh(range(len(q_labels)), q_means, color=q_colors, alpha=0.8, edgecolor="white")
    ax.set_yticks(range(len(q_labels)))
    ax.set_yticklabels(q_labels, fontsize=8)
    ax.set_xlabel("Mean Rel")
    ax.set_title("Per-Question Mean Rel (Green=Correct, Red=Incorrect)", fontweight="bold")
    ax.axvline(x=np.mean(q_means), color="gray", linestyle="--", alpha=0.5)
    for bar, m in zip(bars, q_means):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{m:.3f}", va="center", fontsize=8)

    # 3. Violin plot
    ax = axes[0, 2]
    violin_data = []
    violin_labels = []
    violin_colors = []
    if all_correct:
        violin_data.append(all_correct)
        violin_labels.append(f"Correct\n(n={len(all_correct)})")
        violin_colors.append("#2ecc71")
    if all_incorrect:
        violin_data.append(all_incorrect)
        violin_labels.append(f"Incorrect\n(n={len(all_incorrect)})")
        violin_colors.append("#e74c3c")
    if violin_data:
        parts = ax.violinplot(violin_data, positions=range(len(violin_data)), showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(violin_colors[i])
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(violin_data)))
        ax.set_xticklabels(violin_labels)
        ax.set_ylabel("Rel Score")
        ax.set_title("Rel Distribution (Violin)", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.7, color="red", linestyle="--", alpha=0.3, label="0.7")
        ax.axhline(y=0.5, color="blue", linestyle="--", alpha=0.3, label="0.5")
        ax.legend(fontsize=8)

    # 4. CDF comparison
    ax = axes[1, 0]
    for rels, label, color in [
        (all_correct, "Correct", "#2ecc71"),
        (all_incorrect, "Incorrect", "#e74c3c"),
    ]:
        if rels:
            sorted_r = np.sort(rels)
            cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
            ax.plot(sorted_r, cdf, label=label, color=color, linewidth=2)
    ax.set_xlabel("Rel Score")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of Rel: Correct vs Incorrect", fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.4)

    # 5. Per-question box plot, colored by correctness
    ax = axes[1, 1]
    box_data = [q["rels"] for q in all_per_q if q["rels"]]
    box_labels = [q["label"] for q in all_per_q if q["rels"]]
    box_colors = ["#2ecc71" if q["is_correct"] else "#e74c3c" for q in all_per_q if q["rels"]]
    if box_data:
        bp = ax.boxplot(box_data, patch_artist=True, vert=True, widths=0.6)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticklabels(box_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Rel Score")
        ax.set_title("Per-Question Rel (Box Plot)", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="blue", linestyle="--", alpha=0.3)

    # 6. Stacked percentage histogram
    ax = axes[1, 2]
    bins_labels_short = ["0-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    for rels, label, color, y_offset in [
        (all_incorrect, "Incorrect", "#e74c3c", 0),
        (all_correct, "Correct", "#2ecc71", None),
    ]:
        if rels:
            counts, _ = np.histogram(rels, bins=REL_BINS)
            pcts = counts / len(rels) * 100
            if y_offset is not None:
                bottoms = np.zeros(len(counts))
            else:
                bottoms = np.array([c2 / len(all_incorrect) * 100 for c2 in np.histogram(all_incorrect, bins=REL_BINS)[0]]) if all_incorrect else np.zeros(len(counts))
            ax.bar(bins_labels_short, pcts, bottom=bottoms, label=label, color=color, alpha=0.7, edgecolor="white")
    ax.set_xlabel("Rel Score Range")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Rel Distribution (Percentage)", fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout(pad=2.0)
    fig_path = output_dir / "rel_grouped_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
