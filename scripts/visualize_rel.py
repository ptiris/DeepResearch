"""
Visualize Rel = cosine_sim(emb_question, emb_response_snippet) results.

Reads the JSON output from compute_rel.py and produces:
  1. Rel distribution histogram per question
  2. Per-search-call box plot
  3. Cross-question comparison
  4. Top/Bottom snippet examples with Rel scores

Usage:
    uv run python scripts/visualize_rel.py --input output/deepseek-v4-pro/bc-zn10-control/rel_results.json
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
    print(f"Using Chinese font: {CHINESE_FONT}")
else:
    print("Warning: No Chinese font found, labels may not render correctly")


REL_BINS = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.01]
REL_LABELS = ["[0,0.3)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,1.0]"]
BIN_COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]


def plot_histogram(ax, rels, title, color="#3498db"):
    if not rels:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    counts, _, patches = ax.hist(rels, bins=REL_BINS, edgecolor="white", linewidth=1.2, alpha=0.85)
    for patch, c in zip(patches, BIN_COLORS):
        patch.set_facecolor(c)
    for i, cnt in enumerate(counts):
        if cnt > 0:
            ax.text(
                (REL_BINS[i] + REL_BINS[i + 1]) / 2,
                cnt + 0.3,
                str(int(cnt)),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Rel Score")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1.05)


def plot_box_per_call(ax, per_call_stats, title):
    data = []
    labels = []
    for call_idx in sorted(per_call_stats.keys(), key=int):
        cs = per_call_stats[call_idx]
        rels = cs["individual_rels"]
        if rels:
            data.append(rels)
            q_preview = cs["search_queries"][0][:25] if cs["search_queries"] else f"Call {call_idx}"
            labels.append(f"C{call_idx}: {q_preview}")
    if not data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    bp = ax.boxplot(data, patch_artist=True, vert=True, widths=0.6)
    colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Rel Score")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.7, color="red", linestyle="--", alpha=0.4, label="0.7 threshold")
    ax.legend(fontsize=8)


def plot_top_bottom_examples(ax, top3, bottom3, title):
    items = list(reversed(bottom3)) + top3
    rels = [item["rel"] for item in items]
    colors = ["#e74c3c"] * len(bottom3) + ["#2ecc71"] * len(top3)
    y_labels = []
    for i, item in enumerate(items):
        q = item.get("search_query", item.get("search_queries", ["?"])[0] if isinstance(item.get("search_queries"), list) else "?")[:30]
        snippet = item["snippet_preview"][:40].replace("\n", " ")
        label = f"{'TOP' if i >= len(bottom3) else 'BOT'} | {q}"
        y_labels.append(label)
    y_pos = range(len(items))
    bars = ax.barh(y_pos, rels, color=colors, alpha=0.8, edgecolor="white")
    for bar, rel in zip(bars, rels):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{rel:.3f}", va="center", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Rel Score")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axvline(x=0.7, color="red", linestyle="--", alpha=0.4)


def plot_cross_comparison(ax, results):
    if len(results) < 2:
        ax.text(0.5, 0.5, "Need >= 2 questions", ha="center", va="center", transform=ax.transAxes)
        return
    data = []
    labels = []
    colors_list = []
    palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for i, r in enumerate(results):
        if r["all_rels"]:
            data.append(r["all_rels"])
            q_preview = r["question"][:40]
            labels.append(f"Q{i+1}(L{r['line_idx']}): {q_preview}")
            colors_list.append(palette[i % len(palette)])

    parts = ax.violinplot(data, positions=range(len(data)), showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.6)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Rel Score")
    ax.set_title("Cross-Question Rel Distribution", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.7, color="red", linestyle="--", alpha=0.4)


def visualize(input_path, output_dir=None):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    if not results:
        print("No results to visualize")
        return

    if output_dir is None:
        output_dir = str(Path(input_path).parent)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_questions = len(results)
    fig, axes = plt.subplots(n_questions, 4, figsize=(28, 6 * n_questions))
    if n_questions == 1:
        axes = axes.reshape(1, -1)

    for i, r in enumerate(results):
        q_short = r["question"][:30]
        rels = r["all_rels"]
        plot_histogram(axes[i, 0], rels, f"Q{i+1} (Line {r['line_idx']}): Rel Distribution\n{q_short}...")
        plot_box_per_call(axes[i, 1], r["per_call_stats"], f"Q{i+1}: Rel per Search Call")
        plot_top_bottom_examples(axes[i, 2], r["top3_highest_rel"], r["bottom3_lowest_rel"], f"Q{i+1}: Top3 (green) / Bottom3 (red)")
        all_call_rels = []
        for cs in r["per_call_stats"].values():
            all_call_rels.extend(cs["individual_rels"])
        if all_call_rels:
            sorted_rels = sorted(all_call_rels)
            axes[i, 3].plot(range(len(sorted_rels)), sorted_rels, "b-", linewidth=1.5, alpha=0.8)
            axes[i, 3].fill_between(range(len(sorted_rels)), sorted_rels, alpha=0.15)
            axes[i, 3].axhline(y=0.7, color="red", linestyle="--", alpha=0.4, label="0.7")
            axes[i, 3].set_xlabel("Snippet Index (sorted)")
            axes[i, 3].set_ylabel("Rel Score")
            axes[i, 3].set_title(f"Q{i+1}: Sorted Rel Curve", fontsize=11, fontweight="bold")
            axes[i, 3].set_ylim(0, 1.05)
            axes[i, 3].legend(fontsize=8)

    plt.tight_layout(pad=2.0)
    fig_path = output_dir / "rel_distribution.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    if len(results) >= 2:
        fig2, ax2 = plt.subplots(1, 2, figsize=(16, 6))
        plot_cross_comparison(ax2[0], results)

        all_rels_all = []
        labels_all = []
        palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
        for i, r in enumerate(results):
            all_rels_all.append(r["all_rels"])
            labels_all.append(f"Q{i+1}(L{r['line_idx']})")
        ax2[1].hist(
            all_rels_all,
            bins=REL_BINS,
            label=labels_all,
            alpha=0.7,
            edgecolor="white",
            color=palette[: len(results)],
        )
        ax2[1].set_xlabel("Rel Score")
        ax2[1].set_ylabel("Count")
        ax2[1].set_title("Overlaid Rel Histograms", fontsize=11, fontweight="bold")
        ax2[1].legend(fontsize=9)
        ax2[1].set_xlim(0, 1.05)

        plt.tight_layout()
        fig2_path = output_dir / "rel_cross_comparison.png"
        fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {fig2_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Rel results")
    parser.add_argument("--input", required=True, help="Path to rel_results.json")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots")
    args = parser.parse_args()
    visualize(args.input, args.output_dir)


if __name__ == "__main__":
    main()
