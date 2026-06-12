"""
Aggregate Rel statistics by correct/incorrect answer groups.

Usage:
    python3 scripts/aggregate_rel_by_correctness.py \
        --results output/deepseek-v4-pro/bc-zn10-control/rel_results_all.json \
        --scored output/deepseek-v4-pro/bc-zn10-control/iter1_scored.jsonl
"""

import json
import argparse
import numpy as np
from pathlib import Path

REL_BINS = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.01]
REL_LABELS = ["[0,0.3)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,1.0]"]


def compute_stats(values):
    a = np.array(values, dtype=np.float64)
    if len(a) == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {
        "count": int(len(a)),
        "mean": round(float(np.mean(a)), 4),
        "std": round(float(np.std(a)), 4),
        "min": round(float(np.min(a)), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "median": round(float(np.median(a)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "max": round(float(np.max(a)), 4),
    }


def compute_histogram(values):
    a = np.array(values, dtype=np.float64)
    if len(a) == 0:
        return {l: 0 for l in REL_LABELS}
    c, _ = np.histogram(a, bins=REL_BINS)
    return {REL_LABELS[i]: int(c[i]) for i in range(len(REL_LABELS))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--scored", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        results_data = json.load(f)

    correctness = {}
    with open(args.scored, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            d = json.loads(line)
            correctness[i] = d.get("is_correct", None)

    results = results_data["results"]
    correct_rels = []
    incorrect_rels = []
    correct_questions = []
    incorrect_questions = []

    per_question = []
    for r in results:
        line_idx = r["line_idx"]
        is_correct = correctness.get(line_idx, None)
        rels = r["all_rels"]
        entry = {
            "line_idx": line_idx,
            "question": r["question"][:60],
            "answer": r["answer"],
            "is_correct": is_correct,
            "num_snippets": r["total_snippets"],
            "num_search_calls": r["num_search_calls"],
            "mean_rel": r["overall_stats"]["mean"],
            "median_rel": r["overall_stats"]["median"],
            "max_rel": r["overall_stats"]["max"],
            "min_rel": r["overall_stats"]["min"],
        }
        per_question.append(entry)

        if is_correct is True:
            correct_rels.extend(rels)
            correct_questions.append(entry)
        elif is_correct is False:
            incorrect_rels.extend(rels)
            incorrect_questions.append(entry)

    print("=" * 90)
    print("Per-Question Summary")
    print("=" * 90)
    print(f"{'Line':>4} {'Correct':>7} {'Snips':>5} {'Calls':>5} {'Mean':>6} {'Med':>6} {'Min':>6} {'Max':>6}  Question")
    print("-" * 90)
    for q in per_question:
        mark = "Y" if q["is_correct"] else "N"
        print(f"{q['line_idx']:>4} {mark:>7} {q['num_snippets']:>5} {q['num_search_calls']:>5} "
              f"{q['mean_rel']:>6.4f} {q['median_rel']:>6.4f} {q['min_rel']:>6.4f} {q['max_rel']:>6.4f}  {q['question']}")

    print("\n" + "=" * 90)
    print("CORRECT Answers Group")
    print("=" * 90)
    cs = compute_stats(correct_rels)
    ch = compute_histogram(correct_rels)
    print(f"  Questions: {len(correct_questions)}")
    print(f"  Total snippets: {len(correct_rels)}")
    print(f"  Stats: count={cs['count']}  mean={cs['mean']}  std={cs['std']}")
    print(f"         min={cs['min']}  p25={cs['p25']}  median={cs['median']}  p75={cs['p75']}  max={cs['max']}")
    print(f"  Histogram: {ch}")
    pct_above_05 = sum(1 for r in correct_rels if r >= 0.5) / len(correct_rels) * 100 if correct_rels else 0
    pct_above_07 = sum(1 for r in correct_rels if r >= 0.7) / len(correct_rels) * 100 if correct_rels else 0
    print(f"  Rel >= 0.5: {pct_above_05:.1f}%")
    print(f"  Rel >= 0.7: {pct_above_07:.1f}%")

    print("\n" + "=" * 90)
    print("INCORRECT Answers Group")
    print("=" * 90)
    ist = compute_stats(incorrect_rels)
    ih = compute_histogram(incorrect_rels)
    print(f"  Questions: {len(incorrect_questions)}")
    print(f"  Total snippets: {len(incorrect_rels)}")
    print(f"  Stats: count={ist['count']}  mean={ist['mean']}  std={ist['std']}")
    print(f"         min={ist['min']}  p25={ist['p25']}  median={ist['median']}  p75={ist['p75']}  max={ist['max']}")
    print(f"  Histogram: {ih}")
    pct_above_05i = sum(1 for r in incorrect_rels if r >= 0.5) / len(incorrect_rels) * 100 if incorrect_rels else 0
    pct_above_07i = sum(1 for r in incorrect_rels if r >= 0.7) / len(incorrect_rels) * 100 if incorrect_rels else 0
    print(f"  Rel >= 0.5: {pct_above_05i:.1f}%")
    print(f"  Rel >= 0.7: {pct_above_07i:.1f}%")

    print("\n" + "=" * 90)
    print("Comparison: Correct vs Incorrect")
    print("=" * 90)
    if cs["mean"] is not None and ist["mean"] is not None:
        delta_mean = cs["mean"] - ist["mean"]
        delta_median = cs["median"] - ist["median"]
        print(f"  Mean Rel:    Correct={cs['mean']:.4f}  Incorrect={ist['mean']:.4f}  Delta={delta_mean:+.4f}")
        print(f"  Median Rel:  Correct={cs['median']:.4f}  Incorrect={ist['median']:.4f}  Delta={delta_median:+.4f}")
        print(f"  Std Rel:     Correct={cs['std']:.4f}  Incorrect={ist['std']:.4f}")
        print(f"  Rel>=0.5:    Correct={pct_above_05:.1f}%  Incorrect={pct_above_05i:.1f}%")
        print(f"  Rel>=0.7:    Correct={pct_above_07:.1f}%  Incorrect={pct_above_07i:.1f}%")

    output = {
        "correct": {
            "num_questions": len(correct_questions),
            "total_snippets": len(correct_rels),
            "stats": cs,
            "histogram": ch,
            "pct_above_05": round(pct_above_05, 2),
            "pct_above_07": round(pct_above_07, 2),
            "per_question": correct_questions,
        },
        "incorrect": {
            "num_questions": len(incorrect_questions),
            "total_snippets": len(incorrect_rels),
            "stats": ist,
            "histogram": ih,
            "pct_above_05": round(pct_above_05i, 2),
            "pct_above_07": round(pct_above_07i, 2),
            "per_question": incorrect_questions,
        },
    }

    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.results).parent / "rel_grouped_stats.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
