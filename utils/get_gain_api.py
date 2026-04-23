"""
读取 get_query_results.py 输出的 JSONL，计算 query pair 之间的相似度和信息增益。
（使用阿里云百炼 text-embedding-v4 API 版本）

对每个 turn 内的 query 组成 pair (qi, qj)，以及当前 turn query 与历史所有跨轮 query 组成 pair，
计算：
  - sim_query:   query embedding 余弦相似度
  - sim_results: results embedding 余弦相似度
  - gain:        qj 相对于 qi 的信息增益
      url_overlap = |urls(qi) ∩ urls(qj)| / |urls(qi)|
      cover_j_i   = mean( max_sim(r, r') for r in qj_individual_results )
      gain        = α*(1-cover_j_i) + (1-α)*(1-url_overlap)

输出三个文件：
  - {output}             详细 pair 数据
  - {output}_stats.json  intra/cross 分开的分布统计
  - {output}_all.csv     所有 pair 数据 CSV，方便画 CDF

用法:
    uv run python scripts/get_gain_api.py --input ./output/*_query_results.jsonl
"""

import json
import re
import csv
import argparse
import os
import time
import numpy as np
from urllib.parse import urlparse
from openai import OpenAI

DASHSCOPE_API_KEY = "sk-0086925a04174bfb82bda5841b5af595"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024
ALPHA = 1


def get_client():
    return OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
    )


def encode_texts(texts, batch_size=6):
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    if batch_size > 10:
        batch_size = 10
    client = get_client()
    all_embs = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIM,
        )
        for item in resp.data:
            all_embs.append(item.embedding)
        print(f"    encoded: {end}/{len(texts)}", flush=True)
        if end < len(texts):
            time.sleep(0.1)
    return np.array(all_embs, dtype=np.float32)


def extract_urls(text):
    urls = set()
    for _, url in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text):
        try:
            p = urlparse(url)
            if p.scheme and p.netloc:
                urls.add(url)
        except Exception:
            pass
    for m in re.finditer(r"^URL:\s*(.+)$", text, re.MULTILINE):
        url = m.group(1).strip()
        try:
            p = urlparse(url)
            if p.scheme and p.netloc:
                urls.add(url)
        except Exception:
            pass
    return urls


def split_individual_results(results_text):
    if not results_text:
        return []
    if "## Web Results" in results_text:
        text = re.sub(r"^## Web Results\n", "", results_text)
        entries = re.split(r"\n\n(?=\d+\.\s)", text)
    elif "Title:" in results_text:
        entries = re.split(r"\n---+\n", results_text)
        entries = [e for e in entries if e.strip()]
    else:
        entries = []
    return [e.strip() for e in entries if e.strip()]


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def compute_pair(qi, qj):
    sim_query = cosine_sim(qi["q_emb"], qj["q_emb"])

    if qi["results"] and qj["results"]:
        sim_results = cosine_sim(qi["r_emb"], qj["r_emb"])
    else:
        sim_results = 0.0

    url_overlap = len(qi["urls"] & qj["urls"]) / len(qi["urls"]) if qi["urls"] else 0.0

    qj_ind = qj["ind_embs"]
    qi_ind = qi["ind_embs"]
    if len(qj_ind) > 0 and len(qi_ind) > 0:
        qj_n = qj_ind / (np.linalg.norm(qj_ind, axis=1, keepdims=True) + 1e-10)
        qi_n = qi_ind / (np.linalg.norm(qi_ind, axis=1, keepdims=True) + 1e-10)
        sim_mat = np.dot(qj_n, qi_n.T)
        cover_j_i = float(np.mean(np.max(sim_mat, axis=1)))
        ratio_j_i = float(np.mean(np.max(sim_mat, axis=1) >= 0.6))
        sim_mat_i_j = np.dot(qi_n, qj_n.T)
        cover_i_j = float(np.mean(np.max(sim_mat_i_j, axis=0)))
        ratio_i_j = float(np.mean(np.max(sim_mat_i_j, axis=0) >= 0.6))
    else:
        cover_j_i = 0.0
        cover_i_j = 0.0
        ratio_j_i = 0.0
        ratio_i_j = 0.0

    sim_results = (cover_j_i + cover_i_j) / 2.0
    sim_results_ratio = (ratio_j_i + ratio_i_j) / 2.0
    gain = ALPHA * (1 - cover_j_i) + (1 - ALPHA) * (1 - url_overlap)

    return {
        "pair": [qi["query"], qj["query"]],
        "sim_query": round(sim_query, 4),
        "sim_results": round(sim_results, 4),
        "sim_results_ratio": round(sim_results_ratio, 4),
        "gain": round(gain, 4),
        "redundant": is_redundant(
            {
                "sim_query": round(sim_query, 4),
                "sim_results": round(sim_results, 4),
                "gain": round(gain, 4),
            }
        ),
    }


def is_redundant(p):
    return p["sim_query"] >= 0.7 and p["sim_results"] >= 0.8


def print_stats(
    intra_sq,
    intra_sr,
    intra_srr,
    intra_g,
    cross_sq,
    cross_sr,
    cross_srr,
    cross_g,
    redundancy_summary,
    stats_fp,
):
    def stat(arr):
        a = np.array(arr)
        if len(a) == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "p25": None,
                "median": None,
                "p75": None,
                "max": None,
            }
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

    bins = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.01]
    labels = [
        "[0,0.3)",
        "[0.3,0.5)",
        "[0.5,0.7)",
        "[0.7,0.8)",
        "[0.8,0.9)",
        "[0.9,1.0]",
    ]

    def hist(arr):
        if len(arr) == 0:
            return {l: 0 for l in labels}
        c, _ = np.histogram(arr, bins=bins)
        return {labels[i]: int(c[i]) for i in range(len(labels))}

    result = {}
    for tag, sq, sr, srr, sg in [
        ("intra", intra_sq, intra_sr, intra_srr, intra_g),
        ("cross", cross_sq, cross_sr, cross_srr, cross_g),
    ]:
        result[tag] = {}
        for name, arr in [
            ("sim_query", sq),
            ("sim_results", sr),
            ("sim_results_ratio", srr),
            ("gain", sg),
        ]:
            s = stat(arr)
            s["histogram"] = hist(arr)
            result[tag][name] = s

    result["redundancy_summary"] = redundancy_summary

    print("=" * 60)
    print("Distribution Summary")
    print("=" * 60)
    for tag in ["intra", "cross"]:
        print(f"\n  [{tag}]")
        for name, s in result[tag].items():
            print(f"    {name}:")
            print(f"      count={s['count']}  mean={s['mean']}  std={s['std']}")
            print(
                f"      min={s['min']}  p25={s['p25']}  median={s['median']}  p75={s['p75']}  max={s['max']}"
            )
            print(f"      histogram: {s['histogram']}")

    print(f"\n  [Redundancy]  (sim_query>=0.7 & sim_results>=0.7 & gain<=0.3)")
    print(
        f"    {'question':<55s} {'red':>4s} {'intra':>5s} {'cross':>5s} {'both':>4s} {'total':>5s} {'ratio':>6s} {'t_red':>5s} {'t_tot':>5s} {'t_rat':>6s}"
    )
    print(
        f"    {'-' * 55} {'-' * 4} {'-' * 5} {'-' * 5} {'-' * 4} {'-' * 5} {'-' * 6} {'-' * 5} {'-' * 5} {'-' * 6}"
    )
    for item in redundancy_summary["per_question"]:
        print(
            f"    {item['question'][:55]:<55s} {item['redundant_count']:>4d} {item['intra_only_count']:>5d} {item['cross_only_count']:>5d} {item['both_count']:>4d} {item['total_queries']:>5d} {item['ratio']:>6.2%} {item['turns_with_intra_redundant']:>5d} {item['total_turns']:>5d} {item['turns_with_intra_ratio']:>6.2%}"
        )
    s = redundancy_summary
    print(
        f"    {'TOTAL':<55s} {s['total_redundant']:>4d} {s['total_intra_only']:>5d} {s['total_cross_only']:>5d} {s['total_both']:>4d} {s['total_queries']:>5d} {s['total_ratio']:>6.2%} {s['total_turns_with_intra_red']:>5d} {s['total_turns']:>5d} {s['turns_with_intra_ratio']:>6.2%}"
    )
    print(f"\n    intra_only / total_red  = {s['intra_only_ratio']:.2%}")
    print(f"    cross_only / total_red  = {s['cross_only_ratio']:.2%}")
    print(f"    both        / total_red = {s['both_ratio']:.2%}")

    with open(stats_fp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nStats saved to {stats_fp}")


def process(input_fp, output_fp):
    print(f"Using API model: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    output_fp = input_fp.replace(".jsonl", "_gain.jsonl")
    stats_fp = output_fp.replace(".jsonl", "_stats.json")
    csv_fp = output_fp.replace(".jsonl", "_all.csv")

    intra_sq, intra_sr, intra_srr, intra_g = [], [], [], []
    cross_sq, cross_sr, cross_srr, cross_g = [], [], [], []
    redundancy_summary = {
        "per_question": [],
        "total_redundant": 0,
        "total_queries": 0,
        "total_ratio": 0.0,
    }

    with (
        open(input_fp, "r", encoding="utf-8") as fin,
        open(output_fp, "w", encoding="utf-8") as fout,
        open(csv_fp, "w", encoding="utf-8", newline="") as fcsv,
    ):
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "type",
                "question",
                "turn",
                "qi",
                "qj",
                "sim_query",
                "sim_results",
                "sim_results_ratio",
                "gain",
                "redundant",
            ]
        )

        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line)
            question = data["question"]
            turns = data.get("turns", [])
            if not turns:
                continue

            # ---- collect all texts for this question ----
            all_q_texts, all_r_texts = [], []
            all_ind_texts = []
            query_ind_ranges = []

            for t in turns:
                rmap = {}
                for qr in t["query_results"]:
                    rmap.update(qr)
                for q in t["query_list"]:
                    all_q_texts.append(q)
                    r = rmap.get(q, "")
                    all_r_texts.append(r)
                    indiv = split_individual_results(r)
                    start = len(all_ind_texts)
                    all_ind_texts.extend(indiv)
                    query_ind_ranges.append((start, len(all_ind_texts)))

            print(
                f"  Encoding {len(all_q_texts)} queries, {len(all_r_texts)} results, {len(all_ind_texts)} individual entries ..."
            )

            # ---- encode via API ----
            print("  Encoding queries ...")
            q_embs = encode_texts(all_q_texts, batch_size=6)
            print("  Encoding results ...")
            r_embs = encode_texts(all_r_texts, batch_size=6)

            print("  Encoding individual results ...")
            ind_all_embs = encode_texts(all_ind_texts, batch_size=6)

            # ---- build per-query entry ----
            idx = 0
            turn_entries = []
            all_queries_this_question = []
            for t in turns:
                rmap = {}
                for qr in t["query_results"]:
                    rmap.update(qr)
                entries = []
                for q in t["query_list"]:
                    r = rmap.get(q, "")
                    urls = extract_urls(r)
                    s, e = query_ind_ranges[idx]
                    ind_embs = (
                        ind_all_embs[s:e] if e > s else np.zeros((0, EMBEDDING_DIM))
                    )
                    entries.append(
                        {
                            "query": q,
                            "results": r,
                            "urls": urls,
                            "ind_embs": ind_embs,
                            "q_emb": q_embs[idx],
                            "r_emb": r_embs[idx],
                        }
                    )
                    all_queries_this_question.append(q)
                    idx += 1
                turn_entries.append((t["turn"], entries))

            # ---- compute pairs ----
            history = []
            out_turns = []
            redundant_queries = {}
            turns_with_intra_red = 0
            total_turns_count = 0

            for turn, entries in turn_entries:
                intra, cross = [], []
                has_intra_red = False
                total_turns_count += 1

                for i in range(len(entries)):
                    for j in range(i + 1, len(entries)):
                        p = compute_pair(entries[i], entries[j])
                        intra.append(p)
                        intra_sq.append(p["sim_query"])
                        intra_sr.append(p["sim_results"])
                        intra_srr.append(p["sim_results_ratio"])
                        intra_g.append(p["gain"])
                        if p["redundant"]:
                            qj = entries[j]["query"]
                            redundant_queries.setdefault(qj, set()).add("intra")
                            has_intra_red = True
                        writer.writerow(
                            [
                                "intra",
                                question,
                                turn,
                                entries[i]["query"],
                                entries[j]["query"],
                                p["sim_query"],
                                p["sim_results"],
                                p["sim_results_ratio"],
                                p["gain"],
                                p["redundant"],
                            ]
                        )

                for curr in entries:
                    for h in history:
                        p = compute_pair(h, curr)
                        cross.append(p)
                        cross_sq.append(p["sim_query"])
                        cross_sr.append(p["sim_results"])
                        cross_srr.append(p["sim_results_ratio"])
                        cross_g.append(p["gain"])
                        if p["redundant"]:
                            q = curr["query"]
                            redundant_queries.setdefault(q, set()).add("cross")
                        writer.writerow(
                            [
                                "cross",
                                question,
                                turn,
                                h["query"],
                                curr["query"],
                                p["sim_query"],
                                p["sim_results"],
                                p["sim_results_ratio"],
                                p["gain"],
                                p["redundant"],
                            ]
                        )

                if has_intra_red:
                    turns_with_intra_red += 1

                out_turns.append(
                    {
                        "turn": turn,
                        "intra_pairs": intra,
                        "cross_pairs": cross,
                    }
                )
                history.extend(entries)

            total_q = len(all_queries_this_question)
            red_count = len(redundant_queries)
            intra_only = sum(1 for v in redundant_queries.values() if v == {"intra"})
            cross_only = sum(1 for v in redundant_queries.values() if v == {"cross"})
            both = sum(1 for v in redundant_queries.values() if v == {"intra", "cross"})
            redundancy_summary["per_question"].append(
                {
                    "question": question,
                    "redundant_count": red_count,
                    "total_queries": total_q,
                    "ratio": round(red_count / total_q, 4) if total_q > 0 else 0.0,
                    "intra_only_count": intra_only,
                    "cross_only_count": cross_only,
                    "both_count": both,
                    "turns_with_intra_redundant": turns_with_intra_red,
                    "total_turns": total_turns_count,
                    "turns_with_intra_ratio": round(
                        turns_with_intra_red / total_turns_count, 4
                    )
                    if total_turns_count > 0
                    else 0.0,
                    "redundant_queries": {
                        q: sorted(v) for q, v in redundant_queries.items()
                    },
                }
            )

            fout.write(
                json.dumps(
                    {
                        "question": question,
                        "turns": out_turns,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    grand_total = sum(
        item["total_queries"] for item in redundancy_summary["per_question"]
    )
    grand_red = sum(
        item["redundant_count"] for item in redundancy_summary["per_question"]
    )
    grand_intra_only = sum(
        item["intra_only_count"] for item in redundancy_summary["per_question"]
    )
    grand_cross_only = sum(
        item["cross_only_count"] for item in redundancy_summary["per_question"]
    )
    grand_both = sum(item["both_count"] for item in redundancy_summary["per_question"])
    grand_turns = sum(
        item["total_turns"] for item in redundancy_summary["per_question"]
    )
    grand_turns_intra = sum(
        item["turns_with_intra_redundant"]
        for item in redundancy_summary["per_question"]
    )
    redundancy_summary["total_queries"] = grand_total
    redundancy_summary["total_redundant"] = grand_red
    redundancy_summary["total_ratio"] = (
        round(grand_red / grand_total, 4) if grand_total > 0 else 0.0
    )
    redundancy_summary["total_intra_only"] = grand_intra_only
    redundancy_summary["total_cross_only"] = grand_cross_only
    redundancy_summary["total_both"] = grand_both
    redundancy_summary["intra_only_ratio"] = (
        round(grand_intra_only / grand_red, 4) if grand_red > 0 else 0.0
    )
    redundancy_summary["cross_only_ratio"] = (
        round(grand_cross_only / grand_red, 4) if grand_red > 0 else 0.0
    )
    redundancy_summary["both_ratio"] = (
        round(grand_both / grand_red, 4) if grand_red > 0 else 0.0
    )
    redundancy_summary["total_turns"] = grand_turns
    redundancy_summary["total_turns_with_intra_red"] = grand_turns_intra
    redundancy_summary["turns_with_intra_ratio"] = (
        round(grand_turns_intra / grand_turns, 4) if grand_turns > 0 else 0.0
    )

    print(f"Done. -> {output_fp}")
    print(f"CSV   -> {csv_fp}")
    print_stats(
        intra_sq,
        intra_sr,
        intra_srr,
        intra_g,
        cross_sq,
        cross_sr,
        cross_srr,
        cross_g,
        redundancy_summary,
        stats_fp,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="计算 query pair 相似度和信息增益 (API版)")
    p.add_argument("--input", required=True, help="get_query_results.py 输出的 JSONL")
    args = p.parse_args()
    process(args.input, None)
