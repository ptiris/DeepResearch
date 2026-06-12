"""
Compute Rel = cosine_sim(emb_question, emb_response_snippet) for DeepResearch results.

Usage:
    uv run python scripts/compute_rel.py \
        --input output/deepseek-v4-pro/bc-zn10-control/iter1.jsonl \
        --lines 1 2 \
        --output scripts/rel_results.json

Reads EMBEDDING_PROVIDER / DASHSCOPE_API_KEY from .env (via dotenv or env).
"""

import json
import re
import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path
from openai import OpenAI

_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", os.getenv("PROVIDER", "dashscope"))

_PROVIDER_API_CONFIGS = {
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url_env": "OPENROUTER_BASE_URL",
        "base_url_default": "https://openrouter.ai/api/v1",
    },
    "dashscope": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "base_url_default": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_API_BASE",
        "base_url_default": "https://api.openai.com/v1",
    },
}

_config = _PROVIDER_API_CONFIGS.get(_EMBEDDING_PROVIDER, _PROVIDER_API_CONFIGS["dashscope"])
API_KEY = os.getenv(_config["api_key_env"], "")
BASE_URL = os.getenv(_config["base_url_env"], _config["base_url_default"])
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024


def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


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
        try:
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
        except Exception as e:
            print(f"[Embedding] Error encoding batch {start}-{end}: {e}")
            for _ in range(len(batch)):
                all_embs.append(np.zeros(EMBEDDING_DIM, dtype=np.float32).tolist())
    return np.array(all_embs, dtype=np.float32)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


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


def _parse_search_call_from_content(content):
    content = content.strip()
    if content.startswith("senal") or content.startswith("senal\n"):
        json_str = content.split("\n", 1)[-1].strip() if "\n" in content else ""
    else:
        json_str = content
    try:
        obj = json.loads(json_str)
        if obj.get("name") == "aliyun_search":
            queries = obj.get("arguments", {}).get("query", [])
            if isinstance(queries, str):
                queries = [queries]
            return queries
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def _split_observation_by_query(obs_content):
    header_pat = re.compile(r"An Aliyun IQS search for '([^']+)' found (\d+) results:")
    headers = list(header_pat.finditer(obs_content))
    if not headers:
        return []

    per_query = []
    for i, m in enumerate(headers):
        query = m.group(1)
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(obs_content)
        block = obs_content[start:end].strip()
        snippets = split_individual_results(block)
        per_query.append({"query": query, "snippets": snippets, "raw": block[:500]})

    return per_query


def extract_search_data_from_messages(messages):
    search_call_queries = []
    observation_contents = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant" and content:
            queries = _parse_search_call_from_content(content)
            if queries is not None:
                search_call_queries.append(queries)

        if role == "assistant":
            tc_list = msg.get("tool_calls", [])
            for tc in tc_list:
                fn = tc.get("function", {})
                if fn.get("name") == "aliyun_search":
                    args_str = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args = {}
                    queries = args.get("query", [])
                    if isinstance(queries, str):
                        queries = [queries]
                    search_call_queries.append(queries)

        if role == "user" and "An Aliyun IQS search for" in (content or ""):
            observation_contents.append(content)

    paired = []
    for sc_queries, obs_content in zip(search_call_queries, observation_contents):
        per_query = _split_observation_by_query(obs_content)
        all_snippets = []
        for pq in per_query:
            all_snippets.extend(pq["snippets"])
        paired.append(
            {
                "search_queries": sc_queries,
                "per_query_results": per_query,
                "raw_results": obs_content[:500] if obs_content else "",
                "snippets": all_snippets,
                "num_snippets": len(all_snippets),
            }
        )

    if len(observation_contents) > len(search_call_queries):
        for obs_content in observation_contents[len(search_call_queries):]:
            per_query = _split_observation_by_query(obs_content)
            all_snippets = []
            for pq in per_query:
                all_snippets.extend(pq["snippets"])
            paired.append(
                {
                    "search_queries": [pq["query"] for pq in per_query],
                    "per_query_results": per_query,
                    "raw_results": obs_content[:500],
                    "snippets": all_snippets,
                    "num_snippets": len(all_snippets),
                }
            )

    return paired


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
    bins = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.01]
    labels = ["[0,0.3)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,1.0]"]
    a = np.array(values, dtype=np.float64)
    if len(a) == 0:
        return {l: 0 for l in labels}
    c, _ = np.histogram(a, bins=bins)
    return {labels[i]: int(c[i]) for i in range(len(labels))}


def process_question(data, line_idx):
    question = data["question"]
    answer = data.get("answer", "")
    messages = data.get("messages", [])

    print(f"\n{'='*80}")
    print(f"Line {line_idx}: {question[:80]}...")
    print(f"Answer: {answer}")
    print(f"{'='*80}")

    search_data = extract_search_data_from_messages(messages)
    print(f"  Found {len(search_data)} search calls")

    all_snippet_texts = []
    snippet_meta = []

    for s_idx, sd in enumerate(search_data):
        for pq in sd.get("per_query_results", []):
            if not pq["snippets"]:
                continue
            for sn_idx, snippet in enumerate(pq["snippets"]):
                all_snippet_texts.append(snippet)
                snippet_meta.append(
                    {
                        "search_call_idx": s_idx,
                        "search_query": pq["query"],
                        "snippet_idx_in_query": sn_idx,
                        "snippet_preview": snippet[:300],
                    }
                )

    if not all_snippet_texts:
        for s_idx, sd in enumerate(search_data):
            for sn_idx, snippet in enumerate(sd["snippets"]):
                all_snippet_texts.append(snippet)
                snippet_meta.append(
                    {
                        "search_call_idx": s_idx,
                        "search_query": sd["search_queries"][0] if sd["search_queries"] else "",
                        "snippet_idx_in_query": sn_idx,
                        "snippet_preview": snippet[:300],
                    }
                )

    print(f"  Total snippets: {len(all_snippet_texts)}")

    print("  Encoding question...")
    q_emb = encode_texts([question], batch_size=1)[0]

    print("  Encoding snippets...")
    if all_snippet_texts:
        sn_embs = encode_texts(all_snippet_texts, batch_size=6)
    else:
        sn_embs = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    all_rels = []
    per_call_rels = {}
    per_query_rels = {}

    for i, meta in enumerate(snippet_meta):
        if i < len(sn_embs):
            rel = cosine_sim(q_emb, sn_embs[i])
        else:
            rel = 0.0
        meta["rel"] = round(rel, 4)
        all_rels.append(rel)

        call_idx = meta["search_call_idx"]
        per_call_rels.setdefault(call_idx, []).append(rel)

        sq = meta["search_query"]
        per_query_rels.setdefault(sq, []).append(rel)

    rel_with_meta = list(zip(all_rels, snippet_meta))
    rel_with_meta.sort(key=lambda x: x[0], reverse=True)

    top3 = []
    for rel, meta in rel_with_meta[:3]:
        top3.append(
            {
                "rel": round(rel, 4),
                "search_query": meta["search_query"],
                "snippet_preview": meta["snippet_preview"],
            }
        )

    bottom3 = []
    for rel, meta in rel_with_meta[-3:]:
        bottom3.append(
            {
                "rel": round(rel, 4),
                "search_query": meta["search_query"],
                "snippet_preview": meta["snippet_preview"],
            }
        )

    per_call_stats = {}
    for call_idx, rels in per_call_rels.items():
        per_call_stats[str(call_idx)] = {
            "search_queries": search_data[call_idx]["search_queries"],
            "num_snippets": len(rels),
            "stats": compute_stats(rels),
            "histogram": compute_histogram(rels),
            "individual_rels": [round(r, 4) for r in rels],
        }

    per_query_stats = {}
    for sq, rels in per_query_rels.items():
        per_query_stats[sq] = {
            "num_snippets": len(rels),
            "stats": compute_stats(rels),
            "histogram": compute_histogram(rels),
            "individual_rels": [round(r, 4) for r in rels],
        }

    result = {
        "line_idx": line_idx,
        "question": question,
        "answer": answer,
        "num_search_calls": len(search_data),
        "total_snippets": len(all_snippet_texts),
        "overall_stats": compute_stats(all_rels),
        "overall_histogram": compute_histogram(all_rels),
        "per_call_stats": per_call_stats,
        "per_query_stats": per_query_stats,
        "top3_highest_rel": top3,
        "bottom3_lowest_rel": bottom3,
        "all_rels": [round(r, 4) for r in all_rels],
    }

    return result


def print_result(result):
    print(f"\n{'='*80}")
    print(f"Line {result['line_idx']}: {result['question'][:80]}")
    print(f"Answer: {result['answer']}")
    print(f"Search calls: {result['num_search_calls']}, Total snippets: {result['total_snippets']}")
    print(f"{'='*80}")

    s = result["overall_stats"]
    print(f"\n  Overall Rel Distribution:")
    print(f"    count={s['count']}  mean={s['mean']}  std={s['std']}")
    print(f"    min={s['min']}  p25={s['p25']}  median={s['median']}  p75={s['p75']}  max={s['max']}")
    print(f"    histogram: {result['overall_histogram']}")

    print(f"\n  Per Search Query Breakdown:")
    for sq, qs in result["per_query_stats"].items():
        st = qs["stats"]
        print(f"    Query: \"{sq}\"")
        print(f"      snippets={qs['num_snippets']}  mean={st['mean']}  min={st['min']}  max={st['max']}  median={st['median']}")
        print(f"      rels={qs['individual_rels']}")

    print(f"\n  Per Search Call Breakdown:")
    for call_idx, cs in result["per_call_stats"].items():
        st = cs["stats"]
        print(f"    Call {call_idx}: queries={cs['search_queries']}")
        print(f"      snippets={cs['num_snippets']}  mean={st['mean']}  min={st['min']}  max={st['max']}")

    print(f"\n  Top-3 Highest Rel:")
    for i, t in enumerate(result["top3_highest_rel"]):
        print(f"    #{i+1}  Rel={t['rel']}")
        print(f"      query: {t['search_query']}")
        print(f"      snippet: {t['snippet_preview'][:200]}...")

    print(f"\n  Bottom-3 Lowest Rel:")
    for i, b in enumerate(result["bottom3_lowest_rel"]):
        print(f"    #{i+1}  Rel={b['rel']}")
        print(f"      query: {b['search_query']}")
        print(f"      snippet: {b['snippet_preview'][:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Compute Rel = cosine_sim(emb_question, emb_response_snippet)")
    parser.add_argument("--input", required=True, help="Path to iter1.jsonl")
    parser.add_argument("--lines", type=int, nargs="+", default=[1, 2], help="1-indexed line numbers to analyze")
    parser.add_argument("--output", default=None, help="Output JSON path (default: <input_dir>/rel_results.json)")
    args = parser.parse_args()

    dotenv_path = Path(__file__).parent.parent / ".env"
    if dotenv_path.exists():
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)
        global API_KEY, BASE_URL
        _config2 = _PROVIDER_API_CONFIGS.get(
            os.getenv("EMBEDDING_PROVIDER", os.getenv("PROVIDER", "dashscope")),
            _PROVIDER_API_CONFIGS["dashscope"],
        )
        API_KEY = os.getenv(_config2["api_key_env"], "")
        BASE_URL = os.getenv(_config2["base_url_env"], _config2["base_url_default"])

    print(f"Using embedding model: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"API base: {BASE_URL}")
    print(f"API key: {API_KEY[:8]}...")

    input_path = Path(args.input)
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    for line_idx in args.lines:
        if line_idx < 1 or line_idx > len(lines):
            print(f"Warning: line {line_idx} out of range (1-{len(lines)}), skipping")
            continue
        data = json.loads(lines[line_idx - 1])
        result = process_question(data, line_idx)
        print_result(result)
        results.append(result)

    if len(results) >= 2:
        all_rels_q1 = results[0]["all_rels"]
        all_rels_q2 = results[1]["all_rels"]
        print(f"\n{'='*80}")
        print(f"Cross-Question Comparison:")
        print(f"  Q1 (line {results[0]['line_idx']}): mean={results[0]['overall_stats']['mean']}, "
              f"median={results[0]['overall_stats']['median']}, "
              f"std={results[0]['overall_stats']['std']}")
        print(f"  Q2 (line {results[1]['line_idx']}): mean={results[1]['overall_stats']['mean']}, "
              f"median={results[1]['overall_stats']['median']}, "
              f"std={results[1]['overall_stats']['std']}")

    output_path = args.output
    if output_path is None:
        output_path = str(input_path.parent / "rel_results.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
