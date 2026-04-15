import argparse
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

ITER_FILE_PATTERN = re.compile(r"^iter(?P<iter>\d+)(?:_split(?P<split>\d+)of(?P<total>\d+))?\.jsonl$")

MODEL_GROUPS = ["research_model", "summary_model"]
EXPECTED_TOP_LEVEL_METRIC_KEYS = {
    "model_metrics",
    "search_tool_metrics",
    "other_tool_metrics",
    "prompt_breakdown",
}


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def list_metric_result_files(dataset_dir: str) -> List[str]:
    files: List[Tuple[int, int, str]] = []
    for name in os.listdir(dataset_dir):
        match = ITER_FILE_PATTERN.match(name)
        if not match:
            continue
        iter_idx = safe_int(match.group("iter"), 0)
        split_idx = safe_int(match.group("split"), 0)
        files.append((iter_idx, split_idx, os.path.join(dataset_dir, name)))

    files.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[2] for item in files]


def new_model_bucket() -> Dict[str, Any]:
    return {
        "calls": 0,
        "success_calls": 0,
        "failed_calls": 0,
        "latency_total_ms": 0.0,
        "tokens": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        },
    }


def new_tool_bucket(category: str) -> Dict[str, Any]:
    return {
        "category": category,
        "calls": 0,
        "effective_calls": 0,
        "success_calls": 0,
        "failed_calls": 0,
        "latency_total_ms": 0.0,
    }


def update_model_metrics(aggregator: Dict[str, Any], model_metrics: Dict[str, Any], stats: Dict[str, int]) -> None:
    for group in MODEL_GROUPS:
        source = model_metrics.get(group, {})
        if source and not isinstance(source, dict):
            stats["incomplete_metrics_samples"] += 1
            continue

        target = aggregator[group]
        target["calls"] += safe_int(source.get("calls", 0))
        target["success_calls"] += safe_int(source.get("success_calls", 0))
        target["failed_calls"] += safe_int(source.get("failed_calls", 0))

        latency = source.get("latency_ms", {}) if isinstance(source.get("latency_ms", {}), dict) else {}
        target["latency_total_ms"] += safe_float(latency.get("total", 0.0))

        token_obj = source.get("tokens", {}) if isinstance(source.get("tokens", {}), dict) else {}
        target["tokens"]["prompt_tokens"] += safe_int(token_obj.get("prompt_tokens", 0))
        target["tokens"]["completion_tokens"] += safe_int(token_obj.get("completion_tokens", 0))
        target["tokens"]["total_tokens"] += safe_int(token_obj.get("total_tokens", 0))
        target["tokens"]["cached_tokens"] += safe_int(token_obj.get("cached_tokens", 0))


def update_tool_metrics(
    aggregator: Dict[str, Dict[str, Dict[str, Any]]],
    metrics_section: Dict[str, Any],
    default_category: str,
    stats: Dict[str, int],
) -> None:
    if not isinstance(metrics_section, dict):
        stats["incomplete_metrics_samples"] += 1
        return

    by_tool = metrics_section.get("by_tool", {})
    if not isinstance(by_tool, dict):
        stats["incomplete_metrics_samples"] += 1
        return

    for tool_name, tool_data in by_tool.items():
        if not isinstance(tool_data, dict):
            stats["incomplete_metrics_samples"] += 1
            continue

        category = str(tool_data.get("category", default_category))
        target = aggregator.setdefault(tool_name, new_tool_bucket(category))

        target["calls"] += safe_int(tool_data.get("calls", 0))
        target["effective_calls"] += safe_int(tool_data.get("effective_calls", 0))
        target["success_calls"] += safe_int(tool_data.get("success_calls", 0))
        target["failed_calls"] += safe_int(tool_data.get("failed_calls", 0))

        latency = tool_data.get("latency_ms", {}) if isinstance(tool_data.get("latency_ms", {}), dict) else {}
        target["latency_total_ms"] += safe_float(latency.get("total", 0.0))


def update_prompt_breakdown(
    aggregator: Dict[str, Dict[str, int]],
    prompt_breakdown: Dict[str, Any],
    stats: Dict[str, int],
) -> None:
    if not isinstance(prompt_breakdown, dict):
        stats["incomplete_metrics_samples"] += 1
        return

    for group in MODEL_GROUPS:
        source_group = prompt_breakdown.get(group, {})
        if source_group and not isinstance(source_group, dict):
            stats["incomplete_metrics_samples"] += 1
            continue

        target_group = aggregator[group]
        for category, payload in source_group.items():
            if category == "_total_prompt_tokens":
                continue
            if isinstance(payload, dict):
                token_value = safe_int(payload.get("tokens", 0))
            else:
                token_value = safe_int(payload, 0)
            target_group[category] = target_group.get(category, 0) + token_value


def finalize_model_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for group, bucket in raw.items():
        calls = safe_int(bucket.get("calls", 0))
        success_calls = safe_int(bucket.get("success_calls", 0))
        failed_calls = safe_int(bucket.get("failed_calls", 0))
        latency_total = safe_float(bucket.get("latency_total_ms", 0.0))

        success_rate = (success_calls / calls * 100.0) if calls > 0 else 0.0
        latency_avg = (latency_total / calls) if calls > 0 else 0.0

        tokens = bucket.get("tokens", {})
        output[group] = {
            "calls": calls,
            "success_calls": success_calls,
            "failed_calls": failed_calls,
            "success_rate": round(success_rate, 4),
            "latency_ms": {
                "total": round(latency_total, 4),
                "average": round(latency_avg, 4),
            },
            "tokens": {
                "prompt_tokens": safe_int(tokens.get("prompt_tokens", 0)),
                "completion_tokens": safe_int(tokens.get("completion_tokens", 0)),
                "total_tokens": safe_int(tokens.get("total_tokens", 0)),
                "cached_tokens": safe_int(tokens.get("cached_tokens", 0)),
            },
        }
    return output


def finalize_tool_buckets(by_tool: Dict[str, Dict[str, Any]], category: str) -> Dict[str, Any]:
    finalized_by_tool: Dict[str, Any] = {}

    total_calls = 0
    total_effective_calls = 0
    total_success_calls = 0
    total_failed_calls = 0
    total_latency = 0.0

    for tool_name in sorted(by_tool.keys()):
        bucket = by_tool[tool_name]
        calls = safe_int(bucket.get("calls", 0))
        success_calls = safe_int(bucket.get("success_calls", 0))
        failed_calls = safe_int(bucket.get("failed_calls", 0))
        effective_calls = safe_int(bucket.get("effective_calls", 0))
        latency_total = safe_float(bucket.get("latency_total_ms", 0.0))

        success_rate = (success_calls / calls * 100.0) if calls > 0 else 0.0
        latency_avg = (latency_total / calls) if calls > 0 else 0.0

        finalized_by_tool[tool_name] = {
            "category": str(bucket.get("category", category)),
            "calls": calls,
            "effective_calls": effective_calls,
            "success_calls": success_calls,
            "failed_calls": failed_calls,
            "success_rate": round(success_rate, 4),
            "latency_ms": {
                "total": round(latency_total, 4),
                "average": round(latency_avg, 4),
            },
        }

        total_calls += calls
        total_effective_calls += effective_calls
        total_success_calls += success_calls
        total_failed_calls += failed_calls
        total_latency += latency_total

    agg_success_rate = (total_success_calls / total_calls * 100.0) if total_calls > 0 else 0.0
    agg_latency_avg = (total_latency / total_calls) if total_calls > 0 else 0.0

    return {
        "by_tool": finalized_by_tool,
        "aggregated": {
            "category": category,
            "calls": total_calls,
            "effective_calls": total_effective_calls,
            "success_calls": total_success_calls,
            "failed_calls": total_failed_calls,
            "success_rate": round(agg_success_rate, 4),
            "latency_ms": {
                "total": round(total_latency, 4),
                "average": round(agg_latency_avg, 4),
            },
        },
    }


def finalize_prompt_breakdown(raw: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for group, categories in raw.items():
        group_out: Dict[str, Any] = {}
        total_tokens = sum(max(0, safe_int(v)) for v in categories.values())
        for category in sorted(categories.keys()):
            tokens = max(0, safe_int(categories[category]))
            percentage = (tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0
            group_out[category] = {
                "tokens": tokens,
                "percentage": round(percentage, 4),
            }
        group_out["_total_prompt_tokens"] = total_tokens
        output[group] = group_out
    return output


def summarize_dataset(dataset_dir: str, strict: bool = False) -> Dict[str, Any]:
    warnings: List[str] = []
    source_files = list_metric_result_files(dataset_dir)

    if not source_files:
        warnings.append(f"No iter result files found in {dataset_dir}")

    raw_model_metrics = {group: new_model_bucket() for group in MODEL_GROUPS}
    raw_search_tool_metrics: Dict[str, Dict[str, Any]] = {}
    raw_other_tool_metrics: Dict[str, Dict[str, Any]] = {}
    raw_prompt_breakdown = {group: {} for group in MODEL_GROUPS}

    stats = {
        "total_lines": 0,
        "invalid_json_lines": 0,
        "error_samples": 0,
        "success_without_metrics": 0,
        "valid_metric_samples": 0,
        "incomplete_metrics_samples": 0,
    }

    for file_path in source_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stats["total_lines"] += 1
                text = line.strip()
                if not text:
                    continue

                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    stats["invalid_json_lines"] += 1
                    warnings.append(f"Invalid JSON at {file_path}:{line_no}")
                    continue

                if not isinstance(record, dict):
                    stats["invalid_json_lines"] += 1
                    warnings.append(f"Non-object JSON at {file_path}:{line_no}")
                    continue

                if "error" in record:
                    stats["error_samples"] += 1
                    continue

                metrics = record.get("metrics")
                if metrics is None:
                    stats["success_without_metrics"] += 1
                    continue

                if not isinstance(metrics, dict):
                    stats["success_without_metrics"] += 1
                    stats["incomplete_metrics_samples"] += 1
                    continue

                if not any(key in metrics for key in EXPECTED_TOP_LEVEL_METRIC_KEYS):
                    stats["success_without_metrics"] += 1
                    stats["incomplete_metrics_samples"] += 1
                    continue

                missing_top_level = EXPECTED_TOP_LEVEL_METRIC_KEYS - set(metrics.keys())
                if missing_top_level:
                    stats["incomplete_metrics_samples"] += 1

                update_model_metrics(raw_model_metrics, metrics.get("model_metrics", {}), stats)
                update_tool_metrics(
                    raw_search_tool_metrics,
                    metrics.get("search_tool_metrics", {}),
                    default_category="search",
                    stats=stats,
                )
                update_tool_metrics(
                    raw_other_tool_metrics,
                    metrics.get("other_tool_metrics", {}),
                    default_category="other",
                    stats=stats,
                )
                update_prompt_breakdown(raw_prompt_breakdown, metrics.get("prompt_breakdown", {}), stats)

                stats["valid_metric_samples"] += 1

    if strict and stats["valid_metric_samples"] == 0:
        raise RuntimeError("Strict mode: no valid metrics samples found.")

    merged_metrics = {
        "model_metrics": finalize_model_metrics(raw_model_metrics),
        "search_tool_metrics": finalize_tool_buckets(raw_search_tool_metrics, category="search"),
        "other_tool_metrics": finalize_tool_buckets(raw_other_tool_metrics, category="other"),
        "prompt_breakdown": finalize_prompt_breakdown(raw_prompt_breakdown),
    }

    return {
        "metadata": {
            "dataset_dir": os.path.abspath(dataset_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_files": [os.path.abspath(path) for path in source_files],
            "source_file_count": len(source_files),
        },
        "summary_statistics": stats,
        "metrics": merged_metrics,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize merged metrics from Tongyi inference JSONL outputs.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing iter*.jsonl outputs.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Output summary JSON path. Default: <dataset_dir>/metrics_summary.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if no valid metrics sample is found.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir

    if not os.path.isdir(dataset_dir):
        print(f"Error: dataset_dir does not exist or is not a directory: {dataset_dir}")
        return 1

    output_file = args.output_file or os.path.join(dataset_dir, "metrics_summary.json")

    try:
        summary = summarize_dataset(dataset_dir=dataset_dir, strict=args.strict)
    except RuntimeError as e:
        print(str(e))
        return 2

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Summary written to: {output_file}")
    print(
        "Stats: total_lines={total_lines}, valid_metric_samples={valid_metric_samples}, "
        "error_samples={error_samples}, success_without_metrics={success_without_metrics}, invalid_json_lines={invalid_json_lines}".format(
            **summary["summary_statistics"]
        )
    )
    if summary["warnings"]:
        print(f"Warnings: {len(summary['warnings'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
