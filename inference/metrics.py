import math
from typing import Any, Dict, List, Optional

import tiktoken


class MetricsCollector:
    """Collects per-sample metrics for model calls, tool calls, and prompt breakdown."""

    SEARCH_TOOL_NAMES = {"search", "aliyun_search", "google_scholar"}

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._model_stats: Dict[str, Dict[str, Any]] = {
            "research_model": self._new_model_bucket(),
            "summary_model": self._new_model_bucket(),
        }
        self._tool_stats: Dict[str, Dict[str, Any]] = {}
        self._prompt_breakdown: Dict[str, Dict[str, float]] = {
            "research_model": {},
            "summary_model": {},
        }

    @staticmethod
    def usage_to_dict(usage: Any) -> Dict[str, Any]:
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return usage
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if hasattr(usage, "dict"):
            return usage.dict()
        return {}

    def record_model_call(
        self,
        model_group: str,
        success: bool,
        latency_ms: float,
        usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        bucket = self._model_stats.setdefault(model_group, self._new_model_bucket())
        bucket["calls"] += 1
        bucket["success_calls"] += 1 if success else 0
        bucket["failed_calls"] += 0 if success else 1
        bucket["total_latency_ms"] += max(latency_ms, 0.0)

        usage = usage or {}
        bucket["prompt_tokens"] += self._extract_int(usage, ["prompt_tokens"])
        bucket["completion_tokens"] += self._extract_int(usage, ["completion_tokens"])
        bucket["total_tokens"] += self._extract_int(usage, ["total_tokens"])
        bucket["cached_tokens"] += self._extract_cached_tokens(usage)

    def record_prompt_breakdown(
        self,
        model_group: str,
        messages: List[Dict[str, Any]],
        usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        category_tokens: Dict[str, int] = {}
        for msg in messages:
            role = str(msg.get("role", ""))
            content = str(msg.get("content", ""))
            category = self._classify_prompt_chunk(role=role, content=content)
            token_count = self._count_tokens(content)
            category_tokens[category] = category_tokens.get(category, 0) + token_count

        if not category_tokens:
            return

        usage_prompt_tokens = self._extract_int((usage or {}), ["prompt_tokens"])
        local_total = sum(category_tokens.values())
        normalized: Dict[str, float] = {}

        if usage_prompt_tokens > 0 and local_total > 0:
            for k, v in category_tokens.items():
                normalized[k] = float(usage_prompt_tokens) * (float(v) / float(local_total))
        else:
            for k, v in category_tokens.items():
                normalized[k] = float(v)

        bucket = self._prompt_breakdown.setdefault(model_group, {})
        for k, v in normalized.items():
            bucket[k] = bucket.get(k, 0.0) + v

    def record_tool_call(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float,
        effective_calls: int = 1,
        status_code: Optional[int] = None,
    ) -> None:
        category = "search" if tool_name in self.SEARCH_TOOL_NAMES else "other"
        bucket = self._tool_stats.setdefault(tool_name, self._new_tool_bucket(category))

        bucket["calls"] += 1
        bucket["effective_calls"] += max(effective_calls, 0)
        bucket["success_calls"] += 1 if success else 0
        bucket["failed_calls"] += 0 if success else 1
        bucket["total_latency_ms"] += max(latency_ms, 0.0)
        if status_code is not None:
            if "status_codes" not in bucket:
                bucket["status_codes"] = {}
            bucket["status_codes"][status_code] = bucket["status_codes"].get(status_code, 0) + 1

    @staticmethod
    def infer_tool_success(result: Any) -> bool:
        if result is None:
            return False
        if not isinstance(result, str):
            return True

        text = result.strip().lower()
        if not text:
            return False

        failure_signals = [
            "invalid request format",
            "tool call is not a valid json",
            "tool not found",
            "openrouter error",
            " failed",
            "error:",
            "timeout",
            "could not be processed",
            "could not be accessed",
        ]
        return not any(signal in text for signal in failure_signals)

    def to_dict(self) -> Dict[str, Any]:
        model_stats = {
            model_group: self._finalize_model_bucket(bucket)
            for model_group, bucket in self._model_stats.items()
        }

        search_tools = {
            name: self._finalize_tool_bucket(bucket)
            for name, bucket in self._tool_stats.items()
            if bucket.get("category") == "search"
        }
        other_tools = {
            name: self._finalize_tool_bucket(bucket)
            for name, bucket in self._tool_stats.items()
            if bucket.get("category") == "other"
        }

        prompt_breakdown = {
            model_group: self._finalize_prompt_breakdown(bucket)
            for model_group, bucket in self._prompt_breakdown.items()
        }

        return {
            "model_metrics": model_stats,
            "search_tool_metrics": {
                "by_tool": search_tools,
                "aggregated": self._aggregate_tool_buckets(search_tools, "search"),
            },
            "other_tool_metrics": {
                "by_tool": other_tools,
                "aggregated": self._aggregate_tool_buckets(other_tools, "other"),
            },
            "prompt_breakdown": prompt_breakdown,
        }

    @staticmethod
    def _new_model_bucket() -> Dict[str, Any]:
        return {
            "calls": 0,
            "success_calls": 0,
            "failed_calls": 0,
            "total_latency_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }

    @staticmethod
    def _new_tool_bucket(category: str) -> Dict[str, Any]:
        return {
            "category": category,
            "calls": 0,
            "effective_calls": 0,
            "success_calls": 0,
            "failed_calls": 0,
            "total_latency_ms": 0.0,
            "status_codes": {},
        }

    @staticmethod
    def _extract_int(payload: Dict[str, Any], path: List[str]) -> int:
        cur: Any = payload
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return 0
            cur = cur[key]
        try:
            return int(cur)
        except (TypeError, ValueError):
            return 0

    def _extract_cached_tokens(self, usage: Dict[str, Any]) -> int:
        candidates = [
            ["cached_tokens"],
            ["prompt_tokens_details", "cached_tokens"],
            ["input_tokens_details", "cached_tokens"],
        ]
        for path in candidates:
            value = self._extract_int(usage, path)
            if value > 0:
                return value
        return 0

    def _classify_prompt_chunk(self, role: str, content: str) -> str:
        text = content.lower()
        if role == "system":
            if "tool" in text and ("parameter" in text or "description" in text):
                return "Tool Definition"
            return "System Instruction"
        if role == "user":
            if "<tool_response>" in text:
                return "Observation"
            return "User Query"
        if role == "assistant":
            if "<tool_call>" in text:
                return "Tool Call"
            if "<answer>" in text:
                return "Report"
            return "Reasoning"
        return "Other"

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def _finalize_model_bucket(self, bucket: Dict[str, Any]) -> Dict[str, Any]:
        calls = int(bucket.get("calls", 0))
        success_calls = int(bucket.get("success_calls", 0))
        total_latency_ms = float(bucket.get("total_latency_ms", 0.0))
        success_rate = (float(success_calls) / float(calls) * 100.0) if calls > 0 else 0.0
        avg_latency_ms = (total_latency_ms / float(calls)) if calls > 0 else 0.0

        return {
            "calls": calls,
            "success_calls": success_calls,
            "failed_calls": int(bucket.get("failed_calls", 0)),
            "success_rate": round(success_rate, 4),
            "latency_ms": {
                "total": round(total_latency_ms, 4),
                "average": round(avg_latency_ms, 4),
            },
            "tokens": {
                "prompt_tokens": int(bucket.get("prompt_tokens", 0)),
                "completion_tokens": int(bucket.get("completion_tokens", 0)),
                "total_tokens": int(bucket.get("total_tokens", 0)),
                "cached_tokens": int(bucket.get("cached_tokens", 0)),
            },
        }

    def _finalize_tool_bucket(self, bucket: Dict[str, Any]) -> Dict[str, Any]:
        calls = int(bucket.get("calls", 0))
        success_calls = int(bucket.get("success_calls", 0))
        total_latency_ms = float(bucket.get("total_latency_ms", 0.0))
        success_rate = (float(success_calls) / float(calls) * 100.0) if calls > 0 else 0.0
        avg_latency_ms = (total_latency_ms / float(calls)) if calls > 0 else 0.0

        result = {
            "category": bucket.get("category", "other"),
            "calls": calls,
            "effective_calls": int(bucket.get("effective_calls", 0)),
            "success_calls": success_calls,
            "failed_calls": int(bucket.get("failed_calls", 0)),
            "success_rate": round(success_rate, 4),
            "latency_ms": {
                "total": round(total_latency_ms, 4),
                "average": round(avg_latency_ms, 4),
            },
        }

        status_codes = bucket.get("status_codes", {})
        if status_codes:
            result["status_codes"] = status_codes

        return result

    def _aggregate_tool_buckets(self, finalized_buckets: Dict[str, Dict[str, Any]], category: str) -> Dict[str, Any]:
        calls = sum(v.get("calls", 0) for v in finalized_buckets.values())
        success_calls = sum(v.get("success_calls", 0) for v in finalized_buckets.values())
        failed_calls = sum(v.get("failed_calls", 0) for v in finalized_buckets.values())
        effective_calls = sum(v.get("effective_calls", 0) for v in finalized_buckets.values())
        total_latency_ms = sum(v.get("latency_ms", {}).get("total", 0.0) for v in finalized_buckets.values())
        success_rate = (float(success_calls) / float(calls) * 100.0) if calls > 0 else 0.0
        avg_latency_ms = (float(total_latency_ms) / float(calls)) if calls > 0 else 0.0

        return {
            "category": category,
            "calls": int(calls),
            "effective_calls": int(effective_calls),
            "success_calls": int(success_calls),
            "failed_calls": int(failed_calls),
            "success_rate": round(success_rate, 4),
            "latency_ms": {
                "total": round(float(total_latency_ms), 4),
                "average": round(float(avg_latency_ms), 4),
            },
        }

    def _finalize_prompt_breakdown(self, bucket: Dict[str, float]) -> Dict[str, Any]:
        total = float(sum(bucket.values()))
        out: Dict[str, Any] = {}
        for category, tokens in bucket.items():
            percentage = (tokens / total * 100.0) if total > 0 else 0.0
            out[category] = {
                "tokens": int(round(tokens)),
                "percentage": round(percentage, 4),
            }
        out["_total_prompt_tokens"] = int(round(total))
        return out
