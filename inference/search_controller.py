from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Enum Classes : Action and Tool Status

class SearchAction(str, Enum):
    EXECUTE = "execute"
    REUSE_CACHE = "reuse_cache"
    MERGE = "merge"
    REDUCE_TOPK = "reduce_topk"
    SKIP_DUPLICATE = "skip_duplicate"
    REWRITE_REQUEST = "rewrite_request"


class ToolStatus(str, Enum):
    SUCCESS_NONEMPTY = "success_nonempty"
    EMPTY = "empty"
    FAILED = "failed"
    CACHED = "cached"
    SKIPPED = "skipped"


class SearchRequest:
    
    def __init__(
        self,
        task_id: str,
        turn_id: int,
        tool_name: str,
        search_engine: str,
        query_list: list[str],
        original_question: str,
        raw_args: dict,
        requested_topk: int | None = None,
    ):
        self.task_id = task_id
        self.turn_id = turn_id
        self.tool_name = tool_name
        self.search_engine = search_engine
        self.query_list = query_list
        self.original_question = original_question
        self.raw_args = raw_args
        self.requested_topk = requested_topk

# Configured Classes

class ContextProfile:
    def __init__(
        self,
        policy: str,
        replay_factor: float,
        admission_policy: str,
        cache_replay_policy: str,
    ):
        self.policy = policy
        self.replay_factor = replay_factor
        self.admission_policy = admission_policy
        self.cache_replay_policy = cache_replay_policy


# ============================================================
# Budget & Memory (Sections 5, 6.6)
# ============================================================

@dataclass
class BudgetState:
    effective_search_calls: int = 0
    search_call_budget: int = 10
    observation_tokens_used: int = 0
    observation_token_budget: int = 8000
    current_turn: int = 0
    max_turns: int = 10

    @property
    def pressure(self) -> float:
        """Pressure = max(N_search/B_search, N_obs/B_obs, t/T_max)"""
        return max(
            self.effective_search_calls / max(self.search_call_budget, 1),
            self.observation_tokens_used / max(self.observation_token_budget, 1),
            self.current_turn / max(self.max_turns, 1),
        )

    @property
    def pressure_level(self) -> str:
        p = self.pressure
        if p < 0.4:
            return "low"
        elif p < 0.7:
            return "medium"
        return "high"


@dataclass
class SearchMemoryEntry:
    query: str
    query_embedding: list[float]
    turn_id: int
    tool_name: str
    search_engine: str
    action: SearchAction
    executed_external_api: bool
    tool_status: ToolStatus
    raw_result: str = ""
    returned_observation: str = ""
    url_list: list[str] = field(default_factory=list)
    domain_list: list[str] = field(default_factory=list)
    title_snippet_list: list[dict] = field(default_factory=list)
    url_hashes: list[str] = field(default_factory=list)
    snippet_hashes: list[str] = field(default_factory=list)
    raw_observation_tokens: int = 0
    returned_observation_tokens: int = 0
    latency_ms: float = 0.0


class SearchMemory:
    def __init__(self):
        self.entries: list[SearchMemoryEntry] = []

    def add(self, entry: SearchMemoryEntry) -> None:
        self.entries.append(entry)

    def find_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[SearchMemoryEntry, float]]:
        """top-k similar historical queries by cosine similarity."""
        scored = [
            (entry, _cosine_similarity(query_embedding, entry.query_embedding))
            for entry in self.entries
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ============================================================
# Pre-search output (Section 4 interface)
# ============================================================

@dataclass
class QueryDecision:
    query: str
    original_index: int
    action: SearchAction
    reason: str
    cache_entry: Optional[SearchMemoryEntry] = None
    merged_with_index: Optional[int] = None
    adjusted_topk: Optional[int] = None


@dataclass
class QueryBlock:
    queries: list[str]
    original_indices: list[int]
    action: SearchAction
    reason: str
    cache_entry: Optional[SearchMemoryEntry] = None
    adjusted_topk: Optional[int] = None
    merged_query: Optional[str] = None


@dataclass
class PreSearchDecision:
    task_id: str
    turn_id: int
    query_blocks: list[QueryBlock]


@dataclass
class ControllerState:
    memory: SearchMemory
    budget_state: BudgetState


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)



class SearchController:
    def __init__(
        self,
        params: dict | None = None,
        embed_fn=None,
        context_profile: ContextProfile | None = None,
    ):
        
        self.params = params or {}
        self.embed_fn = embed_fn  # Callable[[str], list[float]]
        self.context_profile = context_profile or ContextProfile(
            policy="append_history",
            replay_factor=2.0,
            admission_policy="pass_through",
            cache_replay_policy="pointer",
        )

        self.high_sim = self.params.get("high_similarity", 0.90)
        self.medium_sim = self.params.get("medium_similarity", 0.75)
        self.pointer_turn_dist = self.params.get("pointer_turn_distance", 3)

        self.cost_per_search = self.params.get("cost_per_search", 0.01)
        self.cost_per_obs_token = self.params.get("cost_per_obs_token", 0.00001)
        self.estimated_obs_tokens = self.params.get("estimated_obs_tokens", 800)
        self.estimated_replay_factor = self.params.get("estimated_replay_factor", 2.0)
        self.latency_cost_coeff = self.params.get("latency_cost_coeff", 0.001)
        self.estimated_latency_ms = self.params.get("estimated_latency_ms", 500.0)
        self.cost_weight = self.params.get("cost_weight", 1.0)

        # Section 10.1: pre-search utility weights
        self.w_new_query = self.params.get("w_new_query", 1.0)
        self.w_no_cache = self.params.get("w_no_cache", 0.5)
        self.w_redundancy = self.params.get("w_redundancy", 0.8)
        self.w_repeated_failure = self.params.get("w_repeated_failure", 0.6)

        self.last_intra_sim: list[list[float]] = []
        self._execution_info: dict = {}

   
    def pre_search(
        self,
        request: SearchRequest,
        state: ControllerState,
    ) -> PreSearchDecision:
        query_list = request.query_list
        n = len(query_list)

        embeddings = self._embed_queries(query_list)
        intra_sim = self._compute_intra_similarity(embeddings)
        print(f"[Search Controller] Query similarity matrix for {n} queries:")
        for i in range(n):
            print(f"  Query[{i}]: {query_list[i][:50]}...")
            for j in range(n):
                print(f"    sim[{i}][{j}] = {intra_sim[i][j]:.4f}")
        self.last_intra_sim = intra_sim
        cache_hits = [
            state.memory.find_similar(emb, top_k=3) for emb in embeddings
        ]
        pressure = state.budget_state.pressure
        pressure_level = state.budget_state.pressure_level

        assigned: set[int] = set()
        blocks: list[QueryBlock] = []


        for i in range(n):
            if i in assigned:
                continue

            merge_group = [i]
            for k in range(i + 1, n):
                if k not in assigned and intra_sim[i][k] >= self.high_sim:
                    merge_group.append(k)

            if len(merge_group) > 1:
                blocks.append(QueryBlock(
                    queries=[query_list[idx] for idx in merge_group],
                    original_indices=merge_group,
                    action=SearchAction.MERGE,
                    reason=f"high intra-call similarity among queries {merge_group}",
                ))
                assigned.update(merge_group)
                continue

            decision = self._decide_single_query(
                index=i,
                query_list=query_list,
                intra_sim=intra_sim,
                cache_hits=cache_hits,
                pressure=pressure,
                pressure_level=pressure_level,
                budget_state=state.budget_state,
            )

            blocks.append(QueryBlock(
                queries=[query_list[i]],
                original_indices=[i],
                action=decision.action,
                reason=decision.reason,
                cache_entry=decision.cache_entry,
                adjusted_topk=decision.adjusted_topk,
            ))
            assigned.add(i)
        print("[Search Controller] Final query blocks are")
        for block in blocks :
            print(f"Block {i} with query : {block.queries}")
            print(f"Actions: {block.action}")
        return PreSearchDecision(
            task_id=request.task_id,
            turn_id=request.turn_id,
            query_blocks=blocks,
        )


    def _embed_queries(self, query_list: list[str]) -> list[list[float]]:
        if self.embed_fn is None:
            return [[] for _ in query_list]
        return [self.embed_fn(q) for q in query_list]


    def _compute_intra_similarity(
        self, embeddings: list[list[float]]
    ) -> list[list[float]]:
        n = len(embeddings)
        sim = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                s = _cosine_similarity(embeddings[i], embeddings[j])
                sim[i][j] = s
                sim[j][i] = s
        return sim


    def _best_cache_hit(
        self,
        cache_hits: list[tuple[SearchMemoryEntry, float]],
    ) -> tuple[Optional[SearchMemoryEntry], float]:
        if not cache_hits:
            return None, 0.0
        return cache_hits[0]


    def _estimate_pre_cost(self) -> float:
        """C_pre = p_search + n_obs*p_in + r_replay*n_obs*p_in + mu*t_search"""
        n_obs = self.estimated_obs_tokens
        return (
            self.cost_per_search
            + n_obs * self.cost_per_obs_token
            + self.estimated_replay_factor * n_obs * self.cost_per_obs_token
            + self.latency_cost_coeff * self.estimated_latency_ms
        )


    def _compute_pre_utility(
        self,
        is_new_query: bool,
        no_cache_hit: bool,
        redundancy_score: float,
        repeated_failure_risk: float,
    ) -> float:
        """U_pre = a1*NewQuery + a2*NoCacheHit - a3*QueryRedundancy - a4*RepeatedFailureRisk"""
        return (
            self.w_new_query * (1.0 if is_new_query else 0.0)
            + self.w_no_cache * (1.0 if no_cache_hit else 0.0)
            - self.w_redundancy * redundancy_score
            - self.w_repeated_failure * repeated_failure_risk
        )


    def _decide_single_query(
        self,
        index: int,
        query_list: list[str],
        intra_sim: list[list[float]],
        cache_hits: list[list[tuple[SearchMemoryEntry, float]]],
        pressure: float,
        pressure_level: str,
        budget_state: BudgetState,
    ) -> QueryDecision:
        query = query_list[index]

        best_entry, best_sim = self._best_cache_hit(cache_hits[index])

        if best_entry is not None and best_sim >= self.high_sim:
            turn_dist = budget_state.current_turn - best_entry.turn_id # Forget when turn_dist is sometimes too far

            if best_entry.tool_status == ToolStatus.SUCCESS_NONEMPTY:
                return QueryDecision(
                    query=query,
                    original_index=index,
                    action=SearchAction.REUSE_CACHE,
                    reason=f"cache hit (sim={best_sim:.3f}, turn_dist={turn_dist}) with successful query: '{best_entry.query}'",
                    cache_entry=best_entry,
                )

            if best_entry.tool_status in (ToolStatus.EMPTY, ToolStatus.FAILED):
                if best_sim >= 0.95:
                    return QueryDecision(
                        query=query,
                        original_index=index,
                        action=SearchAction.EXECUTE,
                        reason=f"near-exact repeat of {best_entry.tool_status.value} query (sim={best_sim:.3f}): '{best_entry.query}'",
                        cache_entry=best_entry,
                    )
                

        if pressure_level == "high" and self.context_profile.admission_policy == "budgeted":
            for j in range(index):
                if self.medium_sim <= intra_sim[index][j] < self.high_sim:
                    adjusted_topk = self._reduce_topk(budget_state)
                    return QueryDecision(
                        query=query,
                        original_index=index,
                        action=SearchAction.REDUCE_TOPK,
                        reason=f"medium intra-call similarity ({intra_sim[index][j]:.3f}) with query[{j}] under high budget pressure",
                        adjusted_topk=adjusted_topk,
                    )

            adjusted_topk = self._reduce_topk(budget_state)
            return QueryDecision(
                query=query,
                original_index=index,
                action=SearchAction.REDUCE_TOPK,
                reason=f"no cache hit, high budget pressure ({pressure:.2f})",
                adjusted_topk=adjusted_topk,
            )

        return QueryDecision(
            query=query,
            original_index=index,
            action=SearchAction.EXECUTE,
            reason="no cache hit, normal execution",
        )

    # ----------------------------------------------------------
    # Internal: top-k reduction under pressure (Section 6.6)
    # ----------------------------------------------------------

    def _reduce_topk(
        self,
        budget_state: BudgetState,
    ) -> int:
        base_topk = self.params.get("default_topk", 10)
        pressure = budget_state.pressure
        reduced = max(3, int(base_topk * (1.0 - 0.7 * pressure)))
        return reduced

    def post_search(
        self,
        request: SearchRequest,
        raw_result: str,
        state: ControllerState,
    ) -> str:
        # TODO: 解析 raw_result → url_list, domain_list, title_snippet_list, token counts
        # TODO: 构建 SearchMemoryEntry
        # TODO: 根据 context_profile 决定返回给 agent 的 observation 长度
        # TODO: state.memory.add(entry)
        return raw_result  # pass-through for now

    def update_memory(
        self,
        entry: SearchMemoryEntry,
        state: ControllerState,
    ) -> None:
        # TODO: 查重/压缩后加入 state.memory
        # TODO: 更新 BudgetState (effective_search_calls, observation_tokens_used)
        state.memory.add(entry)
