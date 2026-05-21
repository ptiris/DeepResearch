from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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
    query: list[str]
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
class PreSearchDecision:
    task_id: str
    turn_id: int
    query_decisions: list[QueryDecision]


@dataclass
class ControllerState:
    memory: SearchMemory
    budget_state: BudgetState
    context_profile: ContextProfile


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
    ):
        self.params = params or {}
        self.embed_fn = embed_fn  # Callable[[str], list[float]]

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

   
    def pre_search(
        self,
        request: SearchRequest,
        state: ControllerState,
    ) -> PreSearchDecision:
        """Section 6: pre-search decides per-query action before any API call."""
        query_list = request.query_list
        n = len(query_list)

        # Embed all queries
        embeddings = self._embed_queries(query_list)

        # Step 1: Intra-call redundancy (Section 6.1)
        intra_sim = self._compute_intra_similarity(embeddings)

        # Step 2: Cross-turn cache lookup (Section 6.2)
        cache_hits = [
            state.memory.find_similar(emb, top_k=3) for emb in embeddings
        ]

        # Step 3: Budget pressure (Section 6.6)
        pressure = state.budget_state.pressure
        pressure_level = state.budget_state.pressure_level

        # Step 4: Decide per-query action (Section 6.7 decision table)
        decisions: list[Optional[QueryDecision]] = [None] * n

        for i in range(n):
            if decisions[i] is not None:
                continue
            decisions[i] = self._decide_query(
                index=i,
                query_list=query_list,
                embeddings=embeddings,
                intra_sim=intra_sim,
                cache_hits=cache_hits,
                pressure=pressure,
                pressure_level=pressure_level,
                context_profile=state.context_profile,
                budget_state=state.budget_state,
                decisions=decisions,
            )

        return PreSearchDecision(
            task_id=request.task_id,
            turn_id=request.turn_id,
            query_decisions=[d for d in decisions if d is not None],
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


    def _decide_query(
        self,
        index: int,
        query_list: list[str],
        embeddings: list[list[float]],
        intra_sim: list[list[float]],
        cache_hits: list[list[tuple[SearchMemoryEntry, float]]],
        pressure: float,
        pressure_level: str,
        context_profile: ContextProfile,
        budget_state: BudgetState,
        decisions: list[Optional[QueryDecision]],
    ) -> QueryDecision:
        query = query_list[index]
        emb = embeddings[index]

        # --- Check 1: Intra-call merge (Section 6.1) ---
        for j in range(index):
            if decisions[j] is not None and decisions[j].action == SearchAction.MERGE:
                # Already merged into something else, skip
                if decisions[j].merged_with_index is not None and decisions[j].merged_with_index < j:
                    continue
            if intra_sim[index][j] >= self.high_sim:
                return QueryDecision(
                    query=query,
                    original_index=index,
                    action=SearchAction.MERGE,
                    reason=f"high intra-call similarity ({intra_sim[index][j]:.3f}) with query[{j}]: '{query_list[j]}'",
                    merged_with_index=j,
                )

        # --- Check 2: Cross-turn cache lookup (Section 6.2) ---
        best_entry, best_sim = self._best_cache_hit(cache_hits[index])

        if best_entry is not None and best_sim >= self.high_sim:
            turn_dist = budget_state.current_turn - best_entry.turn_id

            # Case 2a: cached result was successful non-empty
            if best_entry.tool_status == ToolStatus.SUCCESS_NONEMPTY:
                # Determine replay style based on context visibility
                if context_profile.policy == "iterative_report":
                    # IterResearch: always replay full (Section 6.4)
                    return QueryDecision(
                        query=query,
                        original_index=index,
                        action=SearchAction.REUSE_CACHE,
                        reason=f"cache hit (sim={best_sim:.3f}) with successful query: '{best_entry.query}'",
                        cache_entry=best_entry,
                    )
                else:
                    # ReAct/Tongyi: visibility-based replay (Section 6.5)
                    return QueryDecision(
                        query=query,
                        original_index=index,
                        action=SearchAction.REUSE_CACHE,
                        reason=f"cache hit (sim={best_sim:.3f}, turn_dist={turn_dist}) with successful query: '{best_entry.query}'",
                        cache_entry=best_entry,
                    )

            # Case 2b: cached result was empty or failed
            if best_entry.tool_status in (ToolStatus.EMPTY, ToolStatus.FAILED):
                # Section 6.2: mark as repeated-failure risk; still execute if query changed enough
                if best_sim >= 0.95:
                    # Near-exact repeat of a failed query -> rewrite_request
                    return QueryDecision(
                        query=query,
                        original_index=index,
                        action=SearchAction.REWRITE_REQUEST,
                        reason=f"near-exact repeat of {best_entry.tool_status.value} query (sim={best_sim:.3f}): '{best_entry.query}'",
                        cache_entry=best_entry,
                    )
                else:
                    # Similar but not identical -> execute, but flag the risk
                    pass  # fall through to budget check below

        # --- Check 3: medium-similarity intra-call (Section 6.1) ---
        for j in range(index):
            if decisions[j] is not None and decisions[j].action == SearchAction.MERGE:
                continue
            if self.medium_sim <= intra_sim[index][j] < self.high_sim:
                if pressure_level == "high":
                    # Section 6.1: medium sim + high pressure -> reduce_topk
                    adjusted_topk = self._reduce_topk(budget_state, context_profile)
                    return QueryDecision(
                        query=query,
                        original_index=index,
                        action=SearchAction.REDUCE_TOPK,
                        reason=f"medium intra-call similarity ({intra_sim[index][j]:.3f}) with query[{j}] under high budget pressure",
                        adjusted_topk=adjusted_topk,
                    )

        # --- Check 4: Budget pressure (Section 6.6) ---
        if pressure_level == "high":
            adjusted_topk = self._reduce_topk(budget_state, context_profile)
            return QueryDecision(
                query=query,
                original_index=index,
                action=SearchAction.REDUCE_TOPK,
                reason=f"no cache hit, high budget pressure ({pressure:.2f})",
                adjusted_topk=adjusted_topk,
            )

        # --- Default: execute ---
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
        context_profile: ContextProfile,
    ) -> int:
        """Reduce requested top-k proportionally to pressure level."""
        base_topk = self.params.get("default_topk", 10)
        pressure = budget_state.pressure
        # Linearly scale down: at pressure=1.0, return 30% of base
        reduced = max(3, int(base_topk * (1.0 - 0.7 * pressure)))
        return reduced
