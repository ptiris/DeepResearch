from __future__ import annotations

import datetime
import json
import math
import os
import statistics
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Enum Classes : Action and Tool Status

class SearchAction(str, Enum):
    """Actions the search controller can take for a query.
    
    Attributes
    ----------
    EXECUTE : str
        Run the search query normally.
    REUSE_CACHE : str
        Return cached results from a similar previous query.
    MERGE : str
        Combine multiple similar queries into a single search.
    REDUCE_TOPK : str
        Reduce the number of results returned due to budget pressure.
    SKIP_DUPLICATE : str
        Skip this query as a duplicate within the same call.
    REWRITE_REQUEST : str
        Rewrite the query before execution.
    """
    EXECUTE = "execute"
    REUSE_CACHE = "reuse_cache"
    MERGE = "merge"
    REDUCE_TOPK = "reduce_topk"
    SKIP_DUPLICATE = "skip_duplicate"
    REWRITE_REQUEST = "rewrite_request"


class ToolStatus(str, Enum):
    """Status of a search tool execution result.
    
    Attributes
    ----------
    SUCCESS_NONEMPTY : str
        Search returned results.
    EMPTY : str
        Search returned no results.
    FAILED : str
        Search execution failed.
    CACHED : str
        Results were served from cache.
    SKIPPED : str
        Search was skipped (duplicate, budget, etc.).
    """
    SUCCESS_NONEMPTY = "success_nonempty"
    EMPTY = "empty"
    FAILED = "failed"
    CACHED = "cached"
    SKIPPED = "skipped"


class SearchRequest:
    """A search request to be processed by the SearchController.
    
    Attributes
    ----------
    task_id : str
        Unique identifier for the task.
    turn_id : int
        Current conversation turn number.
    tool_name : str
        Name of the search tool to use (e.g., 'search', 'google_scholar').
    search_engine : str
        Search engine identifier (e.g., 'serper', 'aliyun').
    query_list : list[str]
        List of search queries to execute.
    original_question : str
        The original user question for context.
    raw_args : dict
        Raw arguments passed to the search tool.
    requested_topk : int or None
        Number of results to retrieve (optional).
    """
    
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
    """Configuration profile for context handling policy.
    
    Attributes
    ----------
    policy : str
        Context management policy (e.g., 'append_history').
    replay_factor : float
        Factor for estimating replay token costs.
    admission_policy : str
        Policy for admitting queries under pressure
        (e.g., 'pass_through', 'budgeted').
    cache_replay_policy : str
        How to handle cache replay (e.g., 'pointer', 'copy').
    """
    
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
    """Tracks search budget consumption and pressure level.

    Attributes
    ----------
    effective_search_calls : int
        Number of search calls executed.
    search_call_budget : int
        Maximum allowed search calls.
    current_turn : int
        Current conversation turn.
    max_turns : int
        Maximum conversation turns allowed.
    step_gain_history : list[float]
        Mean MMR scores of selected results per search step, used for
        dynamic K_obs z-score computation.
    """
    effective_search_calls: int = 0
    search_call_budget: int = 10
    current_turn: int = 0
    max_turns: int = 10
    step_gain_history: list[float] = field(default_factory=list)

    @property
    def pressure(self) -> float:
        """Compute overall budget pressure as max of normalized metrics.

        Returns
        -------
        float
            Pressure ratio in [0, 1+], where higher means more constrained.
        """
        return max(
            self.effective_search_calls / max(self.search_call_budget, 1),
            self.current_turn / max(self.max_turns, 1),
        )

    @property
    def pressure_level(self) -> str:
        """Categorize pressure level for policy decisions.

        Returns
        -------
        str
            'low' (<0.4), 'medium' (0.4-0.7), or 'high' (>0.7).
        """
        p = self.pressure
        if p < 0.4:
            return "low"
        elif p < 0.7:
            return "medium"
        return "high"

    
    


@dataclass
class SearchMemoryEntry:
    """A single historical search entry stored in SearchMemory.
    
    Attributes
    ----------
    query : str
        The search query string.
    query_embedding : list[float]
        Vector embedding of the query for similarity search.
    turn_id : int
        Conversation turn when this search was executed.
    tool_name : str
        Name of the search tool used.
    search_engine : str
        Search engine identifier.
    action : SearchAction
        The SearchAction taken for this query.
    executed_external_api : bool
        Whether an external API was called.
    tool_status : ToolStatus
        Result status of the search.
    raw_result : str
        Raw result string from the search.
    returned_observation : str
        Processed observation returned to the agent.
    url_list : list[str]
        List of URLs retrieved.
    domain_list : list[str]
        List of domains from retrieved URLs.
    title_snippet_list : list[dict]
        List of title/snippet dicts from results.
    url_hashes : list[str]
        Content hashes of URLs for deduplication.
    snippet_hashes : list[str]
        Content hashes of snippets for deduplication.
    raw_observation_tokens : int
        Token count of raw observation.
    returned_observation_tokens : int
        Token count of returned observation.
    latency_ms : float
        Search latency in milliseconds.
    """
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
    selected_result_indices: list[int] = field(default_factory=list)
    selected_result_texts: list[str] = field(default_factory=list)
    selected_result_embeddings: list[list[float]] = field(default_factory=list)


class SearchMemory:
    """In-memory store of historical search entries for caching and deduplication.
    
    Provides similarity-based lookup to reuse previous search results
    and avoid redundant API calls.
    
    Attributes
    ----------
    entries : list[SearchMemoryEntry]
        List of SearchMemoryEntry objects.
    """
    
    def __init__(self):
        self.entries: list[SearchMemoryEntry] = []

    def add(self, entry: SearchMemoryEntry) -> None:
        """Add a new search entry to memory.

        Parameters
        ----------
        entry : SearchMemoryEntry
            The SearchMemoryEntry to store.
        """
        self.entries.append(entry)

    def find_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[SearchMemoryEntry, float]]:
        """Find top-k similar historical queries by cosine similarity.

        Parameters
        ----------
        query_embedding : list[float]
            Query vector to compare against.
        top_k : int, optional
            Maximum number of results to return.

        Returns
        -------
        list[tuple[SearchMemoryEntry, float]]
            List of (entry, similarity_score) tuples, sorted by score descending.
        """
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
    """Decision for a single query in the pre-search phase.
    
    Attributes
    ----------
    query : str
        The query string.
    original_index : int
        Index in the original query list.
    action : SearchAction
        SearchAction to take.
    reason : str
        Human-readable explanation for the decision.
    cache_entry : SearchMemoryEntry or None
        Cached SearchMemoryEntry if reusing cache.
    merged_with_index : int or None
        Index of query merged with (if MERGE action).
    adjusted_topk : int or None
        Adjusted top-k if REDUCE_TOPK action.
    """
    query: str
    original_index: int
    action: SearchAction
    reason: str
    cache_entry: Optional[SearchMemoryEntry] = None
    merged_with_index: Optional[int] = None
    adjusted_topk: Optional[int] = None


@dataclass
class QueryBlock:
    """A block of queries to be processed together with a single action.
    
    Attributes
    ----------
    queries : list[str]
        List of query strings in this block.
    original_indices : list[int]
        Original indices of these queries.
    action : SearchAction
        The SearchAction for all queries in this block.
    reason : str
        Explanation for why queries were grouped.
    cache_entry : SearchMemoryEntry or None
        Cached entry if reusing cache.
    adjusted_topk : int or None
        Adjusted top-k if REDUCE_TOPK.
    merged_query : str or None
        The combined query string if action is MERGE.
    """
    queries: list[str]
    original_indices: list[int]
    action: SearchAction
    reason: str
    cache_entry: Optional[SearchMemoryEntry] = None
    adjusted_topk: Optional[int] = None
    merged_query: Optional[str] = None
    raw_result: str = ""
    returned_observation: str = ""
    selected_result_indices: list[int] = field(default_factory=list)
    query_embedding: list[float] = field(default_factory=list)
    selected_result_texts: list[str] = field(default_factory=list)
    selected_result_embeddings: list[list[float]] = field(default_factory=list)
    raw_observation_tokens: int = 0
    returned_observation_tokens: int = 0
    reuse_pointer: bool = False


@dataclass
class PreSearchDecision:
    """Output of the pre-search planning phase.
    
    Contains the task/turn identifiers and the list of query blocks
    to be executed in this turn.
    
    Attributes
    ----------
    task_id : str
        Unique task identifier.
    turn_id : int
        Current turn number.
    query_blocks : list[QueryBlock]
        List of QueryBlock objects to execute.
    """
    task_id: str
    turn_id: int
    query_blocks: list[QueryBlock]

    
@dataclass
class ControllerState:
    """Combined state for search controller decision making.
    
    Wraps memory and budget state together for easy passing.
    
    Attributes
    ----------
    memory : SearchMemory
        The SearchMemory instance for caching.
    budget_state : BudgetState
        The BudgetState instance for budget tracking.
    """
    memory: SearchMemory
    budget_state: BudgetState


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Parameters
    ----------
    a : list[float]
        First vector.
    b : list[float]
        Second vector.

    Returns
    -------
    float
        Cosine similarity score in [-1, 1]. Returns 0.0 if either vector is zero.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class SearchController:
    """Coordinates search execution with caching, deduplication, and budget management.
    
    The SearchController decides how to handle a list of search queries:
    - Which queries to execute vs. reuse from cache
    - Which queries to merge together
    - How many results to retrieve based on budget pressure

    Uses dynamic observation budget (K_obs) based on z-score of step-level
    marginal gain to control how many search results are selected per step.
    
    Parameters
    ----------
    params : dict or None
        Configuration parameters for thresholds.
        - high_similarity: Threshold for cache reuse (default 0.90)
        - medium_similarity: Threshold for merge detection (default 0.70)
        - pointer_turn_distance: Max turns to look back for cache (default 3)
        - dynamic_k_obs_enabled: Enable dynamic K_obs budget (default True)
    embed_fn : callable or None
        Callable[[str], list[float]] for query embedding. If None,
        similarity computation is skipped.
    context_profile : ContextProfile or None
        ContextProfile for admission and replay policies.
    """

    # Dynamic observation budget table from design.md Section 5
    K_OBS_TABLE = {
        SearchAction.EXECUTE: {"k_min": 3, "k_base": 5, "k_max": 7, "k_extra": 1},
        SearchAction.REDUCE_TOPK: {"k_min": 1, "k_base": 3, "k_max": 5, "k_extra": 1},
        SearchAction.REUSE_CACHE: {"k_min": 3, "k_base": 5, "k_max": 7, "k_extra": 1},
        SearchAction.MERGE: {"k_min": 3, "k_base": 5, "k_max": 7, "k_extra": 1},
        SearchAction.SKIP_DUPLICATE: {"k_min": 1, "k_base": 1, "k_max": 2, "k_extra": 1},
        SearchAction.REWRITE_REQUEST: {"k_min": 3, "k_base": 5, "k_max": 7, "k_extra": 1},
    }
    
    def __init__(
        self,
        params: dict | None = None,
        embed_fn=None,
        context_profile: ContextProfile | None = None,
        disable_search_controller: bool = False,
    ):

        self.params = params or {}
        self.embed_fn = embed_fn
        self.disable_search_controller = disable_search_controller or self.params.get("disable_search_controller", False)
        self.context_profile = context_profile or ContextProfile(
            policy="append_history",
            replay_factor=2.0,
            admission_policy="pass_through",
            cache_replay_policy="pointer",
        )

        self.mode = self.params.get("mode", os.getenv("SEARCH_CONTROLLER_MODE", "default"))

        self.high_sim = self.params.get("high_similarity", _env_float("SEARCH_CONTROLLER_HIGH_SIM", 0.90))
        self.medium_sim = self.params.get("medium_similarity", _env_float("SEARCH_CONTROLLER_MEDIUM_SIM", 0.70))
        self.pointer_turn_dist = self.params.get("pointer_turn_distance", _env_int("SEARCH_CONTROLLER_REUSE_POINTER_WINDOW", 3))
        self.reduce_topk = self.params.get("reduce_topk", _env_int("SEARCH_CONTROLLER_REDUCE_TOPK", 7))

        self.dynamic_k_obs_enabled = self.params.get(
            "dynamic_k_obs_enabled",
            os.getenv("SEARCH_DYNAMIC_K_OBS", "true").lower() == "true",
        )

        self.mmr_alpha = self.params.get("mmr_alpha", _env_float("SEARCH_MMR_ALPHA", 0.35))
        self.mmr_beta = self.params.get("mmr_beta", _env_float("SEARCH_MMR_BETA", 0))
        self.mmr_threshold = self.params.get("mmr_threshold", _env_float("SEARCH_MMR_THRESHOLD", 0.05))
        self.mmr_min_results = self.params.get("mmr_min_results", _env_int("SEARCH_MMR_MIN_RESULTS", 5))

        # Section 10.1: pre-search utility weights
        self.w_new_query = self.params.get("w_new_query", 1.0)
        self.w_no_cache = self.params.get("w_no_cache", 0.5)
        self.w_redundancy = self.params.get("w_redundancy", 0.8)
        self.w_repeated_failure = self.params.get("w_repeated_failure", 0.6)

        self._mmr_log_path: str | None = None
        self._mmr_log_lock = threading.Lock()

        self.last_intra_sim: list[list[float]] = []
        self._execution_info: dict = {}
        self._text_embedding_cache: dict[str, list[float]] = {}

   
    def pre_search(
        self,
        request: SearchRequest,
        state: ControllerState,
    ) -> PreSearchDecision:
        """Plan search execution strategy for a list of queries.

        Analyzes query similarity, checks cache for matches, and decides
        which queries to execute, merge, or serve from cache based on
        budget pressure and similarity thresholds.

        Parameters
        ----------
        request : SearchRequest
            The SearchRequest containing the queries to plan.
        state : ControllerState
            Current ControllerState with memory and budget.

        Returns
        -------
        PreSearchDecision
            PreSearchDecision containing query blocks with assigned actions.
        """
        if self.disable_search_controller:
            blocks = [
                QueryBlock(
                    queries=[q],
                    original_indices=[i],
                    action=SearchAction.EXECUTE,
                    reason="search controller disabled, force execute",
                )
                for i, q in enumerate(request.query_list)
            ]
            return PreSearchDecision(
                task_id=request.task_id,
                turn_id=request.turn_id,
                query_blocks=blocks,
            )

        if self.mode == "reduce_topk":
            print(f"[Search Controller] reduce_topk mode: forcing all {len(request.query_list)} queries to REDUCE_TOPK with topk={self.reduce_topk}")
            blocks = [
                QueryBlock(
                    queries=[q],
                    original_indices=[i],
                    action=SearchAction.REDUCE_TOPK,
                    reason=f"reduce_topk mode: force topk={self.reduce_topk}",
                    adjusted_topk=self.reduce_topk,
                )
                for i, q in enumerate(request.query_list)
            ]
            return PreSearchDecision(
                task_id=request.task_id,
                turn_id=request.turn_id,
                query_blocks=blocks,
            )

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
        assigned: set[int] = set()
        blocks: list[QueryBlock] = []


        for i in range(n):
            if i in assigned:
                continue

            merge_group = [i]
            for k in range(i + 1, n):
                if k not in assigned and all(
                    intra_sim[zk][k] >= self.high_sim for zk in merge_group
                ):
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
                budget_state=state.budget_state,
            )

            blocks.append(QueryBlock(
                queries=[query_list[i]],
                original_indices=[i],
                action=decision.action,
                reason=decision.reason,
                cache_entry=decision.cache_entry,
                adjusted_topk=decision.adjusted_topk,
                query_embedding=embeddings[i],
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
    

    def _compute_dynamic_k_obs(
        self,
        action: SearchAction,
        budget_state: BudgetState,
    ) -> int:
        """Compute dynamic K_obs based on z-score of step gain history.

        K_obs(t) = clip(K_base + round(K_extra * z_t), K_min, K_max)

        When fewer than 2 history entries exist, returns K_base.

        Parameters
        ----------
        action : SearchAction
            The action for the current query block.
        budget_state : BudgetState
            Current budget state containing step_gain_history.

        Returns
        -------
        int
            Dynamic K_obs value clipped to [K_min, K_max].
        """
        table = self.K_OBS_TABLE.get(action, self.K_OBS_TABLE[SearchAction.EXECUTE])
        history = budget_state.step_gain_history
        if len(history) < 2:
            return table["k_base"]
        mu = statistics.mean(history)
        sigma = statistics.stdev(history)
        if sigma == 0:
            return table["k_base"]
        z_t = (history[-1] - mu) / sigma
        k_obs = table["k_base"] + round(table["k_extra"] * z_t)
        return max(table["k_min"], min(k_obs, table["k_max"]))

    def _record_step_gain(
        self,
        selected_items: list[dict],
        query_emb: list[float],
        budget_state: BudgetState,
    ) -> None:
        """Record the mean MMR score of selected results as step gain.

        G_t = (1/L) * sum(MG(r_i)) for each selected result, where MG is
        the MMR score at selection time. Appends G_t to step_gain_history.

        Parameters
        ----------
        selected_items : list[dict]
            Selected result items from MMR, each with '_mmr_score' key.
        query_emb : list[float]
            Query embedding vector.
        budget_state : BudgetState
            Budget state to update step_gain_history.
        """
        if not selected_items or not query_emb:
            return
        scores = [
            item["_mmr_score"] for item in selected_items
            if "_mmr_score" in item
        ]
        if scores:
            budget_state.step_gain_history.append(statistics.mean(scores))


    def _embed_queries(self, query_list: list[str]) -> list[list[float]]:
        """Embed queries using the provided embedding function.
        
        Parameters
    ----------
    query_list : list[str]
        List of query strings to embed.

    Returns
    -------
    list[list[float]]
        List of embedding vectors. Returns empty vectors if embed_fn is None.
        """
        if self.embed_fn is None:
            return [[] for _ in query_list]
        return self._embed_texts(query_list)


    def _compute_intra_similarity(
        self, embeddings: list[list[float]]
    ) -> list[list[float]]:
        """Compute pairwise cosine similarity matrix between query embeddings.
        
        Parameters
        ----------
        embeddings : list[list[float]]
            List of query embedding vectors.

        Returns
        -------
        list[list[float]]
        NxN similarity matrix where sim[i][j] = similarity between i and j.
        """
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
        """Select the best cache hit from candidate matches.

        Parameters
        ----------
        cache_hits : list[tuple[SearchMemoryEntry, float]]
            List of (entry, similarity) tuples from find_similar.

        Returns
        -------
        tuple[SearchMemoryEntry or None, float]
            Tuple of (best_entry, best_similarity). Returns (None, 0.0) if no hits.
        """
        if not cache_hits:
            return None, 0.0
        return cache_hits[0]


    def _decide_single_query(
        self,
        index: int,
        query_list: list[str],
        intra_sim: list[list[float]],
        cache_hits: list[list[tuple[SearchMemoryEntry, float]]],
        budget_state: BudgetState,
    ) -> QueryDecision:
        """Make a decision for a single query based on cache and budget state.

        Parameters
        ----------
        index : int
            Index of the query in the query list.
        query_list : list[str]
            List of all queries in this turn.
        intra_sim : list[list[float]]
            Pre-computed similarity matrix between queries.
        cache_hits : list[list[tuple[SearchMemoryEntry, float]]]
            Cache hit candidates from memory lookup.
        pressure : float
            Current budget pressure ratio.
        pressure_level : str
            'low', 'medium', or 'high'.
        budget_state : BudgetState
            Current BudgetState for decisions.

        Returns
        -------
        QueryDecision
            QueryDecision with the chosen action and reasoning.
        """
        query = query_list[index]

        best_entry, best_sim = self._best_cache_hit(cache_hits[index])

        if best_entry is not None and best_sim >= self.high_sim:
            turn_dist = budget_state.current_turn - best_entry.turn_id # Forget when turn_dist is sometimes too far

            if best_entry.tool_status in (ToolStatus.SUCCESS_NONEMPTY, ToolStatus.CACHED):
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
                

        if best_entry is not None and self.medium_sim <= best_sim < self.high_sim:
            return QueryDecision(
                query=query,
                original_index=index,
                action=SearchAction.REDUCE_TOPK,
                reason=f"medium cache similarity ({best_sim:.3f}) with historical query: '{best_entry.query}'",
                cache_entry=best_entry,
                adjusted_topk=self._reduce_topk(budget_state),
            )

        for j in range(index):
            if self.medium_sim <= intra_sim[index][j] < self.high_sim:
                return QueryDecision(
                    query=query,
                    original_index=index,
                    action=SearchAction.REDUCE_TOPK,
                    reason=f"medium intra-call similarity ({intra_sim[index][j]:.3f}) with query[{j}]",
                    adjusted_topk=self._reduce_topk(budget_state),
                )

        return QueryDecision(
            query=query,
            original_index=index,
            action=SearchAction.EXECUTE,
            reason="no cache hit, normal execution",
        )

    # ----------------------------------------------------------
    # Internal: top-k reduction under pressure
    # ----------------------------------------------------------

    def _reduce_topk(
        self,
        budget_state: BudgetState,
    ) -> int:
        """Reduce top-k based on current budget pressure.

        Parameters
        ----------
        budget_state : BudgetState
            Current budget state to compute pressure from.

        Returns
        -------
        int
            Reduced top-k value, minimum 5.
        """
        return max(1, int(self.reduce_topk))

    def post_search(
        self,
        request: SearchRequest,
        raw_result: str,
        state: ControllerState,
        block: QueryBlock | None = None,
    ) -> str:
        """Return the observation admitted to agent context for a search block."""
        block = block or QueryBlock(
            queries=request.query_list,
            original_indices=list(range(len(request.query_list))),
            action=SearchAction.EXECUTE,
            reason="post_search fallback block",
        )

        source_raw = raw_result or ""
        if block.action == SearchAction.REUSE_CACHE and block.cache_entry is not None:
            turn_dist = request.turn_id - block.cache_entry.turn_id
            if turn_dist <= self.pointer_turn_dist:
                pointer = (
                    "[CACHE HIT] Current query is similar to a recent search. "
                    f"See turn {block.cache_entry.turn_id} search result for query: "
                    f"{block.cache_entry.query}"
                )
                block.raw_result = block.cache_entry.raw_result
                block.returned_observation = pointer
                block.selected_result_indices = []
                block.selected_result_texts = []
                block.selected_result_embeddings = []
                block.raw_observation_tokens = self._count_tokens(block.raw_result)
                block.returned_observation_tokens = self._count_tokens(pointer)
                block.reuse_pointer = True
                return pointer
            source_raw = block.cache_entry.raw_result or block.cache_entry.returned_observation

        entries = self.parse_search_results(source_raw)
        if not entries:
            block.raw_result = source_raw
            block.returned_observation = source_raw
            block.selected_result_indices = []
            block.selected_result_texts = []
            block.selected_result_embeddings = []
            block.raw_observation_tokens = self._count_tokens(source_raw)
            block.returned_observation_tokens = block.raw_observation_tokens
            return source_raw

        candidate_entries = entries
        if not self.dynamic_k_obs_enabled:
            if block.action == SearchAction.REDUCE_TOPK:
                candidate_entries = entries[: max(1, block.adjusted_topk or self.reduce_topk)]

        query = block.merged_query or (block.queries[0] if block.queries else "")
        if not block.query_embedding and query:
            block.query_embedding = self._embed_text(query)
        selected = self._select_results_mmr(
            query=query,
            candidates=candidate_entries,
            history_entries=state.memory.entries,
            action=block.action,
            budget_state=state.budget_state,
            max_results=len(candidate_entries),
            query_emb=block.query_embedding,
        )
        if not selected:
            selected = candidate_entries[: min(len(candidate_entries), self.mmr_min_results)]
            self._attach_result_embeddings(selected)

        if self.dynamic_k_obs_enabled:
            self._record_step_gain(selected, block.query_embedding, state.budget_state)

        returned = self._format_selected_observation(source_raw, selected)
        block.raw_result = source_raw
        block.returned_observation = returned
        block.selected_result_indices = [item["index"] for item in selected]
        block.selected_result_texts = [self._result_similarity_text(item) for item in selected]
        block.selected_result_embeddings = [
            list(item.get("_similarity_embedding", [])) for item in selected
        ]
        block.raw_observation_tokens = self._count_tokens(source_raw)
        block.returned_observation_tokens = self._count_tokens(returned)
        block.reuse_pointer = False
        return returned

    def parse_search_results(self, raw_result: str) -> list[dict]:
        """Parse formatted search tool output into indexed result dictionaries."""
        if not raw_result or not raw_result.strip():
            return []

        pattern = re.compile(
            r"(?:^|\n)(\d+)\.\s+\[(.*?)\]\((.*?)\)(.*?)(?=\n\d+\.\s+\[|\n=======|\Z)",
            re.DOTALL,
        )
        entries = []
        for match in pattern.finditer(raw_result):
            body = match.group(4).strip()
            text = f"{match.group(1)}. [{match.group(2)}]({match.group(3)})"
            if body:
                text = f"{text}\n{body}"
            entries.append({
                "index": len(entries) + 1,
                "display_index": int(match.group(1)),
                "title": match.group(2).strip(),
                "url": match.group(3).strip(),
                "body": body,
                "text": text.strip(),
            })
        return entries

    def _select_results_mmr(
        self,
        query: str,
        candidates: list[dict],
        history_entries: list[SearchMemoryEntry],
        action: SearchAction,
        budget_state: BudgetState,
        max_results: int,
        query_emb: list[float] | None = None,
    ) -> list[dict]:
        if not candidates:
            return []

        query_emb = query_emb if query_emb is not None else self._embed_text(query)
        candidate_embs = self._embed_texts([self._result_similarity_text(item) for item in candidates])
        for item, emb in zip(candidates, candidate_embs):
            item["_similarity_embedding"] = emb
        history_embs = self._selected_history_result_embeddings(history_entries)

        # Compute initial K_obs from action table
        k_obs_table = self.K_OBS_TABLE.get(action, self.K_OBS_TABLE[SearchAction.EXECUTE])
        initial_k_obs = self._compute_dynamic_k_obs(action, budget_state)

        selected_positions: list[int] = []
        selected_scores: list[float] = []
        remaining = set(range(len(candidates)))
        max_results = max(1, min(max_results, len(candidates)))

        # TODO: AQ admission quality filtering will be added here
        # For now, candidates go directly to MMR selection

        print(f"\n{'='*80}")
        print(f"[MMR] Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"[MMR] Candidates: {len(candidates)} | History ents: {len(history_embs)} | Max slots: {max_results}")
        print(f"[MMR] Action: {action.value} | K_obs table: {k_obs_table} | Initial K_obs: {initial_k_obs}")
        print(f"[MMR] Params: alpha={self.mmr_alpha} beta={self.mmr_beta} "
              f"threshold={self.mmr_threshold} min_results={self.mmr_min_results}")
        print(f"[MMR] Step gain history: {budget_state.step_gain_history}")

        n_stopped_threshold = 0
        n_stopped_k_obs = 0
        round_num = 0
        mmr_rounds: list[dict] = []

        while remaining and len(selected_positions) < max_results:
            round_num += 1
            best_pos = None
            best_score = -float("inf")
            best_rel = 0.0
            best_red_cur = 0.0
            best_red_hist = 0.0
            round_candidates: list[dict] = []

            for pos in remaining:
                rel = _cosine_similarity(candidate_embs[pos], query_emb)
                red_cur = 0.0
                if selected_positions:
                    red_cur = max(
                        _cosine_similarity(candidate_embs[pos], candidate_embs[selected_pos])
                        for selected_pos in selected_positions
                    )
                red_hist = 0.0
                if history_embs:
                    red_hist = max(_cosine_similarity(candidate_embs[pos], hist_emb) for hist_emb in history_embs)
                score = rel - self.mmr_alpha * red_cur - self.mmr_beta * red_hist

                title = candidates[pos].get("title", "")[:80]
                round_candidates.append({
                    "idx": pos, "score": round(score, 6), "rel": round(rel, 6),
                    "red_cur": round(red_cur, 6), "red_hist": round(red_hist, 6),
                    "title": title,
                })

                if score > best_score:
                    best_score = score
                    best_pos = pos
                    best_rel = rel
                    best_red_cur = red_cur
                    best_red_hist = red_hist

            if best_pos is None:
                break

            round_candidates.sort(key=lambda x: x["score"], reverse=True)
            print(f"\n[MMR] --- Round {round_num} ---")
            for rc in round_candidates:
                marker = " >> SELECTED <<" if rc["idx"] == best_pos else ""
                print(f"[MMR]   #{rc['idx']:2d} score={rc['score']:+.4f} rel={rc['rel']:.4f} "
                      f"red_cur={rc['red_cur']:.4f} red_hist={rc['red_hist']:.4f} "
                      f"| {rc['title']}{marker}")

            mmr_rounds.append({
                "round": round_num,
                "selected_idx": best_pos,
                "best_score": round(best_score, 6),
                "best_rel": round(best_rel, 6),
                "best_red_cur": round(best_red_cur, 6),
                "best_red_hist": round(best_red_hist, 6),
                "candidates": round_candidates,
            })

            if best_score < self.mmr_threshold and len(selected_positions) >= min(self.mmr_min_results, len(candidates)):
                print(f"[MMR] STOP: best_score={best_score:.4f} < threshold={self.mmr_threshold} "
                      f"(selected={len(selected_positions)} >= min_results={self.mmr_min_results})")
                n_stopped_threshold += 1
                break

            # Store MMR score on the candidate for step gain recording
            candidates[best_pos]["_mmr_score"] = best_score

            selected_positions.append(best_pos)
            selected_scores.append(best_score)
            remaining.remove(best_pos)

            # Recompute dynamic K_obs from running G_t and step_gain_history
            running_g_t = statistics.mean(selected_scores)
            history = budget_state.step_gain_history
            if len(history) >= 2:
                mu = statistics.mean(history)
                sigma = statistics.stdev(history)
                if sigma > 0:
                    z_t = (running_g_t - mu) / sigma
                    current_k_obs = max(k_obs_table["k_min"],
                                        min(k_obs_table["k_base"] + round(k_obs_table["k_extra"] * z_t),
                                            k_obs_table["k_max"]))
                else:
                    current_k_obs = k_obs_table["k_base"]
            else:
                current_k_obs = k_obs_table["k_base"]

            print(f"[MMR]   G_t={running_g_t:.4f} z_t={((running_g_t - statistics.mean(history)) / statistics.stdev(history)) if len(history) >= 2 and statistics.stdev(history) > 0 else 'N/A'} "
                  f"K_obs={current_k_obs} selected={len(selected_positions)}")

            if len(selected_positions) >= current_k_obs:
                print(f"[MMR] STOP: selected {len(selected_positions)} >= dynamic K_obs={current_k_obs}")
                n_stopped_k_obs += 1
                break

        # --- MMR statistics summary ---
        selected_rel_range: tuple[float, float] = (0.0, 0.0)
        selected_rel_avg = 0.0
        if selected_positions:
            scores = []
            for pos in selected_positions:
                rel = _cosine_similarity(candidate_embs[pos], query_emb)
                scores.append(rel)
            selected_rel_range = (round(min(scores), 6), round(max(scores), 6))
            selected_rel_avg = round(sum(scores) / len(scores), 6)

        print(f"\n[MMR] ===== Summary =====")
        print(f"[MMR] Selected: {len(selected_positions)}/{len(candidates)} candidates "
              f"in {round_num} rounds")
        print(f"[MMR] Stopped (threshold): {n_stopped_threshold} | Stopped (K_obs): {n_stopped_k_obs} "
              f"| Remaining: {len(remaining)}")
        if selected_positions:
            print(f"[MMR] Selected relevance range: [{selected_rel_range[0]:.4f}, {selected_rel_range[1]:.4f}] "
                  f"avg={selected_rel_avg:.4f}")
        print(f"[MMR] Step gain history before update: {budget_state.step_gain_history}")
        print(f"{'='*80}\n")

        # --- Write structured MMR stats to file ---
        if self._mmr_log_path:
            mmr_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "query": query,
                "num_candidates": len(candidates),
                "num_history_embeddings": len(history_embs),
                "max_slots": max_results,
                "k_obs_table": k_obs_table,
                "initial_k_obs": initial_k_obs,
                "params": {
                    "alpha": self.mmr_alpha,
                    "beta": self.mmr_beta,
                    "threshold": self.mmr_threshold,
                    "min_results": self.mmr_min_results,
                },
                "step_gain_history": budget_state.step_gain_history,
                "rounds": mmr_rounds,
                "summary": {
                    "selected_count": len(selected_positions),
                    "total_candidates": len(candidates),
                    "stopped_threshold": n_stopped_threshold,
                    "stopped_k_obs": n_stopped_k_obs,
                    "remaining": len(remaining),
                    "total_rounds": round_num,
                    "selected_relevance_range": selected_rel_range,
                    "selected_relevance_avg": selected_rel_avg,
                },
            }
            with self._mmr_log_lock:
                with open(self._mmr_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(mmr_record, ensure_ascii=False) + "\n")

        return [candidates[pos] for pos in selected_positions]

    def _selected_history_result_embeddings(self, entries: list[SearchMemoryEntry]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for entry in entries:
            if entry.selected_result_embeddings:
                embeddings.extend(entry.selected_result_embeddings)
                continue

            parsed = self.parse_search_results(entry.raw_result)
            if not parsed:
                continue
            by_index = {item["index"]: item for item in parsed}
            selected_texts: list[str] = []
            for result_index in entry.selected_result_indices:
                item = by_index.get(result_index)
                if item is not None:
                    selected_texts.append(self._result_similarity_text(item))

            entry.selected_result_texts = selected_texts
            entry.selected_result_embeddings = self._embed_texts(selected_texts) if selected_texts else []
            embeddings.extend(entry.selected_result_embeddings)
        return embeddings

    def _attach_result_embeddings(self, items: list[dict]) -> None:
        missing = [
            item for item in items
            if "_similarity_embedding" not in item
        ]
        if not missing:
            return
        embeddings = self._embed_texts([self._result_similarity_text(item) for item in missing])
        for item, emb in zip(missing, embeddings):
            item["_similarity_embedding"] = emb

    def _result_similarity_text(self, item: dict) -> str:
        return "\n".join(part for part in [item.get("title", ""), item.get("url", ""), item.get("body", "")] if part)

    def _embed_text(self, text: str) -> list[float]:
        return self._embed_texts([text])[0]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.embed_fn is None:
            return [[] for _ in texts]
        embeddings: list[list[float]] = []
        for text in texts:
            if text in self._text_embedding_cache:
                embeddings.append(list(self._text_embedding_cache[text]))
                continue
            try:
                emb = list(self.embed_fn(text))
                self._text_embedding_cache[text] = emb
                embeddings.append(list(emb))
            except Exception as exc:
                print(f"[Search Controller] Embedding failed, using zero vector: {exc}")
                embeddings.append([])
        return embeddings

    def _format_selected_observation(self, raw_result: str, selected: list[dict]) -> str:
        first_line = raw_result.strip().splitlines()[0] if raw_result.strip() else "Search results"
        body = "\n\n".join(item["text"] for item in selected)
        return f"{first_line}\n\n## Controlled Search Results\n{body}" if body else first_line

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(re.findall(r"\S+", text))

    def set_mmr_log_path(self, path: str) -> None:
        self._mmr_log_path = path

    def update_memory(
        self,
        entry: SearchMemoryEntry,
        state: ControllerState,
    ) -> None:
        """Add a search entry to memory after post_search processing.

        Performs deduplication/compression before adding to memory.
        Updates budget state for successful external API calls.

        Parameters
        ----------
        entry : SearchMemoryEntry
            The processed SearchMemoryEntry to add.
        state : ControllerState
            ControllerState with memory to update.
        """
        if entry.executed_external_api and entry.tool_status == ToolStatus.SUCCESS_NONEMPTY:
            state.budget_state.effective_search_calls += 1

        state.memory.add(entry)
