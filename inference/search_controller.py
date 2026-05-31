from __future__ import annotations

import math
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
    observation_tokens_used : int
        Tokens consumed by observations.
    observation_token_budget : int
        Maximum observation tokens allowed.
    current_turn : int
        Current conversation turn.
    max_turns : int
        Maximum conversation turns allowed.
    """
    effective_search_calls: int = 0
    search_call_budget: int = 10
    observation_tokens_used: int = 0
    observation_token_budget: int = 8000
    current_turn: int = 0
    max_turns: int = 10

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
            self.observation_tokens_used / max(self.observation_token_budget, 1),
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


class SearchController:
    """Coordinates search execution with caching, deduplication, and budget management.
    
    The SearchController decides how to handle a list of search queries:
    - Which queries to execute vs. reuse from cache
    - Which queries to merge together
    - How many results to retrieve based on budget pressure
    
    Parameters
    ----------
    params : dict or None
        Configuration parameters for thresholds and cost estimates.
        - high_similarity: Threshold for cache reuse (default 0.90)
        - medium_similarity: Threshold for merge detection (default 0.75)
        - pointer_turn_distance: Max turns to look back for cache (default 3)
        - cost_per_search: Estimated cost per search call (default 0.01)
        - cost_per_obs_token: Cost per observation token (default 0.00001)
        - default_topk: Default number of results (default 10)
    embed_fn : callable or None
        Callable[[str], list[float]] for query embedding. If None,
        similarity computation is skipped.
    context_profile : ContextProfile or None
        ContextProfile for admission and replay policies.
    """
    
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
        return [self.embed_fn(q) for q in query_list]


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


    def _estimate_pre_cost(self) -> float:
        """Estimate pre-search cost.

        Returns
        -------
        float
            Estimated cost in arbitrary units for executing a single query.

        Formula: C_pre = p_search + n_obs*p_in + r_replay*n_obs*p_in + mu*t_search
        """
        n_obs = self.estimated_obs_tokens
        return (
            self.cost_per_search
            + n_obs * self.cost_per_obs_token
            + self.estimated_replay_factor * n_obs * self.cost_per_obs_token
            + self.latency_cost_coeff * self.estimated_latency_ms
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
        base_topk = self.params.get("default_topk", 10)
        pressure = budget_state.pressure
        reduced = max(5, int(base_topk * (1.0 - 0.7 * pressure)))
        return reduced

    def post_search(
        self,
        request: SearchRequest,
        raw_result: str,
        state: ControllerState,
    ) -> str:
        """Process raw search results and update memory.

        Parses the raw search result, extracts URLs/snippets/token counts,
        builds a SearchMemoryEntry, and updates the controller state.

        Parameters
        ----------
        request : SearchRequest
            Original SearchRequest for context.
        raw_result : str
            Raw result string from the search tool.
        state : ControllerState
            ControllerState to update with new entry and budget.

        Returns
        -------
        str
            Processed observation string to return to the agent.
        """
        # TODO: Parse raw_result → url_list, domain_list, title_snippet_list, token counts
        # TODO: Build SearchMemoryEntry
        # TODO: Update budget state (effective_search_calls, observation_tokens_used)
        # TODO: Decide observation length based on context_profile
        return raw_result  # pass-through for now

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
            state.budget_state.observation_tokens_used += entry.returned_observation_tokens

        state.memory.add(entry)
