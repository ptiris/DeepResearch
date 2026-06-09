import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))

from search_controller import (  # noqa: E402
    BudgetState,
    ControllerState,
    QueryBlock,
    SearchAction,
    SearchController,
    SearchMemory,
    SearchMemoryEntry,
    SearchRequest,
    ToolStatus,
)


def unit(x, y):
    n = math.sqrt(x * x + y * y)
    return [x / n, y / n]


class FakeEmbedder:
    def __call__(self, text):
        text = text.lower()
        if "alpha exact" in text or "same topic" in text:
            return [1.0, 0.0]
        if "alpha medium" in text or "related topic" in text:
            return unit(0.8, 0.6)
        if "beta" in text or "fresh" in text:
            return [0.0, 1.0]
        if "query" in text:
            return [1.0, 0.0]
        if "duplicate" in text:
            return [1.0, 0.0]
        if "novel" in text:
            return [0.0, 1.0]
        if "item 1" in text:
            return [1.0, 0.0]
        if "item 2" in text:
            return unit(0.9, 0.43589)
        if "item 3" in text:
            return [0.0, 1.0]
        return unit(0.5, 0.866)


class CountingEmbedder(FakeEmbedder):
    def __init__(self):
        self.calls = {}

    def __call__(self, text):
        self.calls[text] = self.calls.get(text, 0) + 1
        return super().__call__(text)


def controller(**params):
    defaults = {
        "high_similarity": 0.90,
        "medium_similarity": 0.70,
        "reduce_topk": 7,
        "mmr_min_results": 5,
    }
    defaults.update(params)
    return SearchController(params=defaults, embed_fn=FakeEmbedder())


def request(queries, turn_id=10):
    return SearchRequest(
        task_id="task",
        turn_id=turn_id,
        tool_name="search",
        search_engine="search",
        query_list=queries,
        original_question="question",
        raw_args={"query": queries},
    )


def memory_entry(query="same topic", emb=None, turn_id=1, raw_result=""):
    return SearchMemoryEntry(
        query=query,
        query_embedding=emb or [1.0, 0.0],
        turn_id=turn_id,
        tool_name="search",
        search_engine="search",
        action=SearchAction.EXECUTE,
        executed_external_api=True,
        tool_status=ToolStatus.SUCCESS_NONEMPTY,
        raw_result=raw_result,
        returned_observation=raw_result,
        selected_result_indices=[1],
    )


SERPER_RAW = """A Google search for 'query' found 3 results:\n\n## Web Results\n1. [Item 1 duplicate](https://a.example)\nSource: A\nSnippet duplicate\n\n2. [Item 2 related](https://b.example)\nSource: B\nSnippet related\n\n3. [Item 3 novel](https://c.example)\nSource: C\nSnippet novel"""

ALIYUN_RAW = """An Aliyun IQS search for 'query' found 2 results:\n\n## Web Results\n1. [Aliyun One](https://aliyun.example/1)\nDate published: 2024-01-01\nSource: host1\nSummary one\n\n2. [Aliyun Two](https://aliyun.example/2)\nSource: host2\nSummary two"""

SCHOLAR_RAW = """A Google scholar for 'query' found 2 results:\n\n## Scholar Results\n1. [Paper One](pdfUrl: https://paper.example/1.pdf)\npublicationInfo: Journal\nDate published: 2021\ncitedBy: 5\nAbstract one\n\n2. [Paper Two](no available link)\npublicationInfo: Conf\nDate published: 2022\nAbstract two"""

HISTORY_RAW = """A Google search for 'history' found 1 result:\n\n## Web Results\n1. [History Only](https://history.example)\nHistorical snippet"""


class SearchControllerTests(unittest.TestCase):
    def test_parse_search_result_formats(self):
        ctrl = controller()
        self.assertEqual([e["title"] for e in ctrl.parse_search_results(SERPER_RAW)], ["Item 1 duplicate", "Item 2 related", "Item 3 novel"])
        self.assertEqual([e["url"] for e in ctrl.parse_search_results(ALIYUN_RAW)], ["https://aliyun.example/1", "https://aliyun.example/2"])
        scholar = ctrl.parse_search_results(SCHOLAR_RAW)
        self.assertEqual(scholar[0]["url"], "pdfUrl: https://paper.example/1.pdf")
        self.assertEqual(scholar[1]["url"], "no available link")

    def test_pre_search_merge_reuse_reduce_and_execute(self):
        ctrl = controller()
        state = ControllerState(memory=SearchMemory(), budget_state=BudgetState(current_turn=10))
        decision = ctrl.pre_search(request(["alpha exact", "same topic"]), state)
        self.assertEqual(decision.query_blocks[0].action, SearchAction.MERGE)

        state.memory.add(memory_entry())
        decision = ctrl.pre_search(request(["alpha exact"]), state)
        self.assertEqual(decision.query_blocks[0].action, SearchAction.REUSE_CACHE)

        decision = ctrl.pre_search(request(["alpha medium"]), state)
        self.assertEqual(decision.query_blocks[0].action, SearchAction.REDUCE_TOPK)

        decision = ctrl.pre_search(request(["fresh beta"]), ControllerState(memory=SearchMemory(), budget_state=BudgetState()))
        self.assertEqual(decision.query_blocks[0].action, SearchAction.EXECUTE)
        self.assertEqual(decision.query_blocks[0].query_embedding, [0.0, 1.0])

    def test_pre_search_embeds_duplicate_query_once(self):
        embedder = CountingEmbedder()
        ctrl = SearchController(
            params={
                "high_similarity": 0.90,
                "medium_similarity": 0.70,
                "reduce_topk": 7,
                "mmr_min_results": 5,
            },
            embed_fn=embedder,
        )
        state = ControllerState(memory=SearchMemory(), budget_state=BudgetState(current_turn=10))
        ctrl.pre_search(request(["alpha exact", "alpha exact"]), state)
        self.assertEqual(embedder.calls["alpha exact"], 1)

    def test_intra_call_medium_similarity_triggers_reduce_topk(self):
        ctrl = controller()
        state = ControllerState(memory=SearchMemory(), budget_state=BudgetState())
        decision = ctrl.pre_search(request(["alpha exact", "alpha medium"]), state)
        self.assertEqual([b.action for b in decision.query_blocks], [SearchAction.EXECUTE, SearchAction.REDUCE_TOPK])

    def test_mmr_prefers_novel_result_when_history_has_duplicate(self):
        ctrl = controller(mmr_min_results=1, mmr_threshold=-1.0, mmr_beta=2.0)
        state = ControllerState(memory=SearchMemory(), budget_state=BudgetState(observation_token_budget=1000))
        state.memory.add(memory_entry(raw_result=SERPER_RAW))
        entries = ctrl.parse_search_results(SERPER_RAW)
        selected = ctrl._select_results_mmr("query", entries[:3], state.memory.entries, 1000, 1)
        self.assertEqual(selected[0]["index"], 3)

    def test_history_result_embeddings_are_saved_on_entry(self):
        embedder = CountingEmbedder()
        ctrl = SearchController(
            params={
                "high_similarity": 0.90,
                "medium_similarity": 0.70,
                "reduce_topk": 7,
                "mmr_min_results": 1,
                "mmr_threshold": -1.0,
            },
            embed_fn=embedder,
        )
        history = memory_entry(raw_result=HISTORY_RAW)
        state = ControllerState(memory=SearchMemory(), budget_state=BudgetState(observation_token_budget=1000))
        state.memory.add(history)

        block = QueryBlock(queries=["query"], original_indices=[0], action=SearchAction.EXECUTE, reason="test")
        ctrl.post_search(request(["query"]), SERPER_RAW, state, block)
        history_text = "History Only\nhttps://history.example\nHistorical snippet"
        self.assertEqual(embedder.calls[history_text], 1)
        self.assertEqual(len(history.selected_result_embeddings), 1)

        ctrl._text_embedding_cache.clear()
        block = QueryBlock(queries=["query"], original_indices=[0], action=SearchAction.EXECUTE, reason="test")
        ctrl.post_search(request(["query"]), SERPER_RAW, state, block)
        self.assertEqual(embedder.calls[history_text], 1)

    def test_mmr_forces_minimum_five_below_threshold(self):
        raw = "A Google search for 'query' found 6 results:\n\n## Web Results\n" + "\n\n".join(
            f"{i}. [Low Item {i}](https://e.example/{i})\nSnippet" for i in range(1, 7)
        )
        ctrl = controller(mmr_threshold=2.0, mmr_min_results=5)
        state = ControllerState(memory=SearchMemory(), budget_state=BudgetState(observation_token_budget=1000))
        block = QueryBlock(queries=["query"], original_indices=[0], action=SearchAction.EXECUTE, reason="test")
        returned = ctrl.post_search(request(["query"]), raw, state, block)
        self.assertEqual(len(block.selected_result_indices), 5)
        self.assertIn("Controlled Search Results", returned)

    def test_reduce_topk_limits_candidates_to_seven(self):
        raw = "A Google search for 'query' found 9 results:\n\n## Web Results\n" + "\n\n".join(
            f"{i}. [Item {i}](https://e.example/{i})\nSnippet" for i in range(1, 10)
        )
        ctrl = controller(mmr_threshold=-1.0)
        state = ControllerState(memory=SearchMemory(), budget_state=BudgetState(observation_token_budget=1000))
        block = QueryBlock(queries=["query"], original_indices=[0], action=SearchAction.REDUCE_TOPK, reason="test", adjusted_topk=7)
        ctrl.post_search(request(["query"]), raw, state, block)
        self.assertLessEqual(max(block.selected_result_indices), 7)

    def test_reuse_pointer_window_and_old_cache_replay(self):
        ctrl = controller()
        recent = memory_entry(turn_id=8, raw_result=SERPER_RAW)
        state = ControllerState(memory=SearchMemory(), budget_state=BudgetState(observation_token_budget=1000))
        block = QueryBlock(queries=["alpha exact"], original_indices=[0], action=SearchAction.REUSE_CACHE, reason="test", cache_entry=recent)
        returned = ctrl.post_search(request(["alpha exact"], turn_id=10), recent.raw_result, state, block)
        self.assertTrue(block.reuse_pointer)
        self.assertIn("See turn 8", returned)

        old = memory_entry(turn_id=1, raw_result=SERPER_RAW)
        block = QueryBlock(queries=["alpha exact"], original_indices=[0], action=SearchAction.REUSE_CACHE, reason="test", cache_entry=old)
        returned = ctrl.post_search(request(["alpha exact"], turn_id=10), old.raw_result, state, block)
        self.assertFalse(block.reuse_pointer)
        self.assertIn("Controlled Search Results", returned)
        self.assertTrue(block.selected_result_indices)


if __name__ == "__main__":
    unittest.main()
