import json
import json5
import os
import re
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from datetime import datetime
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
from prompt import *
import time
import asyncio
import numpy as np
from metrics import MetricsCollector, QueryProcessLogger
from model_client import ModelClient
from search_controller import (
    SearchController, SearchRequest, ControllerState, SearchMemory,
    BudgetState, ContextProfile, SearchAction, SearchMemoryEntry, PreSearchDecision,
    QueryBlock, ToolStatus
)

from tool_file import *
from tool_scholar import *
from tool_python import *
from tool_search import *
from tool_visit import *

# Import query deduplication modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.embedding_client import EmbeddingClient


OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))

TOOL_CLASS = [
    FileParser(),
    Scholar(),
    Visit(),
    Search(),
    AliyunSearch(),
    PythonInterpreter(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}
import random
import datetime


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


class DummyLLM:
    model = "dummy"

class MultiTurnReactAgent(FnCallAgent):
    def __init__(
        self,
        function_list: Optional[Union[List[Union[str, Dict, BaseTool]]]],
        llm_cfg: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        disable_search_controller: bool = False,
    ) -> None:
        super().__init__(llm=DummyLLM(), function_list=function_list)
        self.llm_generate_cfg = llm_cfg.get("generate_cfg", {}) if llm_cfg else {}
        self.custom_system_prompt = system_prompt
        self._function_list = function_list if function_list else []
        self.disable_search_controller = disable_search_controller

        # Initialize SearchController
        self.embedding_client = EmbeddingClient()
        self.search_controller = SearchController(
            params={
                "high_similarity": float(os.getenv("SEARCH_CONTROLLER_HIGH_SIM", "0.90")),
                "medium_similarity": float(os.getenv("SEARCH_CONTROLLER_MEDIUM_SIM", "0.70")),
                "default_topk": 10,
                "reduce_topk": int(os.getenv("SEARCH_CONTROLLER_REDUCE_TOPK", "7")),
                "pointer_turn_distance": int(os.getenv("SEARCH_CONTROLLER_REUSE_POINTER_WINDOW", "3")),
                "disable_search_controller": self.disable_search_controller,
            },
            embed_fn=lambda q: self.embedding_client.encode([q])[0].tolist(),
            context_profile=ContextProfile(
                policy="append_history",
                replay_factor=2.0,
                admission_policy="pass_through",
                cache_replay_policy="pointer",
            ),
        )
        self.controller_state = ControllerState(
            memory=SearchMemory(),
            budget_state=BudgetState(
                search_call_budget=10,
                observation_token_budget=8000,
                max_turns=MAX_LLM_CALL_PER_RUN,
            ),
        )
        self.turn_id = 0
        self.query_logger: Optional[QueryProcessLogger] = None

    def set_query_logger(self, output_dir: str):
        log_path = os.path.join(output_dir, "query_process_log.jsonl")
        self.query_logger = QueryProcessLogger(log_path)

    def call_server(self, msgs, max_tries=10, metrics: Optional[MetricsCollector] = None):
        model_client = ModelClient('research')
        provider = model_client.provider
        model = model_client.model
        if not model_client.api_key:
            return "Error: API_KEY not set in environment"

        client = model_client.get_client()

        base_sleep_time = 1
        for attempt in range(max_tries):
            call_start = time.perf_counter()
            try:
                print(f"--- Attempting to call {model}, try {attempt + 1}/{max_tries} ---")
                chat_response = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    max_tokens=10000,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1)
                )
                content = chat_response.choices[0].message.content
                usage = MetricsCollector.usage_to_dict(getattr(chat_response, "usage", None))
                latency_ms = (time.perf_counter() - call_start) * 1000.0

                if content and content.strip():
                    print(f"--- {provider} call successful, received a valid response ---")
                    if metrics:
                        metrics.record_model_call(
                            model_group="research_model",
                            success=True,
                            latency_ms=latency_ms,
                            usage=usage,
                        )
                        metrics.record_prompt_breakdown(
                            model_group="research_model",
                            messages=msgs,
                            usage=usage,
                        )
                    return {
                        "content": content.strip(),
                        "usage": usage,
                    }
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response.")
                    if metrics:
                        metrics.record_model_call(
                            model_group="research_model",
                            success=False,
                            latency_ms=latency_ms,
                            usage=usage,
                        )
                        metrics.record_prompt_breakdown(
                            model_group="research_model",
                            messages=msgs,
                            usage=usage,
                        )

            except APIError as e:
                # Check if it's a context length error (400 with max context length message) - non-retryable
                error_msg = ''
                try:
                    if hasattr(e, 'response') and e.response is not None:
                        error_body = e.response.json()
                        error_msg = error_body.get('error', {}).get('message', '')
                        error_code = error_body.get('error', {}).get('code', '')
                        if (error_code == 400 or 'maximum context length' in error_msg.lower()) and 'context length' in error_msg.lower():
                            print(f"Error: Context length exceeded (non-retryable): {e}")
                            if metrics:
                                metrics.record_model_call(
                                    model_group="research_model",
                                    success=False,
                                    latency_ms=(time.perf_counter() - call_start) * 1000.0,
                                    usage=None,
                                )
                            return {
                                "content": f"Error: Context length exceeded - {error_msg}",
                                "usage": None,
                            }
                except:
                    pass
                # Fall through to normal retry handling for other APIErrors
                print(f"Error: Attempt {attempt + 1} failed with an API error: {e}")
                if metrics:
                    metrics.record_model_call(
                        model_group="research_model",
                        success=False,
                        latency_ms=(time.perf_counter() - call_start) * 1000.0,
                        usage=None,
                    )
            except (APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
                if metrics:
                    metrics.record_model_call(
                        model_group="research_model",
                        success=False,
                        latency_ms=(time.perf_counter() - call_start) * 1000.0,
                        usage=None,
                    )
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")
                if metrics:
                    metrics.record_model_call(
                        model_group="research_model",
                        success=False,
                        latency_ms=(time.perf_counter() - call_start) * 1000.0,
                        usage=None,
                    )

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)

                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")

        return {
            f"content": "{provider} error!!!",
            "usage": None,
        }

    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        metrics = MetricsCollector()
        try:
            question = data['item']['question']
        except:
            raw_msg = data['item']['messages'][1]["content"]
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg

        start_time = time.time()
        answer = data['item']['answer']
        self.user_prompt = question
        if self.custom_system_prompt:
            system_prompt = self.custom_system_prompt
        else:
            system_prompt = SYSTEM_PROMPT
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        
        self.turn_id = 0
        json_retry_count = 0
        
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        while num_llm_calls_available > 0:
            if time.time() - start_time > 150 * 60:
                prediction = 'No answer found after 2h30mins'
                termination = 'No answer found after 2h30mins'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result
            round += 1
            num_llm_calls_available -= 1
            metrics.record_context_by_source(
                model_group="research_model",
                messages=messages,
                round_num=round,
            )
            model_response = self.call_server(messages, metrics=metrics)
            content = model_response.get("content", "")
            print(f'Round {round}: {content}')
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            messages.append({"role": "assistant", "content": content.strip()})
            action_found = False
            if '<tool_call>' in content and '</tool_call>' in content:
                action_found = True
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                try:
                    if "python" in tool_call.lower():
                        try:
                            code_raw = content.split('<tool_call>')[1].split('</tool_call>')[0].split('<code>')[1].split('</code>')[0].strip()
                            result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                        except:
                            result = "[Python Interpreter Error]: Formatting error."
                    else:
                        tool_name = None
                        tool_args = None
                        
                        # 检测 XML 属性格式: <tool_call name="search">
                        xml_attr_match = re.search(r'<tool_call\s+name="([^"]+)"', content)
                        if xml_attr_match:
                            tool_name = xml_attr_match.group(1)
                            json_match = re.search(r'<tool_call[^>]*>([\s\S]+?)</tool_call>', content)
                            if json_match:
                                try:
                                    inner_json = json5.loads(json_match.group(1))
                                    tool_args = inner_json.get('arguments', inner_json)
                                    print(f"[TOOL_CALL_PARSE] XML attribute format detected")
                                    print(f"[TOOL_CALL_PARSE]   tool_name: {tool_name}")
                                    print(f"[TOOL_CALL_PARSE]   tool_args: {str(tool_args)[:100]}...")
                                except Exception as e:
                                    print(f"[TOOL_CALL_PARSE] XML format JSON parse failed: {e}")
                                    tool_args = None
                        else:
                            # 标准 JSON 格式
                            try:
                                tool_call_json = json5.loads(tool_call)
                                tool_name = tool_call_json.get('name', '')
                                tool_args = tool_call_json.get('arguments', {})
                                print(f"[TOOL_CALL_PARSE] Standard JSON format detected")
                                print(f"[TOOL_CALL_PARSE]   tool_name: {tool_name}")
                                print(f"[TOOL_CALL_PARSE]   tool_args: {str(tool_args)[:100]}...")
                            except Exception as e:
                                print(f"[TOOL_CALL_PARSE] JSON parse failed: {e}")
                                tool_name = None
                                tool_args = None
                        
                        # 验证工具调用参数合法性，不合法则触发重试
                        if not tool_name or tool_args is None:
                            raise ValueError(
                                f"Invalid tool_call: cannot extract valid tool_name and arguments. "
                                f"tool_name={tool_name}, tool_args={tool_args}. "
                                f"Please use format: <tool_call>{{\"name\": \"tool_name\", \"arguments\": {{...}}}}</tool_call>"
                            )
                        
                        result = self.custom_call_tool(tool_name, tool_args, _metrics=metrics)
                except Exception as e:
                    max_json_retries = 5
                    if json_retry_count < max_json_retries:
                        json_retry_count += 1
                        print(f"[TOOL_CALL_PARSE] JSON parse FAILED")
                        print(f"[TOOL_CALL_PARSE]   Attempt: {json_retry_count}/{max_json_retries}")
                        print(f"[TOOL_CALL_PARSE]   Error: {e}")
                        print(f"[TOOL_CALL_PARSE]   Content preview: {tool_call[:200]}...")
                        messages.append({"role": "user", "content": FORMAT_GUARD_PROMPT})
                        num_llm_calls_available += 1
                        round -= 1
                        continue
                    else:
                        print(f"[TOOL_CALL_PARSE] Max retries ({max_json_retries}) exceeded")
                        print(f"[TOOL_CALL_PARSE]   Error: {e}")
                        print(f"[TOOL_CALL_PARSE]   Content: {tool_call[:200]}...")
                        print(f"[TOOL_CALL_PARSE]   Continuing with error info in context...")
                        error_context = (
                            f"ERROR: Tool call parse failed after {max_json_retries} retries.\n"
                            f"Parse error: {str(e)}\n"
                            f"Invalid content: {tool_call[:500]}...\n"
                            f"Please use correct <tool_call> format with valid JSON."
                        )
                        messages.append({"role": "user", "content": error_context})
                        result = f'Parse failed after {max_json_retries} retries'
                result = "<tool_response>\n" + result + "\n</tool_response>"
                messages.append({"role": "user", "content": result})
            if '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                action_found = True
                break
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                action_found = True
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'
            if action_found == False and num_llm_calls_available > 0:
                if self.is_context_length_error(content):
                    print("[CONTEXT_OVERFLOW] Context length exceeded. Stop this task.")
                    prediction = "No answer found."
                    termination = "context_length_exceeded"
                    result = {
                        "question": question,
                        "answer": answer,
                        "messages": messages,
                        "prediction": prediction,
                        "termination": termination,
                        "metrics": metrics.to_dict(),
                    }
                    return result
                print(f"[TOOL_CALL_PARSE] No valid tool_call or answer found in response")
                print(f"[TOOL_CALL_PARSE] Content preview: {content[:200]}...")
                messages.append({"role": "user", "content": FORMAT_GUARD_PROMPT})
                round -= 1
                continue


        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "metrics": metrics.to_dict(),
        }
        return result

    def is_context_length_error(self, text: str) -> bool:
        s = str(text).lower()
        return (
            "context length exceeded" in s
            or "maximum context length" in s
            or ("requested about" in s and "tokens" in s)
        )

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name == "search" and "search" not in self._function_list and "aliyun_search" in self._function_list:
            print(f"[DEBUG] Remapping 'search' to 'aliyun_search' (Serper not available, using Aliyun IQS)")
            tool_name = "aliyun_search"

        metrics: Optional[MetricsCollector] = kwargs.get("_metrics")
        tool_start = time.perf_counter()
        success = False
        effective_calls = 1
        status_code = None
        
        # Increment turn_id for each tool call
        self.turn_id += 1

        # ========== Branch A: Search tools → SearchController ==========
        if tool_name in MetricsCollector.SEARCH_TOOL_NAMES:
            return self._execute_search_with_controller(
                tool_name, tool_args, metrics, tool_start, **kwargs
            )

        # ========== Branch B: Non-search tools → Original dispatch ==========
        if tool_name in TOOL_MAP:
            print(f"[DEBUG] custom_call_tool invoked with tool_name: '{tool_name}', args: {tool_args}")
            tool_args["params"] = tool_args
            try:
                if "python" in tool_name.lower():
                    result = TOOL_MAP['PythonInterpreter'].call(tool_args)
                elif tool_name == "parse_file":
                    params = {"files": tool_args["files"]}
                    raw_result = asyncio.run(
                        TOOL_MAP[tool_name].call(
                            params,
                            file_root_path="./eval_data/file_corpus",
                            **kwargs,
                        )
                    )
                    result = raw_result
                    if not isinstance(raw_result, str):
                        result = str(raw_result)
                else:
                    raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
                    if isinstance(raw_result, tuple) and len(raw_result) == 2:
                        result, status_code = raw_result
                    else:
                        result = raw_result
                success = MetricsCollector.infer_tool_success(result)
                return result
            except Exception as e:
                result = f"Error: Tool {tool_name} failed with exception: {str(e)}"
                success = False
                return result
            finally:
                if metrics:
                    metrics.record_tool_call(
                        tool_name=tool_name,
                        success=success,
                        latency_ms=(time.perf_counter() - tool_start) * 1000.0,
                        effective_calls=effective_calls,
                        status_code=status_code,
                    )
        else:
            result = f"Error: Tool {tool_name} not found"
            if metrics:
                metrics.record_tool_call(
                    tool_name=tool_name,
                    success=False,
                    latency_ms=(time.perf_counter() - tool_start) * 1000.0,
                    effective_calls=effective_calls,
                    status_code=status_code,
                )
            return result

    def _execute_search_with_controller(
        self,
        tool_name: str,
        tool_args: dict,
        metrics: Optional[MetricsCollector],
        tool_start: float,
        **kwargs,
    ) -> str:
        query_list = tool_args.get("query", [])
        if isinstance(query_list, str):
            query_list = [query_list]

        if not query_list:
            if metrics:
                metrics.record_tool_call(
                    tool_name=tool_name,
                    success=False,
                    latency_ms=(time.perf_counter() - tool_start) * 1000.0,
                    effective_calls=0,
                    status_code=400,
                )
            return "[SearchController] Error: empty query list"

        request = SearchRequest(
            task_id=str(id(self)),
            turn_id=self.turn_id,
            tool_name=tool_name,
            search_engine=tool_name,
            query_list=query_list,
            original_question=self.user_prompt,
            raw_args=tool_args,
            requested_topk=None,
        )
        self.controller_state.budget_state.current_turn = self.turn_id

        decision = self.search_controller.pre_search(request, self.controller_state)

        block_results, block_latencies = self._execute_query_blocks(tool_name, tool_args, request, decision, metrics, tool_start, **kwargs)

        if self.query_logger is not None:
            self._log_query_processing(request.task_id, request.turn_id, query_list, decision, block_results)

        for block, result_str, latency in zip(decision.query_blocks, block_results, block_latencies):
            entry = self._build_memory_entry(tool_name, block, result_str, request, latency)
            self.search_controller.update_memory(entry, self.controller_state)

        if metrics:
            effective_calls = sum(
                1 for block in decision.query_blocks
                if block.action in (SearchAction.EXECUTE, SearchAction.MERGE, SearchAction.REDUCE_TOPK)
            )
            reuse_count = sum(
                len(block.queries) for block in decision.query_blocks
                if block.action == SearchAction.REUSE_CACHE
            )
            merge_saved_count = sum(
                len(block.queries) - 1 for block in decision.query_blocks
                if block.action == SearchAction.MERGE
            )
            reduce_topk_count = sum(
                1 for block in decision.query_blocks
                if block.action == SearchAction.REDUCE_TOPK
            )
            total_latency = sum(block_latencies)
            has_error = any("Error" in result_str for result_str in block_results)
            metrics.record_tool_call(
                tool_name=tool_name,
                success=not has_error,
                latency_ms=total_latency,
                effective_calls=effective_calls,
                status_code=None,
                reuse_count=reuse_count,
                merge_saved_count=merge_saved_count,
                reduce_topk_count=reduce_topk_count,
            )

        return "\n=======\n".join(block_results)

    def _log_query_processing(
        self,
        task_id: str,
        turn_id: int,
        query_list: list[str],
        decision: PreSearchDecision,
        block_results: list[str],
    ):
        blocks_info = []
        for block, result_str in zip(decision.query_blocks, block_results):
            block_info = {
                "action": block.action.value,
                "original_queries": block.queries,
                "success": "Error" not in result_str,
                "selected_result_indices": block.selected_result_indices,
                "raw_observation_tokens": block.raw_observation_tokens,
                "returned_observation_tokens": block.returned_observation_tokens,
                "reuse_pointer": block.reuse_pointer,
            }
            if block.action == SearchAction.MERGE:
                block_info["merged_query"] = block.merged_query
            if block.action == SearchAction.REUSE_CACHE and block.cache_entry:
                block_info["cache_info"] = {
                    "cached_query": block.cache_entry.query,
                    "sim": 0.0,
                    "turn_dist": turn_id - block.cache_entry.turn_id,
                }
            blocks_info.append(block_info)

        self.query_logger.record_turn(
            task_id=task_id,
            turn_id=turn_id,
            query_list=query_list,
            intra_sim=self.search_controller.last_intra_sim,
            blocks=blocks_info,
        )
        self.query_logger.flush()

    def _execute_query_blocks(
        self,
        tool_name: str,
        tool_args: dict,
        request: SearchRequest,
        decision: PreSearchDecision,
        metrics: Optional[MetricsCollector],
        tool_start: float,
        **kwargs,
    ) -> tuple[list[str], list[float]]:
        block_results = []
        block_latencies = []

        for block in decision.query_blocks:
            block_start = time.perf_counter()
            if block.action == SearchAction.EXECUTE or block.action == SearchAction.REDUCE_TOPK:
                block_args = tool_args.copy()
                block_args["query"] = block.queries
                try:
                    raw = TOOL_MAP[tool_name].call(block_args, **kwargs)
                    if isinstance(raw, tuple) and len(raw) == 2:
                        raw_result, _ = raw
                    else:
                        raw_result = raw
                    block_result = self.search_controller.post_search(
                        request, str(raw_result), self.controller_state, block
                    )
                except Exception as e:
                    block_result = f"Error: Search query failed: {str(e)}"
                    block.raw_result = block_result
                    block.returned_observation = block_result
                block_results.append(block_result)
                block_latencies.append((time.perf_counter() - block_start) * 1000.0)

            elif block.action == SearchAction.MERGE:
                merged_query = self._merge_queries(block.queries, metrics)
                block.merged_query = merged_query
                block_args = tool_args.copy()
                block_args["query"] = [merged_query]
                try:
                    raw = TOOL_MAP[tool_name].call(block_args, **kwargs)
                    if isinstance(raw, tuple) and len(raw) == 2:
                        raw_result, _ = raw
                    else:
                        raw_result = raw
                    block_result = self.search_controller.post_search(
                        request, str(raw_result), self.controller_state, block
                    )
                except Exception as e:
                    block_result = f"Error: Merged search query failed: {str(e)}"
                    block.raw_result = block_result
                    block.returned_observation = block_result
                block_results.append(block_result)
                block_latencies.append((time.perf_counter() - block_start) * 1000.0)

            elif block.action == SearchAction.REUSE_CACHE:
                if block.cache_entry:
                    block_result = self.search_controller.post_search(
                        request, block.cache_entry.raw_result, self.controller_state, block
                    )
                else:
                    block_result = "Error: Empty cache entry found"
                    block.raw_result = block_result
                    block.returned_observation = block_result
                block_results.append(block_result)
                block_latencies.append(0.0)

            elif block.action == SearchAction.SKIP_DUPLICATE:
                block_results.append("[SKIPPED] Duplicate query")
                block_latencies.append(0.0)

        return block_results, block_latencies

    def _build_memory_entry(
        self,
        tool_name: str,
        block: QueryBlock,
        result_str: str,
        request: SearchRequest,
        latency_ms: float = 0.0,
    ) -> SearchMemoryEntry:
        raw_result = block.raw_result or result_str
        returned_observation = block.returned_observation or result_str
        url_list = self._extract_urls(raw_result)
        raw_tokens = block.raw_observation_tokens or self._count_tokens(raw_result)
        returned_tokens = block.returned_observation_tokens or self._count_tokens(returned_observation)

        if block.action == SearchAction.REUSE_CACHE:
            tool_status = ToolStatus.CACHED
        elif block.action == SearchAction.SKIP_DUPLICATE:
            tool_status = ToolStatus.SKIPPED
        elif "Error" in returned_observation:
            tool_status = ToolStatus.FAILED
        elif not returned_observation.strip():
            tool_status = ToolStatus.EMPTY
        else:
            tool_status = ToolStatus.SUCCESS_NONEMPTY

        query = block.merged_query or block.queries[0]
        query_embedding = block.query_embedding
        if not query_embedding:
            query_embedding = self.search_controller._embed_text(query)

        return SearchMemoryEntry(
            query=query,
            query_embedding=query_embedding,
            turn_id=request.turn_id,
            tool_name=tool_name,
            search_engine=request.search_engine,
            action=block.action,
            executed_external_api=block.action in (SearchAction.EXECUTE, SearchAction.MERGE, SearchAction.REDUCE_TOPK),
            tool_status=tool_status,
            raw_result=raw_result,
            returned_observation=returned_observation,
            url_list=url_list,
            raw_observation_tokens=raw_tokens,
            returned_observation_tokens=returned_tokens,
            latency_ms=latency_ms,
            selected_result_indices=block.selected_result_indices,
            selected_result_texts=block.selected_result_texts,
            selected_result_embeddings=block.selected_result_embeddings,
        )

    def _extract_urls(self, result: str) -> list[str]:
        return re.findall(r'\]\((https?://[^\s)]+)\)', result)

    _tiktoken_encoder = None

    def _count_tokens(self, text: str) -> int:
        if self._tiktoken_encoder is None:
            import tiktoken
            MultiTurnReactAgent._tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        return len(self._tiktoken_encoder.encode(text))

    def _merge_queries(self, queries: list[str], metrics: Optional[MetricsCollector] = None) -> str:
        if len(queries) <= 1:
            return queries[0] if queries else ""

        rephrase_client = ModelClient('rephrase')
        if not rephrase_client.model or not rephrase_client.api_key:
            print(f"[MERGE] Rephrase model not available, using first query: '{queries[0]}'")
            return queries[0]

        client = rephrase_client.get_client()

        # Format all queries into a numbered list for the multi-query prompt
        queries_text = "\n".join(f"Query {i+1}: {q}" for i, q in enumerate(queries))
        prompt_content = MERGE_MULTI_PROMPT.format(queries=queries_text)
        messages = [{"role": "user", "content": prompt_content}]

        start_time = time.time()
        try:
            print(f"[MERGE] Merging {len(queries)} queries at once: {[q[:40] for q in queries]}")
            chat_response = client.chat.completions.create(
                model=rephrase_client.model,
                messages=messages,
                temperature=0.2,
                max_tokens=40960,
            )
            latency_ms = (time.time() - start_time) * 1000
            content = chat_response.choices[0].message.content
            if content is None:
                content = chat_response.choices[0].message.reasoning
            merged = content.strip()
            print(f"[MERGE] Result: '{merged[:80]}'")

            if metrics:
                usage = MetricsCollector.usage_to_dict(getattr(chat_response, "usage", None))
                metrics.record_model_call(
                    model_group="rephrase_model",
                    success=True,
                    latency_ms=latency_ms,
                    usage=usage,
                )
                metrics.record_prompt_breakdown(
                    model_group="rephrase_model",
                    messages=messages,
                    usage=usage,
                )

            return merged
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            print(f"[MERGE] Error: {e}, falling back to first query")
            if metrics:
                metrics.record_model_call(
                    model_group="rephrase_model",
                    success=False,
                    latency_ms=latency_ms,
                    usage=None,
                )
            return queries[0]
