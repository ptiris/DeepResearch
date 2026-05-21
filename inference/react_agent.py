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
from metrics import MetricsCollector
from model_client import ModelClient

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
from utils.query_deduplication import QueryHistory

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
        enable_redundancy_check: bool = False,
        redundancy_strategy: str = "rephase",
        redundancy_scope: str = "single_turn",
        redundancy_similarity_threshold: float = 0.8,
        redundancy_max_retries: int = 2) -> None:
        super().__init__(llm=DummyLLM(), function_list=function_list)
        self.llm_generate_cfg = llm_cfg.get("generate_cfg", {}) if llm_cfg else {}
        self.custom_system_prompt = system_prompt
        self._function_list = function_list if function_list else []
        
        # Query deduplication configuration
        self.enable_redundancy_check = enable_redundancy_check
        self.redundancy_strategy = redundancy_strategy
        self.redundancy_scope = redundancy_scope
        self.redundancy_similarity_threshold = redundancy_similarity_threshold
        self.redundancy_max_retries = redundancy_max_retries
        
        # Initialize embedding client (will be initialized per question in _run)
        self.embedding_client = None
        self.query_history = None

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
            f"content": "{privider} error!!!",
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
        
        # Initialize query deduplication components
        if self.enable_redundancy_check:
            print(f"[REDUNDANCY] Initializing redundancy check (strategy: {self.redundancy_strategy}, scope: {self.redundancy_scope}, threshold: {self.redundancy_similarity_threshold})")
            self.embedding_client = EmbeddingClient()
            self.query_history = QueryHistory()
        else:
            print(f"[REDUNDANCY] Redundancy check is disabled")
        self.turn_id = 0
        rephase_retry_count = 0
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
                        
                        # Check if result contains REDUNDANT_REPHASE marker
                        if isinstance(result, str) and "[REDUNDANT_REPHASE]" in result:
                            if rephase_retry_count < self.redundancy_max_retries:
                                rephase_retry_count += 1
                                print(f"[REDUNDANCY] Rephasing attempt {rephase_retry_count}/{self.redundancy_max_retries}")
                                
                                # Add feedback to messages and retry
                                messages.append({"role": "user", "content": result})
                                model_response = self.call_server(messages, metrics=metrics)
                                content = model_response.get("content", "")
                                print(f'Round {round}.{rephase_retry_count}: {content}')
                                if OBS_END in content:
                                    pos = content.find(OBS_END)
                                    content = content[:pos]
                                messages.append({"role": "assistant", "content": content.strip()})
                                
                                # Continue to next iteration to parse new tool call
                                num_llm_calls_available += 1  # Don't count this as a new LLM call
                                continue
                            else:
                                print(f"[REDUNDANCY] Max retries ({self.redundancy_max_retries}) exceeded, proceeding with execution")
                                # Proceed with current result (will be added to messages below)
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

        # Check if redundancy checking is enabled for search tools
        if self.enable_redundancy_check and tool_name in MetricsCollector.SEARCH_TOOL_NAMES:
            query_list = tool_args.get("query", [])
            if isinstance(query_list, str):
                query_list = [query_list]

            if query_list:
                print(f"[REDUNDANCY] Checking {len(query_list)} queries for redundancy (scope: {self.redundancy_scope})")
                # Perform redundancy check
                deduplication_result = self._check_query_redundancy(query_list)
                print(f"[REDUNDANCY] Found {len(deduplication_result['redundant_queries'])} redundant queries, {len(deduplication_result['non_redundant_queries'])} non-redundant")
                
                if deduplication_result["has_redundant"]:
                    # Handle based on strategy
                    if self.redundancy_strategy == "rephase":
                        merged_result = self._handle_rephase_strategy(deduplication_result, metrics)
                        if isinstance(merged_result, list):
                            print(f"[REDUNDANCY] Executing merged queries: {merged_result}")
                            new_tool_args = tool_args.copy()
                            new_tool_args["query"] = merged_result
                            result = self._call_tool_and_update_history(
                                tool_name, new_tool_args, metrics=metrics, tool_start=tool_start, **kwargs
                            )
                        else:
                            result = merged_result
                    elif self.redundancy_strategy == "skip":
                        result = self._handle_skip_strategy(tool_name, tool_args, deduplication_result, metrics, tool_start, **kwargs)
                    elif self.redundancy_strategy == "cache":
                        result = self._handle_cache_strategy(tool_name, tool_args, deduplication_result, metrics, tool_start, **kwargs)
                    else:
                        result = self._call_tool_and_update_history(tool_name, tool_args, metrics=metrics, tool_start=tool_start, **kwargs)

                    return result
        else:
            if tool_name in MetricsCollector.SEARCH_TOOL_NAMES:
                print(f"[REDUNDANCY] Skipped: enable_redundancy_check={self.enable_redundancy_check} (search tool '{tool_name}' called without redundancy check)")
        
        if tool_name in TOOL_MAP:
            print(f"[DEBUG] custom_call_tool invoked with tool_name: '{tool_name}', args: {tool_args}")
            if tool_name == "search" and "search" not in self._function_list and "aliyun_search" in self._function_list:
                print(f"[DEBUG] Remapping 'search' to 'aliyun_search' (Serper not available, using Aliyun IQS)")
                tool_name = "aliyun_search"
            tool_args["params"] = tool_args
            if tool_name in MetricsCollector.SEARCH_TOOL_NAMES:
                query = tool_args.get("query", [])
                if isinstance(query, list):
                    effective_calls = len(query)
                elif isinstance(query, str):
                    effective_calls = 1
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
    
    def _check_query_redundancy(self, query_list: List[str]) -> Dict:
        """
        Check for redundant queries in the query list

        Args:
            query_list: List of query strings

        Returns:
            Dict containing redundant and non-redundant queries
        """
        print(f"[REDUNDANCY] Computing embeddings for {len(query_list)} queries")
        # Compute embeddings for all queries
        embeddings = self.embedding_client.encode(query_list)

        # Find similar queries
        redundant_queries = []
        non_redundant_queries = []

        # In single_turn mode, maintain a history list and check each query against it
        seen_representative_queries = []  # List of (query_text, embedding) that are already kept

        for i, (query, emb) in enumerate(zip(query_list, embeddings)):
            print(f"\n[REDUNDANCY] === Checking query {i}: '{query[:60]}' ===")
            if self.redundancy_scope == "single_turn":
                # Check against previously kept queries in current turn
                sim_queries = []
                if not seen_representative_queries:
                    print(f"[REDUNDANCY]   No history to compare (first query in this turn)")
                for rep_text, rep_emb in seen_representative_queries:
                    sim = self.embedding_client.similarity(emb, rep_emb)
                    status = "REDUNDANT" if sim >= self.redundancy_similarity_threshold else "unique"
                    print(f"[REDUNDANCY]   vs '{rep_text[:50]}' -> sim={sim:.4f} (threshold={self.redundancy_similarity_threshold}) => {status}")
                    if sim >= self.redundancy_similarity_threshold:
                        sim_queries.append({
                            "query": rep_text,
                            "similarity": sim,
                            "is_executed": None,
                            "results": None
                        })
            else:  # global
                # Compare with all historical queries
                history_size = self.query_history.get_history_size()
                print(f"[REDUNDANCY]   Global scope: comparing against {history_size} historical queries")
                sim_queries = self.query_history.find_similar_queries(
                    emb,
                    self.redundancy_similarity_threshold,
                    scope="global"
                )

            if sim_queries:
                print(f"[REDUNDANCY] Query {i}: '{query[:60]}' => REDUNDANT (found {len(sim_queries)} similar)")
                for sim_query in sim_queries:
                    print(f"[REDUNDANCY]   Similar to: '{sim_query['query'][:50]}' (sim={sim_query['similarity']:.4f})")
                redundant_queries.append({
                    "index": i,
                    "query": query,
                    "embedding": emb,
                    "similar_to": sim_queries
                })
            else:
                print(f"[REDUNDANCY] Query {i}: '{query[:50]}...' is unique (no similar queries found)")
                non_redundant_queries.append({
                    "index": i,
                    "query": query,
                    "embedding": emb
                })
                if self.redundancy_scope == "single_turn":
                    seen_representative_queries.append((query, emb))

        return {
            "has_redundant": len(redundant_queries) > 0,
            "redundant_queries": redundant_queries,
            "non_redundant_queries": non_redundant_queries,
            "query_list": query_list,
            "embeddings": embeddings
        }
    
    def _merge_queries_with_rephrase(self, q1: str, q2: str, metrics: Optional[MetricsCollector] = None) -> str:
        """
        Merge two queries using the rephrase model and REPHASE_PROMPT.

        Args:
            q1: First query
            q2: Second query
            metrics: Metrics collector

        Returns:
            str: Merged query
"""
        import time
        rephrase_client = ModelClient('rephrase')
        if not rephrase_client.model:
            print("[REDUNDANCY] REPHASE_MODEL not set, returning q2 unchanged")
            return q2
        if not rephrase_client.api_key:
            print("[REDUNDANCY] No API key available, returning q2 unchanged")
            return q2

        prompt_content = REPHASE_PROMPT.format(q1=q1, q2=q2)
        client = rephrase_client.get_client()

        messages = [{"role": "user", "content": prompt_content}]

        start_time = time.time()
        try:
            print(f"[REDUNDANCY] Calling rephrase model: {rephrase_client.model}")
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
            merged_query = content.strip()
            print(f"[REDUNDANCY] Merged '{q1}' + '{q2}' -> '{merged_query}'")

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

            return merged_query
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            print(f"[REDUNDANCY] Error calling rephrase model: {e}, returning q2 unchanged")

            if metrics:
                metrics.record_model_call(
                    model_group="rephrase_model",
                    success=False,
                    latency_ms=latency_ms,
                    usage=None,
                )

            return q2

    def _handle_rephase_strategy(self, deduplication_result: Dict, metrics: Optional[MetricsCollector] = None) -> str:
        """
        Handle rephase strategy for redundant queries.

        For single_turn scope: directly merge similar queries using rephrase model.
        For global scope: fall back to the original feedback-based approach.

        Args:
            deduplication_result: Result from _check_query_redundancy
            metrics: Metrics collector

        Returns:
            str: Merged query list or feedback message
        """
        print(f"[REDUNDANCY] Using rephase strategy to handle redundant queries")

        if self.redundancy_scope != "single_turn":
            redundant_queries = deduplication_result["redundant_queries"]
            similar_queries = [q["similar_to"][0]["query"] for q in redundant_queries if q["similar_to"]]
            feedback = f"You have already proposed similar queries: {similar_queries}. "
            feedback += "Please make sure your new queries are not similar to these previous queries."
            return f"[REDUNDANCY] {feedback}"

        query_list = deduplication_result["query_list"]
        embeddings = deduplication_result["embeddings"]

        processed_queries = []
        processed_embeddings = []

        for i, (query, emb) in enumerate(zip(query_list, embeddings)):
            merged = False
            for j, rep_emb in enumerate(processed_embeddings):
                sim = self.embedding_client.similarity(emb, rep_emb)
                if sim >= self.redundancy_similarity_threshold:
                    rep_query = processed_queries[j]
                    print(f"[REDUNDANCY] Query '{query}' is similar to '{rep_query}' (similarity: {sim:.4f}), merging")
                    merged_query = self._merge_queries_with_rephrase(rep_query, query, metrics)
                    processed_queries.pop(j)
                    processed_embeddings.pop(j)
                    merged_emb = self.embedding_client.encode([merged_query])[0]
                    processed_queries.append(merged_query)
                    processed_embeddings.append(merged_emb)
                    merged = True
                    break

            if not merged:
                processed_queries.append(query)
                processed_embeddings.append(emb)

        print(f"[REDUNDANCY] Merged query list: {processed_queries}")

        if len(processed_queries) == 1:
            return processed_queries[0]
        return processed_queries
    
    def _handle_skip_strategy(self, tool_name: str, tool_args: Dict, deduplication_result: Dict,
                             metrics: Optional[MetricsCollector], tool_start: float, **kwargs) -> str:
        """
        Handle skip strategy for redundant queries

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            deduplication_result: Result from _check_query_redundancy
            metrics: Metrics collector
            tool_start: Tool start time

        Returns:
            str: Combined results with skipped queries marked
        """
        print(f"[REDUNDANCY] Using skip strategy: executing {len(deduplication_result['non_redundant_queries'])} unique queries, skipping {len(deduplication_result['redundant_queries'])} redundant queries")
        redundant_queries = deduplication_result["redundant_queries"]
        non_redundant = deduplication_result["non_redundant_queries"]
        query_list = deduplication_result["query_list"]

        # Execute non-redundant queries
        results = {}
        for item in non_redundant:
            idx = item["index"]
            query = item["query"]
            emb = item["embedding"]

            new_tool_args = tool_args.copy()
            new_tool_args["query"] = [query] if len(query_list) > 1 else query

            print(f"[REDUNDANCY] Executing unique query {idx}: '{query[:50]}...'")
            result = self._call_tool_and_update_history(
                tool_name, new_tool_args, metrics=metrics, tool_start=tool_start, query_emb=emb, **kwargs
            )
            results[idx] = result

        # Build final results, mark skipped queries
        final_results = []
        for i, query in enumerate(query_list):
            redundant_item = next((q for q in redundant_queries if q["index"] == i), None)
            if redundant_item:
                print(f"[REDUNDANCY] Skipping redundant query {i}: '{query[:50]}...'")
                final_results.append(f"[SKIPPED] Duplicate of query: \"{query}\"")
                # Do NOT add to history for skip strategy
            else:
                final_results.append(results[i])

        return "\n=======\n".join(final_results)
    
    def _handle_cache_strategy(self, tool_name: str, tool_args: Dict, deduplication_result: Dict,
                              metrics: Optional[MetricsCollector], tool_start: float, **kwargs) -> str:
        """
        Handle cache strategy for redundant queries

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            deduplication_result: Result from _check_query_redundancy
            metrics: Metrics collector
            tool_start: Tool start time

        Returns:
            str: Combined results with cached queries
        """
        print(f"[REDUNDANCY] Using cache strategy: executing {len(deduplication_result['non_redundant_queries'])} unique queries, using cached results for {len(deduplication_result['redundant_queries'])} redundant queries")
        redundant_queries = deduplication_result["redundant_queries"]
        non_redundant = deduplication_result["non_redundant_queries"]
        query_list = deduplication_result["query_list"]
        embeddings = deduplication_result["embeddings"]

        # Execute non-redundant queries
        results = {}
        for item in non_redundant:
            idx = item["index"]
            query = item["query"]
            emb = item["embedding"]

            new_tool_args = tool_args.copy()
            new_tool_args["query"] = [query] if len(query_list) > 1 else query

            print(f"[REDUNDANCY] Executing unique query {idx}: '{query[:50]}...'")
            result = self._call_tool_and_update_history(
                tool_name, new_tool_args, metrics=metrics, tool_start=tool_start, query_emb=emb, **kwargs
            )
            results[idx] = result
        
        # Handle redundant queries (use cache)
        final_results = []
        for i, (query, emb) in enumerate(zip(query_list, embeddings)):
            redundant_item = next((q for q in redundant_queries if q["index"] == i), None)
            if redundant_item:
                # Find first executed similar query
                executed_query = self.query_history.find_first_executed_similar_query(
                    redundant_item["similar_to"]
                )

                if executed_query:
                    # Use cached result
                    cached_result = executed_query["results"]
                    print(f"[REDUNDANCY] Using cached result for query {i}: '{query[:50]}...' (from: '{executed_query['query'][:50]}...')")
                    final_results.append(cached_result)

                    # Add to history with is_executed=False
                    self.query_history.add_query(
                        query=query,
                        embedding=emb,
                        results=cached_result,
                        success=executed_query["success"],
                        is_executed=False,
                        turn_id=self.turn_id
                    )
                else:
                    # No executed similar query found, execute current query
                    print(f"[REDUNDANCY] No cached result found for query {i}: '{query[:50]}...', executing anyway")
                    new_tool_args = tool_args.copy()
                    new_tool_args["query"] = [query] if len(query_list) > 1 else query

                    result = self._call_tool_and_update_history(
                        tool_name, new_tool_args, metrics=metrics, tool_start=tool_start, query_emb=emb, **kwargs
                    )
                    final_results.append(result)
            else:
                final_results.append(results[i])

        return "\n=======\n".join(final_results)
    
    def _call_tool_and_update_history(self, tool_name: str, tool_args: Dict, metrics: Optional[MetricsCollector] = None,
                                     tool_start: float = None, query_emb: np.ndarray = None, **kwargs):
        """
        Call tool and update query history if it's a search tool
        
        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            metrics: Metrics collector
            tool_start: Tool start time (for metrics)
            query_emb: Query embedding (if already computed)
            
        Returns:
            Tool result
        """
        if tool_start is None:
            tool_start = time.perf_counter()
        
        # Call tool
        if tool_name not in TOOL_MAP:
            result = f"Error: Tool {tool_name} not found"
            if metrics:
                metrics.record_tool_call(
                    tool_name=tool_name,
                    success=False,
                    latency_ms=(time.perf_counter() - tool_start) * 1000.0,
                    effective_calls=1,
                    status_code=None,
                )
            return result
        
        try:
            tool_args["params"] = tool_args
            raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
            
            if isinstance(raw_result, tuple) and len(raw_result) == 2:
                result, status_code = raw_result
            else:
                result = raw_result
                status_code = None
            
            success = MetricsCollector.infer_tool_success(result)
            
            # Update history if it's a search tool
            if tool_name in MetricsCollector.SEARCH_TOOL_NAMES and self.enable_redundancy_check:
                query = tool_args.get("query", "")
                if isinstance(query, list):
                    query = query[0]

                # Compute embedding if not provided
                if query_emb is None:
                    query_emb = self.embedding_client.encode([query])[0]

                # Add to history
                print(f"[REDUNDANCY] Adding query to history: '{query[:50]}...' (turn_id: {self.turn_id}, success: {success})")
                self.query_history.add_query(
                    query=query,
                    embedding=query_emb,
                    results=result,
                    success=success,
                    is_executed=True,
                    turn_id=self.turn_id
                )
            
            # Record metrics
            if metrics:
                effective_calls = 1
                if tool_name in MetricsCollector.SEARCH_TOOL_NAMES:
                    query = tool_args.get("query", [])
                    if isinstance(query, list):
                        effective_calls = len(query)
                    elif isinstance(query, str):
                        effective_calls = 1
                
                metrics.record_tool_call(
                    tool_name=tool_name,
                    success=success,
                    latency_ms=(time.perf_counter() - tool_start) * 1000.0,
                    effective_calls=effective_calls,
                    status_code=status_code,
                )
            
            return result
            
        except Exception as e:
            result = f"Error: Tool {tool_name} failed with exception: {str(e)}"
            success = False
            
            if metrics:
                metrics.record_tool_call(
                    tool_name=tool_name,
                    success=False,
                    latency_ms=(time.perf_counter() - tool_start) * 1000.0,
                    effective_calls=1,
                    status_code=None,
                )
            
            return result
