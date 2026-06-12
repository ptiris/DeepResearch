# Agents Context

## Project Overview

Tongyi DeepResearch is an agentic LLM (30B total, 3.3B activated) for deep information-seeking tasks. It uses a ReAct inference paradigm calling external APIs — not local vLLM serving — via a multi-provider `ModelClient`.

## Key Commands

### Setup
```bash
conda create -n react_infer_env python=3.10.0
conda activate react_infer_env
pip install -r requirements.txt
cp .env.example .env  # Then edit .env with your keys and model config
```

### Inference
```bash
bash inference/run_react_infer.sh
```
This script sources `.env`, validates `OPENROUTER_API_KEY`, then runs `run_multi_react.py`. It does **not** start vLLM servers — the agent calls an external OpenAI-compatible API.

Configurable via env vars: `DATASET`, `DATA_FILE`, `OUTPUT_PATH`, `MAX_WORKERS`, `TEMPERATURE`, `PRESENCE_PENALTY`, `ROLLOUT_COUNT`, `WORLD_SIZE`, `RANK`.

### Evaluation
```bash
# For HLE benchmark
export API_KEY=... BASE_URL=...
python evaluation/eval_hle_old_react.py --input_fp <dir> --model_path <model>

# For other benchmarks
export OPENAI_API_KEY=... OPENAI_API_BASE=... API_KEY=... BASE_URL=... Qwen2_5_7B_PATH=...
python evaluation/evaluate_all_official.py --input_fp <dir> --dataset <name>
```

## Architecture

- `inference/react_agent.py` — Core `MultiTurnReactAgent` class with ReAct tool-calling loop
- `inference/model_client.py` — Multi-provider routing (openrouter/dashscope/openai) with per-stage config (`ModelClient`)
- `inference/search_controller.py` — `SearchController` for query deduplication, caching, merging, and dynamic K_obs observation budget management
- `inference/prompt.py` / `inference/prompt_new.py` — Two prompt variants; toggled by `PROMPT_NEW=True` env var
- `inference/run_multi_react.py` — Multi-worker inference orchestration with parallel rollout support
- `inference/run_react_infer.sh` — Entry point; sources `.env` → runs `run_multi_react.py`
- `inference/tool_*.py` — Tool implementations: `search`, `aliyun_search`, `visit`, `google_scholar`, `PythonInterpreter`, `parse_file`
- `inference/file_tools/` — File parsing subsystem (file_parser, idp, video_analysis, video_agent)
- `inference/metrics.py` — `MetricsCollector` and `QueryProcessLogger` for API call tracking
- `utils/embedding_client.py` — `EmbeddingClient` for similarity-based query dedup
- `evaluation/` — Benchmark eval scripts (evaluate_deepsearch_official.py, evaluate_hle_official.py)
- `scripts/` — Post-processing: cleanup, relevance computation, visualization
- `WebAgent/` — Individual web agent papers/models (WebDancer, WebSailor, etc.)
- `Agent/` — AgentFounder/AgentScaler (gitignored, not part of main pipeline)

## Multi-Provider Configuration

`ModelClient` routes LLM calls by stage, each using a separate provider/model:

| Stage | Env Var | Default Provider |
|-------|---------|-----------------|
| research | `RESEARCH_MODEL` | `PROVIDER` (default: openrouter) |
| rephrase | `REPHASE_MODEL` | `PROVIDER` (default: openrouter) |
| summary | `SUMMARY_MODEL` | `PROVIDER` (default: openrouter) |
| embedding | `EMBEDDING_MODEL` | dashscope |

Override provider per stage: `RESEARCH_PROVIDER=dashscope`, etc. Each provider reads its own `*_API_KEY` and `*_BASE_URL` env vars.

## Tools Selection

Controlled by `AVAILABLE_TOOLS` env var (comma-separated). Default: `search,visit,google_scholar,PythonInterpreter`. Options: `search`, `aliyun_search`, `visit`, `google_scholar`, `PythonInterpreter`, `parse_file`.

## Data Format

JSONL recommended (JSON accepted):
```jsonl
{"question": "What is X?", "answer": "Reference answer for evaluation"}
```

For file processing: prepend filename to question, place files in `eval_data/file_corpus/`:
```jsonl
{"question": "(Uploaded 1 file: ['report.pdf'])\n\nWhat are the findings?", "answer": "..."}
```

## Required Environment Variables

**Provider API keys** (set whichever provider you use):
- `OPENROUTER_API_KEY` — OpenRouter (default provider)
- `DASHSCOPE_API_KEY` — Aliyun DashScope
- `OPENAI_API_KEY` / `OPENAI_API_BASE` — OpenAI-compatible fallback

**Search/web tools**:
- `SERPER_KEY_ID` — Serper.dev (for `search` tool)
- `ALIYUN_IQS_API_KEY` — Aliyun IQS (for `aliyun_search` tool)
- `JINA_API_KEYS` — Jina.ai (for `visit` tool page reading)

**Other services**:
- `API_KEY` / `API_BASE` / `SUMMARY_MODEL_NAME` — Summarization model
- `DASHSCOPE_API_KEY` — Also used for file parsing and video analysis
- `SANDBOX_FUSION_ENDPOINT` — Python interpreter sandbox (comma-separated for multiple)

**Feature flags**:
- `DISABLE_SEARCH_CONTROLLER` — Set `true` to bypass SearchController
- `PROMPT_NEW` — Set `True` to use `prompt_new.py` instead of `prompt.py`
- `REDUNDANCY_ENABLED`, `REDUNDANCY_STRATEGY`, `REDUNDANCY_SCOPE`, `REDUNDANCY_SIMILARITY_THRESHOLD` — Query deduplication
- `SEARCH_DYNAMIC_K_OBS` — Set `true` (default) to enable dynamic K_obs observation budget; `false` uses static topk

## Important Notes

- Python 3.10.0 required (other versions may cause dependency issues)
- No test suite, lint, or typecheck config in this repo
- `.env` is gitignored — never commit secrets
- Output is resumable: already-processed questions are skipped on re-run (checked by question text)
- Rollout support: `ROLLOUT_COUNT` controls how many independent passes per question
- Distributed support: `WORLD_SIZE`/`RANK` for splitting work across machines
- Max LLM calls per query: 100 (configurable via `MAX_LLM_CALL_PER_RUN` env var)
- Max context: ~110K tokens; timeout: 1800s per query
- `data/`, `output/`, `cache/`, `Agent/`, `models/` directories are gitignored
- CI only generates pdoc docs on push to main (`.github/workflows/docs.yml`)