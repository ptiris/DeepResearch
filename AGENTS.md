# Agents Context

## Project Overview

Tongyi DeepResearch is an agentic LLM (30B total, 3.3B activated) for deep information-seeking tasks. It builds on the WebAgent family and uses a ReAct inference paradigm.

## Key Commands

### Setup
```bash
conda create -n react_infer_env python=3.10.0
conda activate react_infer_env
pip install -r requirements.txt
cp .env.example .env  # Then edit .env with your keys
```

### Inference
```bash
bash inference/run_react_infer.sh
```

The script starts 8 vLLM servers (ports 6001-6008), waits for them to be ready, then runs `run_multi_react.py`.

### Manual single-run
```bash
# Start vLLM manually
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6001

# Run inference
python -u inference/run_multi_react.py --dataset your_data.jsonl --output ./outputs --max_workers 30 --model $MODEL_PATH
```

## Architecture

- `inference/react_agent.py` - Core ReAct agent with tool calling loop
- `inference/tool_*.py` - Tools: search, visit, google_scholar, PythonInterpreter, parse_file
- `inference/run_multi_react.py` - Multi-worker inference orchestration
- `inference/run_react_infer.sh` - Full startup script (vLLM servers + inference)
- `evaluation/` - Benchmark evaluation scripts (evaluate_deepsearch_official.py, evaluate_hle_official.py)
- `WebAgent/` - Individual web agent papers/models (WebDancer, WebSailor, WebWalker, etc.)

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

- `MODEL_PATH` - Path to model weights
- `SERPER_KEY_ID` - Serper.dev for web search
- `JINA_API_KEYS` - Jina.ai for page reading
- `API_KEY/API_BASE` - OpenAI-compatible API for summarization
- `DASHSCOPE_API_KEY` - Aliyun for file parsing
- `SANDBOX_FUSION_ENDPOINT` - Python interpreter sandbox

## Important Notes

- Python 3.10.0 required (other versions may cause dependency issues)
- No test suite exists in this repo
- No lint/typecheck configuration
- `.env` is gitignored - never commit secrets
- vLLM servers use exponential backoff retry (max 10 attempts, up to 30s delay)
- Max context: 110K tokens; runtime limit: 150 minutes per query
