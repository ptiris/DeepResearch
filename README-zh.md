# Tongyi DeepResearch 实验说明

本文档记录当前仓库相较于原版新增或改动的启动方式、环境变量和实验参数。当前推理链路通过 `ModelClient` 调用外部 OpenAI-compatible API，不再在本地启动 vLLM 服务。

## 快速启动

1. 创建环境并安装依赖：

```bash
conda create -n react_infer_env python=3.10.0
conda activate react_infer_env
pip install -r requirements.txt
```

2. 准备 `.env`：

```bash
cp .env.example .env
```

然后在 `.env` 中填入模型、数据、工具 API key 和输出路径。

3. 如启用 `PythonInterpreter`，先启动 SandboxFusion：

```bash
docker run -it -p 8080:8080 vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

如果已有 tmux session，可以用下面的方式检查：

```bash
tmux a -t sandbox
```

4. 运行推理：

```bash
bash inference/run_react_infer.sh
```

`run_react_infer.sh` 会自动读取项目根目录下的 `.env`，检查 `OPENROUTER_API_KEY` 是否配置，然后进入 `inference/` 目录执行 `run_multi_react.py`。脚本不会启动 vLLM 或本地模型服务。

## 启动脚本参数

`inference/run_react_infer.sh` 会把 `.env` 中的变量转换为 `run_multi_react.py` 参数：

| `.env` 变量 | 传入参数 | 说明 |
| --- | --- | --- |
| `DATASET` | `--dataset` | 实验/数据集名称，也会作为输出目录名的一部分 |
| `DATA_FILE` | `--data_file` | 输入 JSON/JSONL 文件路径 |
| `OUTPUT_PATH` | `--output` | 输出根目录 |
| `MAX_WORKERS` | `--max_workers` | 并发 worker 数 |
| `TEMPERATURE` | `--temperature` | 主模型采样温度 |
| `PRESENCE_PENALTY` | `--presence_penalty` | 主模型 presence penalty |
| `ROLLOUT_COUNT` | `--roll_out_count` | 每道题独立 rollout 次数 |
| `WORLD_SIZE` | `--total_splits` | 分布式切分总份数，默认 1 |
| `RANK` | `--worker_split` | 当前 worker 的 0-based rank，脚本会转成 1-based split |

也可以直接运行 Python 脚本：

```bash
python -u inference/run_multi_react.py \
  --dataset bc-zn10 \
  --data_file /home/liuqian/DR/Tongyi/data/bc-zn10.jsonl \
  --output /home/liuqian/DR/Tongyi/output \
  --max_workers 30 \
  --temperature 0.85 \
  --presence_penalty 1.1 \
  --roll_out_count 3
```

## 模型与 Provider 配置

当前统一通过 `inference/model_client.py` 的 `ModelClient` 按阶段路由模型调用。支持的 provider：

| Provider | API key | Base URL 变量 | 默认 Base URL |
| --- | --- | --- | --- |
| `openrouter` | `OPENROUTER_API_KEY` | `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| `dashscope` | `DASHSCOPE_API_KEY` | `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `openai` | `OPENAI_API_KEY` | `OPENAI_API_BASE` | `https://api.openai.com/v1` |

全局默认 provider：

```bash
PROVIDER=openrouter
```

不同阶段可以单独覆盖 provider 和模型：

| 阶段 | Provider 变量 | Model 变量 | 用途 |
| --- | --- | --- | --- |
| research | `RESEARCH_PROVIDER` | `RESEARCH_MODEL` | ReAct 主推理模型 |
| rephrase | `REPHASE_PROVIDER` | `REPHASE_MODEL` | 查询合并/改写 |
| summary | `SUMMARY_PROVIDER` | `SUMMARY_MODEL` | 页面摘要阶段 |
| embedding | `EMBEDDING_PROVIDER` | 固定 `text-embedding-v4` | 相似度、去重和 Search Controller |

示例：

```bash
PROVIDER=openrouter
RESEARCH_MODEL=alibaba/tongyi-deepresearch-30b-a3b

REPHASE_PROVIDER=dashscope
REPHASE_MODEL=qwen-plus

SUMMARY_PROVIDER=openai
SUMMARY_MODEL=gpt-4o-mini

EMBEDDING_PROVIDER=dashscope
```

注意：当前 `run_react_infer.sh` 会强制检查 `OPENROUTER_API_KEY`，即使主模型 provider 改成了 `dashscope` 或 `openai`，也需要在 `.env` 中给出该变量，或自行调整脚本检查逻辑。

## 工具配置

模型可见工具由 `AVAILABLE_TOOLS` 控制，逗号分隔：

```bash
AVAILABLE_TOOLS=aliyun_search,visit,google_scholar,PythonInterpreter
```

可选工具：

| 工具 | 说明 | 依赖变量 |
| --- | --- | --- |
| `search` | Serper 搜索 | `SERPER_KEY_ID` |
| `aliyun_search` | 阿里云 IQS 搜索 | `ALIYUN_IQS_API_KEY` |
| `visit` | 网页读取/摘要 | `JINA_API_KEYS`, `API_KEY`, `API_BASE`, `SUMMARY_MODEL_NAME` |
| `google_scholar` | Google Scholar | `SERPER_KEY_ID` |
| `PythonInterpreter` | 代码执行 | `SANDBOX_FUSION_ENDPOINT` |

关于搜索工具的特殊逻辑：

- 如果启用了 `aliyun_search`，模型调用 `search` 时会被 remap 到 `aliyun_search`。
- 当前逻辑不适合同时让模型在 Serper `search` 和阿里云 `aliyun_search` 之间自主选择。
- 如果只想使用 Serper，配置为：

```bash
AVAILABLE_TOOLS=search,visit,google_scholar,PythonInterpreter
```

如果要处理上传文件，在问题前写入文件名，并把文件放到 `eval_data/file_corpus/`：

```jsonl
{"question": "(Uploaded 1 file: ['report.pdf'])\n\nWhat are the findings?", "answer": "..."}
```

## Search Controller 新增参数

Search Controller 负责搜索查询合并、缓存复用、预算控制和动态观察结果选择。可通过 `.env` 调整：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `DISABLE_SEARCH_CONTROLLER` | `false` | 设为 `true` 后绕过 Controller，所有 query 直接执行 |
| `SEARCH_CONTROLLER_MODE` | `default` | `default` 正常策略；`reduce_topk` 强制降 topk。禁用请使用 `DISABLE_SEARCH_CONTROLLER=true` |
| `SEARCH_CONTROLLER_HIGH_SIM` | `0.90` | 高相似阈值，用于缓存复用/合并 |
| `SEARCH_CONTROLLER_MEDIUM_SIM` | `0.70` | 中等相似阈值，用于降低观察结果 |
| `SEARCH_CONTROLLER_REDUCE_TOPK` | `7` | reduce_topk 模式保留结果数 |
| `SEARCH_CONTROLLER_REUSE_POINTER_WINDOW` | `3` | 缓存复用向前查找的 turn 窗口 |
| `SEARCH_DYNAMIC_K_OBS` | `true` | 是否启用动态 K_obs |
| `SEARCH_MMR_ALPHA` | `0.50` | MMR 相关性权重 |
| `SEARCH_MMR_BETA` | `0.50` | MMR 记忆/冗余惩罚权重 |
| `SEARCH_MMR_THRESHOLD` | `0.00` | MMR 选择阈值 |
| `SEARCH_MMR_MIN_RESULTS` | `5` | MMR 至少保留结果数 |

Controller 的主要动作：

| 动作 | 含义 |
| --- | --- |
| `EXECUTE` | 正常执行搜索 |
| `MERGE` | turn 内多个相似 query 合并 |
| `REUSE_CACHE` | 复用历史相似 query 的结果 |
| `REDUCE_TOPK` | 降低返回结果数 |
| `SKIP_DUPLICATE` | 跳过重复 query |
| `REWRITE_REQUEST` | 请求改写 query |

推理过程中会在输出目录写入：

- `query_process_log.jsonl`：搜索 query、动作和结果记录
- `mmr_stats.jsonl`：MMR 与动态 K_obs 相关统计

## Query Redundancy 参数

查询冗余削减可独立配置：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `REDUNDANCY_ENABLED` | `False` | 是否启用 query 冗余检测 |
| `REDUNDANCY_STRATEGY` | `rephase` | `rephase` 改写；`skip` 跳过；`cache` 使用缓存 |
| `REDUNDANCY_SCOPE` | `single_turn` | `single_turn` 仅当前工具调用；`global` 整个解题过程 |
| `REDUNDANCY_SIMILARITY_THRESHOLD` | `0.8` | 判定冗余的相似度阈值 |
| `REDUNDANCY_MAX_RETRIES` | `2` | rephase 最大重试次数 |

## 数据与输出

输入推荐使用 JSONL，每行包含 `question` 和 `answer`：

```jsonl
{"question": "What is X?", "answer": "Reference answer for evaluation"}
```

也支持 JSON 数组格式。

输出目录结构：

```text
${OUTPUT_PATH}/${RESEARCH_MODEL}/${DATASET}/
```




其中 `RANK` 是 0-based，脚本会传给 `run_multi_react.py` 的 `--worker_split $(RANK + 1)`。

## 常用实验流程

例如跑一个新的数据集/参数设置：

1. 修改 `.env` 中的 `DATASET`，例如 `bc-zn10-single-turn-redundant-rephase-0.8-50`。
2. 设置 `DATA_FILE`、`OUTPUT_PATH`、`RESEARCH_MODEL`、`AVAILABLE_TOOLS` 和 Search Controller 参数。
3. 确认 `PythonInterpreter` 需要的 sandbox 正常运行。
4. 如搜索/访问服务需要代理，在终端中设置好代理环境变量。
5. 启动推理：

```bash
tmux a -t tongyi
source /mnt/data_4/envs/.venv/bin/activate
bash inference/run_react_infer.sh
```

## 评测与统计

BrowseComp / GAIA 等 benchmark 可用：

```bash
python3 evaluation/evaluate_deepsearch_official.py \
  --input_folder output/alibaba/tongyi-deepresearch-30b-a3b/bc-zn10 \
  --judge_model dashscope/qwen3.5-plus \
  --judge_prompt browsecomp \
  --num_rounds 1
```


汇总推理指标：

```bash
python3 inference/summarize_metrics.py \
  --dataset_dir output/alibaba/tongyi-deepresearch-30b-a3b/bc-zn10 \
  --strict
```

## 重要注意事项

- Python 建议使用 `3.10.0`。
- `.env` 已被 gitignore，不能提交真实 key。
- `MAX_LLM_CALL_PER_RUN` 默认 `100`，单题超时逻辑按 1800s 处理。
- `PROMPT_NEW=True` 时使用 `inference/prompt_new.py`，否则使用 `inference/prompt.py`。
- `data/`、`output/`、`cache/`、`Agent/`、`models/` 等目录通常不进入版本控制。
