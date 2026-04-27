# Exp Settings 

相较于原版，在 `.env` 文件中重要的参数：

## 模型调用/工具相关
- OPENROUTER_MODEL 设置在 OpenRouter上调用的主要模型
- DASHSCOPE_MODEL 作 Deepseek 对比实验的时候添加的Dashscope的模型参数
- PROVIDER `openrouter` 或者`dashscope` 主要区别API调用商，因为 Openrouter 上面也有ds，但是框架大修之后可能设置为dashscope会有问题
- REPHASE_MODEL 用于合并query的名称，需要填和 openrouer 上面的官方名称

- SANDBOX_FUSION_ENDPOINT 在使用 python interpreter 之前需要把 sandbox fusion 打开，默认是运行的，可以通过 `tmux a -t sandbox` 检查运行状态。可以通过下面的方法启动：

```zsh
docker run -it -p 8080:8080 vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```
- AVAILABLE_TOOLS 设置模型能够看见的工具，例如：

```
# Use aliyun search
AVAILABLE_TOOLS=aliyun_search,visit,PythonInterpreter

# Use google search and scholar
AVAILABLE_TOOLS=aliyun_search,visit,PythonInterpreter,search,scholar
```
由于这里的 Tongyi模型被微调过，它不能够区分： `search` 和 `aliyun_search` （它在搜索的时候总是调用 search），所以手动做了 remapping ， 如果存在 aliyun_search 则模型调用任何的search实际上对应aliyun_search。
所以目前的代码逻辑不能够处理：同时使用谷歌search和aliyun search让模型自己选择。

在搜索的时候默认调用的是 Serper API 这个没有改变仓库原来的逻辑


## 实验数据/参数/保存相关

- DATASET 对应实际上保存的数据的最终位置
- DATA_FILE=/home/liuqian/DR/Tongyi/data/bc-zn50.jsonl 对应输入的 jsonl 文件的名字
- OUTPUT_PATH=/home/liuqian/DR/Tongyi/output 输出的目标位置

最后 iter1.jsonl 会保存在： {$OUTPUT_PATH} / ${$MODEL_PRODUCER} / {$MODEL_NAME} / {$DATASET} 下面

这里是会读取  {$OUTPUT_PATH} / ${$MODEL_PRODUCER} / {$MODEL_NAME} / {$DATASET} 下面已有的结果，最终会实际上跑的是 **input data file 中有的问题，但目前目标文件中没有记录完成的问题**，对于已经在目标文件中的问题不会跑（可以断点续跑）

## 使用样例

例如跑一个新的数据集/参数设置的实验

1. 首先修改 DATASET 为实验的名称，例如 bc-zn10-single-turn-redundant-rephase-0.8-50
2. 修改参数 、 输入的数据路径
3. **确认python interpreter**是正常运行的
4. 需要在终端启用梯子
5. 运行实验

```bash
tmux a -t tongyi
source /mnt/data_4/envs/.venv/bin/activate 
inference/run_react_infer.sh   
```

6. 统计最终的正确率、数据

```bash

# Evaluation
python3 evaluation/evaluate_deepsearch_official.py \
--input_folder output/alibaba/tongyi-deepresearch-30b-a3b/bc-zn10-single-turn-redundant-rephase-0.8-50 \
--judge_model dashscope/qwen3.5-plus \
--judge_prompt browsecomp \
--num_rounds 1

# Summerize
python3 inference/summarize_metrics.py --dataset_dir output/alibaba/tongyi-deepresearch-30b-a3b/bc-zn10-single-turn-redundant-rephase-0.8 --strict
```