# 2026-03-10 Benchmark 脚本重构与配置分离记录

## 1. 改动背景与目的

在先前的版本中，`run/benchmark.py` 内部存在大量模型权重的绝对路径硬编码（如 `/data1/zzy/roberta-base`, `/data_sda/zhiyuan/models/falcon-7b` 等）以及固定的超参数（如 `batch_size=64`, `epoch=3` 等）。这导致跨服务器部署或复现时，使用者必须侵入源码修改路径，严重影响了 Benchmark 的灵活性。

本次改动的目的是：

- **解耦环境依赖**：将模型路径提取至独立的环境变量配置文件 `.env`。
- **动态读入超参数**：提取硬编码在内部的超参数进入独立配置文件 `run/config.json`，方便通过传参动态指定，而无需修改 Python 代码。
- **输出代码阅读文档**：梳理与输出各 Baseline 方法详细实现原理（详见 `baseline_analysis.md`）。

## 2. 具体改动清单

### 2.1 新增依赖

- 在 `requirements.txt` 中添加了 `python-dotenv`，用于支持无缝加载 `.env` 变量至 `os.environ`。

### 2.2 环境模板更新 (`.env.example`)

补充和统一了所有主流及补充模型的路径名称预设：

- `ROBERTA_BASE`, `DISTILBERT_BASE`
- `FALCON_7B`, `FALCON_7B_INSTRUCT`
- `LLAMA2_CHAT`
- `RADAR_MODEL`
- `CHATGPT_ROBERTA`
- 设定了统一的输出基目录 `DATASET_DIR_SAVE`。

### 2.3 新增配置 JSON (`run/config.json`)

引入字典化的运行期超参数管理工具，支持以下功能分支：

- `global` 节点：管理随机种子 `seed`，及快速评估时截断的 `train_size`、`test_size`。
- 其他针对监督类模型如 `LM-D` 和双模型打分 `Binoculars` 的专属参数（`epoch`、`batch_size`、`threshold_mode` 等）。

### 2.4 重构主入口 (`run/benchmark.py`)

- 文件头引入 `dotenv` 的 `load_dotenv()`。
- 将硬编码声明全部替换为安全取值的 `os.environ.get(KEY, DEFAULT)` 形式。
- 添加 `argparse` 参数 `--config_path`，动态拉取 `config.json`。
- `experiment` 函数的内部循环条件改写为根据 `config` 内设定的对应算法树动态调参。

### 2.5 SwanLab 集成与训练参数解耦 (`mgtbench/methods/supervised.py`)

- 提取了原本硬接线在代码内部的 `TrainingArguments`（例如 `logging_steps`, `weight_decay`, `save_total_limit`）并转移到了外部的 `config.json` 里的 `LM-D` 实体下，由 `SupervisedConfig` 读取。
- 新增库依赖 `swanlab`，并在方法 `finetune()` 的 `Trainer` 初始化阶段手动注入 `SwanLabCallback`，用于在任何监督型检测器任务中实时打点跟踪并生成实验控制面板。

## 3. 后续指南

新的运行方式与以前兼容，但允许通过配置外挂实现完全不同的子实验。
若需改动模型目录，请直接修改根目录下的 `.env` 文件。
若需控制跑测数据量或监督训练轮数，请修改 `run/config.json`。
