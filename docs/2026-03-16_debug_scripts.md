# 单方法调试脚本目录 `run/debug/`

> **改动范围**：新增 `run/debug/` 目录
> **改动日期**：2026-03-16
> **改动目标**：为项目中每个检测方法创建独立的最小可运行脚本，便于单独调试和测试，无需通过 `benchmark.py` 批量执行。

---

## 一、背景与动机

### 1.1 现状

`run/benchmark.py` 是项目的统一入口：通过命令行参数选择方法名后，内部的 `experiment()` 函数根据配置分发到不同的实验流程。这种设计适合批量跑实验，但在**调试单个方法**时存在明显痛点：

| 问题               | 说明                                                                           |
| ------------------ | ------------------------------------------------------------------------------ |
| **启动成本高**     | 每次都经过完整的参数解析、配置加载、多 topic 循环                              |
| **不便于断点调试** | 函数嵌套层次深，需要在分发逻辑中定位到目标方法                                 |
| **参数不透明**     | 方法用到了哪些参数、走了哪条分支，需要反复查看 `benchmark.py` 和 `config.json` |
| **修改风险**       | 调试时临时修改 `benchmark.py` 可能影响其他方法                                 |

### 1.2 目标

创建 `run/debug/` 目录，为每个检测方法提供一个**独立、自包含、可直接运行**的 Python 脚本，满足：

1. 从项目根目录 `python run/debug/test_xxx.py` 即可执行
2. 配置直接从 `run/config.json` 读取
3. 默认只跑一个 topic（`STEM`）+ 一个 LLM（`Moonshot`），快速验证
4. 代码量少、结构清晰，方便在 IDE 中断点调试

---

## 二、新增文件结构

```
run/debug/
├── __init__.py              # 包标识文件
├── _common.py               # 公共辅助模块（环境初始化、数据加载、结果打印）
│
├── test_ll.py               # Log-Likelihood          (threshold)
├── test_rank.py             # Rank                    (threshold)
├── test_rank_gltr.py        # Rank-GLTR               (threshold)
├── test_entropy.py          # Entropy                 (threshold)
├── test_lrr.py              # LRR                     (threshold)
├── test_binoculars.py       # Binoculars              (threshold, 无 base_models)
│
├── test_detectgpt.py        # DetectGPT               (perturb)
├── test_fast_detectgpt.py   # Fast-DetectGPT          (perturb)
├── test_npr.py              # NPR                     (perturb)
├── test_dna_gpt.py          # DNA-GPT                 (perturb)
│
├── test_lm_d.py             # LM-D                    (supervised + checkpoint)
├── test_radar.py            # RADAR                   (supervised, 仅推理)
├── test_openai_d.py         # OpenAI-D                (supervised, 需微调)
├── test_conda.py            # ConDA                   (supervised, 需微调)
├── test_chatgpt_d.py        # ChatGPT-D               (supervised, 仅推理)
│
├── test_demasq.py           # Demasq                  (demasq)
├── test_gptzero.py          # GPTZero                 (gptzero / API)
├── test_incremental.py      # Incremental             (incremental)
│
├── test_baseline.py         # Baseline                (fewshot)
├── test_generate.py         # Generate                (fewshot)
└── test_rn.py               # RN (Relation Network)   (fewshot)
```

> 共 23 个文件：1 个 `__init__.py` + 1 个 `_common.py` + **21 个方法脚本**。

---

## 三、公共模块 `_common.py` 说明

从 `benchmark.py` 中提取共用逻辑，避免每个脚本重复编写。

| 函数                                       | 功能       | 说明                                                                                       |
| ------------------------------------------ | ---------- | ------------------------------------------------------------------------------------------ |
| `init_env(gpu_id, config_path)`            | 环境初始化 | 设置 `CUDA_VISIBLE_DEVICES`、加载 `config.json`、设置随机种子                              |
| `load_demo_data(config, topic, detectLLM)` | 数据加载   | 从 `config["global"]` 读取参数，调用 `dataloader.load()` 并截断到 `train_size`/`test_size` |
| `print_results(results)`                   | 结果打印   | 格式化输出每个 `DetectOutput` 的 train/test 指标                                           |
| `resolve_model_path(path)`                 | 路径解析   | 支持 `$ENV_VAR` 形式的环境变量引用                                                         |

**路径自动处理**：`_common.py` 在导入时自动将项目根目录加入 `sys.path`，确保无论从哪个目录执行，`easymgtd` 包都能被正确导入。

---

## 四、各方法脚本按实验类型分组

### 4.1 Threshold 类（5 个脚本）

**对应方法**：`ll`、`rank`、`rank_GLTR`、`entropy`、`LRR`

**共同流程**：

1. 读取 `config.json` 中的 `base_models` 列表
2. 对每个 base model：创建 detector → threshold 实验 → 加载截断数据 → launch → 打印结果

**脚本结构统一**，仅 `METHOD_NAME` 不同。

### 4.2 Binoculars（1 个脚本）

**特点**：同属 threshold 实验，但不需要迭代 `base_models`。detector 直接从 `detector_args` 中获取 `observer_model` 和 `performer_model`。

### 4.3 Perturb 类（4 个脚本）

**对应方法**：`detectGPT`、`fast-detectGPT`、`NPR`、`DNA-GPT`

**共同流程**：

1. 从 `detector_args` 创建 detector
2. 创建 perturb 实验
3. 加载截断数据 → launch（带 `experiment_args`，如 `span_length`、`n_perturbations` 等） → 打印结果

### 4.4 Supervised 类（5 个脚本）

| 脚本                | 方法      | 是否微调 | 特殊逻辑                                                  |
| ------------------- | --------- | -------- | --------------------------------------------------------- |
| `test_lm_d.py`      | LM-D      | 是       | checkpoint 管理（无 checkpoint 训练，有 checkpoint 评估） |
| `test_radar.py`     | RADAR     | 否       | 预训练模型直接推理                                        |
| `test_openai_d.py`  | OpenAI-D  | 是       | 标准微调流程                                              |
| `test_conda.py`     | ConDA     | 是       | 标准微调流程                                              |
| `test_chatgpt_d.py` | ChatGPT-D | 否       | 预训练模型直接推理                                        |

**LM-D 特殊处理**：保留了 `benchmark.py` 中完整的 checkpoint 查找和管理逻辑，包括 `get_path()` 函数。

### 4.5 其他类型

| 脚本                  | 实验类型    | 说明                              |
| --------------------- | ----------- | --------------------------------- |
| `test_demasq.py`      | demasq      | Demasq 专用实验流程               |
| `test_gptzero.py`     | gptzero     | 需要 API key，脚本会检查并提醒    |
| `test_incremental.py` | incremental | 使用完整数据（不截断）            |
| `test_baseline.py`    | fewshot     | k-shot + SVM 分类器，使用完整数据 |
| `test_generate.py`    | fewshot     | k-shot + 数据生成                 |
| `test_rn.py`          | fewshot     | Relation Network                  |

---

## 五、使用方法

### 5.1 基本用法

```bash
# 进入项目根目录
cd /data/liyang/MGTD-Baselines/EasyMGTD

# 激活 conda 环境
conda activate mgtd

# 运行某个方法的调试脚本
python run/debug/test_ll.py
```

### 5.2 修改调试参数

每个脚本顶部有可直接修改的配置常量：

```python
METHOD_NAME = "ll"       # 方法名
GPU_ID = "0"             # GPU 编号
DEBUG_TOPIC = "STEM"     # 测试 topic（可改为 "Humanities"、"Art" 等）
DEBUG_LLM = "Moonshot"   # 目标 LLM（可改为 "gpt35"、"Llama3" 等）
```

### 5.3 修改 train/test 数据量

在 `run/config.json` 的 `global` 段中修改：

```json
{
  "global": {
    "train_size": 100,
    "test_size": 200,
    ...
  }
}
```

---

## 六、与 `benchmark.py` 的关系

| 维度     | `benchmark.py`       | `run/debug/test_*.py` |
| -------- | -------------------- | --------------------- |
| 用途     | 批量跑实验、生成 CSV | 单方法调试和测试      |
| 数据范围 | 遍历多个 topics      | 默认单个 topic        |
| 结果输出 | CSV 文件             | stdout 打印           |
| 配置来源 | 同一个 `config.json` | 同一个 `config.json`  |
| 是否影响 | 不影响               | 不影响                |

两者**完全独立**，互不影响。

---

## 七、环境维护与修复记录（2026-03-16）

在进行调试脚本编写与测试过程中，发现并解决了以下环境与代码兼容性问题：

### 7.1 Transformers 升降级导致的 AdamW 错误

**问题描述**：运行 `test_ll.py` 等脚本时，抛出 `ImportError: cannot import name 'AdamW' from 'transformers'`。
**原因分析**：新版 `transformers` (4.45+) 已移除 `AdamW`，建议直接使用 `torch.optim.AdamW`。
**修复方案**：

- 修改文件：
  - `easymgtd/methods/supervised.py`
  - `easymgtd/methods/supervised_incremental.py`
  - `easymgtd/methods/supervised_fewshot.py`
- 改动内容：将 `from transformers import AdamW` 替换为 `from torch.optim import AdamW`，并清理了重复的导入语句。

### 7.2 补齐环境缺失依赖项

为确保所有调试脚本正常运行，在 `mgtd` 环境中补齐了以下缺失依赖：

| 包名            | 版本   | 用途                                      |
| :-------------- | :----- | :---------------------------------------- |
| `captum`        | 0.8.0  | `DemasqDetector` 模型解释与特征归因       |
| `marshmallow`   | 4.2.2  | `supervised_fewshot` 相关数据处理         |
| `sentencepiece` | 0.2.1  | 某些 Tokenizer 的依赖（如 Llama/DeBERTa） |
| `xgboost`       | 3.2.0  | 传统机器学习基准方法                      |
| `protobuf`      | 5.29.3 | 模型序列化格式支持                        |

**执行建议**：若在其他环境部署，建议运行：

```bash
pip install captum marshmallow sentencepiece xgboost protobuf
```

### 7.3 数据集加载逻辑重构（适配新版 datasets）

**问题描述**：在修复环境依赖后，新版 `datasets` 库 (4.2.0) 在运行 `test_ll.py` 时抛出 `RuntimeError: Dataset scripts are no longer supported`，完全禁用了原 `AI_Polish_clean.py` 的数据加载功能。
**原因分析**：这是 Hugging Face 出于安全考量在新版本中强制实施的策略，以往通过指定本地 `repo` 路径并开启 `trust_remote_code=True` 的脚本加载方式已全部失效。
**修复方案**：

- 修改文件：`easymgtd/loading/dataloader.py`
- 改动内容：
  - 新增 `_load_aitextdetect(repo, name, split)` 函数。
  - **直接绕过 `datasets` 底层的加载器**，按照特定目录结构（如 `Human/` 和 `{LLM}_new/`）扫描并读取本地的 `.json` 数据文件。
  - 将配套的 `tiktoken` 文本截断逻辑从外部脚本一并内嵌至该函数中。
  - 修改 `load_subject_data`, `load_topic_data`, `download_data`, `prepare_attribution` 等入口函数，使其优先调用新函数加载本地 AITextDetect 数据。
- **结果**：调试脚本 `test_ll.py` 成功越过数据加载阶段，顺利完成数据组装并输出了各检测方法的最终评测指标（Acc, F1, AUC）。这使得项目彻底摆脱了对旧加载脚本的依赖，保障了后续调试的顺畅运行。
