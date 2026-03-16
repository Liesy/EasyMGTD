# 2026-03-13 检测方法配置参数解耦改造文档

> **改动范围**：`run/config.json`、`run/benchmark.py`
> **改动目标**：将所有检测方法的参数配置从代码硬编码中解耦出来，达到「一个方法一个配置节」的效果
> **不涉及改动**：`mgtbench/` 内部的方法实现和实验框架

---

## 目录

1. [改动背景与动机](#1-改动背景与动机)
2. [改造前的问题分析](#2-改造前的问题分析)
3. [新 config.json 结构详解](#3-新-configjson-结构详解)
   - 3.1 [总体结构](#31-总体结构)
   - 3.2 [global 全局配置](#32-global-全局配置)
   - 3.3 [各方法配置节字段说明](#33-各方法配置节字段说明)
   - 3.4 [方法分类及配置详解](#34-方法分类及配置详解)
4. [benchmark.py 改造详解](#4-benchmarkpy-改造详解)
   - 4.1 [改造前的代码结构](#41-改造前的代码结构)
   - 4.2 [改造后的代码结构](#42-改造后的代码结构)
   - 4.3 [关键函数说明](#43-关键函数说明)
   - 4.4 [分发流程图](#44-分发流程图)
5. [参数来源追溯表（完整版）](#5-参数来源追溯表完整版)
6. [使用指南](#6-使用指南)
   - 6.1 [基本用法](#61-基本用法)
   - 6.2 [如何修改某方法的参数](#62-如何修改某方法的参数)
   - 6.3 [环境变量覆盖机制](#63-环境变量覆盖机制)
   - 6.4 [可用方法列表](#64-可用方法列表)
7. [向后兼容性说明](#7-向后兼容性说明)

---

## 1. 改动背景与动机

在 `2026-03-10` 的首次重构中，我们将硬编码的模型路径提取至 `.env` 环境变量，并引入了 `config.json` 管理部分超参数。但那次改动只覆盖了 `global`、`LM-D`、`Binoculars` 三个配置节。

**遗留问题**：项目注册了 **21 个检测方法**（见 `mgtbench/auto.py` 中的 `DETECTOR_MAPPING`），但绝大多数方法的参数仍然硬编码在 `benchmark.py` 的 `experiment()` 函数中。使用者面临以下困境：

1. **不知道某方法需要哪些参数**——必须先阅读方法源码才能理解
2. **无法灵活调整参数**——改参数就得改 Python 代码
3. **方法之间参数混杂**——模型路径散落在文件顶部，与具体方法的关联不明确
4. **部分方法没有运行入口**——`demasq`、`incremental`、`baseline`、`generate`、`rn`、`GPTZero` 等在 `DETECTOR_MAPPING` 中注册了，但 `benchmark.py` 没有对应的执行分支

本次改造彻底解决以上问题。

---

## 2. 改造前的问题分析

### 2.1 参数散落在三个层面

| 层面             | 位置                                     | 举例                                                |
| ---------------- | ---------------------------------------- | --------------------------------------------------- |
| **检测器初始化** | `AutoDetector.from_detector_name()` 传参 | `model_name_or_path`、`observer_model_name_or_path` |
| **实验层配置**   | 各 `*Config` dataclass 的字段            | `epochs`、`batch_size`、`lr`、`criterion_score`     |
| **脚本硬编码**   | `benchmark.py` 中的 `if-elif` 分支       | 模型路径变量、topic 循环方式                        |

### 2.2 原 experiment() 函数的硬编码分支

```python
# 改造前的 experiment() 函数结构（伪代码）
def experiment(csv_file, method, detectLLM, config):
    if method in ["ll", "rank", "LRR", "rank_GLTR", "entropy"]:
        base_models = [llama2_chat]                     # ← 硬编码变量
        detector = AutoDetector.from_detector_name(
            method, model_name_or_path=base_model       # ← 硬编码传参
        )
        ...
    elif method in ["LM-D"]:
        epoch = config.get("LM-D", {}).get("epoch", 3)  # ← 仅此方法读配置
        ...
    elif method in ["fast-detectGPT"]:
        scoring_model = gpt_neo                          # ← 硬编码变量
        reference_model = gpt_j                          # ← 硬编码变量
        ...
    elif method in ["Binoculars"]:
        ...                                              # ← 部分读配置，部分硬编码
    # 没有 demasq、incremental 等方法的分支！
```

---

## 3. 新 config.json 结构详解

### 3.1 总体结构

```
config.json
├── global                 # 全局共享配置
├── ll                     # 方法：Log-Likelihood
├── rank                   # 方法：Rank
├── rank_GLTR              # 方法：GLTR Rank
├── entropy                # 方法：Entropy
├── LRR                    # 方法：Log-Rank Ratio
├── Binoculars             # 方法：Binoculars
├── fast-detectGPT         # 方法：Fast-DetectGPT
├── detectGPT              # 方法：DetectGPT
├── NPR                    # 方法：NPR
├── DNA-GPT                # 方法：DNA-GPT
├── LM-D                   # 方法：LM-D (Supervised)
├── RADAR                  # 方法：RADAR
├── OpenAI-D               # 方法：OpenAI-D (Supervised)
├── ConDA                  # 方法：ConDA (Supervised)
├── ChatGPT-D              # 方法：ChatGPT-D (Supervised)
├── demasq                 # 方法：DEMASQ
├── GPTZero                # 方法：GPTZero API
├── incremental            # 方法：Incremental
├── baseline               # 方法：Baseline (Few-Shot)
├── generate               # 方法：Generate (Few-Shot)
└── rn                     # 方法：Relation Network (Few-Shot)
```

### 3.2 global 全局配置

```json
{
  "global": {
    "train_size": 1000,
    "test_size": 2000,
    "seed": 3407,
    "topics": ["STEM", "Humanities", "Social_sciences"],
    "dataset": "AITextDetect"
  }
}
```

| 字段         | 类型      | 说明                                              | 默认值                                      |
| ------------ | --------- | ------------------------------------------------- | ------------------------------------------- |
| `train_size` | int       | 训练集截取大小（项目使用 `get_demo_data()` 截取） | 1000                                        |
| `test_size`  | int       | 测试集截取大小                                    | 2000                                        |
| `seed`       | int       | 全局随机种子                                      | 3407                                        |
| `topics`     | list[str] | 要遍历的数据主题列表                              | `["STEM", "Humanities", "Social_sciences"]` |
| `dataset`    | str       | 数据集名称，传入 `load()` 函数                    | `"AITextDetect"`                            |

> **新增字段**：`topics` 和 `dataset` 是本次新增的，使得数据集选择也可配置化，不再硬编码。

### 3.3 各方法配置节字段说明

每个方法的配置节统一包含以下字段（按需出现）：

| 字段               | 类型      | 说明                                                                                                                                          |
| ------------------ | --------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment_type`  | str       | 实验类型。决定使用 `AutoExperiment` 的哪个实验类。可选值：`threshold`、`perturb`、`supervised`、`demasq`、`incremental`、`fewshot`、`gptzero` |
| `detector_args`    | dict      | 传给 `AutoDetector.from_detector_name()` 的关键字参数                                                                                         |
| `experiment_args`  | dict      | 传给 `experiment.launch()` 的关键字参数（最终会更新对应的 `*Config` dataclass）                                                               |
| `base_models`      | list[str] | （仅 metric-based 方法）评分基座模型列表，每个模型会分别运行                                                                                  |
| `base_model_names` | list[str] | （仅 LM-D）要训练/评估的模型名列表                                                                                                            |
| `model_paths`      | dict      | （仅 LM-D）模型名到 HuggingFace 路径的映射                                                                                                    |
| `train_size_max`   | int       | （仅 LM-D）从头训练时的最大训练数据量                                                                                                         |

### 3.4 方法分类及配置详解

---

#### 3.4.1 Metric-Based 方法（阈值实验）

**涵盖方法**：`ll`、`rank`、`rank_GLTR`、`entropy`、`LRR`

这些方法使用一个因果语言模型作为「基座模型」来计算文本的统计指标（如 log-likelihood、token rank 等），然后通过阈值或逻辑回归分类。

```json
{
  "ll": {
    "experiment_type": "threshold",
    "base_models": ["meta-llama/Llama-2-7b-chat-hf"],
    "detector_args": {
      "load_in_k_bit": null
    }
  }
}
```

**`detector_args` 字段说明**：

| 参数            | 类型        | 说明                                  | 默认值 |
| --------------- | ----------- | ------------------------------------- | ------ |
| `load_in_k_bit` | int \| null | 量化位数（4 或 8），`null` 表示不量化 | null   |

> **注意**：`model_name_or_path` 不在 `detector_args` 中，因为它从 `base_models` 列表中逐个迭代传入。这样做是因为可能需要对比多个基座模型的效果。

**`base_models` 说明**：

列表中的每个路径都会依次用于创建检测器并运行全部 topic。如需测试多模型对比，只需添加：

```json
"base_models": [
  "meta-llama/Llama-2-7b-chat-hf",
  "openai-community/gpt2-medium"
]
```

---

#### 3.4.2 Binoculars

Binoculars 方法需要两个模型（observer + performer），且有自己的阈值策略。

```json
{
  "Binoculars": {
    "experiment_type": "threshold",
    "detector_args": {
      "observer_model_name_or_path": "tiiuae/falcon-7b",
      "performer_model_name_or_path": "tiiuae/falcon-7b-instruct",
      "max_length": 1024,
      "mode": "accuracy",
      "threshold": "new"
    }
  }
}
```

**`detector_args` 字段说明**：

| 参数                           | 类型 | 说明                                                             | 默认值                        |
| ------------------------------ | ---- | ---------------------------------------------------------------- | ----------------------------- |
| `observer_model_name_or_path`  | str  | Observer 模型路径                                                | `"tiiuae/falcon-7b"`          |
| `performer_model_name_or_path` | str  | Performer 模型路径                                               | `"tiiuae/falcon-7b-instruct"` |
| `max_length`                   | int  | Tokenizer 最大长度                                               | 512                           |
| `mode`                         | str  | 阈值优化目标：`"accuracy"` 或 `"low-fpr"`                        | `"accuracy"`                  |
| `threshold`                    | str  | 阈值策略：`"default"` 使用论文预设值，`"new"` 在训练集上重新搜索 | `"default"`                   |

---

#### 3.4.3 Perturbation-Based 方法

**涵盖方法**：`fast-detectGPT`、`detectGPT`、`NPR`、`DNA-GPT`

这些方法通过对文本进行扰动（masking + regeneration）来判别机器生成文本。

##### fast-detectGPT

```json
{
  "fast-detectGPT": {
    "experiment_type": "perturb",
    "detector_args": {
      "scoring_model_name_or_path": "EleutherAI/gpt-neo-2.7B",
      "reference_model_name_or_path": "EleutherAI/gpt-j-6b",
      "discrepancy_analytic": false
    },
    "experiment_args": {
      "span_length": 2,
      "buffer_size": 1,
      "mask_top_p": 1,
      "pct_words_masked": 0.3,
      "random_fills": false,
      "n_perturbation_rounds": 1,
      "n_perturbations": 10,
      "criterion_score": "z"
    }
  }
}
```

**`detector_args` 字段说明**：

| 参数                           | 类型 | 说明                                        |
| ------------------------------ | ---- | ------------------------------------------- |
| `scoring_model_name_or_path`   | str  | 评分模型路径（必填）                        |
| `reference_model_name_or_path` | str  | 参考模型路径（选填，不填则与 scoring 相同） |
| `discrepancy_analytic`         | bool | 是否使用解析版 discrepancy（更快但近似）    |

**`experiment_args` 字段说明**（对应 `PerturbConfig` dataclass）：

| 参数                    | 类型  | 说明                                            | 默认值 |
| ----------------------- | ----- | ----------------------------------------------- | ------ |
| `span_length`           | int   | 每次 mask 的 span 长度                          | 2      |
| `buffer_size`           | int   | span 间的最小缓冲区                             | 1      |
| `mask_top_p`            | float | T5 生成时的 top_p 采样参数                      | 1.0    |
| `pct_words_masked`      | float | 被 mask 的词比例                                | 0.3    |
| `random_fills`          | bool  | 是否使用随机填充代替模型生成                    | false  |
| `n_perturbation_rounds` | int   | 扰动轮数                                        | 1      |
| `n_perturbations`       | int   | 每个文本的扰动次数                              | 10     |
| `criterion_score`       | str   | 评分标准：`"d"` (difference) 或 `"z"` (z-score) | `"z"`  |

##### detectGPT / NPR

```json
{
  "detectGPT": {
    "experiment_type": "perturb",
    "detector_args": {
      "model_name_or_path": "openai-community/gpt2-medium",
      "mask_model_name_or_path": "t5-large"
    },
    "experiment_args": { ... }
  }
}
```

| 参数                      | 说明                                 |
| ------------------------- | ------------------------------------ |
| `model_name_or_path`      | 评分/源模型路径（必填）              |
| `mask_model_name_or_path` | Mask 填充模型路径（必填，通常是 T5） |

##### DNA-GPT

```json
{
  "DNA-GPT": {
    "experiment_type": "perturb",
    "detector_args": {
      "base_model_name_or_path": "gpt2",
      "batch_size": 5,
      "regen_number": 5,
      "temperature": 1.0,
      "truncate_ratio": 0.5,
      "mode": "accuracy"
    },
    "experiment_args": {}
  }
}
```

| 参数                      | 类型  | 说明                 | 默认值       |
| ------------------------- | ----- | -------------------- | ------------ |
| `base_model_name_or_path` | str   | 基座模型路径         | `"gpt2"`     |
| `batch_size`              | int   | 重生成批大小         | 5            |
| `regen_number`            | int   | 每个样本的重生成次数 | 5            |
| `temperature`             | float | 生成温度             | 1.0          |
| `truncate_ratio`          | float | 前缀截断比例         | 0.5          |
| `mode`                    | str   | 阈值优化目标         | `"accuracy"` |

---

#### 3.4.4 Supervised 方法

**涵盖方法**：`LM-D`、`RADAR`、`OpenAI-D`、`ConDA`、`ChatGPT-D`

##### LM-D（需要训练的监督检测器）

```json
{
  "LM-D": {
    "experiment_type": "supervised",
    "base_model_names": ["distilbert"],
    "model_paths": {
      "distilbert": "distilbert/distilbert-base-uncased",
      "roberta-base": "FacebookAI/roberta-base"
    },
    "detector_args": {},
    "experiment_args": {
      "need_finetune": true,
      "need_save": true,
      "epochs": 3,
      "batch_size": 64,
      "save_path": "./exp_data/topic_models",
      "lr": 5e-6,
      "logging_steps": 30,
      "weight_decay": 0.01,
      "save_total_limit": 2,
      "gradient_accumulation_steps": 1,
      "swanlab_project": "EasyMGTD-SFT",
      "disable_tqdm": true,
      "pos_bit": 1
    },
    "train_size_max": 10000
  }
}
```

**LM-D 特有字段**：

| 字段               | 说明                                                                  |
| ------------------ | --------------------------------------------------------------------- |
| `base_model_names` | 要训练的模型名列表（如 `["distilbert", "roberta-base"]`），会逐个迭代 |
| `model_paths`      | 模型名到 HuggingFace 路径的映射字典                                   |
| `train_size_max`   | 从头训练时使用的最大训练集样本数（不同于 `global.train_size`）        |

**`experiment_args` 字段说明**（对应 `SupervisedConfig` dataclass）：

| 参数                          | 类型  | 说明                               | 默认值         |
| ----------------------------- | ----- | ---------------------------------- | -------------- |
| `need_finetune`               | bool  | 是否需要微调（`false` 时直接推理） | false          |
| `need_save`                   | bool  | 是否保存 checkpoint                | true           |
| `epochs`                      | int   | 训练轮数                           | 3              |
| `batch_size`                  | int   | 训练批大小                         | 16             |
| `save_path`                   | str   | Checkpoint 保存路径                | `"finetuned/"` |
| `lr`                          | float | 学习率                             | 5e-6           |
| `logging_steps`               | int   | 日志打印步数间隔                   | 30             |
| `weight_decay`                | float | 权重衰减系数                       | 0.01           |
| `save_total_limit`            | int   | 最多保留几个 checkpoint            | 2              |
| `gradient_accumulation_steps` | int   | 梯度累积步数                       | 1              |
| `swanlab_project`             | str   | SwanLab 项目名（训练日志追踪）     | `"EasyMGTD"`   |
| `pos_bit`                     | int   | 正类在 softmax 输出中的位置        | 1              |
| `disable_tqdm`                | bool  | 是否禁用进度条                     | false          |

> **LM-D 的 checkpoint 管理逻辑**：`benchmark.py` 会自动检查 `save_path` 下是否已有对应的 checkpoint 目录（格式 `{model_name}_{detectLLM}_{topic}_{epoch}_{batch_size}`），若存在则直接加载评估，否则从头训练。

##### RADAR / ChatGPT-D（预训练模型直接推理）

```json
{
  "RADAR": {
    "experiment_type": "supervised",
    "detector_args": {
      "model_name_or_path": "TrustSafeAI/RADAR-Vicuna-7B",
      "tokenizer_path": "TrustSafeAI/RADAR-Vicuna-7B"
    },
    "experiment_args": {
      "need_finetune": false
    }
  }
}
```

这类方法本身已经是训练好的检测模型，通常 `need_finetune` 设为 `false`，直接推理。

---

#### 3.4.5 Demasq

```json
{
  "demasq": {
    "experiment_type": "demasq",
    "detector_args": {},
    "experiment_args": {
      "need_finetune": true,
      "need_save": true,
      "batch_size": 1,
      "save_path": "model_weight/",
      "epoch": 12
    }
  }
}
```

**`experiment_args` 字段**（对应 `DemasqConfig`）：

| 参数            | 类型 | 说明               | 默认值            |
| --------------- | ---- | ------------------ | ----------------- |
| `need_finetune` | bool | 是否需要训练       | true              |
| `need_save`     | bool | 训练后是否保存权重 | true              |
| `batch_size`    | int  | 训练批大小         | 1                 |
| `save_path`     | str  | 权重保存路径       | `"model_weight/"` |
| `epoch`         | int  | 训练轮数           | 12                |

**可选 `detector_args`**：

| 参数              | 说明                               |
| ----------------- | ---------------------------------- |
| `state_dict_path` | 已有权重路径（如需加载预训练权重） |

---

#### 3.4.6 GPTZero（API 调用）

```json
{
  "GPTZero": {
    "experiment_type": "gptzero",
    "detector_args": {
      "api_key": ""
    }
  }
}
```

| 参数      | 说明                                                 |
| --------- | ---------------------------------------------------- |
| `api_key` | GPTZero API 密钥（必填，从 https://gptzero.me 获取） |

---

#### 3.4.7 Incremental Learning

```json
{
  "incremental": {
    "experiment_type": "incremental",
    "detector_args": {
      "model_name_or_path": "distilbert/distilbert-base-uncased",
      "num_labels": 2,
      "lwf_reg": 0.5,
      "cache_size": 100,
      "bic": false
    },
    "experiment_args": {
      "need_finetune": true,
      "need_save": true,
      "batch_size": 16,
      "pos_bit": 1,
      "epochs": 1,
      "save_path": "finetuned/",
      "gradient_accumulation_steps": 1,
      "lr": 5e-6,
      "lr_factor": 5
    }
  }
}
```

**`detector_args` 字段**（对应 `IncrementalDetector.__init__`）：

| 参数                 | 类型  | 说明                                   | 默认值 |
| -------------------- | ----- | -------------------------------------- | ------ |
| `model_name_or_path` | str   | 基座模型路径                           | -      |
| `num_labels`         | int   | 初始类别数                             | 2      |
| `lwf_reg`            | float | Learning without Forgetting 正则化系数 | 0.5    |
| `cache_size`         | int   | 每个类别缓存的 exemplar 样本数         | 100    |
| `bic`                | bool  | 是否启用 Bias Correction Layer         | false  |

**`experiment_args` 字段**（对应 `IncrementalConfig`）：

| 参数        | 类型 | 说明                     | 默认值 |
| ----------- | ---- | ------------------------ | ------ |
| `lr_factor` | int  | 增量学习后学习率衰减因子 | 5      |

（其余字段与 `SupervisedConfig` 相同）

---

#### 3.4.8 Few-Shot 方法

**涵盖方法**：`baseline`、`generate`、`rn`

```json
{
  "baseline": {
    "experiment_type": "fewshot",
    "detector_args": {
      "model_name_or_path": "distilbert/distilbert-base-uncased",
      "kshot": 5
    },
    "experiment_args": {
      "need_finetune": false,
      "need_save": false,
      "kshot": 5,
      "batch_size": 16,
      "epochs": 1,
      "lr": 5e-6,
      "gradient_accumulation_steps": 1,
      "save_path": "finetuned/",
      "classifier": "SVM"
    }
  }
}
```

**`detector_args` 字段**：

| 参数                 | 类型 | 说明                      | 默认值 |
| -------------------- | ---- | ------------------------- | ------ |
| `model_name_or_path` | str  | 特征提取基座模型路径      | -      |
| `kshot`              | int  | 每类的支持集样本数        | 5      |
| `num_classes`        | int  | （仅 `rn`）Way 数量       | 6      |
| `num_query`          | int  | （仅 `rn`）每类查询样本数 | 15     |

**`experiment_args` 字段**（对应 `FewShotConfig`）：

| 参数         | 类型 | 说明                                                            | 默认值  |
| ------------ | ---- | --------------------------------------------------------------- | ------- |
| `kshot`      | int  | 每类支持样本数                                                  | 5       |
| `classifier` | str  | 分类器类型（仅 `generate`）：`"SVM"` / `"Regression"` / `"MLP"` | `"SVM"` |

---

## 4. benchmark.py 改造详解

### 4.1 改造前的代码结构

```
benchmark.py (改造前)
├── 模型路径硬编码变量（gpt2, gpt_neo, falcon_7b, ...）
├── log_result() / log_result_lmd()   # CSV 日志记录
├── get_path()                         # checkpoint 查找
├── get_demo_data()                    # 数据截取
├── experiment()                       # 核心函数
│   ├── if method in ["ll", "rank", ...]:    # 硬编码分支 1
│   ├── elif method in ["LM-D"]:             # 硬编码分支 2
│   ├── elif method in ["RADAR", ...]:       # 硬编码分支 3
│   ├── elif method in ["fast-detectGPT"]:   # 硬编码分支 4
│   └── elif method in ["Binoculars"]:       # 硬编码分支 5
└── __main__                           # argparse 入口
```

**问题**：

- 6 个变量（`gpt2`, `gpt_neo`, `gpt_j`, `falcon_7b`, `falcon_7b_instruct`, `llama2_chat`, ...）硬编码模型路径
- 5 个 if-elif 分支，每个分支内部逻辑重复（topic 循环、数据加载、结果记录）
- 只覆盖了部分方法，`demasq` 等方法缺失

### 4.2 改造后的代码结构

```
benchmark.py (改造后)
├── EXPERIMENT_TYPE_MAP               # experiment_type -> 实验名称映射
├── resolve_model_path()              # $ENV_VAR 路径解析
├── log_result() / log_result_lmd()   # CSV 日志记录（保持不变）
├── get_path()                         # checkpoint 查找（保持不变）
├── get_demo_data()                    # 数据截取（保持不变）
├── experiment()                       # 核心函数（重构）
│   ├── 读取 global 配置
│   ├── 读取 method_cfg = config[method]
│   ├── if method in ["ll",...,"entropy"]:  # metric-based + base_models 迭代
│   ├── elif method == "Binoculars":        # Binoculars 特殊初始化
│   ├── elif method == "LM-D":              # LM-D checkpoint 管理
│   ├── elif experiment_type == "supervised": # 其他 supervised 方法通用
│   ├── elif experiment_type == "perturb":   # 全部 perturb 方法通用
│   ├── elif experiment_type == "demasq":    # demasq
│   ├── elif experiment_type == "incremental": # incremental
│   ├── elif experiment_type == "fewshot":   # few-shot 方法通用
│   └── elif experiment_type == "gptzero":   # GPTZero API
└── __main__                           # argparse 入口（choices 自动生成）
```

**关键改进**：

1. **移除所有硬编码模型路径变量**：不再有 `gpt2`, `gpt_neo` 等全局变量
2. **参数全部从 config 读取**：`detector_args` 透传给检测器，`experiment_args` 透传给实验
3. **通用化分支**：同一 `experiment_type` 的方法共用一个分支
4. **choices 自动生成**：`argparse` 的 `--method` 选项从 `DETECTOR_MAPPING.keys()` 自动获取
5. **覆盖全部方法**：新增了 `demasq`、`incremental`、`fewshot`、`gptzero` 的执行分支

### 4.3 关键函数说明

#### `resolve_model_path(path)`

支持 `$ENV_VAR` 语法，允许通过环境变量覆盖配置文件中的模型路径：

```python
# config.json 中写法：
"observer_model_name_or_path": "$FALCON_7B"

# .env 中设置：
FALCON_7B=/your/local/path/falcon-7b

# resolve_model_path("$FALCON_7B") → "/your/local/path/falcon-7b"
```

如果路径不以 `$` 开头，则原样返回。

#### `experiment(csv_file, method, detectLLM, config)`

核心分发函数。流程如下：

1. 从 `config["global"]` 读取全局配置（seed, train_size, test_size, topics, dataset）
2. 从 `config[method]` 读取方法专属配置
3. 提取 `experiment_type`、`detector_args`、`experiment_args` 三个部分
4. 对 `detector_args` 中的所有值调用 `resolve_model_path()` 进行环境变量替换
5. 根据方法名或 `experiment_type` 进入对应的执行分支
6. 在每个分支中，遍历 topics，加载数据，创建检测器和实验，运行并记录结果

### 4.4 分发流程图

```
用户命令行
  │
  ▼
argparse 解析 --method、--detectLLM、--config_path
  │
  ▼
加载 config.json
  │
  ▼
experiment(csv_file, method, detectLLM, config)
  │
  ├─ 读取 config["global"]
  │    → seed, train_size, test_size, topics, dataset
  │
  ├─ 读取 config[method]
  │    → experiment_type, detector_args, experiment_args
  │
  ├─ resolve_model_path() 处理 detector_args
  │
  ▼
根据 method / experiment_type 分发
  │
  ├─ metric-based → AutoDetector + ThresholdExperiment
  ├─ Binoculars   → AutoDetector + ThresholdExperiment
  ├─ LM-D         → AutoDetector + SupervisedExperiment (checkpoint mgmt)
  ├─ supervised    → AutoDetector + SupervisedExperiment
  ├─ perturb       → AutoDetector + PerturbExperiment
  ├─ demasq        → AutoDetector + DemasqExperiment
  ├─ incremental   → AutoDetector + IncrementalExperiment
  ├─ fewshot       → AutoDetector + FewShotExperiment
  └─ gptzero       → AutoDetector + ThresholdExperiment
         │
         ▼
    for topic in topics:
      load(dataset, detectLLM, topic)
      get_demo_data(train_size, test_size)
      exp.load_data(data)
      res = exp.launch(**experiment_args)
      log_result(csv_file, res, ...)
```

---

## 5. 参数来源追溯表（完整版）

下表追溯每个方法的每个参数「从 config.json 的哪个字段 → 传递给代码的哪个位置」。

### 5.1 Metric-Based 方法 (ll / rank / rank_GLTR / entropy / LRR)

| config.json 路径                       | 代码接收位置                                                      | 说明         |
| -------------------------------------- | ----------------------------------------------------------------- | ------------ |
| `{method}.base_models[i]`              | `AutoDetector.from_detector_name(method, model_name_or_path=...)` | 基座模型路径 |
| `{method}.detector_args.load_in_k_bit` | `MetricBasedDetector.__init__(load_in_k_bit=...)`                 | 量化位数     |
| `global.train_size`                    | `get_demo_data(train_size=...)`                                   | 训练集大小   |
| `global.test_size`                     | `get_demo_data(test_size=...)`                                    | 测试集大小   |
| `global.topics`                        | topic 循环                                                        | 数据主题     |

### 5.2 Binoculars

| config.json 路径             | 代码接收位置                                                                                       |
| ---------------------------- | -------------------------------------------------------------------------------------------------- |
| `Binoculars.detector_args.*` | `AutoDetector.from_detector_name("Binoculars", **detector_args)` → `BinocularsDetector.__init__()` |

### 5.3 Perturbation-Based 方法

| config.json 路径             | 代码接收位置                                  |
| ---------------------------- | --------------------------------------------- |
| `{method}.detector_args.*`   | `AutoDetector.from_detector_name(method, **)` |
| `{method}.experiment_args.*` | `exp.launch(**)` → `PerturbConfig.update()`   |

### 5.4 LM-D

| config.json 路径          | 代码接收位置                                                      |
| ------------------------- | ----------------------------------------------------------------- |
| `LM-D.base_model_names`   | model_name 循环                                                   |
| `LM-D.model_paths.{name}` | `AutoDetector.from_detector_name("LM-D", model_name_or_path=...)` |
| `LM-D.experiment_args.*`  | `exp.launch(**)` → `SupervisedConfig.update()`                    |
| `LM-D.train_size_max`     | `get_demo_data(train_size=max_train)`                             |

### 5.5 其他 Supervised (RADAR / OpenAI-D / ConDA / ChatGPT-D)

| config.json 路径                            | 代码接收位置                                                      |
| ------------------------------------------- | ----------------------------------------------------------------- |
| `{method}.detector_args.model_name_or_path` | `AutoDetector.from_detector_name(method, model_name_or_path=...)` |
| `{method}.detector_args.tokenizer_path`     | `AutoDetector.from_detector_name(method, tokenizer_path=...)`     |
| `{method}.experiment_args.*`                | `exp.launch(**)` → `SupervisedConfig.update()`                    |

### 5.6 Demasq / GPTZero / Incremental / Few-Shot

类似上述模式：`detector_args` → 检测器初始化，`experiment_args` → 实验 launch。

---

## 6. 使用指南

### 6.1 基本用法

```bash
# 运行 ll 方法检测 gpt35 生成的文本
python run/benchmark.py \
  --csv_path results/ll_results.csv \
  --method ll \
  --detectLLM gpt35 \
  --config_path run/config.json
```

### 6.2 如何修改某方法的参数

**示例 1**：修改 Binoculars 的 observer 模型

编辑 `run/config.json`：

```json
{
  "Binoculars": {
    "detector_args": {
      "observer_model_name_or_path": "/your/local/path/falcon-7b",
      ...
    }
  }
}
```

**示例 2**：给 metric-based 方法添加第二个基座模型

```json
{
  "ll": {
    "base_models": [
      "meta-llama/Llama-2-7b-chat-hf",
      "openai-community/gpt2-medium"
    ],
    ...
  }
}
```

**示例 3**：修改 LM-D 的训练轮数和学习率

```json
{
  "LM-D": {
    "experiment_args": {
      "epochs": 5,
      "lr": 1e-5,
      ...
    }
  }
}
```

**示例 4**：切换 DNA-GPT 的阈值优化模式

```json
{
  "DNA-GPT": {
    "detector_args": {
      "mode": "low-fpr",
      ...
    }
  }
}
```

### 6.3 环境变量覆盖机制

如果不想在 `config.json` 中暴露完整模型路径（如多机部署场景），可以使用 `$ENV_VAR` 语法：

```json
{
  "Binoculars": {
    "detector_args": {
      "observer_model_name_or_path": "$FALCON_7B",
      "performer_model_name_or_path": "$FALCON_7B_INSTRUCT"
    }
  }
}
```

然后在 `.env`（或系统环境变量）中设置对应值：

```env
FALCON_7B=/data/models/falcon-7b
FALCON_7B_INSTRUCT=/data/models/falcon-7b-instruct
```

`benchmark.py` 中的 `resolve_model_path()` 函数会自动将 `$FALCON_7B` 替换为环境变量的实际值。

### 6.4 可用方法列表

运行 `--help` 可查看所有支持的方法：

```bash
python run/benchmark.py --help
```

完整列表（来自 `DETECTOR_MAPPING`）：

| 方法名           | experiment_type | 说明                        |
| ---------------- | --------------- | --------------------------- |
| `ll`             | threshold       | Log-Likelihood 检测         |
| `rank`           | threshold       | Token Rank 检测             |
| `rank_GLTR`      | threshold       | GLTR Token Rank 分布检测    |
| `entropy`        | threshold       | Entropy 检测                |
| `LRR`            | threshold       | Log-Rank Ratio 检测         |
| `Binoculars`     | threshold       | 双模型交叉熵比较            |
| `fast-detectGPT` | perturb         | Fast-DetectGPT              |
| `detectGPT`      | perturb         | DetectGPT                   |
| `NPR`            | perturb         | Normalized Perturbed Rank   |
| `DNA-GPT`        | perturb         | DNA-GPT                     |
| `LM-D`           | supervised      | 微调语言模型检测器          |
| `RADAR`          | supervised      | RADAR 预训练检测器          |
| `OpenAI-D`       | supervised      | OpenAI Supervised 检测器    |
| `ConDA`          | supervised      | ConDA Supervised 检测器     |
| `ChatGPT-D`      | supervised      | ChatGPT Supervised 检测器   |
| `demasq`         | demasq          | DEMASQ 检测器               |
| `GPTZero`        | gptzero         | GPTZero API 调用            |
| `incremental`    | incremental     | 增量学习检测器              |
| `baseline`       | fewshot         | Few-Shot 基线（余弦相似度） |
| `generate`       | fewshot         | Few-Shot 数据增强检测       |
| `rn`             | fewshot         | Relation Network Few-Shot   |

---

## 7. 向后兼容性说明

| 方面                 | 兼容性      | 说明                                                                                                                  |
| -------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------- |
| `mgtbench/` 内部代码 | ✅ 完全不变 | 本次只改动 `run/` 目录                                                                                                |
| `.env` 环境变量      | ✅ 继续生效 | `$ENV_VAR` 语法依然支持                                                                                               |
| 命令行参数           | ✅ 向后兼容 | `--method`、`--detectLLM`、`--csv_path`、`--config_path` 保持一致                                                     |
| 旧 config.json       | ⚠️ 需替换   | 旧配置格式不再兼容（旧配置只有 `global`、`LM-D`、`Binoculars` 三个节，缺少 `experiment_type` 等字段），需以新格式为准 |
| `--method` choices   | ⚠️ 已扩展   | 现在支持所有 `DETECTOR_MAPPING` 中注册的方法，不再是硬编码子集                                                        |
