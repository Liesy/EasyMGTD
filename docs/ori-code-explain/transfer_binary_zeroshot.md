# transfer_binary_zeroshot.py 代码分析

> **脚本用途**：评估零样本检测方法在二分类任务上的**跨领域**和**跨 LLM** 迁移效果。用在源 topic+源 LLM 上训练好的分类器（Logistic/Threshold），直接在目标 topic 或目标 LLM 的数据上推理评估。

---

## 全局配置（第 1-25 行）

```python
METHODS = ['ll', 'rank', 'rank_GLTR', 'entropy', 'LRR', 'Binoculars', 'fast-detectGPT']
TOPICS = ['STEM', 'Humanities', 'Social_sciences']
LLMS = ['Moonshot', 'Mixtral', 'gpt35', 'Llama3', 'gpt-4omini']
```

定义了模型路径（`gpt_neo`、`gpt_j`、`falcon_7b` 等）供不同方法使用。

---

## 辅助函数

### `log_result`（第 28-50 行）
将迁移实验结果追加写入 CSV，字段包括：方法、分类器、源/目标 topic、源/目标 LLM、基模型、测试指标（acc/precision/recall/F1/AUC）。

### `get_demo_data`（第 53-57 行）
截取数据子集。

---

## `transfer_domain_and_llm` 函数（第 60-180 行）

### 1. 外层循环：遍历 LLM × 源 topic（第 66-106 行）

```python
for detectLLM in LLMS:
    for source_topic in TOPICS:
        config = {
            'logistic_path': ...,
            'threshold_path': ...,
        }
```

- 对每个 (LLM, source_topic) 组合，加载预训练的 Logistic 分类器和阈值文件。
- 根据方法类型创建检测器：
  - `Binoculars`：falcon-7b + falcon-7b-instruct，实验类型 `threshold`
  - 统计方法：Llama-2-7b-chat，实验类型 `threshold`
  - `fast-detectGPT`：gpt-neo + gpt-j，实验类型 `perturb`
- 预训练分类器路径通过检测器构造函数传入。

### 2. 跨领域迁移（第 108-144 行）

```python
for target_topic in TOPICS:
    data_target = load_topic_data(detectLLM=detectLLM, topic=target_topic)
    data_target = get_demo_data(data_target, 10, 100)
    experiment.load_data(data_target)
    exp_config = {'use_pretrained': True}
    res = experiment.launch(**exp_config)
```

- 固定 LLM，变换 topic。
- `use_pretrained=True`：使用预训练分类器推理。
- 如果有多个分类器结果（如 Logistic + Threshold），都记录。

### 3. 跨 LLM 迁移（第 146-180 行）

```python
for target_llm in LLMS:
    data_target = load_topic_data(detectLLM=target_llm, topic=source_topic)
    ...
```

- 固定 topic，变换 LLM。
- 其余逻辑与跨领域迁移相同。

---

## `__main__` 入口（第 183-194 行）

命令行参数：`--method`、`--result_csv`。
