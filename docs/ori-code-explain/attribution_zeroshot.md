# attribution_zeroshot.py 代码分析

> **脚本用途**：使用零样本（zero-shot）方法对源归因（attribution）任务进行检测，按 Topic 遍历三个学科领域，用不同统计指标或模型评分进行分类。

---

## 全局常量与模型路径（第 1-25 行）

```python
MODELS = ['Moonshot', 'gpt35', 'Mixtral', 'Llama3', 'gpt-4omini']
TOPICS = ['STEM', 'Humanities', 'Social_sciences']
METHODS = ['ll', 'rank', 'rank_GLTR', 'entropy', 'LRR', 'Binoculars', 'fast-detectGPT']
```

- `MODELS`：LLM 列表（此脚本中未直接使用，仅作参考）。
- `TOPICS`：三个学科领域。
- `METHODS`：支持的零样本检测方法。
- 下方定义了若干本地模型路径（`gpt_neo`、`gpt_j`、`falcon_7b` 等），供不同方法使用。

---

## `log_result` 函数（第 28-40 行）

将单次实验的结果追加写入 CSV 文件。记录字段包括：方法名、分类器类型、topic、基模型、训练 F1、测试 acc/precision/recall/F1/AUC。

- 对 `Binoculars` 和 `fast-detectGPT` 方法，`base_model` 设为 `None`（因为它们有自己独立的模型）。

---

## `get_demo_data` 函数（第 43-47 行）

从数据中截取前 `train_size` 条训练数据和前 `test_size` 条测试数据，用于快速实验。

---

## `train_attribution` 函数（第 50-103 行）

### 1. 方法选择与检测器初始化（第 53-77 行）

根据 `method` 参数创建不同的检测器和实验：

- **统计类方法**（`ll`、`rank`、`rank_GLTR`、`entropy`、`LRR`、`Binoculars`）：
  - `Binoculars`：使用 `falcon-7b` 作为 observer、`falcon-7b-instruct` 作为 performer。
  - 其余方法：使用指定的 `model_path` 加载模型。
  - 实验类型为 `threshold`（基于阈值的分类）。
- **扰动类方法**（`fast-detectGPT`）：
  - 使用 `gpt-neo-2.7B` 作为打分模型、`gpt-j-6B` 作为参考模型。
  - 实验类型为 `perturb`。

### 2. 遍历 Topic 运行实验（第 80-103 行）

```python
for topic in TOPICS:
    data = load_attribution_topic(topic)
    data = get_demo_data(data, 1000, 2000)
    experiment.load_data(data)
    config = {
        'attribution': True,
        'logistic_path': ...,
        'svm_path': ...,
    }
    res = experiment.launch(**config)
```

- 对每个 topic 加载归因数据，截取 1000 条训练 + 2000 条测试。
- `attribution=True`：指示实验按多分类模式运行。
- 保存训练好的 Logistic 回归和 SVM 分类器到指定路径。
- 如果有两个分类器结果（Logistic + SVM），分别记录。

---

## `__main__` 入口（第 106-117 行）

命令行参数：
| 参数 | 说明 |
|---|---|
| `--model_path` | 基模型路径 |
| `--method` | 零样本方法名 |
| `--output_csv` | 结果 CSV 输出路径 |
