# transfer_attribution_zeroshot.py 代码分析

> **脚本用途**：评估零样本检测方法在源归因任务上的**跨领域迁移**效果。分为两阶段：(1) 对所有 topic 计算并缓存检测分数；(2) 用源 topic 训练的分类器在目标 topic 上评估。

---

## 全局配置（第 1-26 行）

```python
METHODS = ['ll', 'rank', 'rank_GLTR', 'entropy', 'LRR', 'Binoculars', 'fast-detectGPT']
TOPICS = ['STEM', 'Humanities', 'Social_sciences']
TEST_SIZE = 2000
```

定义了 7 种零样本方法、3 个 topic、以及模型路径。

---

## 辅助函数

### `log_result`（第 28-41 行）
记录迁移实验结果到 CSV，字段包括：方法、分类器、源/目标 topic、基模型、测试指标。

### `get_demo_data`（第 43-47 行）
截取数据子集。

---

## `get_method_scores` 函数（第 50-123 行）

**阶段一：预计算并缓存检测分数**

### 1. 检查缓存（第 55-71 行）
遍历所有 topic，检查是否已有保存的分数文件（`.npy`）。如果全部已保存则跳过。

### 2. 初始化检测器（第 73-99 行）
根据方法创建对应的检测器和实验：
- `Binoculars`：falcon-7b + falcon-7b-instruct，实验类型 `threshold`
- 统计方法（`ll`、`rank` 等）：Llama-2-7b-chat，实验类型 `threshold`
- `fast-detectGPT`：gpt-neo + gpt-j，实验类型 `perturb`

### 3. 计算并保存分数（第 103-123 行）
```python
for topic in TOPICS:
    data = load_attribution_topic(topic)
    data = get_demo_data(data, 10, TEST_SIZE)
    exp_config = {
        'attribution': True,
        'save_test_score': True,
        'test_score_x_path': ...,
        'test_score_y_path': ...,
    }
    res = experiment.launch(**exp_config)
```
- 仅用 10 条训练数据（占位）+ 2000 条测试数据。
- `save_test_score=True`：将原始检测分数保存为 `.npy` 文件。

---

## `transfer_domain` 函数（第 126-201 行）

**阶段二：跨领域迁移评估**

### 1. 初始化检测器（第 131-157 行）
与 `get_method_scores` 类似，创建检测器。

### 2. 加载预训练分类器并迁移评估（第 160-201 行）
```python
for source_topic in TOPICS:
    config = {
        'pretrained_logistic_path': ...,
        'pretrained_svm_path': ...,
    }
    detector.load_pretrained_classifier(**config)
    for target_topic in TOPICS:
        exp_config = {
            'attribution': True,
            'use_pretrained': True,
            'use_saved_score': True,
            'test_score_x_path': ...,
            'test_score_y_path': ...,
        }
        res = experiment.launch(**exp_config)
```
- 加载在源 topic 上训练好的 Logistic/SVM 分类器。
- `use_pretrained=True`：使用预训练分类器而非重新训练。
- `use_saved_score=True`：直接使用缓存的检测分数，避免重复计算。
- 记录每个 (source, target) 组合的结果。

---

## `__main__` 入口（第 204-216 行）

依次执行两个阶段：先缓存分数，再做迁移评估。命令行参数：`--method`、`--result_csv`。
