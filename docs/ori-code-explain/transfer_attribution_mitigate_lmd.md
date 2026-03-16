# transfer_attribution_mitigate_lmd.py 代码分析

> **脚本用途**：评估 LM-D 源归因模型在跨领域场景下的**缓解（mitigate）迁移**效果。在源 topic 训练好的模型基础上，使用目标 topic 的少量数据做进一步微调，观察不同微调数据量对迁移性能的提升。

---

## 全局配置（第 1-19 行）

```python
TOPICS = ['STEM', 'Humanities', 'Social_sciences']
llms = ['Moonshot', 'Mixtral', 'gpt35', 'Llama3', 'gpt-4omini']
TRAIN_SIZE = 1000
TEST_SIZE = 2000
MITIGATE_SAVE_DIR = '/data_sda/zhiyuan/transfer_mitigate_attribution_topics'
```

---

## 辅助函数

### `get_demo_data`（第 22-26 行）
截取训练和测试数据子集。

### `log_domain_result`（第 28-51 行）
将详细的实验结果写入 CSV，字段包括：基模型、源/目标 topic、缓解数据量、epoch、batch_size、训练和测试的 acc/precision/recall/F1/AUC。

---

## `transfer_domain_mitigate` 函数（第 54-116 行）

### 1. 模型类型判断（第 58-63 行）
根据 `model_path` 中是否包含 `distil` 或 `roberta` 来确定基模型名称。

### 2. 源 == 目标的情况（第 65-79 行）
如果源 topic 和目标 topic 相同：
- 直接加载模型，用少量数据（10 条训练 + 2000 条测试）做推理评估。
- 记录结果后直接返回，不做微调。

### 3. 源 ≠ 目标：缓解实验（第 82-116 行）

```python
data_sizes = [0, 100, 200, 300, 500, 800]
for size in data_sizes:
    mitigate_data = get_demo_data(data_target, size, TEST_SIZE)
    for epoch in [1]:
        for batch_size in [32]:
            ...
```

- 遍历不同的目标域数据量（0/100/200/300/500/800 条）。
- `size=0` 时不微调（`need_finetune=False`），仅评估零样本迁移基线。
- `size>0` 时用目标域数据微调已有模型，评估缓解效果。
- 每次实验重新加载原始模型（避免累积微调效应）。
- 如果训练结果为 `None`（未微调时），用全零 `Metric` 填充。

---

## `__main__` 入口（第 119-139 行）

命令行参数：`--source_topic`、`--target_topic`、`--model_path`、`--result_csv`。
