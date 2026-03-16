# transfer_binary_mitigate_lmd.py 代码分析

> **脚本用途**：评估 LM-D 二分类模型在**跨领域**和**跨 LLM** 迁移场景下的**缓解（mitigate）** 效果。在已训练模型的基础上，使用不同数量的目标域数据做进一步微调，观察对迁移性能的改善。

---

## 全局配置（第 1-27 行）

```python
TOPICS = ['STEM', 'Humanities', 'Social_sciences']
LLMS = ['Moonshot', 'Mixtral', 'gpt35', 'Llama3', 'gpt-4omini']
TEST_SIZE = 2000
mitigate_save_dir = '/data_sda/zhiyuan/transfer_mitigate_topics'
base_dir = '/data_sda/zhiyuan/topic_models'
```

---

## 辅助函数

### `get_path`（第 30-48 行）
查找指定模型目录下最新的 checkpoint 路径。路径格式为 `{base_dir}/{model}_{llm}_{topic}_3_64`。

### `get_demo_data`（第 51-55 行）
截取数据子集。

### `log_domain_result` / `log_llm_result`（第 58-110 行）
分别记录跨领域和跨 LLM 的缓解实验结果，字段包括：缓解数据量、epoch、batch_size、训练和测试的全部指标。

---

## `transfer_domain` 函数（第 113-175 行）

### 源 == 目标（第 117-133 行）
直接评估，不微调，记录基线结果。

### 源 ≠ 目标（第 135-175 行）

```python
data_sizes = [0, 100, 200, 300, 500, 800]
for size in data_sizes:
    mitigate_data = get_demo_data(data_target, size, TEST_SIZE)
    for epoch in [1]:
        for batch_size in [32]:
            if size == 0:
                if epoch == 2 or batch_size == 64: break
                config['need_finetune'] = False
            detector = AutoDetector.from_detector_name('LM-D', ...)
            experiment = AutoExperiment.from_experiment_name('supervised', ...)
            res = experiment.launch(**config)
```

- `size=0`：不微调，仅评估零样本迁移。对非默认 epoch/batch_size 组合直接跳过。
- `size>0`：用目标域数据微调，每次重新加载原始模型。
- 每轮清空 GPU 缓存。

---

## `transfer_llm` 函数（第 178-238 行）

逻辑与 `transfer_domain` 几乎一致，区别在于：
- 迁移维度是 LLM（源 LLM → 目标 LLM），topic 保持不变。
- `source_llm == target_llm` 时直接评估基线。

---

## `__main__` 入口（第 241-283 行）

通过 `--task` 参数选择：
| task | 说明 |
|---|---|
| `domain` | 跨领域缓解迁移 |
| `llm` | 跨 LLM 缓解迁移 |

结果分别写入 `domain_result_csv` 和 `llm_result_csv`。
