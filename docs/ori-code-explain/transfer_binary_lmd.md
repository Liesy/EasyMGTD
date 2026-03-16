# transfer_binary_lmd.py 代码分析

> **脚本用途**：评估 LM-D 有监督二分类模型在两种迁移场景下的性能：(1) **跨领域迁移**（同一 LLM，不同 topic）；(2) **跨 LLM 迁移**（同一 topic，不同 LLM）。加载已训练模型直接评估，不重新微调。

---

## 全局配置（第 1-16 行）

```python
TOPICS = ['STEM', 'Humanities', 'Social_sciences']
llms = ['Moonshot', 'Mixtral', 'gpt35', 'Llama3', 'gpt-4omini']
```

---

## 辅助函数

### `get_demo_data`（第 19-23 行）
截取数据子集（100 训练 + 2000 测试）。

### `get_path`（第 25-46 行）
在指定目录中查找最新的 HuggingFace checkpoint 文件夹（格式为 `checkpoint-{num}`），返回该路径。使用修改时间（`mtime`）确定最新的 checkpoint。

### `log_domain_result`（第 49-59 行）
记录跨领域迁移结果到 CSV：基模型、检测 LLM、源/目标 topic、测试指标。

### `log_llm_result`（第 62-72 行）
记录跨 LLM 迁移结果到 CSV：基模型、源/目标 LLM、源 topic、测试指标。

---

## `transfer_llm` 函数（第 75-96 行）

**跨 LLM 迁移**：在源 LLM 训练的模型上测试目标 LLM 的数据。

```python
data_target = load_topic_data(topic=source_topic, detectLLM=target_llm)
model_path = f'.../topic_models/{base_model}_{source_llm}_{source_topic}_3_64'
actual_model_path = get_path(model_path)
method = AutoDetector.from_detector_name('LM-D', model_name_or_path=actual_model_path, ...)
res = experiment.launch(need_finetune=False, eval=True)
```

- 加载目标 LLM 的数据，使用在源 LLM 上训练的模型直接推理。
- `need_finetune=False`：只评估不训练。

---

## `transfer_domain` 函数（第 99-120 行）

**跨领域迁移**：在源 topic 训练的模型上测试目标 topic 的数据。

逻辑与 `transfer_llm` 类似，区别在于：
- 数据按 `target_topic` 加载，LLM 保持不变。
- 模型路径按 `source_topic` 定位。

---

## `__main__` 入口（第 123-158 行）

通过 `--task` 参数选择迁移类型：

| task | 必要参数 | 说明 |
|---|---|---|
| `domain` | `--source_topic`, `--target_topic`, `--detectLLM`, `--base_model` | 跨领域迁移 |
| `llm` | `--source_topic`, `--source_llm`, `--target_llm`, `--base_model` | 跨 LLM 迁移 |

入口处对 `base_model` 限制为 `distilbert` 或 `roberta-base`。
