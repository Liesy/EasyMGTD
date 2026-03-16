# transfer_attribution_lmd.py 代码分析

> **脚本用途**：评估使用 LM-D 有监督方法训练的源归因模型在**跨领域（跨 topic）**场景下的迁移性能。加载在源 topic 上训练好的模型，直接在目标 topic 上做推理评估（不微调）。

---

## 全局配置（第 1-14 行）

```python
os.environ["CUDA_VISIBLE_DEVICES"]="0"
TOPICS = ['STEM', 'Humanities', 'Social_sciences']
llms = ['Moonshot', 'Mixtral', 'gpt35', 'Llama3', 'gpt-4omini']
```

---

## `get_demo_data` 函数（第 17-21 行）

截取指定数量的训练和测试数据子集。

---

## `transfer_domain` 函数（第 24-49 行）

### 1. 数据准备（第 25-27 行）
```python
data_target = load_attribution_topic(topic=target_topic)
data_target = get_demo_data(data_target, 100, 2000)
```
加载目标 topic 的归因数据，取 100 条训练 + 2000 条测试。

### 2. 模型加载与推理（第 29-39 行）
```python
method = AutoDetector.from_detector_name('LM-D', model_name_or_path=model_path, tokenizer_path=model_path)
experiment = AutoExperiment.from_experiment_name('supervised', detector=[method])
experiment.load_data(data_target)
res = experiment.launch(need_finetune=False, eval=True)
```
- `model_path` 是在**源 topic** 上已训练好的模型路径。
- `need_finetune=False`：不进行微调，直接评估迁移效果。
- `eval=True`：启用评估模式。

### 3. 结果记录（第 42-49 行）
将模型路径、源 topic、目标 topic 和测试 F1 写入 CSV。

---

## `__main__` 入口（第 53-73 行）

命令行参数：
| 参数 | 说明 |
|---|---|
| `--source_topic` | 源领域（训练时的 topic） |
| `--target_topic` | 目标领域（测试的 topic） |
| `--model_path` | 训练好的模型路径 |
| `--result_csv` | 结果 CSV 路径 |

入口处会校验 `source_topic` 必须出现在 `model_path` 中（确保模型与声称的源 topic 一致）。
