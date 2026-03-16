# fewshot.py 代码分析

> **脚本用途**：在增量学习场景下，先用已知 LLM 的数据微调基模型，然后用 few-shot 方法（`baseline`、`rn`、`generate`）在新 LLM 数据上做少样本检测，评估不同 k-shot 设置下的性能。

---

## 全局配置（第 1-31 行）

```python
os.environ["CUDA_VISIBLE_DEVICES"]="1"
order = [['gpt35', 'Mixtral', 'Moonshot'], ['Llama3', 'gpt-4omini']]
TOPICS = ['STEM', 'Humanities', 'Social_sciences']
models = ['/data1/models/roberta-base', '/data1/models/distilbert-base-uncased', '/data1/models/deberta-v3-base']
methods = ['baseline', 'rn', 'generate']
kshot = [1, 5, 20]
```

- **GPU**：指定 GPU 1。
- **`order`**：LLM 分组——第一组是已知 LLM（用于训练），第二组是新 LLM（用于 few-shot 测试）。
- **`TOPICS`**：三个学科领域。
- **`models`**：三个预训练编码器模型。
- **`methods`**：三种 few-shot 策略。
- **`kshot`**：每类样本数（1/5/20 shot）。

---

## 主循环（第 33-79 行）

### 三层嵌套结构：`cat × model × order`

### 1. 基模型微调（第 40-57 行）

```python
data = load_incremental_topic(order, cat)
nclass = len(set(data['train'][0]['label']))
supervise = AutoDetector.from_detector_name('LM-D', model_name_or_path=model, num_labels=nclass)
supervise.finetune(data['train'][0], cfg)
```

- 使用 `load_incremental_topic` 按增量顺序加载数据。
- `nclass`：从训练标签中推断类别数。
- 训练配置：2 个 epoch、学习率 1e-6、batch_size 64、不保存模型。
- 使用 `SupervisedConfig` 对象管理配置，包含 `lr_factor` 和 `classifier='MLP'`。

### 2. Few-shot 实验（第 58-75 行）

```python
for method in methods:
    metric = AutoDetector.from_detector_name(method, model=supervise.model, tokenizer=supervise.tokenizer,
                                             num_labels=nclass, kshot=1)
    experiment = AutoExperiment.from_experiment_name('fewshot', detector=[metric])
    experiment.load_data(data)
    experiment.launch(**config)  # 预热
    for shot in kshot:
        trials = []
        for _ in range(5):       # 每个 kshot 重复 5 次
            res1 = experiment.launch(**config)
            trials.append(res1)
        result.append({...})
```

- 将已微调模型的 `model` 和 `tokenizer` 传入 few-shot 检测器（复用权重）。
- `need_finetune=False`：不再做全量微调，仅做 few-shot 推理。
- 每个 kshot 设置运行 5 次取消随机性影响。

### 3. 结果保存（第 76-79 行）

```python
os.makedirs(f'fewshot_{basename}_32all', exist_ok=True)
with open(f'fewshot_{basename}_32all/{cat}.pickle', 'wb') as f:
    pickle.dump(result, f)
```

将结果序列化为 pickle 文件，按模型名和学科分别保存。
