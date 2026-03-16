# increamental.py 代码分析

> **脚本用途**：使用增量学习方法（`incremental`）在多个 LLM 按顺序出现的场景下训练检测器，评估不同正则化强度（`lwf_reg`）和缓存大小（`cache_size`）对性能的影响。

---

## 全局配置（第 1-36 行）

```python
os.environ["CUDA_VISIBLE_DEVICES"]="3"
order = [['gpt35', 'Mixtral', 'Moonshot', 'Llama3'], ['gpt-4omini']]
TOPICS = ['STEM', 'Humanities', 'Social_sciences']
model = '/data1/models/deberta-v3-base'
cache_size = [0, 100]
```

- **GPU**：指定 GPU 3。
- **`order`**：增量学习顺序——先训练前 4 个 LLM，然后增量引入 `gpt-4omini`。
- **`model`**：使用 DeBERTa-v3-base 作为骨干模型。
- **`cache_size`**：旧任务样本的缓存大小，0 表示不缓存，100 表示保留 100 条旧样本。

### `setup_seed` 函数（第 20-28 行）

设置全局随机种子（3407），确保实验可复现。覆盖 Python hash、PyTorch、CUDA、Numpy、Random 的随机状态，并关闭 CuDNN 的非确定性行为。

---

## 主循环（第 39-80 行）

### 四层嵌套：`cat × cache_size × reg × order`

```python
for cat in TOPICS:
    for size in cache_size:
        for reg in [0, 0.2]:
            for order in orders:
```

### 1. 数据加载与检测器初始化（第 47-55 行）

```python
data = load_incremental_topic(order, cat)
nclass = len(set(data['train'][0]['label']))
bic = True if cache_size==100 and reg==0 else False
metric = AutoDetector.from_detector_name('incremental',
    model_name_or_path=model,
    num_labels=nclass, lwf_reg=reg, cache_size=size, bic=bic)
```

- `load_incremental_topic` 按照增量顺序加载数据，返回分阶段的训练/测试数据。
- `lwf_reg`：LwF（Learning without Forgetting）正则化系数。0 表示不使用知识蒸馏，0.2 表示使用。
- `bic`：Bias Correction，仅在 `cache_size=100` 且 `reg=0` 时启用。

> ⚠️ **注意**：第 52 行 `if cache_size==100` 使用的是列表变量而非循环变量 `size`，可能是一个 bug。

### 2. 训练配置与启动（第 57-69 行）

```python
config = {
    'need_finetune': True,
    'need_save': False,
    'epochs': 2,
    'lr': 1e-6,
    'batch_size': 64,
    'save_path': '/data1/lyl/mgtout-1/',
    'eval': True,
    'lr_factor': 4
}
experiment.load_data(data)
res1 = experiment.launch(**config)
```

- `lr_factor=4`：学习率缩放因子（增量学习中用于调整新旧任务的学习率比例）。

### 3. 结果收集与保存（第 71-80 行）

每次实验的结果包含：`order`、`lwf_reg`、`use_bic`、`lr_factor`、`cache_size` 和实验结果。按学科保存为 pickle 文件。
