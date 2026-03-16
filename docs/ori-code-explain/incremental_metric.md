# incremental_metric.py 代码分析

> **脚本用途**：在增量学习场景下，使用零样本检测方法（`fast-detectGPT`、`ll`、`rank_GLTR`、`Binoculars`）结合阈值策略做增量检测，并通过多进程并行加速不同方法的实验。

---

## 全局配置（第 1-19 行）

```python
order = [['gpt35', 'Mixtral', 'Moonshot', 'Llama3'], ['gpt-4omini']]
TOPICS = ['STEM', 'Humanities', 'Social_sciences']
```

- 增量学习的 LLM 顺序：前 4 个为已知，`gpt-4omini` 为新引入。
- 设置随机种子 3407 确保可复现。

---

## `get_detector` 函数（第 22-49 行）

根据方法名创建对应的检测器实例：

| 方法 | 模型配置 |
|---|---|
| `fast-detectGPT` | scoring: gpt-neo-2.7B, reference: gpt-j-6B |
| `ll` | Llama-2-7b-chat-hf |
| `rank_GLTR` | Llama-2-7b-chat-hf |
| `Binoculars` | observer: falcon-7b, performer: falcon-7b-instruct, mode: low-fpr |

---

## `run_experiment_sequence` 函数（第 57-67 行）

```python
def run_experiment_sequence(cat, name, cache_sizes, order, device, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    for size in cache_sizes:
        data = load_incremental_topic(order, cat)
        metric = get_detector(name)
        experiment = AutoExperiment.from_experiment_name('incremental_threshold', detector=[metric])
        experiment.load_data(data)
        res1 = experiment.launch()
        results.append({...})
    queue.put(results)
```

- 每个进程绑定到指定 GPU。
- 使用 `incremental_threshold` 实验类型：在增量数据上用阈值分类。
- 结果通过 `Queue` 传回主进程。

---

## 多进程并行执行（第 69-99 行）

```python
cache_sizes = [0]
cuda_devices = [5, 6, 7, 3]

for cat in TOPICS:
    for name in ['fast-detectGPT', 'll', 'rank_GLTR', 'Binoculars']:
        device = cuda_devices[device_index % len(cuda_devices)]
        p = Process(target=run_experiment_sequence, args=(...))
        p.start()
    # 等待所有进程完成，收集结果
    for p in processes:
        p.join()
    while not queue.empty():
        results.extend(queue.get())
    # 保存结果
    with open(f'incremental_metric2/{cat}_result.pickle', 'wb') as f:
        pickle.dump(results, f)
```

- 4 个方法分配到 4 个 GPU 并行运行。
- 每个 topic 的结果单独保存为 pickle 文件。
