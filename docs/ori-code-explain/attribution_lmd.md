# attribution_lmd.py 代码分析

> **脚本用途**：使用有监督方法 LM-D 对多分类（源归因）任务进行微调训练，判断文本由哪个 LLM 生成。

---

## 全局配置（第 1-9 行）

```python
os.environ["CUDA_VISIBLE_DEVICES"]="3"
MODELS = ['Moonshot', 'gpt35', 'Mixtral', 'Llama3', 'gpt-4omini']
```

- 指定使用 GPU 3。
- `MODELS` 列表定义了 5 个 LLM，加上人类写作共 6 个类别，因此模型的 `num_labels = len(MODELS) + 1 = 6`。

---

## `train_attribution` 函数（第 12-53 行）

### 1. 数据加载（第 13 行）
```python
data = load_attribution(category)
```
按 `category`（学科类别）加载源归因数据集。

### 2. 检测器初始化（第 18-21 行）
```python
metric = AutoDetector.from_detector_name('LM-D',
    model_name_or_path=model_name_or_path,
    num_labels=len(MODELS) + 1)
```
创建 LM-D 检测器（有监督分类模型），类别数为 6（5 个 LLM + 1 个 Human）。

### 3. 实验配置与启动（第 22-35 行）
```python
experiment = AutoExperiment.from_experiment_name('supervised', detector=[metric])
experiment.load_data(data)
config = {
    'need_finetune': True,
    'save_path': model_save_dir,
    'epochs': epoch,
    'batch_size': batch_size,
    'disable_tqdm': False
}
res = experiment.launch(**config)
```
- 创建有监督实验，加载数据。
- `need_finetune=True` 表示需要微调训练。
- `model_save_dir` 格式为 `{save_dir}/{category}_{modelName}_{batchSize}_{epoch}`。

### 4. 结果记录（第 37-53 行）
- 打印训练和测试结果。
- 将结果追加写入 CSV 文件，字段包括：模型名、类别、batch_size、epoch、训练 F1、测试 F1。
- 如果 CSV 文件不存在，则先创建并写入表头。

---

## `__main__` 入口（第 56-73 行）

通过 `argparse` 接收 5 个命令行参数：
| 参数 | 说明 |
|---|---|
| `--model_path` | 预训练模型路径 |
| `--category` | 数据类别（如 STEM） |
| `--epoch` | 训练轮数 |
| `--batch_size` | 批大小 |
| `--model_save_dir` | 模型保存目录 |
| `--output_csv` | 结果 CSV 输出路径 |
