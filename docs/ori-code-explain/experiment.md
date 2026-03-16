# experiment.py 代码分析

> **文件用途**：定义了所有实验类型的执行逻辑。每个实验类对应一种检测范式，负责编排"数据准备 → 检测器推理 → 分类/评估"的完整流程。

---

## 文件总览

本文件包含 **7 个实验类** 和 **4 个配置数据类**：

| 实验类                           | 配置类              | 支持的检测器                                                  | 用途                     |
| -------------------------------- | ------------------- | ------------------------------------------------------------- | ------------------------ |
| `ThresholdExperiment`            | —                   | ll, rank, LRR, rank_GLTR, entropy, Binoculars                 | 基于统计指标的零样本检测 |
| `PerturbExperiment`              | `PerturbConfig`     | detectGPT, NPR, fast-detectGPT, DNA-GPT                       | 基于文本扰动的零样本检测 |
| `SupervisedExperiment`           | `SupervisedConfig`  | OpenAI-D, ConDA, ChatGPT-D, LM-D, RADAR                       | 有监督微调检测           |
| `DemasqExperiment`               | `DemasqConfig`      | demasq                                                        | Demasq 专用检测          |
| `GPTZeroExperiment`              | —                   | GPTZero                                                       | GPTZero API 检测         |
| `IncrementalExperiment`          | `IncrementalConfig` | incremental                                                   | 有监督增量学习检测       |
| `IncrementalThresholdExperiment` | —                   | ll, rank, LRR, rank_GLTR, entropy, Binoculars, fast-detectGPT | 零样本增量检测           |
| `FewShotExperiment`              | `FewShotConfig`     | baseline, generate, rn                                        | 少样本学习检测           |

---

## 1. `ThresholdExperiment`（第 18-133 行）

### 用途

使用基于统计指标的零样本方法，通过阈值或 Logistic 回归对检测分数做分类。

### `launch` 方法（第 29-57 行）

- 调用 `predict` 获取预测结果。
- 如果有两套预测（threshold + logistic），分别计算指标，返回两个 `DetectOutput`。
- 如果只有一套（仅 logistic），返回一个 `DetectOutput`。

### `predict` 方法（第 59-133 行）

**数据预测阶段**（第 66-83 行）：

- `rank_GLTR`：直接用 detect 的多维输出（不经过 `data_prepare`）。
- 其他方法：通过 `data_prepare` 将一维分数转为适合分类器的格式。

**分类阶段**（第 91-131 行）：

- **支持阈值的方法**（`Binoculars`、`rank`、`ll`、`LRR`、`entropy`）：
  - 先用 `find_threshold` 搜索最优阈值，基于阈值做二分类。
  - 再训练 Logistic 回归作为第二种分类方式。
  - 返回两套结果 (threshold, logistic)。
  - **方向区分**：`rank`/`LRR`/`entropy`/`Binoculars` 分数越低越像机器（`< threshold`）；`ll` 分数越高越像机器（`> threshold`）。
- **仅 logistic 的方法**（`rank_GLTR`）：
  - 只用 Logistic 回归分类。

---

## 2. `PerturbConfig`（第 136-153 行）

扰动实验的配置数据类：

| 字段                    | 默认值 | 说明             |
| ----------------------- | ------ | ---------------- |
| `span_length`           | 2      | 掩码跨度长度     |
| `buffer_size`           | 1      | 缓冲区大小       |
| `mask_top_p`            | 1.0    | 掩码采样的 top-p |
| `pct_words_masked`      | 0.3    | 掩码词比例       |
| `n_perturbations`       | 10     | 扰动次数         |
| `n_perturbation_rounds` | 1      | 扰动轮数         |
| `criterion_score`       | "z"    | 评分准则         |

---

## 3. `PerturbExperiment`（第 156-269 行）

### 用途

使用基于文本扰动的方法（detectGPT、NPR、fast-detectGPT、DNA-GPT）进行检测。

### `launch` 方法（第 182-210 行）

与 `ThresholdExperiment.launch` 结构相同：处理单套或双套预测结果。

### `predict` 方法（第 212-269 行）

- 将扰动配置传入检测器的 `detect` 方法。
- 对 `NPR`、`fast-detectGPT`、`DNA-GPT`、`detectGPT`：
  - 先搜索最优阈值（分数越高越像机器）。
  - 再训练 Logistic 回归。
  - 返回两套结果。
- 其他方法仅用 Logistic 回归。

---

## 4. `SupervisedConfig`（第 272-291 行）

有监督实验的配置数据类：

| 字段                          | 默认值       | 说明                |
| ----------------------------- | ------------ | ------------------- |
| `need_finetune`               | False        | 是否微调            |
| `need_save`                   | True         | 是否保存模型        |
| `batch_size`                  | 16           | 批大小              |
| `pos_bit`                     | 1            | 正类标签位          |
| `epochs`                      | 3            | 训练轮数            |
| `lr`                          | 5e-6         | 学习率              |
| `save_path`                   | "finetuned/" | 保存路径            |
| `gradient_accumulation_steps` | 1            | 梯度累积            |
| `weight_decay`                | 0.01         | 权重衰减            |
| `save_total_limit`            | 2            | checkpoint 最大数量 |
| `swanlab_project`             | "EasyMGTD"   | SwanLab 项目名      |

---

## 5. `SupervisedExperiment`（第 294-376 行）

### 用途

有监督微调实验（LM-D、OpenAI-D 等），先微调模型再评估。

### `predict` 方法（第 316-376 行）

**微调阶段**（第 325-328 行）：

```python
if self.supervise_config.need_finetune:
    detector.finetune(data_train, self.supervise_config)
```

**评估阶段**：分两种模式：

- **`eval=True`**（第 330-344 行）：仅评估测试集（用于不微调的迁移评估）。
- **`eval=False`**（第 346-375 行）：评估训练集和测试集。

**分类逻辑**（两种模式共享）：

- **二分类**（`num_labels == 2`）：softmax 概率 ≥ 0.5 → 正类。
- **多分类**：直接取 `argmax` 的预测类别。

---

## 6. `DemasqExperiment`（第 393-433 行）

### 用途

Demasq 检测器的专用实验。

### `predict` 方法（第 411-433 行）

- 如果 `need_finetune=True`，先微调。
- 用检测器预测 logits，以 0.5 为阈值做二分类。
- 同时评估训练集和测试集。

---

## 7. `GPTZeroExperiment`（第 436-463 行）

### 用途

GPTZero API 检测实验。

### `predict` 方法（第 445-463 行）

- 不涉及微调，直接调用检测器推理。
- logit > 0.5 → 正类。

---

## 8. `IncrementalConfig`（第 466-482 行）

增量学习配置，额外包含 `lr_factor`（学习率缩放因子，默认 5）。

---

## 9. `IncrementalExperiment`（第 485-583 行）

### 用途

有监督增量学习实验，支持分阶段训练和评估。

### `load_data`（第 507-513 行）

加载增量数据的**最后一个阶段**作为标准训练/测试集。

### `return_output`（第 515-533 行）

通用的输出构造方法，支持从 (text, label) 对或中间结果构造预测输出。

### `predict`（第 535-583 行）

- 如果 `need_finetune=True`，调用检测器的增量微调方法，获得中间结果。
- `eval=True` 模式：评估中间结果和测试集。
- `eval=False` 模式：评估训练集和测试集。

---

## 10. `IncrementalThresholdExperiment`（第 586-765 行）

### 用途

在增量学习场景中使用零样本方法，结合 Logistic 回归分类器做增量检测。**本文件中最复杂的实验类**。

### `__init__`（第 607-618 行）

- 加载一个 `roberta-base` 模型用于特征提取（构建 exampler）。
- `cache_size`：每类缓存的样本数量。

### `load_data`（第 620-626 行）

同 `IncrementalExperiment`，使用最后一个阶段的数据。

### `get_dataset`（第 628-644 行）

将当前阶段数据与旧样本（exampler）合并：

- 如果 `return_exampler=True` 且 `cache_size != 0`，调用 `construct_exampler` 构建旧样本缓存。

### `construct_exampler`（第 647-692 行）

**核心方法**：使用 roberta-base 提取每条文本的 [CLS] 嵌入，然后：

1. 计算每类的均值嵌入。
2. 对每类样本，计算到类均值的余弦距离。
3. 选取距离最近的 `cache_size` 条样本作为 exampler（代表性样本）。

### `increment_classes`（第 694-710 行）

扩展 Logistic 回归分类器以适应新类别：

- 在权重矩阵中添加新类别对应的零初始化行。
- 更新 `classes_` 属性。

### `predict`（第 712-765 行）

**增量训练循环**：

```python
for idx, stage_data in enumerate(stages):
    train_dataset, exampler = self.get_dataset(stage_data, exampler, return_exampler=True)
    test_dataset, _ = self.get_dataset(eval_set[idx], ...)
    if idx != 0:
        self.increment_classes(detector, num_newclass)  # 扩展分类器
    # 用检测器提取分数
    detector.classifier.fit(x_train, y_train)          # 重新训练分类器
    test_result = self.run_clf(detector.classifier, x_test, y_test)
```

每个阶段：

1. 合并当前数据与 exampler。
2. 如果不是第一阶段，扩展分类器维度。
3. 提取检测分数，训练分类器，评估测试集。

---

## 11. `FewShotConfig`（第 768-786 行）

少样本实验配置，额外包含：

- `kshot`：每类样本数（默认 5）。
- `classifier`：分类器类型（默认 "SVM"）。
- `lr_factor`：学习率缩放因子（默认 5）。

---

## 12. `FewShotExperiment`（第 789-880 行）

### 用途

在增量学习先验的基础上，用 few-shot 方法对新类别进行少样本检测。

### `load_data`（第 812-818 行）

同 `IncrementalExperiment`，使用最后阶段的数据。

### `return_output`（第 820-839 行）

与 `IncrementalExperiment.return_output` 类似，构造预测输出。

### `predict`（第 841-880 行）

- 调用检测器的 `finetune` 方法做 few-shot 适配。
- `eval=True`：仅评估测试集。
- `eval=False`：评估训练集和测试集。

---

## 各实验类的共性模式

所有实验类都继承自 `BaseExperiment`，共享以下模式：

1. **`__init__`**：初始化检测器列表和配置对象。
2. **`load_data`**：加载数据，设置 `self.train_text/label` 和 `self.test_text/label`。
3. **`predict`**：核心方法，编排检测器推理和分类逻辑。
4. **`launch`**：调用 `predict`，然后对结果调用 `cal_metrics` 计算评估指标，返回 `DetectOutput` 列表。

> `BaseExperiment` 提供了公共工具方法：`data_prepare`（数据格式转换）、`run_clf`（运行分类器预测）、`cal_metrics`（计算指标）。
