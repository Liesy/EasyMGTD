# transfer_mitigate_zeroshot.py 代码分析

> **脚本用途**：在零样本方法的跨领域/跨 LLM 迁移场景下，通过混入不同数量的目标域数据重新训练轻量分类器（Logistic/SVM/Threshold），评估**缓解（mitigate）** 迁移性能下降的效果。本脚本**不依赖 GPU**，纯基于预先缓存的检测分数（`.npy` 文件）在 CPU 上训练传统分类器。

---

## 全局配置（第 1-29 行）

```python
METHODS / TOPICS / LLMS / ZEROSHOT_METHODS  # 方法、领域、LLM 列表
TRAIN_SIZE = 1000
TEST_SIZE = 2000
```

---

## 核心工具函数（第 31-168 行）

### `run_clf`（第 31-37 行）
用训练好的分类器做预测，返回 (真实标签, 预测标签, 预测概率)。对极端值做 clip 处理防止溢出。

### `cal_metrics`（第 40-58 行）
根据标签数自动选择二分类或多分类的指标计算。
- **二分类**（`max(label) < 2`）：标准 acc/precision/recall/F1/AUC。
- **多分类**：使用 `weighted` 平均，AUC 设为 -1，并额外计算混淆矩阵。

### `train_logistic`（第 61-81 行）
Logistic 回归缓解训练：
```python
x_train = concat(source_x_train, target_x_train[:mitigate_size])
y_train = concat(source_y_train, target_y_train[:mitigate_size])
clf = LogisticRegression().fit(x_train, y_train)
```
将源域全部训练数据与目标域前 `mitigate_size` 条数据合并，训练 Logistic 回归。

### `train_svm`（第 84-110 行）
SVM 缓解训练，逻辑与 `train_logistic` 类似：
- 额外对 `rank` 方法做标准化处理（`StandardScaler`）加速 SVM 收敛。
- 使用线性核 SVM。

### `find_threshold`（第 113-134 行）
在训练集上搜索最优阈值：
- `ll`、`fast-detectGPT`：分数越高越像机器文本（`> threshold` 为正类）。
- `rank`、`LRR`、`entropy`、`Binoculars`：分数越低越像机器文本（`< threshold` 为正类）。
- 遍历所有可能的阈值，选择 F1 最高的。

### `train_threshold`（第 137-168 行）
与 Logistic/SVM 类似，但使用阈值分类器。将源域和目标域数据合并后搜索最优阈值，再在测试集上评估。

---

## 日志记录函数（第 171-217 行）

### `log_domain_result`（第 171-192 行）
记录跨领域缓解结果：method、criterion、detect_llm、源/目标 topic、mitigate_size、训练/测试指标。

### `log_llm_result`（第 195-217 行）
记录跨 LLM 缓解结果：method、criterion、源/目标 LLM、source_topic、mitigate_size、训练/测试指标。

---

## 实验函数

### `get_binary_domain_result`（第 220-259 行）

**二分类跨领域缓解实验**：

```python
for method in ZEROSHOT_METHODS:
    for detectLLM in LLMS:
        for source_topic in TOPICS:
            for target_topic in TOPICS:
                sizes = [0, 100, 200, 300, 500, 800] if source != target else [0]
                # 加载预缓存的 .npy 分数文件
                source_x_train = np.load(...)
                ...
                for mitigate_size in sizes:
                    # Logistic 缓解
                    train_logistic(source, target[:mitigate_size], test)
                    # Threshold 缓解（rank_GLTR 不支持）
                    train_threshold(...)
```

- 源域与目标域相同时仅评估 `size=0`（基线）。
- 每种方法使用两种分类器（Logistic + Threshold），`rank_GLTR` 跳过 Threshold。

### `get_binary_llm_result`（第 262-302 行）

**二分类跨 LLM 缓解实验**：逻辑与跨领域完全对称，迁移维度从 topic 改为 LLM。

### `get_attribution_result`（第 329-365 行）

**源归因（多分类）跨领域缓解实验**：
- 数据路径使用 `_attribution_` 后缀。
- 使用 Logistic + SVM 两种分类器（而非 Threshold，因为多分类不适合单阈值）。

### `log_attribution_result`（第 306-326 行）
记录归因任务的缓解结果。

---

## `__main__` 入口（第 367-370 行）

```python
get_binary_domain_result()
get_binary_llm_result()
get_attribution_result()
```

依次执行三个实验：二分类跨领域缓解 → 二分类跨 LLM 缓解 → 归因跨领域缓解。无需命令行参数。
