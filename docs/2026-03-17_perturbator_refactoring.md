# 扰动器模块重构文档

> **日期**: 2026-03-17
> **分支**: `refactor/decouple-perturbation`
> **影响范围**: `easymgtd/methods/perturb.py`, `easymgtd/methods/perturbators/`

## 1. 概述

本次重构将 `perturb.py` 中与具体扰动实现强绑定的逻辑**解耦**为独立的扰动器（Perturbator）子包。重构遵循**策略模式（Strategy Pattern）**，使每个检测器持有一个可替换的扰动器实例，同时保持所有外部 API（`AutoDetector`、`AutoExperiment`、`config.json`）完全不变。

### 1.1 重构动机

重构前的 `perturb.py` 存在以下问题：

| 问题      | 具体表现                                                                                                               |
| --------- | ---------------------------------------------------------------------------------------------------------------------- |
| T5 强绑定 | 核心函数 `perturb_texts_()` 硬编码了 T5 的 `<extra_id_N>` 占位符协议，无法使用其他掩码模型                             |
| 职责混杂  | 四种截然不同的检测方法（DetectGPT、NPR、Fast-DetectGPT、DNA-GPT）的扰动逻辑共存于同一文件，且扰动+检测混杂在同一个类中 |
| 不可扩展  | 用户无法自定义扰动策略                                                                                                 |

### 1.2 重构后的架构

```
easymgtd/methods/
├── perturbators/                    # [新增] 扰动器子包
│   ├── __init__.py                  # 统一导出
│   ├── _base.py                     # 三层抽象基类
│   ├── t5_span.py                   # T5 掩码填充扰动器
│   ├── logprob_sampling.py          # 对数概率采样扰动器
│   └── truncate_regen.py            # 截断续写扰动器
├── perturb.py                       # [改造] 检测器，持有扰动器实例
├── __init__.py                      # [更新] 新增导出
└── ...
```

---

## 2. 扰动器类型体系

### 2.1 三层继承结构

```
BasePerturbator (ABC)
├── TextPerturbator (ABC)          ← 文本输入、文本输出
│   ├── T5SpanPerturbator          ← DetectGPT / NPR 默认
│   └── TruncateRegenPerturbator   ← DNA-GPT 默认
└── LogitsPerturbator (ABC)        ← 张量输入、标量输出
    └── LogProbSamplingPerturbator ← Fast-DetectGPT 默认
```

### 2.2 设计原则

- **`TextPerturbator`**：输入为文本列表，输出为扰动后的文本列表。同一种类型的扰动器可以在**不同检测器之间互换**。
- **`LogitsPerturbator`**：输入为模型输出的 logits 张量，输出为标量分数。这是一个**纯计算层**，不持有任何模型引用。
- 检测器通过构造函数的 `perturbator` 参数接受自定义扰动器，并在注入时通过 `isinstance` 校验类型约束。

---

## 3. 各扰动器详解

### 3.1 T5SpanPerturbator（掩码填充扰动）

**文件**: `easymgtd/methods/perturbators/t5_span.py`
**继承**: `TextPerturbator`
**使用者**: `DetectGPTDetector`, `NPRDetector`

#### 工作原理

随机选取文本中的若干连续单词片段（span），替换为 T5 模型能识别的 `<extra_id_N>` 占位符，然后调用 T5 模型的 `generate()` 方法生成填充内容，最后将填充内容回填到原文中。

#### 接口签名

```python
class T5SpanPerturbator(TextPerturbator):
    def __init__(self, mask_model, mask_tokenizer, tokenizer=None):
        """
        Args:
            mask_model:      T5 系列模型（用于生成填充文本）
            mask_tokenizer:  对应的 tokenizer
            tokenizer:       评分模型的 tokenizer（仅 random_fills_tokens 模式需要）
        """

    def perturb(self, texts: list[str], config, ceil_pct=False) -> list[str]:
        """
        Args:
            texts:    待扰动的文本列表
            config:   PerturbConfig 实例，需要以下字段：
                      - span_length (int):        掩码跨度长度，如 2
                      - buffer_size (int):        掩码间最小间隔，如 1
                      - mask_top_p (float):       采样 top-p 阈值，如 1.0
                      - pct_words_masked (float): 掩码比例，如 0.3
                      - DEVICE:                   推理设备
                      - random_fills (bool):      是否使用随机填充（消融用）
                      - random_fills_tokens (bool): 是否使用 token 级随机填充
            ceil_pct: 是否向上取整掩码数量

        Returns:
            list[str]: 扰动后的文本列表，长度与输入相同
        """
```

#### 数据流

```
输入文本: "The quick brown fox jumps over the lazy dog"
    ↓ _tokenize_and_mask()
掩码文本: "The <extra_id_0> fox jumps <extra_id_1> dog"
    ↓ _replace_masks() → T5 model.generate()
T5 输出:  "<extra_id_0> fast red <extra_id_1> over the sleepy"
    ↓ _extract_fills()
填充片段: ["fast red", "over the sleepy"]
    ↓ _apply_extracted_fills()
扰动文本: "The fast red fox jumps over the sleepy dog"
```

---

### 3.2 LogProbSamplingPerturbator（对数概率采样扰动）

**文件**: `easymgtd/methods/perturbators/logprob_sampling.py`
**继承**: `LogitsPerturbator`
**使用者**: `FastDetectGPTDetector`

#### 工作原理

基于 Fast-DetectGPT 算法。给定参考模型和评分模型的 logits 输出，通过从参考模型的条件分布中采样替代 token，比较原始 token 的对数似然与替代 token 的对数似然之间的差异（discrepancy），从而判断文本是否为机器生成。

**注意**：这是一个**纯计算层**，不持有任何模型引用。检测器负责运行模型前向传播，然后将 logits 张量传入此扰动器。

#### 接口签名

```python
class LogProbSamplingPerturbator(LogitsPerturbator):
    def __init__(self, analytic=False):
        """
        Args:
            analytic: 若 True，使用解析法（enumerate over vocabulary）计算差异；
                      若 False（默认），使用经验采样法（10000 次蒙特卡洛采样）。
        """

    def perturb(
        self,
        logits_ref: torch.Tensor,   # shape: [1, seq_len, vocab_size]
        logits_score: torch.Tensor,  # shape: [1, seq_len, vocab_size]
        labels: torch.Tensor,        # shape: [1, seq_len]
    ) -> float:
        """
        Args:
            logits_ref:   参考模型的 logits 输出
            logits_score: 评分模型的 logits 输出
            labels:       原始文本的 next-token ids

        Returns:
            float: 差异分数（discrepancy score），值越大越可能是机器生成
        """
```

#### 数据流（在 FastDetectGPTDetector.detect() 中）

```
文本 "Hello world"
    ↓ scoring_tokenizer()
tokenized → labels = input_ids[:, 1:]
    ↓ scoring_model(**tokenized)
logits_score: shape [1, seq_len-1, vocab_size]
    ↓ reference_model(**tokenized)
logits_ref: shape [1, seq_len-1, vocab_size]
    ↓ perturbator.perturb(logits_ref, logits_score, labels)
discrepancy_score: float
```

#### 两种计算模式

| 模式     | 方法                      | 原理                                     | 特点       |
| -------- | ------------------------- | ---------------------------------------- | ---------- |
| 经验采样 | `_discrepancy_sampling()` | 从参考分布采样 10000 次，估计均值/标准差 | 通用但较慢 |
| 解析法   | `_discrepancy_analytic()` | 遍历词表直接计算期望和方差               | 更快更精确 |

---

### 3.3 TruncateRegenPerturbator（截断续写扰动）

**文件**: `easymgtd/methods/perturbators/truncate_regen.py`
**继承**: `TextPerturbator`
**使用者**: `DNAGPTDetector`

#### 工作原理

基于 DNA-GPT 算法。将输入文本截断为前半段（按 `truncate_ratio`），然后使用自回归语言模型（Causal LM）从前半段续写生成后半段。通过比较原始文本与续写文本的对数概率差异来检测机器生成文本。

#### 接口签名

```python
class TruncateRegenPerturbator(TextPerturbator):
    def __init__(self, base_model, base_tokenizer, **kwargs):
        """
        Args:
            base_model:      自回归语言模型（如 GPT-2）
            base_tokenizer:  对应的 tokenizer
            **kwargs:
                batch_size (int):       生成批大小，默认 5
                regen_number (int):     每条文本的重生成数量，默认 5
                temperature (float):    采样温度，默认 1.0
                truncate_ratio (float): 截断比例（保留前多少），默认 0.5
        """

    def perturb(self, texts: list[str], config=None, **kwargs) -> list[str]:
        """
        Args:
            texts:  待扰动的文本列表（通常是同一条文本重复 regen_number 次）
            config: 可选的配置对象（本扰动器不使用，保留接口兼容性）

        Returns:
            list[str]: 续写后的文本列表
        """
```

#### 数据流

```
输入文本: "The quick brown fox jumps over the lazy dog near the river bank"
    ↓ 截断（truncate_ratio=0.5）
前缀:    "The quick brown fox jumps over"
    ↓ base_model.generate()
续写文本: "The quick brown fox jumps over a tall fence and runs away quickly"
    ↓ _trim_to_shorter_length()
输出:     "The quick brown fox jumps over a tall fence and runs"
```

---

## 4. 检测器与扰动器的对应关系

| 检测器                  | 接受的扰动器类型    | 默认扰动器                   | 是否可替换 |
| ----------------------- | ------------------- | ---------------------------- | ---------- |
| `DetectGPTDetector`     | `TextPerturbator`   | `T5SpanPerturbator`          | ✅         |
| `NPRDetector`           | `TextPerturbator`   | `T5SpanPerturbator`          | ✅         |
| `DNAGPTDetector`        | `TextPerturbator`   | `TruncateRegenPerturbator`   | ✅         |
| `FastDetectGPTDetector` | `LogitsPerturbator` | `LogProbSamplingPerturbator` | ✅         |

注入方式：在构造检测器时传入 `perturbator=...` 参数。若类型不匹配，会抛出 `TypeError`。

---

## 5. 自定义扰动器指南

### 5.1 自定义文本扰动器（TextPerturbator）

适用于替换 DetectGPT/NPR/DNA-GPT 的扰动策略。

**步骤**：

1. **继承 `TextPerturbator`**
2. **实现 `perturb(self, texts, config, **kwargs) -> list[str]` 方法\*\*
3. **在构造检测器时传入自定义实例**

**示例：基于 BERT 的掩码填充扰动器**

```python
from easymgtd.methods import TextPerturbator
from transformers import AutoModelForMaskedLM, AutoTokenizer

class BERTPerturbator(TextPerturbator):
    """Use BERT's [MASK] token to perturb text."""

    def __init__(self, model_name="bert-base-uncased"):
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def perturb(self, texts, config, **kwargs):
        """
        Args:
            texts:  list[str], 待扰动的文本列表
            config: PerturbConfig, 可以读取 pct_words_masked 等参数

        Returns:
            list[str], 扰动后的文本（与输入等长）
        """
        perturbed = []
        for text in texts:
            words = text.split()
            # 随机遮蔽一定比例的词
            n_mask = max(1, int(len(words) * config.pct_words_masked))
            mask_indices = sorted(random.sample(range(len(words)), n_mask))

            for idx in mask_indices:
                words[idx] = "[MASK]"

            masked_text = " ".join(words)
            inputs = self.tokenizer(masked_text, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits

            # 用模型预测的 top-1 token 替换 [MASK]
            input_ids = inputs["input_ids"][0]
            for i, token_id in enumerate(input_ids):
                if token_id == self.tokenizer.mask_token_id:
                    predicted_id = predictions[0, i].argmax().item()
                    input_ids[i] = predicted_id

            result = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            perturbed.append(result)

        return perturbed
```

**使用方式**：

```python
from easymgtd import AutoDetector, AutoExperiment

# 创建自定义扰动器
my_perturbator = BERTPerturbator("bert-base-uncased")

# 注入到 DetectGPT 检测器（替换默认的 T5 扰动）
detector = AutoDetector.from_detector_name(
    "detectGPT",
    model_name_or_path="/path/to/scoring-model",
    perturbator=my_perturbator,  # ← 注入自定义扰动器，无需 mask_model_name_or_path
)

# 后续使用方式与原来完全一致
exp = AutoExperiment.from_experiment_name("perturb", detector=[detector])
exp.load_data(data)
results = exp.launch(**experiment_args)
```

**示例：基于 LLM Prompt 的扰动器**

```python
class LLMPromptPerturbator(TextPerturbator):
    """Use an LLM with prompt to paraphrase text."""

    def __init__(self, llm_client):
        self.client = llm_client

    def perturb(self, texts, config, **kwargs):
        perturbed = []
        for text in texts:
            prompt = f"请在不改变原意的前提下，随机替换以下句子中的几个词：\n{text}"
            result = self.client.generate(prompt)
            perturbed.append(result)
        return perturbed
```

---

### 5.2 自定义 Logits 扰动器（LogitsPerturbator）

适用于替换 Fast-DetectGPT 的差异计算策略。

**步骤**：

1. **继承 `LogitsPerturbator`**
2. **实现 `perturb(self, logits_ref, logits_score, labels) -> float` 方法**
3. **在构造检测器时传入自定义实例**

**示例：基于 KL 散度的差异计算器**

```python
import torch
import torch.nn.functional as F
from easymgtd.methods import LogitsPerturbator

class KLDivergencePerturbator(LogitsPerturbator):
    """Compute KL divergence between reference and scoring distributions."""

    def perturb(self, logits_ref, logits_score, labels):
        """
        Args:
            logits_ref:   参考模型 logits, shape [1, seq_len, vocab_size]
            logits_score: 评分模型 logits, shape [1, seq_len, vocab_size]
            labels:       ground-truth token ids, shape [1, seq_len]

        Returns:
            float: KL 散度分数
        """
        # 计算两个分布之间的 KL 散度
        log_probs_ref = F.log_softmax(logits_ref, dim=-1)
        probs_score = F.softmax(logits_score, dim=-1)

        kl_div = F.kl_div(log_probs_ref, probs_score, reduction="batchmean")
        return kl_div.item()
```

**使用方式**：

```python
from easymgtd import AutoDetector

my_perturbator = KLDivergencePerturbator()

detector = AutoDetector.from_detector_name(
    "fast-detectGPT",
    scoring_model_name_or_path="/path/to/scoring-model",
    reference_model_name_or_path="/path/to/reference-model",
    perturbator=my_perturbator,  # ← 替换默认的采样差异计算
)
```

---

### 5.3 关键约束

| 约束                                         | 说明                                                                                                        |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 类型必须匹配                                 | `DetectGPT`/`NPR`/`DNA-GPT` 只接受 `TextPerturbator` 子类；`Fast-DetectGPT` 只接受 `LogitsPerturbator` 子类 |
| `TextPerturbator.perturb()` 必须返回等长列表 | 输入 N 条文本，必须返回 N 条扰动文本                                                                        |
| `LogitsPerturbator.perturb()` 必须返回 float | 必须是一个标量分数                                                                                          |
| `LogitsPerturbator` 不持有模型               | 它是纯计算层，模型的前向传播由检测器负责                                                                    |

---

## 6. 向后兼容性

本次重构**完全向后兼容**，不需要修改任何现有的配置文件或调用代码：

- `config.json` 中的参数结构不变
- `AutoDetector.from_detector_name()` 接口不变
- `AutoExperiment.from_experiment_name("perturb")` 接口不变
- 不传入 `perturbator` 参数时，行为与重构前完全一致

---

## 7. 文件变更清单

| 操作 | 文件                                                | 说明                                                      |
| ---- | --------------------------------------------------- | --------------------------------------------------------- |
| 新增 | `easymgtd/methods/perturbators/__init__.py`         | 子包入口                                                  |
| 新增 | `easymgtd/methods/perturbators/_base.py`            | `BasePerturbator`, `TextPerturbator`, `LogitsPerturbator` |
| 新增 | `easymgtd/methods/perturbators/t5_span.py`          | `T5SpanPerturbator`                                       |
| 新增 | `easymgtd/methods/perturbators/logprob_sampling.py` | `LogProbSamplingPerturbator`                              |
| 新增 | `easymgtd/methods/perturbators/truncate_regen.py`   | `TruncateRegenPerturbator`                                |
| 改造 | `easymgtd/methods/perturb.py`                       | 删除迁移函数，检测器注入扰动器                            |
| 更新 | `easymgtd/methods/__init__.py`                      | 新增 perturbator 导出                                     |
| 更新 | `easymgtd/experiment/perturb_experiment.py`         | 类型判断放宽为 `BaseDetector`                             |

---

## 8. Bug 修复：多GPU兼容性

> **日期**: 2026-03-17
> **分支**: `fix/multi-gpu-compat`（从 `refactor/decouple-perturbation` 签出）

### 8.1 问题描述

`debug/test_*.py` 测试脚本中有 20 个脚本固定 `GPU_ID="0"`（单卡），加载大模型（7B+）时会 OOM。直接改为多卡后，由于 `model_loader.py` 中 `device_map="auto"` 会将模型跨多张卡分布，部分方法在运算时出现 **tensor 不在同一设备上的 `RuntimeError`**。

经审查，风险主要集中在 `FastDetectGPTDetector`：

- `logits_ref`（来自 reference model 最后一层所在的设备）和 `logits_score`（来自 scoring model 最后一层所在的设备）可能在不同 GPU 上
- `labels` 的设备取决于 `tokenized.input_ids` 的设备（模型第一层所在设备），与 logits 输出设备可能不一致
- `assert` 中比较不同设备上的 tensor 也会报错

其余方法经审查：

- **MetricBased 系列**（ll, rank, lrr, entropy 等）：单模型，accelerate 自动处理，安全
- **Binoculars / TDT**：手动指定 `device_map=cuda:N`，整个模型放单卡，已有显式设备对齐，安全
- **DNA-GPT**：单模型操作，安全
- **Supervised 系列**：使用 `device_map="cuda"`（非 `"auto"`），不会跨卡，安全

### 8.2 修复内容

#### `model_loader.py` — `device_map` 参数化

`load_pretrained` 和 `load_pretrained_mask` 的 `device_map` 从硬编码 `"auto"` 改为函数参数（默认值仍为 `"auto"`），方便调用方在需要时指定具体设备避免跨卡。

```diff
-def load_pretrained(model_name_or_path, quantization_bit=None):
+def load_pretrained(model_name_or_path, quantization_bit=None, device_map="auto"):
     ...
-    config_kwargs["device_map"] = "auto"
+    config_kwargs["device_map"] = device_map
```

`load_pretrained_mask` 做相同改动。

#### `perturb.py` — FastDetectGPT 跨设备 tensor 对齐

在 `FastDetectGPTDetector.detect` 中：

1. `assert` 比较移到 CPU，避免跨卡 tensor 比较报错
2. 在调用 `perturbator.perturb()` 前，将 `logits_ref` 和 `labels` 统一对齐到 `logits_score.device`

```diff
-assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
+assert torch.all(tokenized.input_ids[:, 1:].cpu() == labels.cpu()), "Tokenizer is mismatch."
 logits_ref = self.reference_model(**tokenized).logits[:, :-1]

+compute_device = logits_score.device
+logits_ref = logits_ref.to(compute_device)
+labels = labels.to(compute_device)
+
 crit = self.perturbator.perturb(logits_ref, logits_score, labels)
```

#### 测试脚本 — 统一多卡

全部 20 个 `GPU_ID="0"` 的测试脚本统一改为 `GPU_ID="0,1"`。

### 8.3 变更清单

| 操作 | 文件                               | 说明                             |
| ---- | ---------------------------------- | -------------------------------- |
| 修改 | `easymgtd/loading/model_loader.py` | `device_map` 参数化              |
| 修改 | `easymgtd/methods/perturb.py`      | FastDetectGPT 跨设备 tensor 对齐 |
| 修改 | `run/debug/test_*.py`（20 个文件） | `GPU_ID` 从 `"0"` 改为 `"0,1"`   |
