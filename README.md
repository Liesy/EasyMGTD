# EasyMGTD

EasyMGTD 是基于 [MGTBench-2.0](https://github.com/Y-L-LIU/MGTBench-2.0) 的机器生成文本检测评测框架。它旨在提供一个易于扩展、“组装流水线”式的架构，方便研究人员快速接入和测试如 SeqXGPT、TDT、Human-Outlier 等前沿的检测方法，或者利用这些基础设施进行大规模实证实验。

## Supported Methods

Currently, we support the following methods (continuous updating):

- Metric-based methods:
  - Log-Likelihood [[Ref]](https://arxiv.org/abs/1908.09203);
  - Rank [[Ref]](https://arxiv.org/abs/1906.04043);
  - Log-Rank [[Ref]](https://arxiv.org/abs/2301.11305);
  - Entropy [[Ref]](https://arxiv.org/abs/1906.04043);
  - GLTR Test 2 Features (Rank Counting) [[Ref]](https://arxiv.org/abs/1906.04043);
  - DetectGPT [[Ref]](https://arxiv.org/abs/2301.11305);
  - LRR [[Ref]](https://arxiv.org/abs/2306.05540);
  - NPR [[Ref]](https://arxiv.org/abs/2306.05540);
  - DNA-GPT [[Ref]](https://arxiv.org/abs/2305.17359);
  - Fast-DetectGPT [[Ref]](https://arxiv.org/abs/2310.05130);
  - Binoculars [[Ref]](https://arxiv.org/abs/2401.12070);
  - TDT (On the way)
- Model-based methods:
  - OpenAI Detector [[Ref]](https://arxiv.org/abs/1908.09203);
  - ChatGPT Detector [[Ref]](https://arxiv.org/abs/2301.07597);
  - ConDA [[Ref]](https://arxiv.org/abs/2309.03992) [[Model Weights]](https://www.dropbox.com/s/sgwiucl1x7p7xsx/fair_wmt19_chatgpt_syn_rep_loss1.pt?dl=0);
  - GPTZero [[Ref]](https://gptzero.me/);
  - RADAR [[Ref]](https://arxiv.org/abs/2307.03838);
  - LM Detector [[Ref]](https://arxiv.org/abs/1911.00650);
  - SeqXGPT (On the way)
  - Human-Outlier (On the way)

## Supported Datasets

- [AITextDetect](https://huggingface.co/datasets/AITextDetect/AI_Polish_clean)

It contains human written and AI polished text in different categories, including:

- STEM (Physics, Math, Computer, Biology, Chemistry, Electrical, Medicine, Statistics)
- Social Sciences (Education, Management, Economy and Finance)
- Humanities (Art, History, Literature, Philosophy, Law)

From [wiki](https://en.wikipedia.org/wiki/Main_Page), [arxiv](https://arxiv.org/), and [Gutenberg](https://www.gutenberg.org/)

## Quick Start

### Installation

```bash
git clone git@github.com:Liesy/EasyMGTD.git
cd EasyMGTD
conda env create -f mgtbench2.yml;
conda activate mgtbench2;
# you may need mirror for faster installation
pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
pip install -r requirements.txt
```

### Basic Usage

示例见 [`notebook/detection.ipynb`](notebook/detection.ipynb)。

```python
from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load

model_name_or_path = 'openai-community/gpt2-medium'
metric = AutoDetector.from_detector_name('ll',
                                         model_name_or_path=model_name_or_path)
experiment = AutoExperiment.from_experiment_name('threshold',detector=[metric])

data_name = 'AITextDetect'
detectLLM = 'gpt35'
category = 'Art'
data = load(data_name, detectLLM, category)
experiment.load_data(data)
res = experiment.launch()

print('train:', res[0].train)
print('test:', res[0].test)
```

## Dataloader

支持通过分类 (Category) 或主题 (Topic) 加载数据。详情参考 `mgtbench/loading/dataloader.py`。
示例代码可参考原始的：[`notebook/check_dataloader.ipynb`](notebook/check_dataloader.ipynb)

---

## 统一调用接口架构及源码解析

EasyMGTD 的设计理念是一种“组装流水线”。`AutoDetector` 负责提供不同原理的**“打分引擎”**（给一段文本，输出它是 AI 生成的置信度/特征得分），而 `AutoExperiment` 则负责提供标准的**“评价流水线”**（如何切分数据、如何寻找阈值、如何使用逻辑回归分类、如何测算不同维度的各种指标）。

以下是两者的详细骨架级解析：

### 一、 AutoDetector.from_detector_name 详解

这个函数是一个简单的映射工厂：
**代码位置:** `mgtbench/auto.py:165`

```python
@classmethod
def from_detector_name(cls, name, *args, **kargs) -> BaseDetector:
    metric_class_path = cls._detector_mapping[name]
    # 动态导入对象并实例化
    return metric_class(name, *args, **kargs)
```

它会根据传入的 `name` 字符串（定义在 `auto.py` 开头的 `DETECTOR_MAPPING` 中），动态去对应的文件中 import 并实例化子类。所有的子类都继承自 `BaseDetector`，并必须实现 `detect` 方法。

以下为主流 `name` 对应的具体解析：

#### 1. 基础概率类：`'ll'`, `'rank'`, `'entropy'`, `'rank_GLTR'`

**底层类实现：** `MetricBasedDetector` 的子类 (如 `LLDetector`) , 见 `mgtbench/methods/metric_based.py`
**可选 kwargs：**

- `model_name_or_path`: HuggingFace 模型的绝对路径或名字（如 `gpt2-medium`）。
- `load_in_k_bit`: 量化参数，传给内部的 `load_pretrained` 函数（支持 4bit / 8bit）。
  **检测逻辑 (`detect`)：**
- 输入 batch 的文本，进行分词后输入因果语言模型（Causal LM）。
- **LL**: 计算序列每个 token 原词的 log-likelihood，取平均。
- **Rank**: 计算模型预测概率分布下，原词对应的 rank 排名（越小说明模型越“喜欢”这个词），取均值 (`log=True` 时取 log 后均值，默认不取 log)。
- **Entropy**: 计算每个 token 预测分布的信息熵，取平均的负值（`-neg_entropy.sum(-1).mean()`）。
- **Rank_GLTR**: 原论文复现，统计序列中 token 的 rank 落在 Top-10 / 10-100 / 100-1000 / >1000 的频率分布比例，返回这**四个频率值的数组**。
  **阈值搜索 (`find_threshold`)：**
- 在训练集上直接对打分数组进行排序遍历。对 LL 等，寻找使 F1 Score 最大的划分分界线（大于此分数为异常或正常）。

#### 2. 双模型比对类：`'Binoculars'`

**底层类实现：** `BinocularsDetector`, 见 `mgtbench/methods/metric_based.py`
**可选 kwargs：**

- `observer_model_name_or_path`: 观察者模型，计算交叉熵（通常较小，如 `falcon-7b`）。
- `performer_model_name_or_path`: 表演者模型，计算困惑度（通常较强，如 `falcon-7b-instruct`）。
- `max_length`: 截断词数（默认 512）。
- `mode`: 阈值寻找模式，可选 `'accuracy'` (找最高F1) 或 `'low-fpr'` (在假阳性率 FPR 逼近 0.01% 的约束下找阈值)。
- `threshold`: 阈值覆盖策略，选 `'default'` 会写死论文里的常量，给别的值则会根据你的数据集动态重算 `find_threshold`。
  **检测逻辑 (`detect`)：**
- 同时过前向传播，计算 `performer` 的困惑度 `ppl`，除以两模型交叉分布的熵 `x_ppl`，将比值 (`ppl / x_ppl`) 作为最终输出的置信得分。

#### 3. 扰动重采样类：`'detectGPT'`, `'fast-detectGPT'`, `'DNA-GPT'`

**底层类实现：** 继承自 `PerturbBasedDetector` 的各个子类, 见 `mgtbench/methods/perturb.py`
**特殊 kwargs：**

- 除了 `model_name_or_path` (打分模型) 外，还需要 `mask_model_name_or_path` (通常是个 T5 模型，用来给原文本挖空做随机填词，制造扰动样本 `perturbed_text`)。
  _(注：对于 FastDetectGPT 还需要 `scoring_model_name_or_path` 和用于计算采样分布差异的 `reference_model_name_or_path`)_
  **检测逻辑 (`detect` 需要配合 `config`):**
- **DetectGPT**: 对每段原始输入的文本挖空并重新填入，生成 `n` 组（默认 10）扰动后的平行变体。
  1. 给原始文本打分 (likelihood)；
  2. 给那 `n` 组变体分别打分求均值和方差；
  3. 返回原分数与变体分数群均值的归一化差异值 `$z$-score`（AI 文本因为处于模型似然高地附近，随机一扰动分数会剧降，这个差异值会显著偏大）。
- **FastDetectGPT**: 抛弃显式的生成扰动，直接利用公式理论推导原分布和采样分布间的统计差异，速度快几百倍。

#### 4. 特征分类器类（有监督训练）: `'LM-D'`, `'RADAR'`, `'ConDA'` 等等统称

**底层类实现：** `SupervisedDetector`, 见 `mgtbench/methods/supervised.py`
**可选 kwargs：**

- `model_name_or_path`: 你的主干网络，比如 `roberta-base`。
- `use_metric`: (布尔值) 核心！如果设为 `True`，训练时会在交叉熵之上**叠加圆环损失 (Circle Loss)** 来做更激进的特征拉开，由自带的 `MetricTrainer` 代替 HF Trainer 进行前向反向传播。
  **检测逻辑 (`detect`)：**
- 前向推理，取 `logits.softmax(-1)[:, pos_bit]`；如果没有二分类头或是长文本提取任务，取 `torch.argmax`。
  **训练逻辑 (`finetune`)：**
- 接收包装好的字典集 (`encodings` & `labels`) 构造 `torch.utils.data.Dataset`。
- 使用原生 `transformers.Trainer` 配置 `TrainingArguments` 发起微调并按 epoch 保存 checkpoint。

---

### 二、 AutoExperiment.from_experiment_name 运行流程拆解

`AutoExperiment` 的底层实现在 `mgtbench/experiment/experiment.py` 中。它的核心生命周期包含四个关键动作：

**1. 初始化 (`__init__`)**

```python
experiment = AutoExperiment.from_experiment_name('threshold', detector=[detector1, detector2])
```

- 实例化指定的具体子类（如 `ThresholdExperiment`），将传入的探测器列表绑定为 `self.detector`。
- 实例化该实验流独有的配套 Config，如 `PerturbExperiment` 会附带一个 `PerturbConfig` 用于记录随机填词的长度和比例。

**2. 注入数据 (`load_data`)**

- 仅仅做了简单的指针绑定（将形如 `{'train':{'text':...}, 'test':...}` 的字典挂载为 `self.train_text`, `self.train_label` 等状态变量）。

**3. 执行控制流 (`launch`)** -> **4. 核心调度 (`predict`)**
当你调用 `experiment.launch(**config)` 时，会发起以下的链式调用：

```python
# mgtbench/auto.py: BaseExperiment 抽象定义
def launch(self, **config) -> list[DetectOutput]:
    predict_list = self.predict(**config) # <--- 交给子类实现的复杂逻辑
    final_output = []
    # 后处理循环：将预测标签数组 (predictions) 与 真实标签比对，算出 sklearn 的所有指标
    # 封装为 Metric DataClass 返回（包含 acc, precision, recall, f1, auc）
    for detector_predict in predict_list:
        ...
        final_output.append(DetectOutput(...))
    return final_output
```

重点在于不同派生类重写的 `predict()`。下面对比两种最常见的流：

#### 范式 A: 基于无参数打分+阈值切分 (`ThresholdExperiment` / `PerturbExperiment`)

_这些算法 (LL, Binoculars, DetectGPT) 吐出的永远是一维浮点得分。它不是 0/1 的二分类结果。怎么划线定结果呢？_

1. **获取打分:** 调用 `detector.detect()` 把所有训练集、测试集文本灌进去计算，得到一维的 float array `x_train`, `x_test`。
2. **生成评价基准一 (找阈值分类)：**
   - 调用该特征检测器的 `detector.find_threshold(x_train, y_train)`。（比如：循环遍历 `x_train` 的不同阈值点，找到能让当前训练集的 F1 score 最佳的那个截断点 `detector.threshold`）。
   - 根据这根红线，判断 `x_train`, `x_test` 代表的是 0 还是 1。将判定结果包装为 `test_result1`。
3. **生成评价基准二 (逻辑回归分类)：**
   - 很多人认为“单一阈值法（画横线）”太过简单粗暴。于是实验基准里还内嵌了一个标准的线性分类器。
   - 调用 `sklearn.linear_model.LogisticRegression()`。通过 `.fit(x_train, y_train)` 把打分作为输入特征训练一个二分类器。
   - 通过 `clf.predict(x_test)` 获取判定结果 `test_result2`。
4. **返回列表:** 组装基准一（返回的 `final_output` 中 `name='threshold'`）和基准二（`name='logistic'`）送给 `launch() `进行指标汇算。这也是为什么第一条回复的代码演示中会看到两套完全不一样指标的原因（他们衡量的是同一种打分能力下的不同利用手段）。

#### 范式 B: 基于有监督模型直出的实验 (`SupervisedExperiment`)

_这种架构自己内化了分类能力，吐出来的就是概率分布和对齐分类好的 Logits。_

1. **动态更新配置 (Update Config):**
   - 把在 `launch(need_finetune=True, epochs=3, ...)` 中传入的实参解包，更新给内部的 `SupervisedConfig` 数据类对象。
2. **训练 (如果需要):**
   - `if self.supervise_config.need_finetune`: 直接调用检测器自带的 `detector.finetune(data, self.supervise_config)` 拉起 HF Trainer 开始反向传播微调权重网络。（此函数由我们在 `AutoDetector详解` 一节里的第4点详细分析过）。
3. **推理并对齐:**
   - 调用 `detector.detect(self.test_text)`，这次直接通过模型得到 logits 输出维度的特征概率 `test_preds`。
   - **自动判断模型头:**
     - 如果这是个二分类模型 (`num_labels == 2`)，对特征向量第一维截断： `np.where(test_preds[:, 0] >= 0.5, 1, 0)` -> 直接划归 1 或 0 。
     - 如果是个多任务分类（包含不同的 AI 类型），则进行维度的降解抽取。
4. **返回列表:** 这里不需要使用 Logistic Regression 或者滑动阈值搜索了，直接将通过微调网络硬推理出来的 `y_test_pred` 判定阵列作为最终答案，交付上层算 F1 和 AUC。

---

本实现项目派生自开源版本 MGTBench-2.0 以供研究探讨。
