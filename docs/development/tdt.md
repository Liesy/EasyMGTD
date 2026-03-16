# TDT (Temporal Discrepancy Tomography) 接入开发文档

本片文档记录了如何将前沿机器生成文本检测方法 **TDT** 源项目（[TDT-Text-Detect](https://github.com/guangshengbao/TDT-Text-Detect)）的特征提取核心逻辑无缝集成至 EasyMGTD 框架的 `AutoDetector` 组件体系内。

## 相关文件修改与新增列表

- `mgtbench/methods/tdt.py`: **[新增]** 这是实现检测器主体特性的核心模块。
- `mgtbench/methods/__init__.py`: **[修改]** 添加对 TDTDetector 的导出引用。
- `mgtbench/auto.py`: **[修改]** 将 `TDTDetector` 暴露至工厂类映射 `DETECTOR_MAPPING`。
- `mgtbench/experiment/experiment.py`: **[修改]** 拓展对 TDT 无参打分引擎在评价环节的支持。

---

## 代码与逻辑比对

### 1. 核心数学抽取机制（完全复用）

从 TDT 原版提取的核心函数完全原封不动地保留到了 `EasyMGTD/mgtbench/methods/tdt.py` 中，从而确保对学术指标的不打折复现：

- `transform_discrete_sequence`: 高斯核密度估计函数 KDE。
- `get_wavelet_features`: 使用 `pywt` 做 CWT 处理，提取基于多层尺度下（词态、句法和语篇）能量信号分布。
- `get_t_discrepancy_analytic`: 利用 t 分布计算和建模置信逻辑并结合上下文，生成 token-level 时序差异得分。

### 2. 探针模型类（`TDTDetector` vs 原始 `TDT`）

在原先的开源代码库中，`TDT` 定义在 `scripts/detectors/tdt.py` 中，他派生于其自带的 `DetectorBase`，内部通过重写 `compute_crit` 输出自带阈值调整（或者基于训练数据的 `perplexity` 进行动态调整）的打分。

在 EasyMGTD 中的架构适配中，做出如下调整：

**[Diff 1：去除了耦合配置和环境加载，改走 kwargs]**
EasyMGTD 支持从顶层通过工厂流透传 `*args` , `**kargs` 替代了硬核配置读取！

```python
# 原始的基于 JSON Config 加载
class TDT(DetectorBase):
    def __init__(self, config_name):
        self.config = load_config(config_name)...
        self.scoring_model = load_model(self.config.scoring_model_name...)

# EasyMGTD 新版的面向对象组件设计：
class TDTDetector(BaseDetector):
    def __init__(self, name, **kargs):
        super().__init__(name)
        ref_model_name_or_path = kargs.get('reference_model_name_or_path', "tiiuae/falcon-7b")
        scoring_model_name_or_path = kargs.get('scoring_model_name_or_path', "tiiuae/falcon-7b-instruct")
        # 并支持原生的 HuggingFace Pipeline 初始化（AutoTokenizer/AutoModelForCausalLM）
```

**[Diff 2：取消其自带模型阈值控制，接管至基准实验器]**

- _原版的 TDT:_ 除了算分之外需要维护一个 `compute_perplexity` 和 `get_dynamic_threshold` 方法。一旦判定样本超出 `base_threshold` 经过困惑度加权的 `dynamic_threshold` 即视作 AI 文本。
- _EasyMGTD 的思路:_ 作为标准 Pipeline，任何纯算特征无全连接侧分类头的算法（例如 Rank, GLTR, Entropy），都不该在类里直接执行“阈值定案”。因此，`TDTDetector` 专注于返回打分：
  1.  `TDTDetector.detect(textList)`: 将 `compute_crit` 产生的不变序列特征返回为 float 数组。
  2.  `TDTDetector.find_threshold()`: 配合 `ThresholdExperiment`，在一维的训练集标量打分空间上搜索最佳 F1 划分阈值（如果检测配置采用了这种切分方式）。我们摒弃了它的动态熵缩放（因为这是原实验中的把控标准不适用于 MGTBench 的二分流）。

### 3. 加入 Experiment 实验路由支持

TDT 是一种 **连续分数打分机制**（即吐出浮点特征数而非直接输出 `0/1` 概率的网络）。它在本质上契合 MGTBench 架构中已有的 `[ll, rank, LRR, entropy, Binoculars]` 这些概率提取类模型逻辑。

因此我们在 `mgtbench/experiment/experiment.py` 的白名单里添加了 `'tdt'` 标识符：

```python
class ThresholdExperiment(BaseExperiment):
    # 将 tdt 加在这，这样 AutoExperiment 就认为它也是使用划横线进行二值的合法模型
    _ALLOWED_detector = ['ll', 'rank', 'LRR', 'rank_GLTR', 'entropy', 'Binoculars', 'tdt']
```

同时修改内部逻辑使“小于号”还是“大于号”成为划分方向（一般 tdt 分值越低代表差异越小/模型似然越高即 AI 迹象更明显）：

```python
if detector.name in ['rank', 'LRR', 'entropy', 'Binoculars', 'tdt']:
    # Score 越小表示越有可能是 AI 生成
    y_train_preds = [x < detector.threshold for x in x_train]
```

同样的修改也同步施加至 `IncrementalThresholdExperiment` 的白名单和调度分发中了。

---

**本整合完全保留了 TDT-Text-Detect 论文作者的核心研究成果和底层波形计算函数。**

---

## 最新集成指标 (easymgtd) 与调试指南

随着项目的全面重构（`mgtbench` -> `easymgtd`），TDT 算法已完成深度适配并将相关依赖和配置标准化。

### 1. 环境依赖补充

TDT 的连续小波变换 (CWT) 指标依赖于小波分析库。在 `mgtd` 环境下需确保安装：

```bash
pip install PyWavelets==1.9.0
```

该依赖已同步更新至项目的 [requirements.txt](file:///data/liyang/MGTD-Baselines/EasyMGTD/requirements.txt)。

### 2. 重构后的核心路径

在最新的组件化架构中，TDT 相关逻辑位于：
- **检测器实现**: [easymgtd/methods/tdt.py](file:///data/liyang/MGTD-Baselines/EasyMGTD/easymgtd/methods/tdt.py)
- **实验调度**: [easymgtd/experiment/threshold_experiment.py](file:///data/liyang/MGTD-Baselines/EasyMGTD/easymgtd/experiment/threshold_experiment.py) (已加入 `_ALLOWED_detector` 白名单)
- **工厂映射**: 在 [easymgtd/auto.py](file:///data/liyang/MGTD-Baselines/EasyMGTD/easymgtd/auto.py) 中注册为 `"tdt"`。

### 3. 参数配置详解 (`run/config.json`)

TDT 采用了典型的“引用-评分”双模型架构，配置示例如下：

```json
"tdt": {
  "experiment_type": "threshold",
  "detector_args": {
    "reference_model_name_or_path": "tiiuae/falcon-7b",
    "scoring_model_name_or_path": "tiiuae/falcon-7b-instruct",
    "max_length": 512,
    "extract_wavelet_features": false
  },
  "experiment_args": {}
}
```

- `reference_model_name_or_path`: 基础参考模型（如 Falcon-7B）。
- `scoring_model_name_or_path`: 评分模型（如 Falcon-7B-Instruct）。
- `extract_wavelet_features`: 是否提取波形特征。设置为 `true` 时，计算开销会显著增加。

### 4. 快速验证与调试

我们提供了独立的调试脚本，用于快速核实 TDT 逻辑在当前环境下的正确性：

```bash
# 进入项目根目录并执行
python run/debug/test_tdt.py
```

**预期行为**:
1. 载入双模型至 GPU（建议使用双卡，脚本会自动尝试 `cuda:0` 和 `cuda:1`）。
2. 加载本地 AITextDetect 数据。
3. 执行打分逻辑并输出 ACC/F1/AUC 等指标。
