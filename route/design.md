# 新闻偏见检测模型设计文档

## 概述

本项目旨在开发一个结合Longformer长文本处理能力和bias_detector偏见分析能力的**文章级框架偏见检测模型**。

**框架偏见定义**: 框架偏见（Framing Bias）是指媒体通过选择特定的叙述角度、语言表达、信息强调和上下文构建来呈现新闻事件，从而系统性地影响读者对事件的理解、态度和判断的偏见类型。这种偏见体现在：
- **选择性强调**: 突出某些事实而忽略其他相关信息
- **语言倾向**: 使用带有情感色彩或价值判断的词汇
- **因果归因**: 对事件原因和后果的特定解释框架
- **视角局限**: 主要从特定立场或群体的角度报道事件

模型将在Kaggle All_the_news_2.0数据集上进行训练和评估，专门检测新闻文章中的框架偏见模式并输出偏见概率分数。

**弱监督学习策略**: 由于现有的bias_detector缺乏文章级分析能力，其输出将被视为弱标签/伪标签。模型将采用弱监督学习方法，包括：
- **标签去噪**: 识别和修正低质量的伪标签
- **置信度加权**: 根据标签质量调整训练权重
- **自训练**: 使用模型预测改进标签质量
- **一致性正则化**: 确保相似文章的预测一致性

## 架构

### 模型架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                  框架偏见检测模型架构                            │
├─────────────────────────────────────────────────────────────────┤
│  输入层 (Input Layer)                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   原始文章文本   │  │   文章元数据     │  │   预处理器       │ │
│  │   (Full Article)│  │   (Metadata)    │  │   (Preprocessor)│ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  编码层 (Encoding Layer)                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Longformer    │  │   注意力掩码     │  │   位置编码       │ │
│  │   Tokenizer     │  │   Generator     │  │   Handler       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  特征提取层 (Feature Extraction Layer)                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Longformer    │  │   滑动窗口       │  │   全局注意力     │ │
│  │   Encoder       │  │   Attention     │  │   Mechanism     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  融合层 (Fusion Layer)                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   特征融合       │  │   Bias Detector │  │   上下文聚合     │ │
│  │   Module        │  │   Integration   │  │   Module        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  输出层 (Output Layer)                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   偏见分类器     │  │   概率计算器     │  │   置信度估计器   │ │
│  │   (Classifier)  │  │   (Probability) │  │   (Confidence)  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 核心组件

1. **长文本编码器 (Long Text Encoder)**
   - 基于Longformer架构处理完整新闻文章（最多4096 tokens）
   - 实现滑动窗口注意力机制处理长序列
   - 支持全局注意力掩码优化关键信息提取

2. **偏见特征提取器 (Bias Feature Extractor)**
   - 集成预训练的bias_detector权重
   - 提取文章中的偏见相关语义特征
   - 实现多层特征融合机制

3. **框架偏见分类器 (Framing Bias Classifier)**
   - 专门检测新闻文章中的框架偏见模式
   - 识别选择性强调、语言倾向、因果归因和视角局限
   - 输出二分类结果（有框架偏见/无框架偏见）和连续概率分数
   - 提供预测置信度估计和偏见类型细分

4. **模型融合模块 (Model Fusion Module)**
   - 结合Longformer和bias_detector的优势
   - 实现权重自适应融合策略
   - 优化长文本和偏见检测的协同效果

5. **弱监督学习模块 (Weak Supervision Module)**
   - 实现伪标签质量评估和去噪算法
   - 设计置信度加权损失函数
   - 提供自训练和标签精炼机制

## 组件和接口

### 模型核心组件

```python
class LongformerBiasDetector(nn.Module):
    def __init__(self, longformer_model: str, bias_detector_model: str):
        self.longformer = LongformerForSequenceClassification.from_pretrained(longformer_model)
        self.bias_detector = AutoModelForSequenceClassification.from_pretrained(bias_detector_model)
        self.fusion_layer = BiasFeatureFusion()
        self.classifier = FrameworkBiasClassifier()
    
    def forward(self, input_ids, attention_mask, global_attention_mask) -> BiasOutput
    def predict_bias(self, article_text: str) -> Tuple[int, float, float]
    def batch_predict(self, articles: List[str]) -> List[BiasResult]
```

### 数据处理组件

```python
class ArticlePreprocessor:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def preprocess_article(self, article: str) -> ProcessedArticle
    def create_attention_masks(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
    def handle_long_articles(self, article: str) -> List[str]
```

### 训练和评估组件

```python
class BiasModelTrainer:
    def __init__(self, model: LongformerBiasDetector, config: TrainingConfig):
        self.model = model
        self.config = config
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
    
    def train_epoch(self, dataloader: DataLoader) -> TrainingMetrics
    def evaluate(self, dataloader: DataLoader) -> EvaluationMetrics
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict) -> None
    def load_checkpoint(self, path: str) -> Dict
```

### 弱监督学习组件

```python
class WeakSupervisionManager:
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.label_quality_estimator = LabelQualityEstimator()
        self.pseudo_labeler = PseudoLabeler()
    
    def assess_label_quality(self, articles: List[str], labels: List[int]) -> List[float]
    def filter_noisy_labels(self, dataset: Dataset, confidence_scores: List[float]) -> Dataset
    def weighted_loss(self, predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor
    def self_training_iteration(self, model: nn.Module, unlabeled_data: Dataset) -> Dataset
    def consistency_regularization(self, model: nn.Module, batch: Dict) -> torch.Tensor
```

### 实验管理组件

```python
class ExperimentManager:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.metrics_tracker = MetricsTracker()
        self.weak_supervision = WeakSupervisionManager()
    
    def run_experiment(self, train_data: Dataset, val_data: Dataset) -> ExperimentResults
    def cross_validate(self, data: Dataset, k_folds: int = 5) -> CrossValidationResults
    def generate_report(self, results: ExperimentResults) -> ResearchReport
    def save_results(self, results: ExperimentResults, path: str) -> None
```

## 数据模型

### 核心数据结构

```python
@dataclass
class Article:
    id: str
    title: str
    content: str
    source: str
    author: Optional[str]
    published_date: datetime
    url: Optional[str]
    word_count: int
    
@dataclass
class ProcessedArticle:
    article_id: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    global_attention_mask: torch.Tensor
    token_count: int
    is_truncated: bool
    
@dataclass
class BiasOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    hidden_states: Optional[torch.Tensor]
    attention_weights: Optional[torch.Tensor]
    
@dataclass
class BiasResult:
    article_id: str
    bias_label: int  # 0 (unbiased) or 1 (biased)
    bias_probability: float  # 0.0 to 1.0
    confidence_score: float  # model confidence
    processing_time: float
    token_count: int
    
@dataclass
class ExperimentResults:
    model_name: str
    dataset_name: str
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: Dict
    training_time: float
    hyperparameters: Dict
    
@dataclass
class CrossValidationResults:
    fold_results: List[ExperimentResults]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    statistical_significance: Dict[str, float]
    
@dataclass
class ResearchReport:
    experiment_id: str
    model_architecture: str
    dataset_statistics: Dict
    performance_metrics: Dict
    ablation_studies: List[Dict]
    error_analysis: Dict
    computational_requirements: Dict
    reproducibility_info: Dict
```
## 正确性属性

*属性是一个特征或行为，应该在系统的所有有效执行中保持为真——本质上是关于系统应该做什么的正式陈述。属性作为人类可读规范和机器可验证正确性保证之间的桥梁。*

### 模型输入处理属性

**属性 1: 文章预处理一致性**
*对于任何*输入的新闻文章，预处理后的token序列应该保持原文的语义完整性
**验证: 需求 1.1**

**属性 2: 长文本处理正确性**
*对于任何*长度超过4096 tokens的文章，滑动窗口处理应该覆盖完整文本内容
**验证: 需求 2.3**

**属性 3: 注意力掩码有效性**
*对于任何*处理的文章，全局注意力掩码应该正确标识关键位置（如[CLS]标记）
**验证: 需求 2.4**

**属性 4: 批处理一致性**
*对于任何*批量输入，每个样本的处理结果应该与单独处理的结果一致
**验证: 需求 1.2**

**属性 5: 数据格式标准化**
*对于任何*输入格式的数据，预处理后应该转换为统一的模型输入格式
**验证: 需求 1.5**

### 模型推理属性

**属性 6: 偏见概率有效性**
*对于任何*输入文章，输出的偏见概率应该在[0.0, 1.0]范围内且符合概率分布特性
**验证: 需求 3.1**

**属性 7: 分类一致性**
*对于任何*偏见概率大于0.5的文章，偏见标签应该为1；否则应该为0
**验证: 需求 3.1**

**属性 8: 模型融合有效性**
*对于任何*输入文章，融合模型的性能应该不低于单独使用任一基础模型的性能
**验证: 需求 2.1**

**属性 9: 推理时间可预测性**
*对于任何*给定长度的文章，推理时间应该在可预测的范围内且与文章长度呈线性关系
**验证: 需求 4.4**

**属性 10: 置信度校准**
*对于任何*预测结果，置信度分数应该与实际预测准确性相关
**验证: 需求 3.4**

### 训练和评估属性

**属性 11: 训练收敛性**
*对于任何*有效的训练配置，模型训练应该在合理的epoch数内收敛
**验证: 需求 7.2**

**属性 12: 评估指标一致性**
*对于任何*测试数据集，计算的准确率、精确率、召回率和F1分数应该数学上一致
**验证: 需求 7.1**

**属性 13: 交叉验证稳定性**
*对于任何*K折交叉验证，不同折之间的性能差异应该在合理范围内
**验证: 需求 7.5**

**属性 14: 超参数敏感性**
*对于任何*超参数的小幅变化，模型性能不应该出现剧烈波动
**验证: 需求 7.2**

**属性 15: 数据泄露防护**
*对于任何*训练过程，验证集和测试集的数据不应该出现在训练集中
**验证: 需求 8.2**

### 实验可重现性属性

**属性 16: 随机种子一致性**
*对于任何*固定的随机种子，多次运行相同实验应该产生完全一致的结果
**验证: 需求 8.2**

**属性 17: 配置记录完整性**
*对于任何*实验运行，所有影响结果的超参数和配置应该被完整记录
**验证: 需求 8.1**

**属性 18: 模型检查点一致性**
*对于任何*保存的模型检查点，加载后的模型应该与保存前的模型行为完全一致
**验证: 需求 8.4**

**属性 19: 实验报告准确性**
*对于任何*生成的实验报告，其中的数值和统计信息应该与实际实验结果完全匹配
**验证: 需求 8.3**

**属性 20: 版本控制追踪**
*对于任何*实验，相关的代码版本、数据版本和模型版本应该被准确记录
**验证: 需求 8.4**

### 数据质量和鲁棒性属性

**属性 21: 噪声数据处理**
*对于任何*包含噪声或格式错误的输入数据，模型应该能够优雅处理而不崩溃
**验证: 需求 5.3**

**属性 22: 边界情况处理**
*对于任何*极短或极长的文章，模型应该能够给出合理的预测结果
**验证: 需求 2.2, 2.3**

**属性 23: 类别平衡敏感性**
*对于任何*不同类别分布的数据集，模型性能不应该严重偏向多数类别
**验证: 需求 7.3**

**属性 24: 领域适应性**
*对于任何*不同新闻领域的文章，模型应该保持相对稳定的检测性能
**���证: 需求 7.3**

**属性 25: 对抗样本鲁棒性**
*对于任何*轻微修改的对抗样本，模型的预测不应该发生剧烈变化
**验证: 需求 7.4**

### 弱监督学习属性

**属性 26: 标签质量评估有效性**
*对于任何*伪标签数据集，质量评估算法应该能够识别低质量和高质量的标签
**验证: 需求 5.3**

**属性 27: 置信度加权一致性**
*对于任何*置信度加权的损失函数，高质量标签应该获得更高的训练权重
**验证: 需求 5.3**

**属性 28: 自训练改进性**
*对于任何*自训练迭代，新生成的伪标签质量应该不低于原始伪标签
**验证: 需求 7.4**

**属性 29: 一致性正则化有效性**
*对于任何*相似的输入文章，模型预测应该保持一致性
**验证: 需求 7.4**

**属性 30: 弱监督收敛性**
*对于任何*弱监督训练过程，模型应该在合理的迭代次数内收敛到稳定性能
**验证: 需求 7.2**

## 错误处理

### 模型训练和推理错误处理

1. **数据相关错误**
   - 文本编码错误：尝试多种编码方式，记录失败样本
   - 文章长度异常：自动截断或分段处理
   - 标签缺失：跳过该样本并记录

2. **模型相关错误**
   - 内存溢出：自动减小批大小并重试
   - 梯度爆炸：应用梯度裁剪和学习率调整
   - 收敛失败：调整学习率或优化器参数

3. **实验管理错误**
   - 检查点损坏：回退到上一个有效检查点
   - 配置冲突：验证配置一致性并报告错误
   - 资源不足：优雅降级或暂停实验

### 错误恢复和日志机制

```python
class ModelErrorHandler:
    def handle_training_error(self, error: TrainingError, context: TrainingContext) -> RecoveryAction
    def handle_inference_error(self, error: InferenceError, sample: Article) -> Optional[BiasResult]
    def handle_data_error(self, error: DataError, batch: List[Article]) -> List[Article]
    def log_experiment_error(self, error: Exception, experiment_id: str) -> None
    def create_error_report(self, errors: List[Exception]) -> ErrorReport
```

## 测试策略

### 双重测试方法

模型开发将采用单元测试和基于属性的测试相结合的方法：

**单元测试**覆盖：
- 模型组件的具体功能测试
- 数据预处理的边界情况
- 训练和推理流程的集成测试
- 性能基准和回归测试

**基于属性的测试**覆盖：
- 模型输出的数学属性验证
- 不同输入条件下的行为一致性
- 训练过程的收敛性和稳定性
- 实验可重现性验证

### 测试框架配置

- **属性测试库**: Hypothesis (Python)
- **最小迭代次数**: 100次
- **测试数据生成**: 
  - 随机长度的新闻文章（100-5000词）
  - 不同偏见程度的合成文本
  - 各种格式和编码的输入数据
  - 边界情况（空文本、超长文本等）

每个基于属性的测试都将使用以下格式标记：
```python
# **Feature: news-bias-analysis, Property X: [property description]**
```

### 模型验证策略

1. **交叉验证**: 5折交叉验证确保模型泛化能力
2. **消融研究**: 验证Longformer和bias_detector融合的有效性
3. **基准对比**: 与现有偏见检测模型进行性能对比
4. **人工评估**: 专家标注样本验证模型预测质量
5. **鲁棒性测试**: 对抗样本和噪声数据测试
6. **计算效率测试**: 推理速度和内存使用优化验证

### 实验设计原则

1. **控制变量**: 每次实验只改变一个关键参数
2. **统计显著性**: 使用适当的统计测试验证结果
3. **可重现性**: 固定随机种子和详细记录实验配置
4. **多指标评估**: 不仅关注准确率，还包括精确率、召回率、F1分数
5. **错误分析**: 深入分析模型失败案例和改进方向