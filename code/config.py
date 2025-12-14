"""
配置文件 - 包含所有模型和训练相关的配置
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch

@dataclass
class ModelConfig:
    """模型配置"""
    # 基础模型配置
    longformer_model: str = "allenai/longformer-base-4096"
    bias_detector_model: str = "himel7/bias-detector"
    max_length: int = 4096
    num_labels: int = 2
    
    # 注意力配置
    attention_window: List[int] = None
    global_attention_positions: List[int] = None
    
    # 融合配置
    fusion_method: str = "weighted_sum"  # "concat", "weighted_sum", "attention"
    fusion_dropout: float = 0.1
    
    def __post_init__(self):
        if self.attention_window is None:
            self.attention_window = [256] * 12  # 12层，每层256窗口大小
        if self.global_attention_positions is None:
            self.global_attention_positions = [0]  # [CLS] token

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # 优化器配置
    optimizer: str = "adamw"
    scheduler: str = "linear"
    max_grad_norm: float = 1.0
    
    # 弱监督学习配置
    confidence_threshold: float = 0.7
    label_smoothing: float = 0.1
    consistency_weight: float = 0.5
    
    # 正则化
    dropout: float = 0.1
    use_gradient_checkpointing: bool = True
    use_fp16: bool = True
    
    # 早停和保存
    early_stopping_patience: int = 3
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100

@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    data_path: str = "E:/Datasets/all-the-news-2-1/all-the-news-2-1_2025-window_bias_scored.csv"
    text_column: str = "content"
    label_column: str = "bias_label"
    source_column: str = "publication"
    
    # 数据处理
    max_samples: Optional[int] = None
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # 数据质量
    min_text_length: int = 100
    max_text_length: int = 8192
    remove_duplicates: bool = True
    
    # 弱监督配置
    pseudo_label_column: str = "bias_probability"
    confidence_column: str = "confidence_score"

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 实验基本信息
    experiment_name: str = "longformer_bias_detector"
    project_name: str = "framing_bias_detection"
    run_name: Optional[str] = None
    
    # 输出路径
    output_dir: str = "./experiments"
    model_save_dir: str = "./models"
    log_dir: str = "./logs"
    
    # 实验跟踪
    use_wandb: bool = True
    wandb_project: str = "framing-bias-detection"
    
    # 随机种子
    seed: int = 42
    
    # 设备配置
    device: str = "auto"  # "auto", "cuda", "cpu"
    num_gpus: int = 1
    
    # 评估配置
    k_fold: int = 5
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Config:
    """主配置类"""
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    experiment: ExperimentConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.experiment is None:
            self.experiment = ExperimentConfig()

# 默认配置实例
default_config = Config()

def load_config(config_path: str) -> Config:
    """从文件加载配置"""
    import json
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    return Config(
        model=ModelConfig(**config_dict.get('model', {})),
        training=TrainingConfig(**config_dict.get('training', {})),
        data=DataConfig(**config_dict.get('data', {})),
        experiment=ExperimentConfig(**config_dict.get('experiment', {}))
    )

def save_config(config: Config, config_path: str):
    """保存配置到文件"""
    import json
    from dataclasses import asdict
    
    config_dict = {
        'model': asdict(config.model),
        'training': asdict(config.training),
        'data': asdict(config.data),
        'experiment': asdict(config.experiment)
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)