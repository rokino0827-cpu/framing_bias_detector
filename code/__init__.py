"""
框架偏见检测模型包
Framing Bias Detector Package

结合Longformer和bias_detector的新闻框架偏见检测系统
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@example.com"

# 核心模块导入
from .config import Config, ModelConfig, TrainingConfig, DataConfig, ExperimentConfig, default_config
from .model import LongformerBiasDetector, BiasFeatureFusion, FrameworkBiasClassifier
from .data_processor import (
    ArticleDataset, ArticlePreprocessor, WeakSupervisionManager, DataManager
)
from .trainer import BiasModelTrainer
from .evaluator import BiasModelEvaluator
from .experiment_manager import ExperimentManager
from .utils import (
    setup_environment, check_requirements, get_device_info, 
    format_time, format_memory, set_random_seed
)

# 便捷函数
def create_model(config=None):
    """创建模型实例"""
    if config is None:
        config = default_config
    return LongformerBiasDetector(config)

def create_experiment(config=None):
    """创建实验管理器"""
    if config is None:
        config = default_config
    return ExperimentManager(config)

def quick_setup(data_path, experiment_name="bias_detection"):
    """快速设置实验"""
    from .utils import create_experiment_config
    
    config_dict = create_experiment_config(
        data_path=data_path,
        experiment_name=experiment_name
    )
    
    config = Config(
        model=ModelConfig(**config_dict['model']),
        training=TrainingConfig(**config_dict['training']),
        data=DataConfig(**config_dict['data']),
        experiment=ExperimentConfig(**config_dict['experiment'])
    )
    
    return create_experiment(config)

# 包级别配置
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# 版本检查
def check_dependencies():
    """检查依赖包版本"""
    try:
        import torch
        import transformers
        import sklearn
        
        print(f"PyTorch: {torch.__version__}")
        print(f"Transformers: {transformers.__version__}")
        print(f"Scikit-learn: {sklearn.__version__}")
        
        if torch.cuda.is_available():
            print(f"CUDA: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
        else:
            print("CUDA: Not available")
            
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False

# 导出的公共API
__all__ = [
    # 配置
    'Config', 'ModelConfig', 'TrainingConfig', 'DataConfig', 'ExperimentConfig', 'default_config',
    
    # 模型
    'LongformerBiasDetector', 'BiasFeatureFusion', 'FrameworkBiasClassifier',
    
    # 数据处理
    'ArticleDataset', 'ArticlePreprocessor', 'WeakSupervisionManager', 'DataManager',
    
    # 训练和评估
    'BiasModelTrainer', 'BiasModelEvaluator',
    
    # 实验管理
    'ExperimentManager',
    
    # 工具函数
    'setup_environment', 'check_requirements', 'get_device_info', 
    'format_time', 'format_memory', 'set_random_seed',
    
    # 便捷函数
    'create_model', 'create_experiment', 'quick_setup', 'check_dependencies'
]