"""
工具函数 - 环境设置、数据处理和其他辅助功能
"""

import os
import sys
import json
import logging
import subprocess
import pkg_resources
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_environment():
    """设置运行环境"""
    
    # 设置环境变量
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 避免tokenizer警告
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        logger.info(f"CUDA可用 - 设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("CUDA不可用，将使用CPU")
    
    # 设置PyTorch性能优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def check_requirements():
    """检查依赖包是否安装"""
    
    required_packages = [
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'datasets>=2.0.0',
        'scikit-learn>=1.0.0',
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'tqdm>=4.60.0'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            pkg_resources.require(package)
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
        except pkg_resources.VersionConflict as e:
            logger.warning(f"版本冲突: {e}")
    
    if missing_packages:
        logger.error(f"缺少依赖包: {missing_packages}")
        logger.error("请运行: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("所有依赖包检查通过")
    return True

def get_device_info() -> Dict[str, Any]:
    """获取设备信息"""
    
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'pytorch_version': torch.__version__,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'devices': []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_cached': torch.cuda.memory_reserved(i)
            }
            info['devices'].append(device_info)
    
    return info

def format_time(seconds: float) -> str:
    """格式化时间显示"""
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"

def format_memory(bytes_size: int) -> str:
    """格式化内存大小显示"""
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"

def save_json(data: Any, filepath: str, indent: int = 2):
    """保存JSON文件"""
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)

def load_json(filepath: str) -> Any:
    """加载JSON文件"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_dir(directory: str):
    """确保目录存在"""
    os.makedirs(directory, exist_ok=True)

def get_model_memory_usage(model: torch.nn.Module) -> Dict[str, int]:
    """获取模型内存使用情况"""
    
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return {
        'parameters': param_size,
        'buffers': buffer_size,
        'total': param_size + buffer_size
    }

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """统计模型参数数量"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }

def set_random_seed(seed: int):
    """设置随机种子"""
    
    import random
    
    # Python随机种子
    random.seed(seed)
    
    # NumPy随机种子
    np.random.seed(seed)
    
    # PyTorch随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_experiment_config(
    data_path: str,
    experiment_name: str = "longformer_bias_detector",
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    num_epochs: int = 10,
    max_length: int = 4096,
    fusion_method: str = "weighted_sum"
) -> Dict[str, Any]:
    """创建实验配置"""
    
    config = {
        "model": {
            "longformer_model": "allenai/longformer-base-4096",
            "bias_detector_model": "himel7/bias-detector",
            "max_length": max_length,
            "num_labels": 2,
            "fusion_method": fusion_method,
            "fusion_dropout": 0.1
        },
        "training": {
            "batch_size": batch_size,
            "gradient_accumulation_steps": 4,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "confidence_threshold": 0.7,
            "label_smoothing": 0.1,
            "consistency_weight": 0.5,
            "dropout": 0.1,
            "use_gradient_checkpointing": True,
            "use_fp16": True,
            "early_stopping_patience": 3,
            "save_steps": 500,
            "eval_steps": 500,
            "logging_steps": 100
        },
        "data": {
            "data_path": data_path,
            "text_column": "content",
            "label_column": "bias_label",
            "source_column": "publication",
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "min_text_length": 100,
            "max_text_length": 8192,
            "remove_duplicates": True,
            "pseudo_label_column": "bias_probability",
            "confidence_column": "confidence_score"
        },
        "experiment": {
            "experiment_name": experiment_name,
            "project_name": "framing_bias_detection",
            "output_dir": "./experiments",
            "model_save_dir": "./models",
            "log_dir": "./logs",
            "use_wandb": True,
            "wandb_project": "framing-bias-detection",
            "seed": 42,
            "device": "auto",
            "num_gpus": 1,
            "k_fold": 5,
            "metrics": ["accuracy", "precision", "recall", "f1", "auc"]
        }
    }
    
    return config

def validate_data_file(filepath: str) -> Tuple[bool, str]:
    """验证数据文件"""
    
    if not os.path.exists(filepath):
        return False, f"文件不存在: {filepath}"
    
    try:
        # 尝试读取文件头部
        df = pd.read_csv(filepath, nrows=5)
        
        # 检查必要的列
        required_columns = ['content', 'publication']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"缺少必要列: {missing_columns}"
        
        return True, "数据文件验证通过"
        
    except Exception as e:
        return False, f"读取文件失败: {str(e)}"

def get_system_info() -> Dict[str, Any]:
    """获取系统信息"""
    
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
    }
    
    # 添加GPU信息
    info.update(get_device_info())
    
    return info

def log_system_info():
    """记录系统信息"""
    
    info = get_system_info()
    
    logger.info("系统信息:")
    logger.info(f"  平台: {info['platform']}")
    logger.info(f"  Python版本: {info['python_version']}")
    logger.info(f"  CPU核心数: {info['cpu_count']}")
    logger.info(f"  内存总量: {format_memory(info['memory_total'])}")
    logger.info(f"  可用内存: {format_memory(info['memory_available'])}")
    logger.info(f"  磁盘使用率: {info['disk_usage']:.1f}%")
    
    if info['cuda_available']:
        logger.info(f"  CUDA版本: {info['cuda_version']}")
        logger.info(f"  PyTorch版本: {info['pytorch_version']}")
        logger.info(f"  GPU数量: {info['device_count']}")
        
        for device in info['devices']:
            logger.info(f"    GPU {device['id']}: {device['name']}")
            logger.info(f"      总显存: {format_memory(device['memory_total'])}")
            logger.info(f"      已分配: {format_memory(device['memory_allocated'])}")

class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total_steps: int, description: str = "Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = None
    
    def start(self):
        """开始跟踪"""
        import time
        self.start_time = time.time()
        logger.info(f"{self.description} 开始 (总步数: {self.total_steps})")
    
    def update(self, step: int = 1):
        """更新进度"""
        self.current_step += step
        
        if self.start_time:
            import time
            elapsed = time.time() - self.start_time
            progress = self.current_step / self.total_steps
            eta = elapsed / progress - elapsed if progress > 0 else 0
            
            logger.info(f"{self.description}: {self.current_step}/{self.total_steps} "
                       f"({progress*100:.1f}%) - 已用时: {format_time(elapsed)} - "
                       f"预计剩余: {format_time(eta)}")
    
    def finish(self):
        """完成跟踪"""
        if self.start_time:
            import time
            elapsed = time.time() - self.start_time
            logger.info(f"{self.description} 完成 - 总用时: {format_time(elapsed)}")

def cleanup_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU内存已清理")

def get_available_models() -> Dict[str, str]:
    """获取可用的预训练模型"""
    
    models = {
        'longformer': {
            'allenai/longformer-base-4096': 'Longformer基础模型 (4096长度)',
            'allenai/longformer-large-4096': 'Longformer大型模型 (4096长度)'
        },
        'bias_detector': {
            'himel7/bias-detector': '偏见检测器模型',
            'unitary/toxic-bert': '毒性检测BERT模型'
        }
    }
    
    return models