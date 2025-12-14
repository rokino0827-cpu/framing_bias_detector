"""
快速测试脚本 - 验证模型基本功能
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import pandas as pd
from transformers import AutoTokenizer

from framing_bias_detector.config import default_config
from framing_bias_detector.model import LongformerBiasDetector
from framing_bias_detector.data_processor import ArticlePreprocessor
from framing_bias_detector.utils import setup_environment, get_device_info, log_system_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """测试模型加载"""
    logger.info("测试模型加载...")
    
    try:
        # 使用较小的配置进行测试
        config = default_config
        config.model.max_length = 512  # 减小长度以便快速测试
        
        model = LongformerBiasDetector(config)
        logger.info("✓ 模型加载成功")
        
        # 获取模型大小信息
        model_size = model.get_model_size()
        logger.info(f"模型参数统计: {model_size}")
        
        return model
        
    except Exception as e:
        logger.error(f"✗ 模型加载失败: {str(e)}")
        return None

def test_data_preprocessing():
    """测试数据预处理"""
    logger.info("测试数据预处理...")
    
    try:
        # 创建测试数据
        test_data = pd.DataFrame({
            'content': [
                "This is a test article about political bias detection.",
                "Another article discussing media bias and framing effects.",
                "A neutral article about technology and innovation."
            ],
            'publication': ['Test News', 'Example Media', 'Tech Today'],
            'bias_probability': [0.8, 0.6, 0.2],
            'confidence_score': [0.9, 0.7, 0.8]
        })
        
        # 测试预处理器
        preprocessor = ArticlePreprocessor(
            tokenizer_name=default_config.model.longformer_model,
            max_length=default_config.model.max_length
        )
        processed_data = preprocessor.preprocess_dataframe(test_data)
        
        logger.info(f"✓ 数据预处理成功 - 处理了 {len(processed_data)} 条记录")
        return processed_data
        
    except Exception as e:
        logger.error(f"✗ 数据预处理失败: {str(e)}")
        return None

def test_model_forward():
    """测试模型前向传播"""
    logger.info("测试模型前向传播...")
    
    try:
        config = default_config
        config.model.max_length = 512
        
        model = LongformerBiasDetector(config)
        model.eval()
        
        # 创建测试输入
        tokenizer = AutoTokenizer.from_pretrained(config.model.longformer_model)
        
        test_text = "This is a test article for bias detection."
        inputs = tokenizer(
            test_text,
            max_length=config.model.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 前向传播
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
        
        logger.info("✓ 模型前向传播成功")
        logger.info(f"输出形状 - logits: {outputs['logits'].shape}, confidence: {outputs['confidence'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 模型前向传播失败: {str(e)}")
        return False

def test_prediction():
    """测试预测功能"""
    logger.info("测试预测功能...")
    
    try:
        config = default_config
        config.model.max_length = 512
        
        model = LongformerBiasDetector(config)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(config.model.longformer_model)
        
        test_texts = [
            "This article presents a balanced view of the political situation.",
            "The government's policies are clearly biased and unfair to citizens.",
            "Technology companies are revolutionizing the way we work."
        ]
        
        for i, text in enumerate(test_texts):
            inputs = tokenizer(
                text,
                max_length=config.model.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            predictions, probabilities, confidence = model.predict_bias(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            
            logger.info(f"文本 {i+1}:")
            logger.info(f"  预测: {predictions.item()}")
            logger.info(f"  概率: {probabilities[0].tolist()}")
            logger.info(f"  置信度: {confidence.item():.3f}")
        
        logger.info("✓ 预测功能测试成功")
        return True
        
    except Exception as e:
        logger.error(f"✗ 预测功能测试失败: {str(e)}")
        return False

def test_gpu_availability():
    """测试GPU可用性"""
    logger.info("测试GPU可用性...")
    
    device_info = get_device_info()
    
    if device_info['cuda_available']:
        logger.info("✓ CUDA可用")
        logger.info(f"GPU数量: {device_info['device_count']}")
        
        for device in device_info['devices']:
            logger.info(f"  GPU {device['id']}: {device['name']}")
            logger.info(f"    总显存: {device['memory_total'] / 1024**3:.1f} GB")
        
        # 测试GPU上的简单计算
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.mm(x, x.t())
            logger.info("✓ GPU计算测试成功")
        except Exception as e:
            logger.warning(f"GPU计算测试失败: {str(e)}")
        
        return True
    else:
        logger.warning("CUDA不可用，将使用CPU")
        return False

def run_all_tests():
    """运行所有测试"""
    logger.info("开始运行快速测试...")
    
    # 设置环境
    setup_environment()
    log_system_info()
    
    test_results = {}
    
    # 测试GPU
    test_results['gpu'] = test_gpu_availability()
    
    # 测试数据预处理
    test_results['data_preprocessing'] = test_data_preprocessing() is not None
    
    # 测试模型加载
    test_results['model_loading'] = test_model_loading() is not None
    
    # 测试前向传播
    test_results['forward_pass'] = test_model_forward()
    
    # 测试预测
    test_results['prediction'] = test_prediction()
    
    # 总结结果
    logger.info("\n" + "="*50)
    logger.info("测试结果总结:")
    logger.info("="*50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    logger.info("="*50)
    logger.info(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！模型基本功能正常")
        return True
    else:
        logger.warning(f"⚠️  有 {total - passed} 个测试失败，请检查环境配置")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
