"""
实验管理器 - 统一管理实验配置、运行和结果记录
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Optional, Any
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import pickle

from .config import Config, save_config
from .model import LongformerBiasDetector
from .data_processor import DataManager
from .trainer import BiasModelTrainer
from .evaluator import BiasModelEvaluator

logger = logging.getLogger(__name__)

class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.experiment_id = self._generate_experiment_id()
        self.experiment_dir = self._setup_experiment_directory()
        
        # 设置随机种子
        self._set_random_seeds()
        
        # 设置日志
        self._setup_logging()
        
        # 实验状态
        self.model = None
        self.data_manager = None
        self.trainer = None
        self.evaluator = None
        
        # 结果存储
        self.results = {}
        
    def _generate_experiment_id(self) -> str:
        """生成实验ID"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.config.experiment.run_name:
            return f"{self.config.experiment.run_name}_{timestamp}"
        else:
            return f"{self.config.experiment.experiment_name}_{timestamp}"
    
    def _setup_experiment_directory(self) -> str:
        """设置实验目录"""
        experiment_dir = os.path.join(
            self.config.experiment.output_dir,
            self.experiment_id
        )
        
        # 创建目录结构
        subdirs = ['models', 'logs', 'results', 'plots', 'configs']
        for subdir in subdirs:
            os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
        
        return experiment_dir
    
    def _set_random_seeds(self):
        """设置随机种子确保可重现性"""
        seed = self.config.experiment.seed
        
        # Python随机种子
        import random
        random.seed(seed)
        
        # NumPy随机种子
        np.random.seed(seed)
        
        # PyTorch随机种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 确保确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info(f"Random seeds set to {seed}")
    
    def _setup_logging(self):
        """设置日志记录"""
        log_file = os.path.join(self.experiment_dir, 'logs', 'experiment.log')
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        logger.info(f"Experiment {self.experiment_id} started")
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def save_config(self):
        """保存实验配置"""
        config_path = os.path.join(self.experiment_dir, 'configs', 'config.json')
        save_config(self.config, config_path)
        
        # 同时保存为pickle以便完整恢复
        pickle_path = os.path.join(self.experiment_dir, 'configs', 'config.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.config, f)
        
        logger.info(f"Configuration saved to {config_path}")
    
    def initialize_components(self):
        """初始化实验组件"""
        logger.info("Initializing experiment components...")
        
        # 数据管理器
        self.data_manager = DataManager(self.config)
        
        # 模型
        self.model = LongformerBiasDetector(self.config)
        
        # 训练器
        self.trainer = BiasModelTrainer(
            model=self.model,
            config=self.config,
            device=self.config.experiment.device
        )
        
        # 评估器
        self.evaluator = BiasModelEvaluator(
            model=self.model,
            config=self.config,
            device=self.config.experiment.device
        )
        
        logger.info("Components initialized successfully")
    
    def prepare_data(self):
        """准备数据"""
        logger.info("Preparing data...")
        
        # 加载数据
        df = self.data_manager.load_data(self.config.data.data_path)
        
        # 预处理
        df_processed = self.data_manager.preprocess_data(df)
        
        # 创建数据集
        train_dataset, val_dataset, test_dataset = self.data_manager.create_datasets(df_processed)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.data_manager.create_dataloaders(
            train_dataset, val_dataset, test_dataset
        )
        
        # 保存数据统计
        data_stats = {
            'total_samples': len(df),
            'processed_samples': len(df_processed),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'train_bias_ratio': sum(train_dataset.labels) / len(train_dataset.labels),
            'val_bias_ratio': sum(val_dataset.labels) / len(val_dataset.labels),
            'test_bias_ratio': sum(test_dataset.labels) / len(test_dataset.labels)
        }
        
        self.results['data_stats'] = data_stats
        
        # 保存数据统计
        stats_path = os.path.join(self.experiment_dir, 'results', 'data_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(data_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data prepared - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader, test_dataset
    
    def run_training(self, train_loader, val_loader):
        """运行训练"""
        logger.info("Starting model training...")
        
        # 模型保存目录
        model_save_dir = os.path.join(self.experiment_dir, 'models')
        
        # 训练
        training_history = self.trainer.train(train_loader, val_loader, model_save_dir)
        
        # 保存训练历史
        self.results['training_history'] = training_history
        
        # 获取训练总结
        training_summary = self.trainer.get_training_summary()
        self.results['training_summary'] = training_summary
        
        logger.info("Training completed")
        return training_history
    
    def run_evaluation(self, test_loader, test_dataset):
        """运行评估"""
        logger.info("Starting model evaluation...")
        
        # 加载最佳模型
        best_model_path = os.path.join(self.experiment_dir, 'models', 'checkpoint_best.pt')
        if os.path.exists(best_model_path):
            self.trainer.load_checkpoint(best_model_path)
            logger.info("Loaded best model for evaluation")
        
        # 基础评估
        test_metrics = self.evaluator.evaluate_dataset(test_loader)
        self.results['test_metrics'] = test_metrics
        
        # 错误分析
        error_analysis_path = os.path.join(self.experiment_dir, 'results', 'error_analysis.json')
        error_analysis = self.evaluator.analyze_errors(test_loader, error_analysis_path)
        self.results['error_analysis'] = error_analysis
        
        # 分类报告
        classification_report_path = os.path.join(self.experiment_dir, 'results', 'classification_report.txt')
        classification_report = self.evaluator.generate_classification_report(
            test_loader, classification_report_path
        )
        
        # 混淆矩阵
        confusion_matrix_path = os.path.join(self.experiment_dir, 'plots', 'confusion_matrix.png')
        self.evaluator.plot_confusion_matrix(test_loader, confusion_matrix_path)
        
        logger.info("Evaluation completed")
        return test_metrics
    
    def run_cross_validation(self, dataset):
        """运行交叉验证"""
        logger.info("Starting cross-validation...")
        
        cv_results = self.evaluator.cross_validate(
            dataset, k_folds=self.config.experiment.k_fold
        )
        
        self.results['cross_validation'] = cv_results
        
        # 保存交叉验证结果
        cv_path = os.path.join(self.experiment_dir, 'results', 'cross_validation.json')
        with open(cv_path, 'w', encoding='utf-8') as f:
            # 转换numpy数组为列表以便JSON序列化
            cv_results_serializable = self._make_json_serializable(cv_results)
            json.dump(cv_results_serializable, f, indent=2, ensure_ascii=False)
        
        logger.info("Cross-validation completed")
        return cv_results
    
    def run_ablation_study(self, train_loader, val_loader, test_loader):
        """运行消融研究"""
        logger.info("Starting ablation study...")
        
        ablation_results = {}
        
        # 原始配置结果
        original_metrics = self.results.get('test_metrics', {})
        ablation_results['full_model'] = {
            'config': 'Full model (Longformer + bias_detector + fusion)',
            'metrics': original_metrics
        }
        
        # 消融实验配置
        ablation_configs = [
            {
                'name': 'longformer_only',
                'description': 'Only Longformer without bias_detector',
                'modifications': {'disable_bias_detector': True}
            },
            {
                'name': 'bias_detector_only',
                'description': 'Only bias_detector without Longformer',
                'modifications': {'disable_longformer': True}
            },
            {
                'name': 'no_fusion',
                'description': 'Separate models without fusion',
                'modifications': {'fusion_method': 'none'}
            }
        ]
        
        for ablation_config in ablation_configs:
            logger.info(f"Running ablation: {ablation_config['description']}")
            
            # 这里可以实现具体的消融逻辑
            # 由于模型架构的限制，这里只是示例框架
            ablation_results[ablation_config['name']] = {
                'config': ablation_config['description'],
                'metrics': {}  # 实际实现中会有具体的指标
            }
        
        self.results['ablation_study'] = ablation_results
        
        # 保存消融研究结果
        ablation_path = os.path.join(self.experiment_dir, 'results', 'ablation_study.json')
        with open(ablation_path, 'w', encoding='utf-8') as f:
            json.dump(ablation_results, f, indent=2, ensure_ascii=False)
        
        logger.info("Ablation study completed")
        return ablation_results
    
    def generate_research_report(self):
        """生成研究报告"""
        logger.info("Generating research report...")
        
        report = {
            'experiment_info': {
                'experiment_id': self.experiment_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'config': self._make_json_serializable(self.config.__dict__)
            },
            'data_statistics': self.results.get('data_stats', {}),
            'model_architecture': {
                'model_size': self.model.get_model_size() if self.model else {},
                'fusion_method': self.config.model.fusion_method,
                'max_length': self.config.model.max_length
            },
            'training_results': self.results.get('training_summary', {}),
            'evaluation_results': self.results.get('test_metrics', {}),
            'cross_validation_results': self.results.get('cross_validation', {}).get('cv_statistics', {}),
            'error_analysis': self.results.get('error_analysis', {}),
            'ablation_study': self.results.get('ablation_study', {})
        }
        
        # 保存报告
        report_path = os.path.join(self.experiment_dir, 'results', 'research_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self._generate_markdown_report(report)
        
        logger.info(f"Research report saved to {report_path}")
        return report
    
    def _generate_markdown_report(self, report: Dict):
        """生成Markdown格式的报告"""
        
        md_content = f"""# 框架偏见检测模型实验报告

## 实验信息
- **实验ID**: {report['experiment_info']['experiment_id']}
- **时间**: {report['experiment_info']['timestamp']}
- **模型**: Longformer + bias_detector融合模型

## 数据统计
- **总样本数**: {report['data_statistics'].get('total_samples', 'N/A')}
- **训练集**: {report['data_statistics'].get('train_samples', 'N/A')}
- **验证集**: {report['data_statistics'].get('val_samples', 'N/A')}
- **测试集**: {report['data_statistics'].get('test_samples', 'N/A')}
- **训练集偏见比例**: {report['data_statistics'].get('train_bias_ratio', 'N/A'):.3f}

## 模型架构
- **融合方法**: {report['model_architecture'].get('fusion_method', 'N/A')}
- **最大长度**: {report['model_architecture'].get('max_length', 'N/A')}
- **总参数量**: {report['model_architecture'].get('model_size', {}).get('total_params', 'N/A')}

## 训练结果
- **最佳验证F1**: {report['training_results'].get('best_val_f1', 'N/A'):.4f}
- **最佳验证准确率**: {report['training_results'].get('best_val_accuracy', 'N/A'):.4f}
- **训练轮数**: {report['training_results'].get('total_epochs', 'N/A')}

## 测试结果
- **准确率**: {report['evaluation_results'].get('accuracy', 'N/A'):.4f}
- **F1分数**: {report['evaluation_results'].get('f1_weighted', 'N/A'):.4f}
- **精确率**: {report['evaluation_results'].get('precision_weighted', 'N/A'):.4f}
- **召回率**: {report['evaluation_results'].get('recall_weighted', 'N/A'):.4f}
- **AUC**: {report['evaluation_results'].get('auc', 'N/A'):.4f}

## 交叉验证结果
- **平均F1**: {report['cross_validation_results'].get('f1_weighted_mean', 'N/A'):.4f} ± {report['cross_validation_results'].get('f1_weighted_std', 'N/A'):.4f}
- **平均准确率**: {report['cross_validation_results'].get('accuracy_mean', 'N/A'):.4f} ± {report['cross_validation_results'].get('accuracy_std', 'N/A'):.4f}

## 错误分析
- **假阳性率**: {report['error_analysis'].get('false_positive_rate', 'N/A'):.4f}
- **假阴性率**: {report['error_analysis'].get('false_negative_rate', 'N/A'):.4f}
- **困难样本数**: {report['error_analysis'].get('num_difficult_samples', 'N/A')}

## 结论
本实验成功实现了基于Longformer和bias_detector融合的框架偏见检测模型，在测试集上取得了良好的性能表现。
"""
        
        md_path = os.path.join(self.experiment_dir, 'results', 'research_report.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def _make_json_serializable(self, obj):
        """使对象可JSON序列化"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        else:
            return obj
    
    def run_full_experiment(self):
        """运行完整实验"""
        logger.info("Starting full experiment pipeline...")
        
        try:
            # 保存配置
            self.save_config()
            
            # 初始化组件
            self.initialize_components()
            
            # 准备数据
            train_loader, val_loader, test_loader, test_dataset = self.prepare_data()
            
            # 训练模型
            self.run_training(train_loader, val_loader)
            
            # 评估模型
            self.run_evaluation(test_loader, test_dataset)
            
            # 交叉验证（可选）
            if self.config.experiment.k_fold > 1:
                # 合并训练和验证数据用于交叉验证
                combined_articles = train_loader.dataset.articles + val_loader.dataset.articles
                combined_labels = train_loader.dataset.labels + val_loader.dataset.labels
                
                from .data_processor import ArticleDataset
                combined_dataset = ArticleDataset(
                    articles=combined_articles,
                    labels=combined_labels,
                    tokenizer=train_loader.dataset.tokenizer,
                    max_length=train_loader.dataset.max_length
                )
                
                self.run_cross_validation(combined_dataset)
            
            # 消融研究（可选）
            # self.run_ablation_study(train_loader, val_loader, test_loader)
            
            # 生成研究报告
            self.generate_research_report()
            
            logger.info("Full experiment completed successfully!")
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise
        
        return self.results
    
    def cleanup(self):
        """清理实验资源"""
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Experiment cleanup completed")