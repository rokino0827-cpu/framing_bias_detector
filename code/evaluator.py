"""
评估器模块 - 包含模型评估、交叉验证和性能分析
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

from .model import LongformerBiasDetector
from .data_processor import ArticleDataset, DataManager

logger = logging.getLogger(__name__)

class BiasModelEvaluator:
    """偏见检测模型评估器"""
    
    def __init__(self, 
                 model: LongformerBiasDetector,
                 config,
                 device: str = 'cuda'):
        
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
    def evaluate_dataset(self, 
                        dataloader: DataLoader,
                        return_predictions: bool = False) -> Dict:
        """评估数据集"""
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_confidences = []
        all_losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                
                # 移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    return_dict=True
                )
                
                # 预测
                predictions, probabilities, confidence = self.model.predict_bias(
                    batch['input_ids'], batch['attention_mask']
                )
                
                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())
                all_losses.append(outputs['loss'].item())
        
        # 计算指标
        metrics = self._compute_metrics(
            all_labels, all_predictions, all_probabilities, all_confidences
        )
        
        metrics['avg_loss'] = np.mean(all_losses)
        
        if return_predictions:
            metrics['predictions'] = {
                'labels': all_labels,
                'predictions': all_predictions,
                'probabilities': all_probabilities,
                'confidences': all_confidences
            }
        
        return metrics
    
    def _compute_metrics(self, 
                        labels: List[int],
                        predictions: List[int],
                        probabilities: List[List[float]],
                        confidences: List[float]) -> Dict:
        """计算评估指标"""
        
        # 基础分类指标
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        # 加权平均指标
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # 宏平均指标
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        
        # AUC
        try:
            probabilities_array = np.array(probabilities)
            if probabilities_array.shape[1] == 2:
                auc = roc_auc_score(labels, probabilities_array[:, 1])
            else:
                auc = 0.0
        except:
            auc = 0.0
        
        # 混淆矩阵
        cm = confusion_matrix(labels, predictions)
        
        # 置信度统计
        avg_confidence = np.mean(confidences)
        confidence_by_class = {}
        for class_idx in [0, 1]:
            class_mask = np.array(predictions) == class_idx
            if np.any(class_mask):
                confidence_by_class[f'confidence_class_{class_idx}'] = np.mean(
                    np.array(confidences)[class_mask]
                )
        
        # 正确/错误预测的置信度
        correct_mask = np.array(labels) == np.array(predictions)
        confidence_correct = np.mean(np.array(confidences)[correct_mask]) if np.any(correct_mask) else 0.0
        confidence_incorrect = np.mean(np.array(confidences)[~correct_mask]) if np.any(~correct_mask) else 0.0
        
        metrics = {
            # 基础指标
            'accuracy': accuracy,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'auc': auc,
            
            # 每类指标
            'precision_class_0': precision[0] if len(precision) > 0 else 0.0,
            'precision_class_1': precision[1] if len(precision) > 1 else 0.0,
            'recall_class_0': recall[0] if len(recall) > 0 else 0.0,
            'recall_class_1': recall[1] if len(recall) > 1 else 0.0,
            'f1_class_0': f1[0] if len(f1) > 0 else 0.0,
            'f1_class_1': f1[1] if len(f1) > 1 else 0.0,
            'support_class_0': support[0] if len(support) > 0 else 0,
            'support_class_1': support[1] if len(support) > 1 else 0,
            
            # 置信度指标
            'avg_confidence': avg_confidence,
            'confidence_correct': confidence_correct,
            'confidence_incorrect': confidence_incorrect,
            **confidence_by_class,
            
            # 混淆矩阵
            'confusion_matrix': cm.tolist(),
            
            # 样本数量
            'num_samples': len(labels)
        }
        
        return metrics
    
    def cross_validate(self, 
                      dataset: ArticleDataset,
                      k_folds: int = 5) -> Dict:
        """K折交叉验证"""
        
        logger.info(f"Starting {k_folds}-fold cross validation...")
        
        # 准备数据
        articles = dataset.articles
        labels = dataset.labels
        
        # 分层K折
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.config.experiment.seed)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(articles, labels)):
            
            logger.info(f"Training fold {fold + 1}/{k_folds}")
            
            # 创建折数据集
            train_articles = [articles[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_articles = [articles[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            
            train_dataset_fold = ArticleDataset(
                articles=train_articles,
                labels=train_labels,
                tokenizer=dataset.tokenizer,
                max_length=dataset.max_length
            )
            
            val_dataset_fold = ArticleDataset(
                articles=val_articles,
                labels=val_labels,
                tokenizer=dataset.tokenizer,
                max_length=dataset.max_length
            )
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset_fold,
                batch_size=self.config.training.batch_size,
                shuffle=True
            )
            
            val_loader = DataLoader(
                val_dataset_fold,
                batch_size=self.config.training.batch_size * 2,
                shuffle=False
            )
            
            # 重新初始化模型（每折都用新模型）
            from .trainer import BiasModelTrainer
            trainer = BiasModelTrainer(self.model, self.config, self.device)
            
            # 训练
            training_history = trainer.train(train_loader, val_loader, f"./cv_fold_{fold}")
            
            # 评估
            fold_metrics = self.evaluate_dataset(val_loader)
            fold_metrics['fold'] = fold
            fold_metrics['train_history'] = training_history['train_history']
            fold_metrics['val_history'] = training_history['val_history']
            
            fold_results.append(fold_metrics)
            
            logger.info(f"Fold {fold + 1} - F1: {fold_metrics['f1_weighted']:.4f}, "
                       f"Accuracy: {fold_metrics['accuracy']:.4f}")
        
        # 计算交叉验证统计
        cv_stats = self._compute_cv_statistics(fold_results)
        
        return {
            'fold_results': fold_results,
            'cv_statistics': cv_stats
        }
    
    def _compute_cv_statistics(self, fold_results: List[Dict]) -> Dict:
        """计算交叉验证统计"""
        
        metrics_to_aggregate = [
            'accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted',
            'precision_macro', 'recall_macro', 'f1_macro', 'auc', 'avg_confidence'
        ]
        
        stats = {}
        
        for metric in metrics_to_aggregate:
            values = [fold[metric] for fold in fold_results if metric in fold]
            
            if values:
                stats[f'{metric}_mean'] = np.mean(values)
                stats[f'{metric}_std'] = np.std(values)
                stats[f'{metric}_min'] = np.min(values)
                stats[f'{metric}_max'] = np.max(values)
        
        return stats
    
    def analyze_errors(self, 
                      dataloader: DataLoader,
                      save_path: Optional[str] = None) -> Dict:
        """错误分析"""
        
        logger.info("Performing error analysis...")
        
        # 获取预测结果
        results = self.evaluate_dataset(dataloader, return_predictions=True)
        predictions_data = results['predictions']
        
        labels = np.array(predictions_data['labels'])
        predictions = np.array(predictions_data['predictions'])
        probabilities = np.array(predictions_data['probabilities'])
        confidences = np.array(predictions_data['confidences'])
        
        # 错误分析
        correct_mask = labels == predictions
        
        # 按类别分析错误
        error_analysis = {
            'total_samples': len(labels),
            'correct_predictions': np.sum(correct_mask),
            'incorrect_predictions': np.sum(~correct_mask),
            'accuracy': np.mean(correct_mask)
        }
        
        # 假阳性和假阴性
        fp_mask = (labels == 0) & (predictions == 1)  # 预测为偏见但实际无偏见
        fn_mask = (labels == 1) & (predictions == 0)  # 预测为无偏见但实际有偏见
        
        error_analysis.update({
            'false_positives': np.sum(fp_mask),
            'false_negatives': np.sum(fn_mask),
            'false_positive_rate': np.sum(fp_mask) / np.sum(labels == 0) if np.sum(labels == 0) > 0 else 0,
            'false_negative_rate': np.sum(fn_mask) / np.sum(labels == 1) if np.sum(labels == 1) > 0 else 0
        })
        
        # 置信度分析
        confidence_analysis = {
            'avg_confidence_correct': np.mean(confidences[correct_mask]) if np.any(correct_mask) else 0,
            'avg_confidence_incorrect': np.mean(confidences[~correct_mask]) if np.any(~correct_mask) else 0,
            'avg_confidence_fp': np.mean(confidences[fp_mask]) if np.any(fp_mask) else 0,
            'avg_confidence_fn': np.mean(confidences[fn_mask]) if np.any(fn_mask) else 0
        }
        
        error_analysis.update(confidence_analysis)
        
        # 困难样本识别（低置信度的错误预测）
        difficult_samples = []
        incorrect_indices = np.where(~correct_mask)[0]
        
        for idx in incorrect_indices:
            if confidences[idx] < 0.6:  # 低置信度阈值
                difficult_samples.append({
                    'index': int(idx),
                    'true_label': int(labels[idx]),
                    'predicted_label': int(predictions[idx]),
                    'confidence': float(confidences[idx]),
                    'probability_class_0': float(probabilities[idx][0]),
                    'probability_class_1': float(probabilities[idx][1])
                })
        
        error_analysis['difficult_samples'] = difficult_samples
        error_analysis['num_difficult_samples'] = len(difficult_samples)
        
        # 保存分析结果
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(error_analysis, f, indent=2, ensure_ascii=False)
            logger.info(f"Error analysis saved to {save_path}")
        
        return error_analysis
    
    def generate_classification_report(self, 
                                     dataloader: DataLoader,
                                     save_path: Optional[str] = None) -> str:
        """生成分类报告"""
        
        results = self.evaluate_dataset(dataloader, return_predictions=True)
        predictions_data = results['predictions']
        
        labels = predictions_data['labels']
        predictions = predictions_data['predictions']
        
        # 生成报告
        report = classification_report(
            labels, predictions,
            target_names=['No Bias', 'Biased'],
            digits=4
        )
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Classification report saved to {save_path}")
        
        return report
    
    def plot_confusion_matrix(self, 
                             dataloader: DataLoader,
                             save_path: Optional[str] = None):
        """绘制混淆矩阵"""
        
        results = self.evaluate_dataset(dataloader, return_predictions=True)
        predictions_data = results['predictions']
        
        labels = predictions_data['labels']
        predictions = predictions_data['predictions']
        
        # 计算混淆矩阵
        cm = confusion_matrix(labels, predictions)
        
        # 绘制
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['No Bias', 'Biased'],
            yticklabels=['No Bias', 'Biased']
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def benchmark_comparison(self, 
                           test_dataloader: DataLoader,
                           baseline_results: Optional[Dict] = None) -> Dict:
        """基准对比"""
        
        # 评估当前模型
        current_results = self.evaluate_dataset(test_dataloader)
        
        comparison = {
            'current_model': {
                'accuracy': current_results['accuracy'],
                'f1_weighted': current_results['f1_weighted'],
                'precision_weighted': current_results['precision_weighted'],
                'recall_weighted': current_results['recall_weighted'],
                'auc': current_results['auc']
            }
        }
        
        # 如果有基准结果，计算改进
        if baseline_results:
            comparison['baseline'] = baseline_results
            comparison['improvement'] = {}
            
            for metric in ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'auc']:
                if metric in baseline_results:
                    current_value = current_results[metric]
                    baseline_value = baseline_results[metric]
                    improvement = ((current_value - baseline_value) / baseline_value) * 100
                    comparison['improvement'][f'{metric}_improvement_pct'] = improvement
        
        return comparison