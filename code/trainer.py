"""
训练器模块 - 包含模型训练、验证和弱监督学习逻辑
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW
from typing import Dict, List, Optional, Tuple
import logging
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import wandb

from .model import LongformerBiasDetector
from .data_processor import WeakSupervisionManager

logger = logging.getLogger(__name__)

class BiasModelTrainer:
    """偏见检测模型训练器"""
    
    def __init__(self, 
                 model: LongformerBiasDetector,
                 config,
                 device: str = 'cuda'):
        
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        # 弱监督管理器
        self.weak_supervision = WeakSupervisionManager(
            confidence_threshold=config.training.confidence_threshold
        )
        
        # 优化器和调度器
        self.optimizer = None
        self.scheduler = None
        self._setup_optimizer()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = 0.0
        self.patience_counter = 0
        
        # 训练历史
        self.train_history = []
        self.val_history = []
        
        # 实验跟踪
        if config.experiment.use_wandb:
            self._setup_wandb()
    
    def _setup_optimizer(self):
        """设置优化器和学习率调度器"""
        
        # 不同参数组的学习率
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.training.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        # 优化器
        if self.config.training.optimizer.lower() == "adamw":
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.config.training.learning_rate,
                eps=1e-8
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
    
    def _setup_scheduler(self, num_training_steps: int):
        """设置学习率调度器"""
        
        if self.config.training.scheduler.lower() == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.training.warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            self.scheduler = None
    
    def _setup_wandb(self):
        """设置Weights & Biases实验跟踪"""
        
        wandb.init(
            project=self.config.experiment.wandb_project,
            name=self.config.experiment.run_name,
            config={
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__,
                'data': self.config.data.__dict__
            }
        )
        
        # 监控模型
        wandb.watch(self.model, log="all", log_freq=100)
    
    def _compute_loss(self, 
                      batch: Dict[str, torch.Tensor],
                      outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失"""
        
        # 基础分类损失
        classification_loss = outputs['loss']
        
        # 置信度加权损失
        if 'confidence' in batch:
            weighted_loss = self.weak_supervision.weighted_loss(
                outputs['logits'],
                batch['labels'],
                batch['confidence']
            )
        else:
            weighted_loss = classification_loss
        
        # 一致性正则化损失
        consistency_loss = self.weak_supervision.consistency_regularization(
            self.model, batch
        )
        
        # 总损失
        total_loss = (
            weighted_loss + 
            self.config.training.consistency_weight * consistency_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'weighted_loss': weighted_loss,
            'consistency_loss': consistency_loss
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        # 损失累积
        loss_components = {
            'total_loss': 0.0,
            'classification_loss': 0.0,
            'weighted_loss': 0.0,
            'consistency_loss': 0.0
        }
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch}")
        
        for step, batch in enumerate(progress_bar):
            
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
            
            # 计算损失
            losses = self._compute_loss(batch, outputs)
            loss = losses['total_loss']
            
            # 梯度累积
            loss = loss / self.config.training.gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 累积损失
            for key, value in losses.items():
                loss_components[key] += value.item()
            
            total_loss += loss.item() * self.config.training.gradient_accumulation_steps
            total_samples += batch['input_ids'].size(0)
            
            # 梯度更新
            if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.max_grad_norm
                )
                
                # 优化器步骤
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # 日志记录
                if self.global_step % self.config.training.logging_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    log_dict = {
                        'train/loss': total_loss / (step + 1),
                        'train/learning_rate': current_lr,
                        'train/epoch': self.current_epoch,
                        'train/step': self.global_step
                    }
                    
                    # 添加损失组件
                    for key, value in loss_components.items():
                        log_dict[f'train/{key}'] = value / (step + 1)
                    
                    if self.config.experiment.use_wandb:
                        wandb.log(log_dict)
                    
                    progress_bar.set_postfix({
                        'loss': f"{total_loss / (step + 1):.4f}",
                        'lr': f"{current_lr:.2e}"
                    })
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        avg_loss_components = {k: v / len(train_loader) for k, v in loss_components.items()}
        
        metrics = {
            'avg_loss': avg_loss,
            **avg_loss_components,
            'samples': total_samples
        }
        
        self.train_history.append(metrics)
        return metrics
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                
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
                
                # 累积损失
                total_loss += outputs['loss'].item()
                
                # 预测
                predictions, probabilities, confidence = self.model.predict_bias(
                    batch['input_ids'], batch['attention_mask']
                )
                
                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # AUC (如果有概率)
        try:
            auc = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1])
        except:
            auc = 0.0
        
        # 平均置信度
        avg_confidence = np.mean(all_confidences)
        
        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'val_auc': auc,
            'val_confidence': avg_confidence
        }
        
        self.val_history.append(metrics)
        
        # 记录到wandb
        if self.config.experiment.use_wandb:
            wandb.log(metrics)
        
        return metrics
    
    def save_checkpoint(self, 
                        save_path: str,
                        metrics: Dict[str, float],
                        is_best: bool = False):
        """保存检查点"""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_score': self.best_val_score,
            'metrics': metrics,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        # 保存检查点
        torch.save(checkpoint, save_path)
        
        # 保存最佳模型
        if is_best:
            best_path = save_path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_score = checkpoint['best_val_score']
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              save_dir: str) -> Dict[str, List[Dict]]:
        """完整训练流程"""
        
        logger.info("Starting training...")
        logger.info(f"Model parameters: {self.model.get_model_size()}")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置调度器
        num_training_steps = len(train_loader) * self.config.training.num_epochs // self.config.training.gradient_accumulation_steps
        self._setup_scheduler(num_training_steps)
        
        # 训练循环
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            logger.info(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Train Loss: {train_metrics['avg_loss']:.4f}")
            
            # 验证
            if (epoch + 1) % (self.config.training.eval_steps // len(train_loader) + 1) == 0:
                val_metrics = self.evaluate(val_loader)
                logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                           f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
                           f"Val F1: {val_metrics['val_f1']:.4f}")
                
                # 检查是否是最佳模型
                current_score = val_metrics['val_f1']  # 使用F1作为主要指标
                is_best = current_score > self.best_val_score
                
                if is_best:
                    self.best_val_score = current_score
                    self.patience_counter = 0
                    logger.info(f"New best model! F1: {current_score:.4f}")
                else:
                    self.patience_counter += 1
                
                # 保存检查点
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                self.save_checkpoint(checkpoint_path, val_metrics, is_best)
                
                # 早停
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                    break
        
        logger.info("Training completed!")
        logger.info(f"Best validation F1: {self.best_val_score:.4f}")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history
        }
    
    def get_training_summary(self) -> Dict:
        """获取训练总结"""
        
        if not self.train_history or not self.val_history:
            return {}
        
        # 最佳验证指标
        best_val_metrics = max(self.val_history, key=lambda x: x['val_f1'])
        
        # 训练曲线统计
        final_train_loss = self.train_history[-1]['avg_loss']
        final_val_loss = self.val_history[-1]['val_loss']
        
        summary = {
            'total_epochs': len(self.train_history),
            'total_steps': self.global_step,
            'best_val_f1': best_val_metrics['val_f1'],
            'best_val_accuracy': best_val_metrics['val_accuracy'],
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'model_size': self.model.get_model_size()
        }
        
        return summary