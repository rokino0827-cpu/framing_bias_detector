"""
核心模型实现 - 结合Longformer和bias_detector的框架偏见检测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    LongformerForSequenceClassification, 
    AutoModelForSequenceClassification,
    LongformerConfig,
    AutoConfig
)
from typing import Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class BiasFeatureFusion(nn.Module):
    """偏见特征融合模块"""
    
    def __init__(self, 
                 longformer_hidden_size: int = 768,
                 bias_detector_hidden_size: int = 768,
                 fusion_method: str = "weighted_sum",
                 dropout: float = 0.1):
        super().__init__()
        
        self.fusion_method = fusion_method
        self.dropout = nn.Dropout(dropout)
        
        if fusion_method == "concat":
            self.fusion_layer = nn.Linear(
                longformer_hidden_size + bias_detector_hidden_size, 
                longformer_hidden_size
            )
        elif fusion_method == "weighted_sum":
            self.longformer_weight = nn.Parameter(torch.tensor(0.5))
            self.bias_detector_weight = nn.Parameter(torch.tensor(0.5))
        elif fusion_method == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=longformer_hidden_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.layer_norm = nn.LayerNorm(longformer_hidden_size)
        
    def forward(self, 
                longformer_features: torch.Tensor,
                bias_detector_features: torch.Tensor) -> torch.Tensor:
        """
        融合两个模型的特征
        
        Args:
            longformer_features: Longformer的[CLS]特征 [batch_size, hidden_size]
            bias_detector_features: bias_detector的[CLS]特征 [batch_size, hidden_size]
            
        Returns:
            融合后的特征 [batch_size, hidden_size]
        """
        
        if self.fusion_method == "concat":
            # 拼接后通过线性层
            fused = torch.cat([longformer_features, bias_detector_features], dim=-1)
            fused = self.fusion_layer(fused)
            
        elif self.fusion_method == "weighted_sum":
            # 加权求和
            weights = F.softmax(torch.stack([self.longformer_weight, self.bias_detector_weight]), dim=0)
            fused = weights[0] * longformer_features + weights[1] * bias_detector_features
            
        elif self.fusion_method == "attention":
            # 注意力融合
            # 将两个特征作为query和key/value
            query = longformer_features.unsqueeze(1)  # [batch_size, 1, hidden_size]
            key_value = torch.stack([longformer_features, bias_detector_features], dim=1)  # [batch_size, 2, hidden_size]
            
            attended, _ = self.attention(query, key_value, key_value)
            fused = self.layer_norm(attended.squeeze(1) + longformer_features)
        
        return self.dropout(fused)

class FrameworkBiasClassifier(nn.Module):
    """框架偏见分类器"""
    
    def __init__(self, 
                 hidden_size: int = 768,
                 num_labels: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # 置信度估计器
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分类和置信度估计
        
        Args:
            features: 融合后的特征 [batch_size, hidden_size]
            
        Returns:
            logits: 分类logits [batch_size, num_labels]
            confidence: 置信度分数 [batch_size, 1]
        """
        
        # 分类
        pooled_output = self.dropout(features)
        pooled_output = self.dense(pooled_output)
        pooled_output = torch.tanh(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # 置信度估计
        confidence = self.confidence_estimator(features)
        
        return logits, confidence

class LongformerBiasDetector(nn.Module):
    """
    结合Longformer和bias_detector的框架偏见检测模型
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.num_labels = config.model.num_labels
        
        # 加载Longformer模型
        logger.info(f"Loading Longformer model: {config.model.longformer_model}")
        self.longformer = LongformerForSequenceClassification.from_pretrained(
            config.model.longformer_model,
            num_labels=config.model.num_labels,
            ignore_mismatched_sizes=True
        )
        
        # 加载bias_detector模型
        logger.info(f"Loading bias_detector model: {config.model.bias_detector_model}")
        self.bias_detector = AutoModelForSequenceClassification.from_pretrained(
            config.model.bias_detector_model,
            num_labels=config.model.num_labels,
            ignore_mismatched_sizes=True
        )
        
        # 冻结bias_detector的部分参数（可选）
        # self._freeze_bias_detector_layers()
        
        # 特征融合模块
        self.feature_fusion = BiasFeatureFusion(
            longformer_hidden_size=self.longformer.config.hidden_size,
            bias_detector_hidden_size=self.bias_detector.config.hidden_size,
            fusion_method=config.model.fusion_method,
            dropout=config.model.fusion_dropout
        )
        
        # 框架偏见分类器
        self.framework_classifier = FrameworkBiasClassifier(
            hidden_size=self.longformer.config.hidden_size,
            num_labels=config.model.num_labels,
            dropout=config.training.dropout
        )
        
        # 替换原始分类头
        self.longformer.classifier = nn.Identity()
        self.bias_detector.classifier = nn.Identity()
        
    def _freeze_bias_detector_layers(self, freeze_layers: int = 6):
        """冻结bias_detector的前几层"""
        for i, layer in enumerate(self.bias_detector.roberta.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                    
    def _prepare_longformer_inputs(self, 
                                   input_ids: torch.Tensor,
                                   attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """准备Longformer的输入，包括全局注意力掩码"""
        
        batch_size, seq_len = input_ids.shape
        
        # 创建全局注意力掩码
        global_attention_mask = torch.zeros_like(attention_mask)
        
        # 为指定位置设置全局注意力
        for pos in self.config.model.global_attention_positions:
            if pos < seq_len:
                global_attention_mask[:, pos] = 1
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'global_attention_mask': global_attention_mask
        }
    
    def _truncate_for_bias_detector(self, 
                                    input_ids: torch.Tensor,
                                    attention_mask: torch.Tensor,
                                    max_length: int = 512) -> Dict[str, torch.Tensor]:
        """为bias_detector截断输入到512长度"""
        
        if input_ids.size(1) <= max_length:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        
        # 保留开头和结尾的token
        start_tokens = max_length // 2
        end_tokens = max_length - start_tokens
        
        truncated_input_ids = torch.cat([
            input_ids[:, :start_tokens],
            input_ids[:, -end_tokens:]
        ], dim=1)
        
        truncated_attention_mask = torch.cat([
            attention_mask[:, :start_tokens],
            attention_mask[:, -end_tokens:]
        ], dim=1)
        
        return {
            'input_ids': truncated_input_ids,
            'attention_mask': truncated_attention_mask
        }
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Union[Dict, Tuple]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ids [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签 [batch_size] (可选)
            return_dict: 是否返回字典格式
            
        Returns:
            包含logits, loss, confidence等的字典或元组
        """
        
        # Longformer前向传播
        longformer_inputs = self._prepare_longformer_inputs(input_ids, attention_mask)
        longformer_outputs = self.longformer(**longformer_inputs, output_hidden_states=True)
        longformer_features = longformer_outputs.hidden_states[-1][:, 0]  # [CLS] token
        
        # bias_detector前向传播
        bias_detector_inputs = self._truncate_for_bias_detector(input_ids, attention_mask)
        bias_detector_outputs = self.bias_detector(**bias_detector_inputs, output_hidden_states=True)
        bias_detector_features = bias_detector_outputs.hidden_states[-1][:, 0]  # [CLS] token
        
        # 特征融合
        fused_features = self.feature_fusion(longformer_features, bias_detector_features)
        
        # 框架偏见分类
        logits, confidence = self.framework_classifier(fused_features)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'confidence': confidence,
                'longformer_features': longformer_features,
                'bias_detector_features': bias_detector_features,
                'fused_features': fused_features
            }
        else:
            outputs = (logits, confidence)
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs
    
    def predict_bias(self, 
                     input_ids: torch.Tensor,
                     attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测偏见
        
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            
        Returns:
            predictions: 预测标签 [batch_size]
            probabilities: 预测概率 [batch_size, num_labels]
            confidence: 置信度 [batch_size]
        """
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, return_dict=True)
            
            logits = outputs['logits']
            confidence = outputs['confidence'].squeeze(-1)
            
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
        return predictions, probabilities, confidence
    
    def get_model_size(self) -> Dict[str, int]:
        """获取模型大小信息"""
        
        longformer_params = sum(p.numel() for p in self.longformer.parameters())
        bias_detector_params = sum(p.numel() for p in self.bias_detector.parameters())
        fusion_params = sum(p.numel() for p in self.feature_fusion.parameters())
        classifier_params = sum(p.numel() for p in self.framework_classifier.parameters())
        
        total_params = longformer_params + bias_detector_params + fusion_params + classifier_params
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'longformer_params': longformer_params,
            'bias_detector_params': bias_detector_params,
            'fusion_params': fusion_params,
            'classifier_params': classifier_params,
            'total_params': total_params,
            'trainable_params': trainable_params
        }