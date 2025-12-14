"""
数据处理模块 - 包含文章预处理和弱监督学习管理
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional, Union
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re

logger = logging.getLogger(__name__)

class ArticleDataset(Dataset):
    """新闻文章数据集"""
    
    def __init__(self, 
                 articles: List[str],
                 labels: List[int],
                 tokenizer: AutoTokenizer,
                 max_length: int = 4096,
                 confidence_scores: Optional[List[float]] = None):
        
        self.articles = articles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.confidence_scores = confidence_scores or [1.0] * len(articles)
        
        assert len(articles) == len(labels), "Articles and labels must have same length"
        assert len(articles) == len(self.confidence_scores), "Articles and confidence scores must have same length"
    
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        article = str(self.articles[idx])
        label = int(self.labels[idx])
        confidence = float(self.confidence_scores[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            article,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'confidence': torch.tensor(confidence, dtype=torch.float),
            'article_text': article
        }

class ArticlePreprocessor:
    """文章预处理器"""
    
    def __init__(self, 
                 tokenizer_name: str = "allenai/longformer-base-4096",
                 max_length: int = 4096):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符（保留基本标点）
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']', ' ', text)
        
        # 移除过短的句子
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        text = '. '.join(sentences)
        
        return text.strip()
    
    def validate_article(self, text: str, min_length: int = 100, max_length: int = 8192) -> bool:
        """验证文章是否符合要求"""
        if not text or len(text) < min_length:
            return False
        
        if len(text) > max_length:
            return False
        
        # 检查是否包含足够的实际内容
        word_count = len(text.split())
        if word_count < 20:
            return False
        
        return True
    
    def preprocess_article(self, article: str) -> Dict[str, Union[torch.Tensor, str, bool]]:
        """预处理单篇文章"""
        
        # 清理文本
        cleaned_text = self.clean_text(article)
        
        # 验证文章
        is_valid = self.validate_article(cleaned_text)
        
        if not is_valid:
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'text': cleaned_text,
                'is_valid': False,
                'token_count': 0
            }
        
        # Tokenize
        encoding = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 计算实际token数量
        token_count = (encoding['attention_mask'] == 1).sum().item()
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': cleaned_text,
            'is_valid': True,
            'token_count': token_count
        }

    def preprocess_dataframe(self,
                             df: pd.DataFrame,
                             text_column: str = "content",
                             remove_invalid: bool = True) -> pd.DataFrame:
        """
        批量预处理数据集
        
        Args:
            df: 输入数据
            text_column: 文本所在列名
            remove_invalid: 是否移除无效文本
        
        Returns:
            包含预处理结果的数据框
        """
        if text_column not in df.columns:
            raise ValueError(f"缺少文本列 {text_column}")
        
        records = []
        invalid_count = 0
        
        for _, row in df.iterrows():
            text = row.get(text_column, "")
            processed = self.preprocess_article(str(text))
            
            record = row.to_dict()
            record['clean_text'] = processed['text']
            record['token_count'] = processed['token_count']
            record['is_valid'] = processed['is_valid']
            record['input_ids'] = processed['input_ids'].tolist()
            record['attention_mask'] = processed['attention_mask'].tolist()
            records.append(record)
            
            if not processed['is_valid']:
                invalid_count += 1
        
        if remove_invalid:
            records = [r for r in records if r['is_valid']]
        
        if invalid_count and remove_invalid:
            logger.info(f"移除了 {invalid_count} 条无效样本")
        
        return pd.DataFrame(records)

    def create_attention_masks(self, 
                               input_ids: torch.Tensor,
                               global_attention_positions: List[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建注意力掩码"""
        
        # 基础注意力掩码
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # 全局注意力掩码
        global_attention_mask = torch.zeros_like(attention_mask)
        
        if global_attention_positions:
            for pos in global_attention_positions:
                if pos < input_ids.size(-1):
                    global_attention_mask[:, pos] = 1
        else:
            # 默认为[CLS] token设置全局注意力
            global_attention_mask[:, 0] = 1
        
        return attention_mask, global_attention_mask

class WeakSupervisionManager:
    """弱监督学习管理器"""
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 noise_detection_method: str = "confidence_based"):
        
        self.confidence_threshold = confidence_threshold
        self.noise_detection_method = noise_detection_method
        
    def assess_label_quality(self, 
                             articles: List[str], 
                             pseudo_labels: List[int],
                             pseudo_scores: List[float]) -> List[float]:
        """评估伪标签质量"""
        
        quality_scores = []
        
        for article, label, score in zip(articles, pseudo_labels, pseudo_scores):
            
            # 基于置信度的质量评估
            if self.noise_detection_method == "confidence_based":
                # 高置信度的预测认为质量更高
                if label == 1:  # 偏见标签
                    quality = score if score > 0.5 else 1 - score
                else:  # 无偏见标签
                    quality = 1 - score if score < 0.5 else score
                    
            # 基于文本长度的质量调整
            text_length_factor = min(len(article) / 1000, 1.0)  # 长文本质量可能更高
            quality = quality * (0.7 + 0.3 * text_length_factor)
            
            quality_scores.append(quality)
        
        return quality_scores
    
    def filter_noisy_labels(self, 
                            dataset: ArticleDataset,
                            quality_scores: List[float]) -> ArticleDataset:
        """过滤噪声标签"""
        
        # 选择高质量样本
        high_quality_indices = [
            i for i, score in enumerate(quality_scores) 
            if score >= self.confidence_threshold
        ]
        
        logger.info(f"Filtered {len(dataset) - len(high_quality_indices)} noisy samples")
        logger.info(f"Remaining high-quality samples: {len(high_quality_indices)}")
        
        # 创建过滤后的数据集
        filtered_articles = [dataset.articles[i] for i in high_quality_indices]
        filtered_labels = [dataset.labels[i] for i in high_quality_indices]
        filtered_confidence = [dataset.confidence_scores[i] for i in high_quality_indices]
        
        return ArticleDataset(
            articles=filtered_articles,
            labels=filtered_labels,
            tokenizer=dataset.tokenizer,
            max_length=dataset.max_length,
            confidence_scores=filtered_confidence
        )
    
    def weighted_loss(self, 
                      predictions: torch.Tensor,
                      targets: torch.Tensor,
                      weights: torch.Tensor) -> torch.Tensor:
        """置信度加权损失函数"""
        
        # 基础交叉熵损失
        base_loss = torch.nn.functional.cross_entropy(predictions, targets, reduction='none')
        
        # 加权损失
        weighted_loss = (base_loss * weights).mean()
        
        return weighted_loss
    
    def consistency_regularization(self, 
                                   model: torch.nn.Module,
                                   batch: Dict[str, torch.Tensor],
                                   augmentation_fn=None) -> torch.Tensor:
        """一致性正则化损失"""
        
        if augmentation_fn is None:
            # 简单的dropout作为数据增强
            model.train()
            
            # 原始预测
            outputs1 = model(batch['input_ids'], batch['attention_mask'])
            
            # 第二次预测（不同的dropout）
            outputs2 = model(batch['input_ids'], batch['attention_mask'])
            
            # 一致性损失
            consistency_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(outputs1['logits'], dim=-1),
                torch.nn.functional.softmax(outputs2['logits'], dim=-1),
                reduction='batchmean'
            )
            
            return consistency_loss
        
        return torch.tensor(0.0)
    
    def self_training_iteration(self, 
                                model: torch.nn.Module,
                                unlabeled_data: DataLoader,
                                device: str = 'cuda') -> List[Tuple[str, int, float]]:
        """自训练迭代"""
        
        model.eval()
        pseudo_labeled_data = []
        
        with torch.no_grad():
            for batch in unlabeled_data:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                articles = batch['article_text']
                
                # 预测
                predictions, probabilities, confidence = model.predict_bias(input_ids, attention_mask)
                
                # 选择高置信度的预测作为伪标签
                for i, (pred, prob, conf, article) in enumerate(zip(
                    predictions.cpu().numpy(),
                    probabilities.cpu().numpy(),
                    confidence.cpu().numpy(),
                    articles
                )):
                    if conf > self.confidence_threshold:
                        pseudo_labeled_data.append((article, int(pred), float(conf)))
        
        logger.info(f"Generated {len(pseudo_labeled_data)} pseudo-labeled samples")
        return pseudo_labeled_data

class DataManager:
    """数据管理器 - 统一管理数据加载、预处理和划分"""
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = ArticlePreprocessor(
            tokenizer_name=config.model.longformer_model,
            max_length=config.model.max_length
        )
        self.weak_supervision = WeakSupervisionManager(
            confidence_threshold=config.training.confidence_threshold
        )
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """加载数据"""
        logger.info(f"Loading data from {data_path}")
        
        try:
            df = pd.read_csv(data_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(data_path, encoding='latin-1')
        
        logger.info(f"Loaded {len(df)} samples")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据"""
        logger.info("Preprocessing data...")
        
        # 基础清理
        df = df.dropna(subset=[self.config.data.text_column])
        df = df.copy()
        
        # 清理文本
        df['cleaned_text'] = df[self.config.data.text_column].apply(self.preprocessor.clean_text)
        
        # 过滤无效文章
        valid_mask = df['cleaned_text'].apply(
            lambda x: self.preprocessor.validate_article(
                x, 
                self.config.data.min_text_length,
                self.config.data.max_text_length
            )
        )
        
        df = df[valid_mask].reset_index(drop=True)
        
        # 移除重复项
        if self.config.data.remove_duplicates:
            df = df.drop_duplicates(subset=['cleaned_text']).reset_index(drop=True)
        
        # 限制样本数量
        if self.config.data.max_samples:
            df = df.head(self.config.data.max_samples)
        
        logger.info(f"After preprocessing: {len(df)} samples")
        return df
    
    def create_datasets(self, df: pd.DataFrame) -> Tuple[ArticleDataset, ArticleDataset, ArticleDataset]:
        """创建训练、验证、测试数据集"""
        
        articles = df['cleaned_text'].tolist()
        
        # 处理标签
        if self.config.data.label_column in df.columns:
            labels = df[self.config.data.label_column].tolist()
        else:
            # 从伪标签概率创建二分类标签
            pseudo_scores = df[self.config.data.pseudo_label_column].tolist()
            labels = [1 if score > 0.5 else 0 for score in pseudo_scores]
        
        # 置信度分数
        if self.config.data.confidence_column in df.columns:
            confidence_scores = df[self.config.data.confidence_column].tolist()
        else:
            # 从伪标签概率计算置信度
            pseudo_scores = df[self.config.data.pseudo_label_column].tolist()
            confidence_scores = [
                score if score > 0.5 else 1 - score 
                for score in pseudo_scores
            ]
        
        # 评估标签质量
        quality_scores = self.weak_supervision.assess_label_quality(
            articles, labels, confidence_scores
        )
        
        # 数据划分
        train_articles, temp_articles, train_labels, temp_labels, train_quality, temp_quality = train_test_split(
            articles, labels, quality_scores,
            test_size=1 - self.config.data.train_split,
            random_state=self.config.experiment.seed,
            stratify=labels
        )
        
        val_size = self.config.data.val_split / (self.config.data.val_split + self.config.data.test_split)
        val_articles, test_articles, val_labels, test_labels, val_quality, test_quality = train_test_split(
            temp_articles, temp_labels, temp_quality,
            test_size=1 - val_size,
            random_state=self.config.experiment.seed,
            stratify=temp_labels
        )
        
        # 创建数据集
        train_dataset = ArticleDataset(
            articles=train_articles,
            labels=train_labels,
            tokenizer=self.preprocessor.tokenizer,
            max_length=self.config.model.max_length,
            confidence_scores=train_quality
        )
        
        val_dataset = ArticleDataset(
            articles=val_articles,
            labels=val_labels,
            tokenizer=self.preprocessor.tokenizer,
            max_length=self.config.model.max_length,
            confidence_scores=val_quality
        )
        
        test_dataset = ArticleDataset(
            articles=test_articles,
            labels=test_labels,
            tokenizer=self.preprocessor.tokenizer,
            max_length=self.config.model.max_length,
            confidence_scores=test_quality
        )
        
        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, 
                           train_dataset: ArticleDataset,
                           val_dataset: ArticleDataset,
                           test_dataset: ArticleDataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器"""
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size * 2,  # 验证时可以用更大的batch size
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
