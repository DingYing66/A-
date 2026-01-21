"""
对比学习预训练模块 (Phase 3)

提供因子对比学习预训练能力：
- FactorContrastiveLearning: 因子序列对比学习
- ContrastiveStockModel: 对比学习 + 监督微调
- 数据增强: 时间掩码、特征 Dropout、行业增强

方法参考:
- TS2Vec: Contrastive Learning for Time Series
- SimCLR: Simple Contrastive Learning
- InfoNCE Loss

用法：
    from src.deep_learning.contrastive import FactorContrastiveLearning, ContrastiveStockModel

    # 预训练
    pretrain = FactorContrastiveLearning(config)
    pretrain.pretrain(train_dataset, epochs=50)
    pretrain.save('pretrained_encoder.pt')

    # 微调
    model = ContrastiveStockModel(config, pretrained_path='pretrained_encoder.pt')
    model.train(X, y)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import copy
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from .base import DeepStockModel, load_deep_config
from .temporal_models import TransformerEncoder, LSTMEncoder
from .losses import get_loss_function

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger, get_project_root, ensure_dir

logger = setup_logger(__name__)


# ===================== 数据增强模块 =====================

class TimeSeriesAugmentation:
    """
    时序数据增强

    支持多种增强策略：
    - 时间掩码 (Time Masking)
    - 特征 Dropout
    - 高斯噪声
    - 时间裁剪
    - 特征混洗
    """

    def __init__(self,
                 time_mask_ratio: float = 0.1,
                 feature_dropout: float = 0.1,
                 noise_std: float = 0.01,
                 crop_ratio: float = 0.0,
                 shuffle_ratio: float = 0.0):
        """
        Args:
            time_mask_ratio: 时间步掩码比例
            feature_dropout: 特征 Dropout 比例
            noise_std: 高斯噪声标准差
            crop_ratio: 随机裁剪比例
            shuffle_ratio: 特征混洗比例
        """
        self.time_mask_ratio = time_mask_ratio
        self.feature_dropout = feature_dropout
        self.noise_std = noise_std
        self.crop_ratio = crop_ratio
        self.shuffle_ratio = shuffle_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用数据增强

        Args:
            x: 输入张量 [batch, seq_len, n_features] 或 [seq_len, n_features]

        Returns:
            增强后的张量
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        # 时间掩码
        if self.time_mask_ratio > 0:
            x = self._time_mask(x)

        # 特征 Dropout
        if self.feature_dropout > 0:
            x = self._feature_dropout(x)

        # 高斯噪声
        if self.noise_std > 0:
            x = self._add_noise(x)

        # 时间裁剪
        if self.crop_ratio > 0:
            x = self._random_crop(x)

        if squeeze_back:
            x = x.squeeze(0)

        return x

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        """时间步掩码"""
        batch, seq_len, n_features = x.shape
        mask_len = int(seq_len * self.time_mask_ratio)

        if mask_len == 0:
            return x

        x = x.clone()

        for b in range(batch):
            start = torch.randint(0, seq_len - mask_len + 1, (1,)).item()
            x[b, start:start + mask_len, :] = 0

        return x

    def _feature_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """特征 Dropout"""
        batch, seq_len, n_features = x.shape
        mask = torch.rand(batch, 1, n_features, device=x.device) > self.feature_dropout
        return x * mask.float()

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def _random_crop(self, x: torch.Tensor) -> torch.Tensor:
        """随机时间裁剪"""
        batch, seq_len, n_features = x.shape
        crop_len = int(seq_len * self.crop_ratio)

        if crop_len == 0:
            return x

        start = torch.randint(0, crop_len + 1, (1,)).item()
        end = seq_len - torch.randint(0, crop_len + 1, (1,)).item()

        return x[:, start:end, :]


class IndustryAugmentation:
    """
    行业增强

    利用行业信息构建正样本对：
    - 同行业股票作为正样本
    - 不同行业股票作为负样本
    """

    def __init__(self, industry_map: Dict[str, str] = None):
        """
        Args:
            industry_map: 行业映射 {code: industry}
        """
        self.industry_map = industry_map or {}
        self._build_industry_groups()

    def _build_industry_groups(self):
        """构建行业分组"""
        self.industry_groups: Dict[str, List[str]] = {}
        for code, industry in self.industry_map.items():
            if industry not in self.industry_groups:
                self.industry_groups[industry] = []
            self.industry_groups[industry].append(code)

    def get_positive_sample(self, code: str) -> Optional[str]:
        """获取同行业的正样本"""
        industry = self.industry_map.get(code)
        if industry is None:
            return None

        group = self.industry_groups.get(industry, [])
        candidates = [c for c in group if c != code]

        if len(candidates) == 0:
            return None

        return np.random.choice(candidates)

    def get_negative_samples(self, code: str, n: int = 5) -> List[str]:
        """获取不同行业的负样本"""
        industry = self.industry_map.get(code)
        candidates = []

        for ind, codes in self.industry_groups.items():
            if ind != industry:
                candidates.extend(codes)

        if len(candidates) == 0:
            return []

        n = min(n, len(candidates))
        return list(np.random.choice(candidates, n, replace=False))


# ===================== 对比学习损失 =====================

class InfoNCELoss(nn.Module):
    """
    InfoNCE 对比学习损失

    L = -log(exp(sim(z_i, z_j) / τ) / Σ exp(sim(z_i, z_k) / τ))

    其中 z_j 是正样本，z_k 是负样本
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: 锚点嵌入 [batch, dim]
            z2: 正样本嵌入 [batch, dim]

        Returns:
            InfoNCE 损失
        """
        batch_size = z1.shape[0]
        device = z1.device

        # L2 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 相似度矩阵
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature

        # 正样本在对角线上
        labels = torch.arange(batch_size, device=device)

        # 交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class NTXentLoss(nn.Module):
    """
    NT-Xent 损失 (Normalized Temperature-scaled Cross Entropy)

    SimCLR 使用的损失函数
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: 第一个视图的嵌入 [batch, dim]
            z2: 第二个视图的嵌入 [batch, dim]

        Returns:
            NT-Xent 损失
        """
        batch_size = z1.shape[0]
        device = z1.device

        # L2 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 拼接两个视图
        z = torch.cat([z1, z2], dim=0)  # [2*batch, dim]

        # 相似度矩阵
        sim = torch.mm(z, z.t()) / self.temperature  # [2*batch, 2*batch]

        # 移除对角线（自身相似度）
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float('-inf'))

        # 正样本标签：(i, i+batch) 和 (i+batch, i) 是正样本对
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(batch_size, device=device)
        ])

        loss = F.cross_entropy(sim, labels)

        return loss


class TripletLoss(nn.Module):
    """
    三元组损失

    L = max(0, d(a, p) - d(a, n) + margin)
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: 锚点嵌入 [batch, dim]
            positive: 正样本嵌入 [batch, dim]
            negative: 负样本嵌入 [batch, dim]

        Returns:
            三元组损失
        """
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)

        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()


# ===================== 对比学习预训练 =====================

class ProjectionHead(nn.Module):
    """投影头"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FactorContrastiveLearning(nn.Module):
    """
    因子对比学习预训练

    使用 TS2Vec 风格的时序对比学习：
    1. 对同一序列应用两种不同的数据增强
    2. 通过编码器获取表示
    3. 使用 InfoNCE 损失训练
    """

    def __init__(self, config: dict = None):
        super().__init__()

        self.config = config or {}
        cl_config = self.config.get('contrastive', {})

        # 编码器配置
        encoder_config = self.config.get('transformer', self.config.get('temporal', {}))
        self.input_dim = encoder_config.get('input_dim', 64)
        self.hidden_dim = encoder_config.get('d_model', encoder_config.get('hidden_dim', 128))
        self.output_dim = cl_config.get('proj_dim', 64)

        # 训练配置
        self.pretrain_epochs = cl_config.get('pretrain_epochs', 50)
        self.temperature = cl_config.get('temperature', 0.07)
        self.lr = cl_config.get('lr', 1e-3)

        # 数据增强配置
        aug_config = cl_config.get('augmentation', {})
        self.augmentation = TimeSeriesAugmentation(
            time_mask_ratio=aug_config.get('time_mask_ratio', 0.1),
            feature_dropout=aug_config.get('feature_dropout', 0.1),
            noise_std=aug_config.get('noise_std', 0.01),
        )

        # 损失类型
        self.loss_type = cl_config.get('loss_type', 'infonce')

        # 模型组件（延迟初始化）
        self.encoder = None
        self.projection = None
        self.criterion = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 特征列
        self.feature_cols = []

        logger.info(f"初始化对比学习: hidden={self.hidden_dim}, proj={self.output_dim}, T={self.temperature}")

    def _init_model(self, input_dim: int):
        """初始化模型组件"""
        self.input_dim = input_dim

        # 编码器
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=self.hidden_dim,
            nhead=4,
            num_layers=2,
            dropout=0.1,
        )

        # 投影头
        self.projection = ProjectionHead(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
        )

        # 损失函数
        if self.loss_type == 'infonce':
            self.criterion = InfoNCELoss(temperature=self.temperature)
        elif self.loss_type == 'ntxent':
            self.criterion = NTXentLoss(temperature=self.temperature)
        else:
            self.criterion = InfoNCELoss(temperature=self.temperature)

        # 移动到设备
        self.encoder = self.encoder.to(self.device)
        self.projection = self.projection.to(self.device)
        self.criterion = self.criterion.to(self.device)

    def pretrain(self,
                 dataset: Dataset,
                 epochs: int = None,
                 batch_size: int = 256) -> Dict[str, Any]:
        """
        预训练

        Args:
            dataset: 序列数据集
            epochs: 预训练轮数
            batch_size: 批大小

        Returns:
            训练结果
        """
        epochs = epochs or self.pretrain_epochs

        logger.info("=" * 60)
        logger.info("开始对比学习预训练")
        logger.info("=" * 60)

        # 获取输入维度
        sample = dataset[0]
        if isinstance(sample, tuple):
            seq = sample[0]
        else:
            seq = sample

        input_dim = seq.shape[-1]
        logger.info(f"输入维度: {input_dim}")

        # 初始化模型
        self._init_model(input_dim)

        # 数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        # 优化器
        params = list(self.encoder.parameters()) + list(self.projection.parameters())
        optimizer = AdamW(params, lr=self.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        # 训练循环
        losses = []

        for epoch in range(epochs):
            self.encoder.train()
            self.projection.train()

            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                if isinstance(batch, tuple):
                    sequences = batch[0]
                else:
                    sequences = batch

                sequences = sequences.to(self.device)

                # 生成两个增强视图
                view1 = self.augmentation(sequences)
                view2 = self.augmentation(sequences)

                # 编码
                h1 = self.encoder(view1)  # [batch, hidden_dim]
                h2 = self.encoder(view2)

                # 投影
                z1 = self.projection(h1)
                z2 = self.projection(h2)

                # 计算损失
                loss = self.criterion(z1, z2)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        logger.info(f"预训练完成, 最终损失: {losses[-1]:.6f}")

        return {
            'losses': losses,
            'final_loss': losses[-1],
            'epochs': epochs,
        }

    def get_embeddings(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        获取嵌入表示

        Args:
            sequences: 输入序列 [batch, seq_len, n_features]

        Returns:
            嵌入向量 [batch, hidden_dim]
        """
        self.encoder.eval()
        with torch.no_grad():
            sequences = sequences.to(self.device)
            embeddings = self.encoder(sequences)
        return embeddings

    def save(self, path: str):
        """保存预训练模型"""
        torch.save({
            'encoder_state': self.encoder.state_dict(),
            'projection_state': self.projection.state_dict(),
            'config': self.config,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
        }, path)
        logger.info(f"预训练模型已保存: {path}")

    @classmethod
    def load(cls, path: str) -> 'FactorContrastiveLearning':
        """加载预训练模型"""
        # PyTorch 2.6+ 需要 weights_only=False
        checkpoint = torch.load(path, weights_only=False)

        model = cls(checkpoint['config'])
        model._init_model(checkpoint['input_dim'])

        model.encoder.load_state_dict(checkpoint['encoder_state'])
        model.projection.load_state_dict(checkpoint['projection_state'])

        model.encoder.eval()
        model.projection.eval()

        logger.info(f"预训练模型已加载: {path}")
        return model


# ===================== 对比学习选股模型 =====================

class ContrastiveStockModel(DeepStockModel):
    """
    对比学习 + 监督微调选股模型

    流程：
    1. 加载预训练编码器
    2. 在编码器上添加预测头
    3. 端到端微调或冻结编码器只训练头部
    """

    def __init__(self, config: dict = None, pretrained_path: str = None):
        """
        Args:
            config: 配置字典
            pretrained_path: 预训练模型路径
        """
        super().__init__(config)

        self.pretrained_path = pretrained_path
        cl_config = self.config.get('contrastive', {})

        # 微调配置
        self.freeze_encoder = cl_config.get('freeze_encoder', False)
        self.finetune_lr = cl_config.get('finetune_lr', 1e-4)
        self.encoder_lr_ratio = cl_config.get('encoder_lr_ratio', 0.1)

        # 训练配置
        train_config = self.config.get('training', {})
        self.epochs = train_config.get('epochs', 50)
        self.patience = train_config.get('patience', 10)
        self.batch_size = train_config.get('batch_size', 256)

        # 损失函数
        self.loss_type = train_config.get('loss', 'listwise')

        # 模型组件
        self.pretrained_encoder = None
        self.head = None
        self.loss_fn = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 特征列
        self.feature_cols = []

        logger.info(f"初始化对比学习选股模型: freeze_encoder={self.freeze_encoder}")

    def _init_model(self, input_dim: int, pretrained: FactorContrastiveLearning = None):
        """初始化模型"""
        if pretrained is not None:
            self.pretrained_encoder = pretrained.encoder
            hidden_dim = pretrained.hidden_dim
        else:
            # 从头开始训练
            hidden_dim = self.config.get('transformer', {}).get('d_model', 128)
            self.pretrained_encoder = TransformerEncoder(
                input_dim=input_dim,
                d_model=hidden_dim,
                nhead=4,
                num_layers=2,
            )

        # 冻结编码器
        if self.freeze_encoder and pretrained is not None:
            for param in self.pretrained_encoder.parameters():
                param.requires_grad = False
            logger.info("编码器已冻结")

        # 预测头
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 损失函数
        self.loss_fn = get_loss_function(self.loss_type)

        # 移动到设备
        self.pretrained_encoder = self.pretrained_encoder.to(self.device)
        self.head = self.head.to(self.device)

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """
        训练模型

        Args:
            X: 因子数据
            y: 标签

        Returns:
            训练结果
        """
        from .sequence_dataset import FactorSequenceDataset

        logger.info("=" * 60)
        logger.info("开始训练对比学习选股模型")
        logger.info("=" * 60)

        # 加载预训练模型
        pretrained = None
        if self.pretrained_path and Path(self.pretrained_path).exists():
            pretrained = FactorContrastiveLearning.load(self.pretrained_path)
            logger.info(f"加载预训练模型: {self.pretrained_path}")

        # 确定特征列
        exclude_cols = {'code', 'name', 'date', 'industry', 'label', 'forward_return', 'total_score'}
        self.feature_cols = [c for c in X.columns
                            if c not in exclude_cols and X[c].dtype in [np.float64, np.float32, np.int64]]

        logger.info(f"特征维度: {len(self.feature_cols)}")

        # 初始化模型
        self._init_model(len(self.feature_cols), pretrained)

        # 准备数据
        X = X.copy()
        X['label'] = y.values

        # 按日期构建批次
        dates = sorted(X['date'].unique())
        logger.info(f"训练日期数: {len(dates)}")

        # 构建样本
        samples = []
        for date in dates:
            date_df = X[X['date'] == date]
            for _, row in date_df.iterrows():
                features = row[self.feature_cols].values.astype(np.float32)
                label = row['label']
                if not np.isnan(label):
                    samples.append({
                        'features': features,
                        'label': label,
                    })

        if len(samples) == 0:
            logger.error("无有效训练样本")
            return {'val_loss': float('inf')}

        logger.info(f"训练样本数: {len(samples)}")

        # 优化器（不同学习率）
        if self.freeze_encoder:
            params = self.head.parameters()
        else:
            params = [
                {'params': self.pretrained_encoder.parameters(), 'lr': self.finetune_lr * self.encoder_lr_ratio},
                {'params': self.head.parameters(), 'lr': self.finetune_lr},
            ]

        optimizer = AdamW(params if isinstance(params, list) else [{'params': params}],
                          lr=self.finetune_lr, weight_decay=1e-4)

        # 训练循环
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []

        for epoch in range(self.epochs):
            self.pretrained_encoder.train()
            self.head.train()

            epoch_loss = 0.0
            n_batches = 0

            # 按日期分批训练
            indices = np.random.permutation(len(dates))

            for date_idx in indices:
                date = dates[date_idx]
                date_samples = [s for i, s in enumerate(samples)
                               if X.iloc[i % len(X)]['date'] == date]

                if len(date_samples) < 10:
                    continue

                # 构建批次
                features = torch.tensor(
                    np.array([s['features'] for s in date_samples]),
                    dtype=torch.float32
                ).to(self.device)

                labels = torch.tensor(
                    [s['label'] for s in date_samples],
                    dtype=torch.float32
                ).to(self.device)

                # 添加序列维度 [batch, 1, features] -> 单步序列
                features = features.unsqueeze(1)

                optimizer.zero_grad()

                # 前向传播
                embeddings = self.pretrained_encoder(features)
                predictions = self.head(embeddings).squeeze(-1)

                # 计算损失
                loss = self.loss_fn(predictions, labels)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.pretrained_encoder.parameters()) + list(self.head.parameters()),
                    max_norm=1.0
                )
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_loss)

            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}, Best: {best_loss:.6f}")

            if patience_counter >= self.patience:
                logger.info(f"早停于 Epoch {epoch + 1}")
                break

        logger.info(f"训练完成, 最终损失: {best_loss:.6f}")

        return {
            'val_loss': best_loss,
            'train_losses': train_losses,
            'epochs_trained': epoch + 1,
        }

    def predict(self, factor_df: pd.DataFrame) -> pd.Series:
        """
        预测

        Args:
            factor_df: 因子数据

        Returns:
            预测分数 Series
        """
        if self.pretrained_encoder is None:
            logger.warning("模型未训练")
            return pd.Series(index=factor_df.index, dtype=float)

        self.pretrained_encoder.eval()
        self.head.eval()

        # 提取特征
        features = factor_df[self.feature_cols].values.astype(np.float32)
        features = torch.tensor(features).unsqueeze(1).to(self.device)  # [batch, 1, features]

        with torch.no_grad():
            embeddings = self.pretrained_encoder(features)
            scores = self.head(embeddings).squeeze(-1).cpu().numpy()

        return pd.Series(scores, index=factor_df.index)

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'encoder_state': self.pretrained_encoder.state_dict(),
            'head_state': self.head.state_dict(),
            'config': self.config,
            'feature_cols': self.feature_cols,
        }, path)
        logger.info(f"模型已保存: {path}")

    def load_model(self, path: str):
        """加载模型"""
        # PyTorch 2.6+ 需要 weights_only=False
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.config = checkpoint['config']
        self.feature_cols = checkpoint['feature_cols']

        self._init_model(len(self.feature_cols))

        self.pretrained_encoder.load_state_dict(checkpoint['encoder_state'])
        self.head.load_state_dict(checkpoint['head_state'])

        self.pretrained_encoder.eval()
        self.head.eval()

        logger.info(f"模型已加载: {path}")


# ===================== 工厂函数 =====================

def get_contrastive_model(config: dict = None,
                          pretrained_path: str = None) -> ContrastiveStockModel:
    """
    获取对比学习选股模型

    Args:
        config: 配置字典
        pretrained_path: 预训练模型路径

    Returns:
        ContrastiveStockModel 实例
    """
    if config is None:
        config = load_deep_config()

    return ContrastiveStockModel(config, pretrained_path=pretrained_path)


def pretrain_contrastive(dataset: Dataset,
                         config: dict = None,
                         epochs: int = 50,
                         save_path: str = None) -> FactorContrastiveLearning:
    """
    运行对比学习预训练

    Args:
        dataset: 序列数据集
        config: 配置字典
        epochs: 预训练轮数
        save_path: 保存路径

    Returns:
        预训练模型
    """
    if config is None:
        config = load_deep_config()

    model = FactorContrastiveLearning(config)
    model.pretrain(dataset, epochs=epochs)

    if save_path:
        model.save(save_path)

    return model
