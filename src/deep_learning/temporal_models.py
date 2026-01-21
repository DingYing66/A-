"""
时序模型模块 (Phase 1)

实现 LSTM 和 Transformer 时序选股模型:
- LSTMStockModel: 基于LSTM的时序模型
- TransformerStockModel: 基于Transformer的时序模型
- TemporalStockModel: 统一接口，支持切换backbone

设计原则:
- 兼容 MLModel 接口 (train/predict/select_stocks_ml)
- 支持排序损失 (Listwise/Pairwise)
- GPU 加速 + 混合精度
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base import DeepStockModel, load_deep_config
from .losses import get_loss_function
from .sequence_dataset import FactorSequenceDataset, FactorSequenceBuilder, create_dataloaders

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger

logger = setup_logger(__name__)


# ===================== LSTM 模型 =====================

class LSTMEncoder(nn.Module):
    """LSTM 编码器"""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            [batch, output_dim] - 最后时刻的隐藏状态
        """
        # output: [batch, seq_len, hidden_dim * num_directions]
        # h_n: [num_layers * num_directions, batch, hidden_dim]
        output, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            # 拼接双向最后层的隐藏状态
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]

        return hidden


# ===================== Transformer 模型 =====================

class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer 编码器"""

    def __init__(self,
                 input_dim: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_len: int = 200):
        super().__init__()

        self.d_model = d_model

        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer Norm
        self.norm = nn.LayerNorm(d_model)

        self.output_dim = d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
            mask: 可选的注意力掩码

        Returns:
            [batch, d_model] - 序列表示
        """
        # 投影到 d_model
        x = self.input_proj(x) * math.sqrt(self.d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer 编码
        x = self.transformer(x, mask=mask)

        # 使用最后位置或均值池化
        # 这里使用最后位置（与LSTM对齐）
        x = self.norm(x[:, -1, :])

        return x


# ===================== PatchTST 模型 =====================

class PatchTSTEncoder(nn.Module):
    """
    PatchTST 编码器

    基于论文 "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"

    核心思想:
    - 将时序切分为 patch (补丁)
    - 每个 patch 作为一个 token 输入 Transformer
    - 显著降低序列长度，提升长序列处理效率

    参数:
    - patch_size: 补丁大小 (默认 16)
    - stride: 补丁步长 (默认 8, 允许重叠)
    - d_model: 模型维度
    - nhead: 注意力头数
    - num_layers: Transformer 层数
    """

    def __init__(self,
                 input_dim: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 patch_size: int = 16,
                 stride: int = 8,
                 max_len: int = 200):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.patch_size = patch_size
        self.stride = stride

        # Patch embedding: 将 [patch_size, input_dim] 展平后投影
        self.patch_embedding = nn.Linear(patch_size * input_dim, d_model)

        # 可学习的位置编码
        max_patches = (max_len - patch_size) // stride + 1
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer Norm
        self.norm = nn.LayerNorm(d_model)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.output_dim = d_model

    def _create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入序列切分为 patches

        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            patches: [batch, n_patches, patch_size * input_dim]
        """
        batch_size, seq_len, input_dim = x.shape

        # 使用 unfold 创建滑动窗口
        # x.unfold(dim, size, step) -> [batch, n_patches, input_dim, patch_size]
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        patches = x.unfold(dimension=2, size=self.patch_size, step=self.stride)
        # patches: [batch, input_dim, n_patches, patch_size]

        patches = patches.permute(0, 2, 3, 1)  # [batch, n_patches, patch_size, input_dim]
        patches = patches.reshape(batch_size, -1, self.patch_size * input_dim)
        # patches: [batch, n_patches, patch_size * input_dim]

        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            [batch, d_model] - 序列表示
        """
        batch_size = x.shape[0]

        # 创建 patches
        patches = self._create_patches(x)  # [batch, n_patches, patch_size * input_dim]
        n_patches = patches.shape[1]

        # Patch embedding
        x = self.patch_embedding(patches)  # [batch, n_patches, d_model]

        # 添加位置编码
        x = x + self.pos_embedding[:, :n_patches, :]

        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, 1 + n_patches, d_model]

        # Transformer 编码
        x = self.transformer(x)

        # 使用 CLS token 的输出作为序列表示
        x = self.norm(x[:, 0, :])

        return x


# ===================== Informer 模型 =====================

class ProbSparseAttention(nn.Module):
    """
    ProbSparse 自注意力机制

    基于论文 "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"

    核心思想:
    - 计算每个 query 的"稀疏度"分数
    - 只对 top-k 最"活跃"的 query 计算完整注意力
    - 其他 query 使用均值近似

    参数:
    - factor: 采样因子，控制 top-k 的数量 (k = factor * log(seq_len))
    """

    def __init__(self, d_model: int, nhead: int, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.factor = factor
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def _compute_sparsity_score(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        计算每个 query 的稀疏度分数

        稀疏度分数 = max(QK^T) - mean(QK^T)
        分数越高表示该 query 的注意力分布越尖锐（更有信息量）
        """
        # Q, K: [batch, heads, seq_len, head_dim]
        # 采样 K 来近似
        sample_size = max(1, int(self.factor * math.log(K.shape[2])))
        sample_idx = torch.randint(0, K.shape[2], (sample_size,), device=K.device)
        K_sample = K[:, :, sample_idx, :]  # [batch, heads, sample_size, head_dim]

        # 计算 Q 与采样 K 的点积
        scores = torch.matmul(Q, K_sample.transpose(-2, -1)) * self.scale
        # scores: [batch, heads, seq_len, sample_size]

        # 稀疏度 = max - mean
        sparsity = scores.max(dim=-1).values - scores.mean(dim=-1)
        # sparsity: [batch, heads, seq_len]

        return sparsity

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # 线性投影
        Q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        # Q, K, V: [batch, heads, seq_len, head_dim]

        # 计算稀疏度分数
        sparsity = self._compute_sparsity_score(Q, K)  # [batch, heads, seq_len]

        # 选择 top-k 最活跃的 queries
        k = max(1, int(self.factor * math.log(seq_len)))
        top_k_indices = sparsity.topk(k, dim=-1).indices  # [batch, heads, k]

        # 对 top-k queries 计算完整注意力
        # 收集 top-k queries
        Q_top = torch.gather(
            Q, 2,
            top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        )  # [batch, heads, k, head_dim]

        # 计算注意力
        attn_scores = torch.matmul(Q_top, K.transpose(-2, -1)) * self.scale
        # attn_scores: [batch, heads, k, seq_len]

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        context_top = torch.matmul(attn_weights, V)  # [batch, heads, k, head_dim]

        # 对其他 queries，使用 V 的均值作为近似
        V_mean = V.mean(dim=2, keepdim=True)  # [batch, heads, 1, head_dim]
        context = V_mean.expand(-1, -1, seq_len, -1).clone()  # [batch, heads, seq_len, head_dim]

        # 将 top-k 的真实注意力结果填回
        context.scatter_(
            2,
            top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
            context_top
        )

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)

        return output


class InformerEncoderLayer(nn.Module):
    """Informer 编码器层"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 256,
                 dropout: float = 0.1, factor: int = 5):
        super().__init__()

        self.self_attn = ProbSparseAttention(d_model, nhead, factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN architecture
        x = x + self.self_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DistillingLayer(nn.Module):
    """
    Informer 的蒸馏层

    使用卷积和池化将序列长度减半，提取主要特征
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len // 2, d_model]
        """
        x = x.permute(0, 2, 1)  # [batch, d_model, seq_len]
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len // 2, d_model]
        return x


class InformerEncoder(nn.Module):
    """
    Informer 编码器

    基于论文 "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"

    特点:
    - ProbSparse 自注意力: O(L log L) 复杂度
    - 自注意力蒸馏: 逐层减半序列长度
    - 适合超长序列预测任务
    """

    def __init__(self,
                 input_dim: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 factor: int = 5,
                 distil: bool = True,
                 max_len: int = 200):
        super().__init__()

        self.d_model = d_model
        self.distil = distil

        # 输入嵌入
        self.input_proj = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # 编码器层
        self.layers = nn.ModuleList([
            InformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, factor)
            for _ in range(num_layers)
        ])

        # 蒸馏层 (每两层之间一个)
        if distil:
            self.distil_layers = nn.ModuleList([
                DistillingLayer(d_model)
                for _ in range(num_layers - 1)
            ])

        # 输出层
        self.norm = nn.LayerNorm(d_model)
        self.output_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            [batch, d_model]
        """
        # 输入投影
        x = self.input_proj(x) * math.sqrt(self.d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # 编码器层 + 蒸馏
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # 蒸馏 (除了最后一层)
            if self.distil and i < len(self.layers) - 1:
                x = self.distil_layers[i](x)

        # 使用最后位置作为输出
        x = self.norm(x[:, -1, :])

        return x


# ===================== 选股模型 =====================

class TemporalStockModel(DeepStockModel):
    """
    时序选股模型

    统一接口，支持 LSTM 和 Transformer backbone。
    """

    def __init__(self, config: dict = None, encoder_type: str = 'transformer'):
        """
        Args:
            config: 配置字典
            encoder_type: 'lstm' 或 'transformer'
        """
        super().__init__(config)
        self.encoder_type = encoder_type

        # 模型参数
        self.input_dim = None  # 运行时确定
        self.encoder: Optional[nn.Module] = None
        self.head: Optional[nn.Module] = None

        logger.info(f"初始化时序模型: {encoder_type}")

    def _build_model(self) -> nn.Module:
        """构建完整模型"""
        if self.input_dim is None:
            raise ValueError("input_dim 未设置，请先调用 _prepare_data")

        if self.encoder_type == 'lstm':
            lstm_cfg = self.config.get('lstm', {})
            self.encoder = LSTMEncoder(
                input_dim=self.input_dim,
                hidden_dim=lstm_cfg.get('hidden_dim', 128),
                num_layers=lstm_cfg.get('num_layers', 2),
                dropout=lstm_cfg.get('dropout', 0.3),
                bidirectional=lstm_cfg.get('bidirectional', False),
            )
        elif self.encoder_type == 'transformer':
            trans_cfg = self.config.get('transformer', {})
            self.encoder = TransformerEncoder(
                input_dim=self.input_dim,
                d_model=trans_cfg.get('d_model', 128),
                nhead=trans_cfg.get('nhead', 8),
                num_layers=trans_cfg.get('num_layers', 4),
                dim_feedforward=trans_cfg.get('dim_feedforward', 512),
                dropout=trans_cfg.get('dropout', 0.1),
            )
        elif self.encoder_type == 'patchtst':
            patch_cfg = self.config.get('patchtst', {})
            self.encoder = PatchTSTEncoder(
                input_dim=self.input_dim,
                d_model=patch_cfg.get('d_model', 128),
                nhead=patch_cfg.get('nhead', 8),
                num_layers=patch_cfg.get('num_layers', 3),
                dim_feedforward=patch_cfg.get('dim_feedforward', 256),
                dropout=patch_cfg.get('dropout', 0.1),
                patch_size=patch_cfg.get('patch_size', 16),
                stride=patch_cfg.get('stride', 8),
            )
        elif self.encoder_type == 'informer':
            informer_cfg = self.config.get('informer', {})
            self.encoder = InformerEncoder(
                input_dim=self.input_dim,
                d_model=informer_cfg.get('d_model', 128),
                nhead=informer_cfg.get('nhead', 8),
                num_layers=informer_cfg.get('num_layers', 3),
                dim_feedforward=informer_cfg.get('dim_feedforward', 256),
                dropout=informer_cfg.get('dropout', 0.1),
                factor=informer_cfg.get('factor', 5),
                distil=informer_cfg.get('distil', True),
            )
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")

        # 预测头
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        # 组合模型
        model = nn.Sequential(
            self.encoder,
            self.head,
        )

        return model

    def _prepare_data(self, X: pd.DataFrame = None, y: pd.Series = None,
                      is_train: bool = True,
                      use_cache: bool = True,
                      universe_id: str = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        准备数据 (C3 修复: 支持直接使用传入的数据)

        数据来源优先级:
        1. 如果 use_cache=False 且 X, y 有效，直接使用传入数据
        2. 否则从因子缓存加载（默认行为，用于 CLI 独立训练）

        Args:
            X: 特征DataFrame，必须包含 'code', 'date' 列
            y: 标签Series（训练时需要）
            is_train: 是否训练模式
            use_cache: 是否使用缓存（默认True用于CLI，False用于MLModel整合）
            universe_id: 股票池标识（用于匹配缓存）

        Returns:
            (train_loader, val_loader) 或 (data_loader, None)
        """
        seq_cfg = self.config.get('sequence', {})
        batch_size = seq_cfg.get('batch_size', 256)
        seq_len = seq_cfg.get('seq_len', 60)

        # C3 修复: 当 use_cache=False 且 X/y 有效时，直接使用传入数据
        if not use_cache and X is not None:
            logger.info("使用传入的 X/y 数据构建数据集 (use_cache=False)")
            dataset = self._build_dataset_from_dataframe(X, y, seq_len)

            if dataset is None or len(dataset) == 0:
                logger.warning("从传入数据构建数据集失败，回退到缓存模式")
            else:
                # 设置 input_dim
                if len(dataset) > 0:
                    sample = dataset[0]
                    self.input_dim = sample[0].shape[1]
                    self.feature_cols = getattr(dataset, 'feature_cols', [])
                    logger.info(f"输入维度: {self.input_dim}, 样本数: {len(dataset)}")

                if is_train:
                    # 80/20 划分
                    train_size = int(0.8 * len(dataset))
                    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
                    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
                    return (
                        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    )
                else:
                    return DataLoader(dataset, batch_size=batch_size, shuffle=False), None

        # 默认行为: 从因子缓存加载
        # 获取日期范围
        if X is not None and 'date' in X.columns:
            dates = pd.to_datetime(X['date'])
            start_date = dates.min().strftime('%Y%m%d')
            end_date = dates.max().strftime('%Y%m%d')
        else:
            # 使用配置中的日期
            data_cfg = self.config.get('data', {})
            start_date = data_cfg.get('start_date', '20180101')
            end_date = data_cfg.get('end_date', datetime.now().strftime('%Y%m%d'))

        if is_train:
            # 分割训练/验证 (80/20)
            if X is not None:
                dates = pd.to_datetime(X['date'])
                split_date = dates.quantile(0.8)
            else:
                split_date = pd.Timestamp(start_date) + (pd.Timestamp(end_date) - pd.Timestamp(start_date)) * 0.8

            train_end = split_date.strftime('%Y%m%d')
            val_start = (split_date + pd.Timedelta(days=1)).strftime('%Y%m%d')

            train_loader, val_loader = create_dataloaders(
                self.config,
                train_start=start_date,
                train_end=train_end,
                val_start=val_start,
                val_end=end_date,
                batch_size=batch_size,
                universe_id=universe_id,
            )

            # 设置 input_dim
            if len(train_loader.dataset) > 0:
                sample = train_loader.dataset[0]
                self.input_dim = sample[0].shape[1]
                self.feature_cols = train_loader.dataset.feature_cols
                logger.info(f"输入维度: {self.input_dim}, 样本数: {len(train_loader.dataset)}")

            return train_loader, val_loader
        else:
            # 推理模式
            dataset = FactorSequenceDataset(
                self.config,
                start_date=start_date,
                end_date=end_date,
                mode='test',
            )
            loader = DataLoader(dataset, batch_size=256, shuffle=False)
            return loader, None

    def _build_dataset_from_dataframe(self, X: pd.DataFrame, y: pd.Series,
                                       seq_len: int) -> Optional[torch.utils.data.Dataset]:
        """
        C3 新增: 从 DataFrame 构建序列数据集

        Args:
            X: 特征 DataFrame (需包含 'code', 'date' 列)
            y: 标签 Series
            seq_len: 序列长度

        Returns:
            PyTorch Dataset 或 None
        """
        try:
            if X is None or len(X) == 0:
                return None

            # 确定特征列
            exclude_cols = {'code', 'name', 'date', 'industry', 'label', 'forward_return', 'total_score'}
            feature_cols = [c for c in X.columns if c not in exclude_cols
                           and X[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

            if len(feature_cols) == 0:
                logger.warning("无有效特征列")
                return None

            # 按股票和日期组织数据
            X = X.copy()
            X['date'] = pd.to_datetime(X['date'])
            X = X.sort_values(['code', 'date'])

            # 构建序列
            sequences = []
            labels = []

            for code in X['code'].unique():
                code_data = X[X['code'] == code].sort_values('date')

                if len(code_data) < seq_len:
                    continue

                # 滑动窗口构建序列
                for i in range(seq_len, len(code_data)):
                    seq = code_data.iloc[i-seq_len:i][feature_cols].values
                    sequences.append(seq)

                    # 获取对应标签
                    if y is not None and len(y) > 0:
                        idx = code_data.index[i]
                        if idx in y.index:
                            labels.append(y[idx])
                        else:
                            labels.append(0.0)
                    else:
                        labels.append(0.0)

            if len(sequences) == 0:
                return None

            # 转换为 tensor
            sequences = np.array(sequences, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)

            # 创建 Dataset
            class SimpleDataset(torch.utils.data.Dataset):
                def __init__(self, sequences, labels, feature_cols):
                    self.sequences = torch.from_numpy(sequences)
                    self.labels = torch.from_numpy(labels)
                    self.feature_cols = feature_cols

                def __len__(self):
                    return len(self.sequences)

                def __getitem__(self, idx):
                    return self.sequences[idx], self.labels[idx]

            dataset = SimpleDataset(sequences, labels, feature_cols)
            logger.info(f"从 DataFrame 构建数据集: {len(dataset)} 个样本, {len(feature_cols)} 个特征")
            return dataset

        except Exception as e:
            logger.warning(f"从 DataFrame 构建数据集失败: {e}")
            return None

    def predict(self, factor_df: pd.DataFrame) -> pd.Series:
        """
        预测分数

        Args:
            factor_df: 因子DataFrame，必须包含 'code' 列

        Returns:
            预测分数Series，索引与 factor_df 对齐
        """
        if not self.is_trained or self.model is None:
            logger.warning("模型未训练")
            return pd.Series(index=factor_df.index, dtype=float)

        self.model.eval()

        # 获取目标日期
        if 'date' in factor_df.columns:
            target_date = pd.to_datetime(factor_df['date'].iloc[0])
        else:
            target_date = pd.Timestamp.now()

        # 尝试构建序列
        builder = FactorSequenceBuilder(self.config)
        codes = factor_df['code'].unique().tolist()
        sequences = builder.build_sequences_for_date(target_date, codes)

        if len(sequences) > 0:
            # 使用序列预测
            scores = {}
            with torch.no_grad():
                for code, seq in sequences.items():
                    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                    score = self.model(seq_tensor).item()
                    scores[code] = score

            result = factor_df['code'].map(scores)
            return result

        # 降级模式：无法构建序列时，使用横截面因子直接预测
        logger.info(f"使用降级模式: 横截面因子预测 ({len(factor_df)} 只股票)")

        # 使用模型保存的特征列（确保与训练时一致）
        if hasattr(self, 'feature_cols') and len(self.feature_cols) > 0:
            feature_cols = [c for c in self.feature_cols if c in factor_df.columns]
            missing_cols = set(self.feature_cols) - set(feature_cols)
            if missing_cols:
                logger.warning(f"因子文件缺少特征列: {missing_cols}")
                # 为缺失的列填充0
                factor_df = factor_df.copy()
                for col in missing_cols:
                    factor_df[col] = 0.0
                feature_cols = list(self.feature_cols)
        else:
            # 回退：从因子文件提取特征
            exclude_cols = {'code', 'name', 'date', 'industry', 'label', 'forward_return', 'total_score'}
            feature_cols = [
                c for c in factor_df.columns
                if c not in exclude_cols
                and not c.endswith('_raw')
                and factor_df[c].dtype in [np.float64, np.float32, np.int64]
            ]

        # 检查特征维度是否匹配
        expected_dim = self.input_dim
        actual_dim = len(feature_cols)

        # 如果有 time_gap 特征，需要补充
        add_time_gap = self.config.get('sequence', {}).get('add_time_gap_feature', True)
        if add_time_gap and actual_dim == expected_dim - 1:
            # 添加 time_gap 特征 (默认为0，表示无时间间隔)
            factor_df = factor_df.copy()
            factor_df['_time_gap'] = 0.0
            feature_cols = list(feature_cols) + ['_time_gap']
            actual_dim += 1

        if actual_dim != expected_dim:
            logger.warning(f"特征维度不匹配: 预期 {expected_dim}, 实际 {actual_dim}")
            # 尝试调整维度
            if actual_dim < expected_dim:
                # 补零
                factor_df = factor_df.copy()
                for i in range(expected_dim - actual_dim):
                    col_name = f'_pad_{i}'
                    factor_df[col_name] = 0.0
                    feature_cols = list(feature_cols) + [col_name]
            else:
                # 截断
                feature_cols = feature_cols[:expected_dim]

        # 构建单步序列 (seq_len=1)
        features = factor_df[feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 扩展为序列形式 [batch, seq_len, features]
        seq_len = self.config.get('sequence', {}).get('seq_len', 20)
        # 复制单步特征形成伪序列
        batch_sequences = np.tile(features[:, np.newaxis, :], (1, seq_len, 1))

        # 批量预测
        scores = []
        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(batch_sequences), batch_size):
                batch = batch_sequences[i:i+batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
                batch_scores = self.model(batch_tensor).squeeze().cpu().numpy()
                if batch_scores.ndim == 0:
                    batch_scores = [batch_scores.item()]
                scores.extend(batch_scores)

        return pd.Series(scores, index=factor_df.index)

    def _compute_batch_loss(self, batch: Tuple, criterion: nn.Module) -> torch.Tensor:
        """??batch??"""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # ????
        pred = self.model(x).squeeze()
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

        # ????
        loss = criterion(pred, y)
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("Detected NaN/Inf loss, returning 0.0")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return loss


class LSTMStockModel(TemporalStockModel):
    """LSTM 选股模型（便捷类）"""

    def __init__(self, config: dict = None):
        super().__init__(config, encoder_type='lstm')


class TransformerStockModel(TemporalStockModel):
    """Transformer 选股模型（便捷类）"""

    def __init__(self, config: dict = None):
        super().__init__(config, encoder_type='transformer')


class PatchTSTStockModel(TemporalStockModel):
    """PatchTST 选股模型（便捷类）"""

    def __init__(self, config: dict = None):
        super().__init__(config, encoder_type='patchtst')


class InformerStockModel(TemporalStockModel):
    """Informer 选股模型（便捷类）"""

    def __init__(self, config: dict = None):
        super().__init__(config, encoder_type='informer')


# ===================== 多尺度融合模型 =====================

class MultiScaleTemporalModel(DeepStockModel):
    """
    多尺度时序模型

    同时使用多个时间窗口 (20/60/120 天)，融合不同尺度特征。
    """

    def __init__(self, config: dict = None, scales: List[int] = None):
        super().__init__(config)
        self.scales = scales or [20, 60, 120]
        self.encoders: Dict[int, nn.Module] = {}

    def _build_model(self) -> nn.Module:
        """构建多尺度模型"""
        if self.input_dim is None:
            raise ValueError("input_dim 未设置")

        trans_cfg = self.config.get('transformer', {})

        # 为每个尺度创建编码器
        encoders = nn.ModuleDict()
        for scale in self.scales:
            encoders[str(scale)] = TransformerEncoder(
                input_dim=self.input_dim,
                d_model=trans_cfg.get('d_model', 128) // len(self.scales),
                nhead=max(2, trans_cfg.get('nhead', 8) // len(self.scales)),
                num_layers=max(2, trans_cfg.get('num_layers', 4) // 2),
                dim_feedforward=trans_cfg.get('dim_feedforward', 512) // len(self.scales),
                dropout=trans_cfg.get('dropout', 0.1),
            )

        self.encoders = encoders

        # 融合层
        total_dim = sum(e.output_dim for e in encoders.values())
        fusion = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        return nn.ModuleDict({'encoders': encoders, 'fusion': fusion})

    def forward(self, inputs: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: {scale: tensor[batch, scale, features]}

        Returns:
            [batch, 1]
        """
        embeddings = []
        for scale in self.scales:
            if scale in inputs:
                emb = self.model['encoders'][str(scale)](inputs[scale])
                embeddings.append(emb)

        fused = torch.cat(embeddings, dim=-1)
        return self.model['fusion'](fused)


# ===================== 工具函数 =====================

def get_temporal_model(model_type: str, config: dict = None) -> TemporalStockModel:
    """
    获取时序模型实例

    Args:
        model_type: 'lstm', 'transformer', 'patchtst', 'informer'
        config: 配置字典

    Returns:
        时序模型实例
    """
    config = config or load_deep_config()

    model_type = model_type.lower()

    if model_type == 'lstm':
        return LSTMStockModel(config)
    elif model_type == 'transformer':
        return TransformerStockModel(config)
    elif model_type == 'patchtst':
        return PatchTSTStockModel(config)
    elif model_type == 'informer':
        return InformerStockModel(config)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            "Use 'lstm', 'transformer', 'patchtst', or 'informer'"
        )
