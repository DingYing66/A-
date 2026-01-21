"""
混合专家模型 (Mixture of Experts, Phase 4)

提供基于 MoE 的选股模型:
- LearnedGatingNetwork: 学习式门控网络
- Expert: 专家模型封装
- MixtureOfExpertsModel: MoE 选股模型

特点:
- 多专家架构 (LSTM/Transformer/GNN)
- Top-K 稀疏门控
- 负载均衡正则化
- 市场状态自适应

用法:
    from src.deep_learning.moe_models import MixtureOfExpertsModel, get_moe_model

    model = get_moe_model(config)
    model.train(X, y)
    predictions = model.predict(factor_df)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from .base import DeepStockModel, load_deep_config
from .temporal_models import LSTMEncoder, TransformerEncoder
from .losses import get_loss_function

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger, get_project_root, ensure_dir

logger = setup_logger(__name__)


# ===================== 市场状态特征 =====================

class MarketStateExtractor:
    """
    市场状态特征提取器

    提取用于门控网络的市场状态特征:
    - market_trend: 市场趋势 (指数涨跌幅)
    - market_volatility: 市场波动率
    - market_breadth: 市场广度 (涨跌家数比)
    - sector_rotation: 板块轮动强度
    """

    def __init__(self, lookback: int = 20):
        """
        Args:
            lookback: 回看窗口
        """
        self.lookback = lookback

    def extract(self, factor_df: pd.DataFrame, date: pd.Timestamp = None) -> torch.Tensor:
        """
        提取市场状态特征

        Args:
            factor_df: 因子数据 (单日横截面)
            date: 当前日期

        Returns:
            市场状态张量 [batch_size, n_features]
        """
        features = []
        batch_size = len(factor_df)

        # 1. 市场趋势: 使用均值收益率作为代理
        if 'forward_return' in factor_df.columns:
            mean_return = factor_df['forward_return'].mean()
            trend = np.tanh(mean_return * 10)  # 归一化到 [-1, 1]
        else:
            trend = 0.0
        features.append(trend)

        # 2. 市场波动率: 使用收益率标准差
        if 'forward_return' in factor_df.columns:
            volatility = factor_df['forward_return'].std()
            volatility = np.clip(volatility * 10, 0, 1)  # 归一化
        else:
            volatility = 0.5
        features.append(volatility)

        # 3. 市场广度: 涨跌家数比
        if 'forward_return' in factor_df.columns:
            up_ratio = (factor_df['forward_return'] > 0).mean()
            breadth = up_ratio * 2 - 1  # 归一化到 [-1, 1]
        else:
            breadth = 0.0
        features.append(breadth)

        # 4. 分散度: 使用因子的离散程度
        numeric_cols = factor_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            dispersion = factor_df[numeric_cols].std().mean()
            dispersion = np.clip(dispersion, 0, 1)
        else:
            dispersion = 0.5
        features.append(dispersion)

        # 扩展到 batch_size
        features = torch.tensor(features, dtype=torch.float32)
        features = features.unsqueeze(0).expand(batch_size, -1)

        return features


# ===================== 门控网络 =====================

class LearnedGatingNetwork(nn.Module):
    """
    学习式门控网络

    根据输入特征动态分配专家权重:
    - 支持 Top-K 稀疏门控
    - 支持负载均衡正则化
    - 支持 Noisy Top-K Gating
    """

    def __init__(self,
                 input_dim: int,
                 n_experts: int,
                 hidden_dim: int = 64,
                 top_k: int = 2,
                 noisy_gating: bool = True,
                 noise_std: float = 1.0):
        """
        Args:
            input_dim: 输入特征维度
            n_experts: 专家数量
            hidden_dim: 隐藏层维度
            top_k: 激活的专家数
            noisy_gating: 是否使用噪声门控
            noise_std: 噪声标准差
        """
        super().__init__()

        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.noisy_gating = noisy_gating
        self.noise_std = noise_std

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_experts),
        )

        # 噪声参数 (可学习)
        if noisy_gating:
            self.w_noise = nn.Linear(input_dim, n_experts, bias=False)

        # 专家使用统计
        self.register_buffer('expert_usage', torch.zeros(n_experts))
        self.register_buffer('total_samples', torch.tensor(0.0))

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算门控权重

        Args:
            x: 输入特征 [batch, input_dim]
            training: 是否训练模式

        Returns:
            gates: 门控权重 [batch, n_experts]
            load_balance_loss: 负载均衡损失
        """
        batch_size = x.shape[0]

        # 计算门控 logits
        logits = self.gate(x)  # [batch, n_experts]

        # 添加噪声 (训练时)
        if self.noisy_gating and training:
            noise = torch.randn_like(logits) * self.noise_std
            noise_weight = F.softplus(self.w_noise(x))
            logits = logits + noise * noise_weight

        # Top-K 稀疏门控
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        # 构建稀疏门控矩阵
        gates = torch.zeros(batch_size, self.n_experts, device=x.device)
        gates.scatter_(1, top_k_indices, top_k_gates)

        # 计算负载均衡损失
        load_balance_loss = self._compute_load_balance_loss(gates)

        # 更新专家使用统计
        if training:
            with torch.no_grad():
                self.expert_usage += gates.sum(dim=0)
                self.total_samples += batch_size

        return gates, load_balance_loss

    def _compute_load_balance_loss(self, gates: torch.Tensor) -> torch.Tensor:
        """
        计算负载均衡损失

        鼓励所有专家被均匀使用:
        L_balance = n_experts * Σ(f_i * P_i)
        其中 f_i 是专家 i 的使用频率, P_i 是专家 i 的平均门控值
        """
        # 每个专家的使用频率
        expert_freq = (gates > 0).float().mean(dim=0)  # [n_experts]

        # 每个专家的平均门控值
        expert_gate_mean = gates.mean(dim=0)  # [n_experts]

        # 负载均衡损失
        loss = self.n_experts * (expert_freq * expert_gate_mean).sum()

        return loss

    def get_expert_usage_stats(self) -> Dict[str, float]:
        """获取专家使用统计"""
        if self.total_samples == 0:
            return {}

        usage = self.expert_usage / self.total_samples
        return {f'expert_{i}': usage[i].item() for i in range(self.n_experts)}

    def reset_stats(self):
        """重置统计"""
        self.expert_usage.zero_()
        self.total_samples.zero_()


# ===================== 专家模型 =====================

class Expert(nn.Module):
    """
    专家模型封装

    支持不同类型的编码器作为专家
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 expert_type: str = 'transformer',
                 **kwargs):
        """
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            expert_type: 专家类型 (lstm/transformer/mlp)
        """
        super().__init__()

        self.expert_type = expert_type
        self.output_dim = output_dim

        if expert_type == 'lstm':
            self.encoder = LSTMEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=kwargs.get('num_layers', 2),
                dropout=kwargs.get('dropout', 0.3),
            )
            encoder_output_dim = self.encoder.output_dim

        elif expert_type == 'transformer':
            self.encoder = TransformerEncoder(
                input_dim=input_dim,
                d_model=hidden_dim,
                nhead=kwargs.get('nhead', 4),
                num_layers=kwargs.get('num_layers', 2),
                dropout=kwargs.get('dropout', 0.1),
            )
            encoder_output_dim = hidden_dim

        elif expert_type == 'mlp':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(kwargs.get('dropout', 0.1)),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            encoder_output_dim = hidden_dim

        else:
            raise ValueError(f"未知专家类型: {expert_type}")

        # 输出投影
        self.output_proj = nn.Linear(encoder_output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入 [batch, seq_len, input_dim] 或 [batch, input_dim]

        Returns:
            [batch, output_dim]
        """
        if self.expert_type == 'mlp':
            # MLP: 展平输入
            if x.dim() == 3:
                x = x.mean(dim=1)  # 时间维度平均
            h = self.encoder(x)
        else:
            # 序列模型: 保持输入维度
            if x.dim() == 2:
                x = x.unsqueeze(1)  # 添加序列维度
            h = self.encoder(x)

        return self.output_proj(h)


# ===================== MoE 模型 =====================

class MixtureOfExpertsLayer(nn.Module):
    """
    MoE 层

    将多个专家通过门控网络组合
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 n_experts: int = 4,
                 top_k: int = 2,
                 expert_types: List[str] = None,
                 gating_input_dim: int = 4,
                 noisy_gating: bool = True):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            n_experts: 专家数量
            top_k: 激活的专家数
            expert_types: 专家类型列表
            gating_input_dim: 门控网络输入维度
            noisy_gating: 是否使用噪声门控
        """
        super().__init__()

        self.n_experts = n_experts
        self.output_dim = output_dim

        # 默认专家类型: 混合不同类型
        if expert_types is None:
            expert_types = ['transformer', 'lstm', 'mlp', 'transformer'][:n_experts]

        # 确保专家类型数量匹配
        if len(expert_types) < n_experts:
            expert_types = expert_types * (n_experts // len(expert_types) + 1)
        expert_types = expert_types[:n_experts]

        # 创建专家
        self.experts = nn.ModuleList([
            Expert(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                expert_type=expert_types[i],
            )
            for i in range(n_experts)
        ])

        # 门控网络
        self.gating = LearnedGatingNetwork(
            input_dim=gating_input_dim + output_dim,  # 市场状态 + 特征
            n_experts=n_experts,
            hidden_dim=64,
            top_k=top_k,
            noisy_gating=noisy_gating,
        )

        # 特征压缩 (用于门控输入)
        self.feature_compress = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

        logger.info(f"MoE 层: {n_experts} 专家, Top-{top_k}, 类型={expert_types}")

    def forward(self,
                x: torch.Tensor,
                market_state: torch.Tensor = None,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入特征 [batch, seq_len, input_dim] 或 [batch, input_dim]
            market_state: 市场状态 [batch, gating_input_dim]
            training: 是否训练模式

        Returns:
            output: MoE 输出 [batch, output_dim]
            load_balance_loss: 负载均衡损失
        """
        batch_size = x.shape[0]

        # 压缩特征用于门控
        if x.dim() == 3:
            x_flat = x.mean(dim=1)  # [batch, input_dim]
        else:
            x_flat = x

        feature_compressed = self.feature_compress(x_flat)  # [batch, output_dim]

        # 构建门控输入
        if market_state is None:
            market_state = torch.zeros(batch_size, 4, device=x.device)

        gating_input = torch.cat([market_state, feature_compressed], dim=-1)

        # 计算门控权重
        gates, load_balance_loss = self.gating(gating_input, training=training)

        # 计算每个专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # [batch, output_dim]
            expert_outputs.append(expert_out)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, n_experts, output_dim]

        # 加权组合
        gates = gates.unsqueeze(-1)  # [batch, n_experts, 1]
        output = (expert_outputs * gates).sum(dim=1)  # [batch, output_dim]

        return output, load_balance_loss


class MixtureOfExpertsModel(DeepStockModel):
    """
    混合专家选股模型

    特点:
    - 多专家架构，不同专家擅长不同市场状态
    - 学习式门控，自动分配专家权重
    - Top-K 稀疏激活，提高效率
    - 负载均衡正则化，防止专家退化
    """

    def __init__(self, config: dict = None):
        """
        Args:
            config: 配置字典
        """
        super().__init__(config)

        moe_config = self.config.get('moe', {})

        # MoE 配置
        self.n_experts = moe_config.get('n_experts', 4)
        self.top_k = moe_config.get('top_k', 2)
        self.reg_lambda = moe_config.get('reg_lambda', 0.01)
        self.expert_types = moe_config.get('expert_types', None)

        # 模型配置
        self.hidden_dim = moe_config.get('hidden_dim',
                                         self.config.get('transformer', {}).get('d_model', 128))
        self.output_dim = moe_config.get('output_dim', 64)

        # 训练配置
        train_config = self.config.get('training', {})
        self.epochs = train_config.get('epochs', 100)
        self.lr = train_config.get('lr', 1e-4)
        self.patience = train_config.get('patience', 10)
        self.batch_size = train_config.get('batch_size', 256)
        self.loss_type = train_config.get('loss', 'listwise')

        # 模型组件 (延迟初始化)
        self.moe_layer = None
        self.prediction_head = None
        self.loss_fn = None

        # 市场状态提取器
        self.market_extractor = MarketStateExtractor()

        # 特征列
        self.feature_cols = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"初始化 MoE 模型: {self.n_experts} 专家, Top-{self.top_k}")

    def _init_model(self, input_dim: int):
        """初始化模型"""
        # MoE 层
        self.moe_layer = MixtureOfExpertsLayer(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_experts=self.n_experts,
            top_k=self.top_k,
            expert_types=self.expert_types,
            gating_input_dim=4,  # 市场状态特征数
            noisy_gating=True,
        )

        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim // 2, 1),
        )

        # 损失函数
        self.loss_fn = get_loss_function(self.loss_type)

        # 移动到设备
        self.moe_layer = self.moe_layer.to(self.device)
        self.prediction_head = self.prediction_head.to(self.device)

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """
        训练模型

        Args:
            X: 因子数据
            y: 标签

        Returns:
            训练结果
        """
        logger.info("=" * 60)
        logger.info("开始训练 MoE 选股模型")
        logger.info("=" * 60)

        # 确定特征列
        exclude_cols = {'code', 'name', 'date', 'industry', 'label', 'forward_return', 'total_score'}
        self.feature_cols = [c for c in X.columns
                            if c not in exclude_cols and X[c].dtype in [np.float64, np.float32, np.int64]]

        logger.info(f"特征维度: {len(self.feature_cols)}")
        logger.info(f"样本数: {len(X)}")

        # 初始化模型
        self._init_model(len(self.feature_cols))

        # 准备数据
        X = X.copy()
        X['label'] = y.values

        # 按日期组织数据
        dates = sorted(X['date'].unique())
        logger.info(f"训练日期数: {len(dates)}")

        # 划分训练/验证集 (时间序列划分)
        val_size = max(1, int(len(dates) * 0.2))
        train_dates = dates[:-val_size]
        val_dates = dates[-val_size:]

        train_X = X[X['date'].isin(train_dates)]
        val_X = X[X['date'].isin(val_dates)]

        logger.info(f"训练集: {len(train_X)}, 验证集: {len(val_X)}")

        # 优化器
        params = list(self.moe_layer.parameters()) + list(self.prediction_head.parameters())
        optimizer = AdamW(params, lr=self.lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            # 训练阶段
            self.moe_layer.train()
            self.prediction_head.train()

            epoch_loss = 0.0
            epoch_lb_loss = 0.0
            n_batches = 0

            # 按日期分批训练
            for date in np.random.permutation(train_dates):
                date_df = train_X[train_X['date'] == date]

                if len(date_df) < 10:
                    continue

                # 准备批次数据
                features = torch.tensor(
                    date_df[self.feature_cols].values.astype(np.float32),
                    device=self.device
                )
                features = torch.nan_to_num(features, nan=0.0)

                labels = torch.tensor(
                    date_df['label'].values.astype(np.float32),
                    device=self.device
                )

                # 提取市场状态
                market_state = self.market_extractor.extract(date_df, date)
                market_state = market_state.to(self.device)

                optimizer.zero_grad()

                # 前向传播
                moe_output, lb_loss = self.moe_layer(features, market_state, training=True)
                predictions = self.prediction_head(moe_output).squeeze(-1)

                # 计算损失
                pred_loss = self.loss_fn(predictions, labels)
                total_loss = pred_loss + self.reg_lambda * lb_loss

                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss += pred_loss.item()
                epoch_lb_loss += lb_loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            avg_lb_loss = epoch_lb_loss / max(n_batches, 1)
            train_losses.append(avg_train_loss)

            # 验证阶段
            val_loss = self._evaluate(val_X)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self._save_best_state()
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                # 获取专家使用统计
                usage = self.moe_layer.gating.get_expert_usage_stats()
                usage_str = ', '.join([f'E{i}:{v:.2f}' for i, v in enumerate(usage.values())])

                logger.info(f"Epoch {epoch + 1}/{self.epochs}")
                logger.info(f"  Train: {avg_train_loss:.6f}, Val: {val_loss:.6f}, LB: {avg_lb_loss:.4f}")
                logger.info(f"  专家使用: {usage_str}")

            if patience_counter >= self.patience:
                logger.info(f"早停于 Epoch {epoch + 1}")
                break

        # 加载最佳模型
        self._load_best_state()

        # 最终统计
        usage = self.moe_layer.gating.get_expert_usage_stats()
        logger.info(f"训练完成, 最佳验证损失: {best_val_loss:.6f}")
        logger.info(f"专家使用统计: {usage}")

        return {
            'val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'expert_usage': usage,
            'epochs_trained': epoch + 1,
        }

    def _evaluate(self, val_X: pd.DataFrame) -> float:
        """评估验证集"""
        self.moe_layer.eval()
        self.prediction_head.eval()

        total_loss = 0.0
        n_batches = 0

        dates = val_X['date'].unique()

        with torch.no_grad():
            for date in dates:
                date_df = val_X[val_X['date'] == date]

                if len(date_df) < 10:
                    continue

                features = torch.tensor(
                    date_df[self.feature_cols].values.astype(np.float32),
                    device=self.device
                )
                features = torch.nan_to_num(features, nan=0.0)

                labels = torch.tensor(
                    date_df['label'].values.astype(np.float32),
                    device=self.device
                )

                market_state = self.market_extractor.extract(date_df, date)
                market_state = market_state.to(self.device)

                moe_output, _ = self.moe_layer(features, market_state, training=False)
                predictions = self.prediction_head(moe_output).squeeze(-1)

                loss = self.loss_fn(predictions, labels)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_best_state(self):
        """保存最佳模型状态"""
        self._best_moe_state = copy.deepcopy(self.moe_layer.state_dict())
        self._best_head_state = copy.deepcopy(self.prediction_head.state_dict())

    def _load_best_state(self):
        """加载最佳模型状态"""
        if hasattr(self, '_best_moe_state'):
            self.moe_layer.load_state_dict(self._best_moe_state)
            self.prediction_head.load_state_dict(self._best_head_state)

    def predict(self, factor_df: pd.DataFrame) -> pd.Series:
        """
        预测

        Args:
            factor_df: 因子数据

        Returns:
            预测分数 Series
        """
        if self.moe_layer is None:
            logger.warning("模型未训练")
            return pd.Series(index=factor_df.index, dtype=float)

        self.moe_layer.eval()
        self.prediction_head.eval()

        # 提取特征
        features = factor_df[self.feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)
        features = torch.tensor(features, device=self.device)

        # 提取市场状态
        market_state = self.market_extractor.extract(factor_df)
        market_state = market_state.to(self.device)

        with torch.no_grad():
            moe_output, _ = self.moe_layer(features, market_state, training=False)
            scores = self.prediction_head(moe_output).squeeze(-1).cpu().numpy()

        return pd.Series(scores, index=factor_df.index)

    def get_expert_weights(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        获取每个样本的专家权重

        Args:
            factor_df: 因子数据

        Returns:
            专家权重 DataFrame [n_samples, n_experts]
        """
        if self.moe_layer is None:
            return pd.DataFrame()

        self.moe_layer.eval()

        features = factor_df[self.feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)
        features = torch.tensor(features, device=self.device)

        market_state = self.market_extractor.extract(factor_df)
        market_state = market_state.to(self.device)

        # 获取门控权重
        with torch.no_grad():
            x_flat = features
            feature_compressed = self.moe_layer.feature_compress(x_flat)
            gating_input = torch.cat([market_state, feature_compressed], dim=-1)
            gates, _ = self.moe_layer.gating(gating_input, training=False)

        weights = gates.cpu().numpy()
        columns = [f'expert_{i}' for i in range(self.n_experts)]

        return pd.DataFrame(weights, index=factor_df.index, columns=columns)

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'moe_state': self.moe_layer.state_dict(),
            'head_state': self.prediction_head.state_dict(),
            'config': self.config,
            'feature_cols': self.feature_cols,
            'n_experts': self.n_experts,
            'top_k': self.top_k,
        }, path)
        logger.info(f"MoE 模型已保存: {path}")

    def load_model(self, path: str):
        """加载模型"""
        # PyTorch 2.6+ 需要 weights_only=False
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.config = checkpoint['config']
        self.feature_cols = checkpoint['feature_cols']
        self.n_experts = checkpoint['n_experts']
        self.top_k = checkpoint['top_k']

        self._init_model(len(self.feature_cols))

        self.moe_layer.load_state_dict(checkpoint['moe_state'])
        self.prediction_head.load_state_dict(checkpoint['head_state'])

        self.moe_layer.eval()
        self.prediction_head.eval()

        logger.info(f"MoE 模型已加载: {path}")


# ===================== 工厂函数 =====================

def get_moe_model(config: dict = None) -> MixtureOfExpertsModel:
    """
    获取 MoE 选股模型

    Args:
        config: 配置字典

    Returns:
        MixtureOfExpertsModel 实例
    """
    if config is None:
        config = load_deep_config()

    return MixtureOfExpertsModel(config)


def create_expert_ensemble(config: dict = None,
                           expert_types: List[str] = None) -> MixtureOfExpertsModel:
    """
    创建自定义专家组合的 MoE 模型

    Args:
        config: 配置字典
        expert_types: 专家类型列表

    Returns:
        MixtureOfExpertsModel 实例
    """
    if config is None:
        config = load_deep_config()

    config = copy.deepcopy(config)

    if expert_types is not None:
        config.setdefault('moe', {})['expert_types'] = expert_types
        config['moe']['n_experts'] = len(expert_types)

    return MixtureOfExpertsModel(config)
