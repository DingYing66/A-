"""
深度学习模型基类

提供与 MLModel 兼容的接口，所有深度模型需继承此基类。
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 复用现有工具（只读依赖）
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger, get_project_root, ensure_dir, load_parquet

logger = setup_logger(__name__)


def load_deep_config() -> dict:
    """加载深度学习配置"""
    import yaml
    config_path = get_project_root() / 'config' / 'deep_learning.yaml'
    if not config_path.exists():
        logger.warning(f"配置文件不存在: {config_path}, 使用默认配置")
        return get_default_config()
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_merged_config() -> dict:
    """
    N2修复: 加载并合并主配置和深度配置

    优先级:
    - 关键口径段 (stock_pool/backtest/execution/liquidity) 从 config.yaml 继承
    - 深度学习专有配置从 deep_learning.yaml 读取
    - 如果 deep_learning.yaml 中显式定义了非null值，则覆盖

    Returns:
        合并后的配置字典
    """
    import yaml

    # 加载深度学习配置
    dl_config = load_deep_config()

    # 尝试加载主配置
    main_config_path = get_project_root() / 'config' / 'config.yaml'
    if main_config_path.exists():
        try:
            with open(main_config_path, 'r', encoding='utf-8') as f:
                main_config = yaml.safe_load(f)

            # N2修复: 关键口径段从主配置继承 (如果深度配置中为null或不存在)
            # R1修复: 添加 rl_adapter 到继承列表
            inherit_keys = ['execution', 'stock_pool', 'data_fetch', 'liquidity', 'backtest', 'rl_adapter']
            for key in inherit_keys:
                if key in main_config:
                    # 如果深度配置中该key为null或不存在，使用主配置
                    if key not in dl_config or dl_config.get(key) is None:
                        dl_config[key] = main_config[key]
                        logger.debug(f"从 config.yaml 继承配置段: {key}")

        except Exception as e:
            logger.warning(f"加载主配置失败: {e}, 使用纯深度配置")

    return dl_config


def get_default_config() -> dict:
    """默认配置"""
    return {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'sequence': {
            'seq_len': 60,
            'stride': 1,
            'batch_size': 256,
            'use_amp': True,
        },
        'transformer': {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 4,
            'dim_feedforward': 512,
            'dropout': 0.1,
        },
        'lstm': {
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'bidirectional': False,
        },
        'training': {
            'epochs': 100,
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'patience': 10,
            'early_stop': True,
            'min_delta': 0.0,
            'min_epochs': 0,
            'loss': 'listwise',  # listwise, pairwise, mse
        },
        'graph': {
            'edge_type': 'hybrid',  # industry, corr, hybrid
            'corr_window': 60,
            'top_k': 10,
        },
    }


class DeepStockModel(ABC):
    """
    深度学习选股模型基类

    接口设计与 MLModel 兼容:
    - train(X, y, **kwargs) -> Dict: 训练模型
    - predict(factor_df) -> pd.Series: 预测分数
    - select_stocks_ml(factor_df, n) -> pd.DataFrame: 选股

    子类需要实现:
    - _build_model(): 构建网络结构
    - _prepare_data(): 准备训练数据
    - forward(): 前向传播 (如果继承 nn.Module)
    """

    def __init__(self, config: dict = None):
        self.config = config or load_deep_config()

        # R8修复: 处理 device: auto 配置
        device_cfg = self.config.get('device', 'auto')
        if device_cfg == 'auto' or device_cfg is None:
            device_cfg = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_cfg)

        self.model: Optional[nn.Module] = None
        self.scaler = None  # 特征标准化器
        self.feature_cols: List[str] = []
        self.is_trained = False

        # 模型保存路径
        self.model_dir = ensure_dir(get_project_root() / 'output' / 'models' / 'deep')

        # 设置随机种子
        self._set_seed(self.config.get('seed', 42))

        logger.info(f"初始化深度模型，设备: {self.device}")

    def _set_seed(self, seed: int):
        """设置随机种子确保可复现"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """构建模型网络结构，子类必须实现"""
        pass

    @abstractmethod
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series = None,
                      is_train: bool = True) -> Any:
        """准备数据 (Dataset/DataLoader)，子类必须实现"""
        pass

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """
        训练模型（兼容 MLModel.train 接口）

        Args:
            X: 特征DataFrame，必须包含 'code', 'date' 列
            y: 标签Series，通常是未来收益率
            **kwargs: 额外参数

        Returns:
            训练结果字典，包含 loss, metrics 等
        """
        logger.info(f"开始训练深度模型，样本数: {len(X)}")

        # R9修复: 先准备数据（设置 input_dim），再构建模型
        train_loader, val_loader = self._prepare_data(X, y, is_train=True)

        # 构建模型（需要 input_dim 已设置）
        if self.model is None:
            self.model = self._build_model()
            self.model = self.model.to(self.device)

        # 训练循环
        training_cfg = self.config.get('training', {})
        epochs = training_cfg.get('epochs', 100)
        lr = training_cfg.get('lr', 1e-4)
        patience = training_cfg.get('patience', 10)
        early_stop = training_cfg.get('early_stop', True)
        min_delta = float(training_cfg.get('min_delta', 0.0) or 0.0)
        min_epochs = int(training_cfg.get('min_epochs', 0) or 0)
        if patience is None:
            patience = 0
        else:
            patience = int(patience)
        if patience <= 0:
            early_stop = False
        use_amp = self.config.get('sequence', {}).get('use_amp', True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=training_cfg.get('weight_decay', 1e-5)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        # 使用新版 PyTorch AMP API (避免 FutureWarning)
        scaler = torch.amp.GradScaler('cuda') if use_amp and self.device.type == 'cuda' else None

        # 获取损失函数
        criterion = self._get_loss_function(training_cfg.get('loss', 'mse'))

        best_val_loss = float('inf')
        patience_counter = 0
        train_history = {'train_loss': [], 'val_loss': []}
        checkpoint_saved = False  # 确保至少保存一次检查点

        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            # 使用 tqdm 显示训练进度
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                optimizer.zero_grad()

                if use_amp and scaler:
                    with torch.amp.autocast('cuda'):
                        loss = self._compute_batch_loss(batch, criterion)
                    scaler.scale(loss).backward()
                    # 梯度裁剪防止梯度爆炸
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = self._compute_batch_loss(batch, criterion)
                    loss.backward()
                    # 梯度裁剪防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                loss_val = loss.item()
                if not np.isnan(loss_val) and not np.isinf(loss_val):
                    train_loss += loss_val
                    n_batches += 1
                pbar.set_postfix({'loss': f'{loss_val:.6f}'})

            train_loss = train_loss / max(n_batches, 1)
            train_history['train_loss'].append(train_loss)

            # 验证
            val_loss = self._validate(val_loader, criterion) if val_loader else train_loss
            train_history['val_loss'].append(val_loss)
            scheduler.step(val_loss)

            # 每个 epoch 都输出日志
            logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # 早停逻辑 + 确保至少保存一次检查点
            improved = (not np.isinf(val_loss)) and (val_loss < best_val_loss - min_delta)
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint('best')
                checkpoint_saved = True
            elif not checkpoint_saved and epoch == 0:
                # 首个 epoch 结束后，即使 loss 是 inf 也保存一次
                self._save_checkpoint('best')
                checkpoint_saved = True
                logger.warning("首个 epoch 损失为 inf，仍保存检查点以便调试")
            else:
                if early_stop and epoch + 1 >= min_epochs:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"早停于 epoch {epoch + 1}")
                        break

        # 加载最佳模型
        self._load_checkpoint('best')
        self.is_trained = True

        return {
            'train_loss': train_history['train_loss'][-1],
            'val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'history': train_history,
        }

    def _get_loss_function(self, loss_type: str) -> nn.Module:
        """获取损失函数"""
        from .losses import ListwiseLoss, PairwiseLoss

        if loss_type == 'listwise':
            return ListwiseLoss()
        elif loss_type == 'pairwise':
            return PairwiseLoss()
        else:
            return nn.MSELoss()

    def _compute_batch_loss(self, batch: Tuple, criterion: nn.Module) -> torch.Tensor:
        """计算batch损失，子类可重写"""
        if len(batch) == 2:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            # 处理 NaN/Inf 数据
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            pred = self.model(x)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            loss = criterion(pred.squeeze(), y)
            # 检测损失是否有效，避免 inf 传播
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("检测到 NaN/Inf 损失，返回零损失")
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            return loss
        elif len(batch) == 3:
            x, y, mask = batch
            x = x.to(self.device)
            y = y.to(self.device)
            mask = mask.to(self.device)
            # 处理 NaN/Inf 数据
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            pred = self.model(x)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            loss = criterion(pred.squeeze(), y, mask)
            # 检测损失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("检测到 NaN/Inf 损失，返回零损失")
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            return loss
        else:
            raise ValueError(f"Unexpected batch format: {len(batch)} elements")

    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """验证"""
        self.model.eval()
        val_loss = 0.0
        valid_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = self._compute_batch_loss(batch, criterion)
                loss_val = loss.item()
                if not np.isnan(loss_val) and not np.isinf(loss_val):
                    val_loss += loss_val
                    valid_batches += 1

        if valid_batches == 0:
            logger.warning("验证集损失全部无效，返回 0 作为占位值")
            return 0.0

        return val_loss / valid_batches

    @abstractmethod
    def predict(self, factor_df: pd.DataFrame) -> pd.Series:
        """
        预测分数（兼容 MLModel.predict 接口）

        Args:
            factor_df: 因子DataFrame，必须包含 'code' 列

        Returns:
            预测分数Series，索引与 factor_df 对齐
        """
        pass

    def select_stocks_ml(self, factor_df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
        """
        选股（兼容 MLModel.select_stocks_ml 接口）

        Args:
            factor_df: 因子DataFrame
            n: 选股数量

        Returns:
            选中的股票DataFrame，包含 'code', 'ml_score', 'rank' 列
        """
        if not self.is_trained:
            logger.warning("模型未训练，返回空结果")
            return pd.DataFrame(columns=['code', 'ml_score', 'rank'])

        df = factor_df.copy()
        df['ml_score'] = self.predict(df)

        # 过滤无效分数
        df = df[df['ml_score'].notna()]

        # 排序选股
        df = df.nlargest(n, 'ml_score')
        df['rank'] = range(1, len(df) + 1)

        return df

    def _save_checkpoint(self, name: str = 'latest'):
        """保存检查点"""
        if self.model is None:
            return
        checkpoint_path = self.model_dir / f'{self.__class__.__name__}_{name}.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_cols': self.feature_cols,
            'config': self.config,
        }, checkpoint_path)
        logger.debug(f"保存检查点: {checkpoint_path}")

    def _load_checkpoint(self, name: str = 'latest'):
        """加载检查点"""
        checkpoint_path = self.model_dir / f'{self.__class__.__name__}_{name}.pt'
        if not checkpoint_path.exists():
            logger.warning(f"检查点不存在: {checkpoint_path}")
            return False

        # PyTorch 2.6+ 需要 weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.feature_cols = checkpoint.get('feature_cols', [])
        logger.debug(f"加载检查点: {checkpoint_path}")
        return True

    def save_model(self, path: str = None):
        """保存模型"""
        if path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = self.model_dir / f'{self.__class__.__name__}_{timestamp}.pt'
        self._save_checkpoint(Path(path).stem)
        logger.info(f"模型已保存: {path}")

    def load_model(self, path: str):
        """加载模型"""
        # PyTorch 2.6+ 需要 weights_only=False
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # 先从检查点恢复 feature_cols
        self.feature_cols = checkpoint.get('feature_cols', [])

        # 从模型权重中获取正确的 input_dim（比 feature_cols 更准确）
        state_dict = checkpoint['model_state_dict']
        for key, value in state_dict.items():
            if 'input_proj.weight' in key or 'input_layer.weight' in key:
                self.input_dim = value.shape[1]
                break
        else:
            # 回退到 feature_cols 长度
            if self.feature_cols:
                self.input_dim = len(self.feature_cols)

        # 恢复配置（如果有）
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            # 合并保存的配置到当前配置
            for key, value in saved_config.items():
                if key not in self.config or self.config[key] is None:
                    self.config[key] = value

        if self.model is None:
            self.model = self._build_model()
            self.model = self.model.to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        logger.info(f"模型已加载: {path}, 特征维度: {self.input_dim}")


class EnhancedPurgedCV:
    """
    增强版防泄露交叉验证 - 支持序列/图构建

    在 PurgedGroupTimeSeriesSplit 基础上增加:
    - 序列窗口检查: 序列构建不能使用未来数据
    - 图构建检查: 相关性计算不能使用未来数据
    """

    def __init__(self, n_splits: int = 5,
                 purge_days: int = 20,
                 seq_len: int = 60,
                 trade_calendar: pd.DatetimeIndex = None):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.seq_len = seq_len
        self.trade_calendar = trade_calendar

    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: pd.Series = None):
        """
        生成训练/验证索引，确保序列构建不泄露

        额外检查: 训练集最后日期 + seq_len < 验证集开始日期
        """
        if groups is None:
            if 'date' in X.columns:
                groups = X['date']
            else:
                raise ValueError("X must contain 'date' column or provide groups")

        groups = pd.to_datetime(groups)
        unique_dates = np.sort(groups.unique())
        n_dates = len(unique_dates)

        val_size = n_dates // (self.n_splits + 1)

        for i in range(self.n_splits):
            val_end_idx = n_dates - i * val_size
            val_start_idx = val_end_idx - val_size

            if val_start_idx < 0 or val_end_idx <= val_start_idx:
                continue

            val_start_date = unique_dates[val_start_idx]

            # 计算序列安全边界: 需要额外减去序列长度
            # 确保训练数据不会在构建序列时使用验证期数据
            total_buffer = self.purge_days + int(self.seq_len * 1.5)

            if self.trade_calendar is not None and len(self.trade_calendar) > 0:
                try:
                    cal_list = self.trade_calendar.tolist()
                    if val_start_date in cal_list:
                        val_start_loc = cal_list.index(val_start_date)
                        cutoff_loc = max(0, val_start_loc - total_buffer)
                        train_end_date = self.trade_calendar[cutoff_loc]
                    else:
                        train_end_date = val_start_date - pd.Timedelta(days=total_buffer * 1.5)
                except:
                    train_end_date = val_start_date - pd.Timedelta(days=total_buffer * 1.5)
            else:
                train_end_date = val_start_date - pd.Timedelta(days=total_buffer * 1.5)

            # 生成索引
            train_mask = groups < train_end_date
            val_mask = (groups >= val_start_date) & (groups < unique_dates[min(val_end_idx, n_dates - 1)])

            train_idx = X.index[train_mask].tolist()
            val_idx = X.index[val_mask].tolist()

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx
