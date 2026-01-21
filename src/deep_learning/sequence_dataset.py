"""
因子序列数据集模块 (Phase 1)

将横截面因子数据转换为时序序列，用于 LSTM/Transformer 训练。

关键设计:
- 防泄露: 只使用 end_date 之前的历史数据
- 缓存机制: 序列数据预计算并缓存，使用 universe_id 和 meta 校验
- 数据契约: 使用 SequenceBatch 作为标准输出格式
- 内存优化: 按需加载，支持分片
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 导入数据契约和策略
from .contracts import (
    CacheMeta, SequenceBatch, FactorFrame, generate_universe_id,
    validate_cache_compatibility
)
from .policies import TradeabilityPolicy, get_default_policies

# 复用现有数据处理模块（只读）
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger, get_project_root, ensure_dir, load_parquet, save_parquet
from src.data_processor import DataProcessor

logger = setup_logger(__name__)


def _sanitize_sequence_values(sequence: np.ndarray) -> np.ndarray:
    if sequence is None:
        return sequence
    seq = np.asarray(sequence, dtype=np.float32)
    if np.isnan(seq).any() or np.isinf(seq).any():
        raise RuntimeError("sequence contains NaN or inf values")
    return seq


class FactorSequenceDataset(Dataset):
    """
    因子序列数据集

    将 (code, date) 的横截面因子扩展为 [seq_len, n_factors] 的时序特征。

    防泄露保证:
    - 给定 end_date，序列只包含 [end_date - seq_len, end_date) 的历史数据
    - 标签使用 end_date 之后的未来收益

    缓存一致性:
    - 使用 universe_id 作为缓存键
    - CacheMeta 记录配置哈希、复权口径、数据源版本
    - 缓存加载时验证元信息一致性

    数据流:
    1. 加载现有因子缓存 (data/processed/factors/)
    2. 按股票和日期对齐
    3. 构建滑动窗口序列
    4. 缓存到 data/processed/factor_seq/
    """

    def __init__(self,
                 config: dict,
                 start_date: str = None,
                 end_date: str = None,
                 mode: str = 'train',
                 cache_dir: str = None,
                 tradeability_policy: TradeabilityPolicy = None,
                 adjust: str = 'hfq',
                 universe_id: str = None):
        """
        Args:
            config: 配置字典
            start_date: 起始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            mode: 'train' 或 'val' 或 'test'
            cache_dir: 缓存目录
            tradeability_policy: 可交易性策略 (用于过滤样本)
            adjust: 复权口径 ('hfq', 'qfq', 'none')
            universe_id: 股票池标识 (用于匹配因子缓存)
        """
        self.config = config
        self.seq_config = config.get('sequence', {})
        self.seq_len = self.seq_config.get('seq_len', 60)
        self.stride = self.seq_config.get('stride', 1)
        self.mode = mode
        self.adjust = adjust
        self._universe_id_hint = universe_id  # 用于匹配因子缓存的提示

        # 柔性序列配置
        self.continuity_mode = self.seq_config.get('continuity_mode', 'strict')
        self.min_coverage = self.seq_config.get('min_coverage', 0.8)
        self.add_time_gap_feature = self.seq_config.get('add_time_gap_feature', True)
        self.max_lookback_multiplier = self.seq_config.get('max_lookback_multiplier', 2.0)

        # 可交易性策略
        self.tradeability_policy = tradeability_policy or TradeabilityPolicy()

        # 数据处理器（只读复用）
        self.processor = DataProcessor()

        # 缓存目录
        self.cache_dir = Path(cache_dir) if cache_dir else \
            get_project_root() / 'data' / 'processed' / 'factor_seq'
        ensure_dir(self.cache_dir)

        # 因子缓存目录
        self.factor_cache_dir = get_project_root() / 'data' / 'processed' / 'factors'

        # 交易日历
        self.trade_calendar = self.processor.trade_calendar

        # 确定日期范围
        if start_date is None:
            start_date = '20180101'
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        # 生成配置哈希 (用于缓存验证)
        self.config_hash = self._compute_config_hash()

        # 获取有效日期
        self.valid_dates = self._get_valid_dates()

        # 加载或构建数据
        self.samples = []  # [(code, date, features, label), ...]
        self.feature_cols = []
        self.universe_id = ''  # 初始化
        self.meta: Optional[CacheMeta] = None

        self._load_or_build_samples()

    def _compute_config_hash(self) -> str:
        """计算配置哈希 (用于缓存一致性验证)"""
        config_str = json.dumps({
            'seq_len': self.seq_len,
            'stride': self.stride,
            'adjust': self.adjust,
            'exclude_st': self.tradeability_policy.exclude_st,
            'check_limit': self.tradeability_policy.check_limit_up,
            # 柔性序列配置
            'continuity_mode': self.continuity_mode,
            'min_coverage': self.min_coverage,
            'add_time_gap_feature': self.add_time_gap_feature,
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _get_valid_dates(self) -> List[pd.Timestamp]:
        """获取有效交易日（有因子缓存的日期）"""
        if not self.factor_cache_dir.exists():
            raise RuntimeError(f"factor cache dir not found: {self.factor_cache_dir}")

        factor_files = list(self.factor_cache_dir.glob('factors_*.parquet'))
        dates = []

        for f in factor_files:
            try:
                # 支持两种格式: factors_YYYYMMDD.parquet 和 factors_YYYYMMDD_universe.parquet
                parts = f.stem.split('_')
                if len(parts) >= 2:
                    date_str = parts[1]
                    date = pd.to_datetime(date_str)
                    if self.start_date <= date <= self.end_date:
                        dates.append(date)
            except ValueError as e:
                logger.debug(f"跳过无法解析的文件 {f.name}: {e}")
                continue

        dates = sorted(set(dates))  # 去重并排序
        logger.info(f"找到 {len(dates)} 个有效因子日期 ({self.start_date.date()} ~ {self.end_date.date()})")
        return dates

    def _load_factor_data(self, date_str: str) -> Optional[pd.DataFrame]:
        """
        加载因子数据，支持新/旧格式兼容 (Bug修复)

        新格式: factors_{date}_{universe_id}.parquet
        旧格式: factors_{date}.parquet

        优先级:
        1. 如果提供了 universe_id 提示，优先匹配
        2. 否则查找新格式文件（使用确定性选择策略）
        3. 回退到旧格式

        Bug修复 R3: 使用确定性选择策略 (股票数量+文件名) 而非修改时间

        Args:
            date_str: 日期字符串 YYYYMMDD

        Returns:
            因子数据 DataFrame 或 None
        """
        if not self._universe_id_hint:
            raise RuntimeError("universe_id is required to load factor cache")
        exact_path = self.factor_cache_dir / f'factors_{date_str}_{self._universe_id_hint}.parquet'
        if not exact_path.exists():
            raise RuntimeError(f"factor cache not found for {date_str}: {exact_path}")
        return load_parquet(exact_path)

    def _load_or_build_samples(self):
        """加载或构建样本（带缓存验证）"""
        cache_file = self._get_cache_path()

        # 检查缓存是否存在且有效
        if cache_file.exists():
            is_valid, reason = self._validate_cache(cache_file)
            if is_valid:
                logger.info(f"加载缓存数据: {cache_file}")
                self._load_cache(cache_file)
                return
            else:
                logger.info(f"缓存无效 ({reason})，重新构建...")

        logger.info("构建序列数据...")
        self._build_samples()
        self._save_cache(cache_file)

    def _get_cache_path(self) -> Path:
        """获取缓存文件路径 (包含 config_hash 确保配置一致性)"""
        start_str = self.start_date.strftime('%Y%m%d')
        end_str = self.end_date.strftime('%Y%m%d')
        return self.cache_dir / f'seq_{self.mode}_{start_str}_{end_str}_{self.seq_len}d_{self.config_hash}.pt'

    def _validate_cache(self, cache_path: Path) -> Tuple[bool, str]:
        """验证缓存有效性"""
        meta_path = cache_path.with_suffix('.meta.json')
        if not meta_path.exists():
            return False, "meta 文件不存在"

        try:
            existing_meta = CacheMeta.load(cache_path)
            if existing_meta is None:
                return False, "无法加载 meta"

            # 验证配置哈希
            if existing_meta.config_hash != self.config_hash:
                return False, f"config_hash 不匹配: {existing_meta.config_hash} vs {self.config_hash}"

            # 验证复权口径
            if existing_meta.adjust != self.adjust:
                return False, f"adjust 不匹配: {existing_meta.adjust} vs {self.adjust}"

            # 过滤旧缓存中包含 *_raw 特征的情况
            if existing_meta.feature_cols and any(c.endswith('_raw') for c in existing_meta.feature_cols):
                return False, "feature_cols 含有 *_raw，需重建缓存"

            return True, "OK"
        except Exception as e:
            return False, f"验证异常: {e}"

    def _build_samples(self):
        """构建序列样本（支持 strict/flexible 模式）"""
        # 根据模式确定最小历史需求
        if self.continuity_mode == 'flexible':
            # 柔性模式：只需要最小覆盖率
            min_history_dates = int(self.seq_len * self.min_coverage)
            logger.info(f"使用柔性序列模式: seq_len={self.seq_len}, min_coverage={self.min_coverage}")
        else:
            # 严格模式：需要完整 seq_len 天
            min_history_dates = self.seq_len

        if len(self.valid_dates) < min_history_dates + 1:
            logger.warning(f"有效日期不足，需要至少 {min_history_dates + 1} 天，当前 {len(self.valid_dates)} 天")
            return

        # 加载所有因子数据到内存 (Bug修复: 使用新方法支持 universe_id)
        all_factors = {}
        for date in tqdm(self.valid_dates, desc="加载因子数据"):
            date_str = date.strftime('%Y%m%d')
            df = self._load_factor_data(date_str)
            if df is not None and len(df) > 0:
                all_factors[date] = df

        if len(all_factors) < min_history_dates:
            logger.warning("因子数据不足以构建序列")
            return

        # 确定特征列（排除非因子列）
        sample_df = list(all_factors.values())[0]
        exclude_cols = {'code', 'name', 'date', 'industry', 'label', 'forward_return', 'total_score'}
        self.feature_cols = [
            c for c in sample_df.columns
            if c not in exclude_cols
            and not c.endswith('_raw')
            and sample_df[c].dtype in [np.float64, np.float32, np.int64]
        ]
        logger.info(f"特征列数: {len(self.feature_cols)}")

        # 获取所有股票代码
        all_codes = set()
        for df in all_factors.values():
            all_codes.update(df['code'].unique())
        all_codes = sorted(all_codes)
        logger.info(f"股票数量: {len(all_codes)}")

        # 按股票构建时序
        # 只对 valid_dates[min_history_dates:] 构建样本（确保有足够历史）
        sample_dates = self.valid_dates[min_history_dates:]

        for date in tqdm(sample_dates, desc="构建序列"):
            date_idx = self.valid_dates.index(date)

            # 根据模式获取历史日期
            if self.continuity_mode == 'flexible':
                history_dates = self._get_flexible_history_dates(date_idx)
            else:
                history_dates = self.valid_dates[date_idx - self.seq_len:date_idx]

            if history_dates is None or len(history_dates) == 0:
                raise RuntimeError("insufficient history dates for sequence build")

            # 获取当天的因子数据作为标签来源
            if date not in all_factors:
                raise RuntimeError(f"missing factor data for {date.date()}")

            current_df = all_factors[date]

            for code in current_df['code'].unique():
                # 获取标签（当天因子中的 forward_return 或计算未来收益）
                code_row = current_df[current_df['code'] == code]
                if len(code_row) == 0:
                    raise RuntimeError(f"missing code row for {code} on {date.date()}")

                # R4 修复: 应用可交易性过滤
                row_series = code_row.iloc[0]
                is_tradeable, reason = self.tradeability_policy.is_tradeable(row_series, action='buy')
                if not is_tradeable:
                    logger.debug(f"过滤 {code}: {reason}")
                    continue

                # 根据模式构建历史序列
                if self.continuity_mode == 'flexible':
                    sequence = self._build_single_sequence_flexible(
                        code, history_dates, all_factors
                    )
                else:
                    sequence = self._build_single_sequence(
                        code, history_dates, all_factors
                    )

                if sequence is None:
                    raise RuntimeError(f"sequence build failed for {code} on {date.date()}")

                # 尝试获取标签
                label = None
                for label_col in ['forward_return', 'label', 'ret_forward']:
                    if label_col in code_row.columns:
                        label = code_row[label_col].values[0]
                        break

                if label is None or np.isnan(label):
                    raise RuntimeError(f"missing label for {code} on {date.date()}")

                self.samples.append({
                    'code': code,
                    'date': date,
                    'sequence': sequence,
                    'label': label,
                })

        logger.info(f"构建完成: {len(self.samples)} 个样本")

    def _build_single_sequence(self,
                                code: str,
                                history_dates: List[pd.Timestamp],
                                all_factors: Dict[pd.Timestamp, pd.DataFrame]) -> Optional[np.ndarray]:
        """
        构建单只股票的因子序列

        Args:
            code: 股票代码
            history_dates: 历史日期列表 [t-seq_len, t-1]
            all_factors: 所有因子数据

        Returns:
            序列数组 [seq_len, n_features] 或 None
        """
        sequence = []

        for date in history_dates:
            if date not in all_factors:
                raise RuntimeError(f"missing factor data for {date.date()}")

            df = all_factors[date]
            code_row = df[df['code'] == code]

            if len(code_row) == 0:
                raise RuntimeError(f"missing factor row for {code} on {date.date()}")

            # 提取特征
            features = code_row[self.feature_cols].values[0]
            sequence.append(features)

        sequence = np.array(sequence, dtype=np.float32)

        if np.isnan(sequence).any() or np.isinf(sequence).any():
            raise RuntimeError(f"sequence contains NaN/inf for {code}")

        return sequence

    def _get_flexible_history_dates(self, date_idx: int) -> Optional[List[pd.Timestamp]]:
        """
        获取柔性历史日期，不要求严格连续

        Args:
            date_idx: 当前日期在 valid_dates 中的索引

        Returns:
            历史日期列表，或 None（如果数据不足）
        """
        target_len = self.seq_len
        max_lookback = int(target_len * self.max_lookback_multiplier)

        start_idx = max(0, date_idx - max_lookback)
        candidate_dates = self.valid_dates[start_idx:date_idx]

        # 取最近的 target_len 个日期
        if len(candidate_dates) >= target_len:
            return candidate_dates[-target_len:]
        elif len(candidate_dates) >= int(target_len * self.min_coverage):
            # 允许不足但超过最低覆盖率
            return candidate_dates
        else:
            raise RuntimeError("insufficient history coverage for flexible sequence")

    def _build_single_sequence_flexible(self,
                                          code: str,
                                          history_dates: List[pd.Timestamp],
                                          all_factors: Dict[pd.Timestamp, pd.DataFrame]) -> Optional[np.ndarray]:
        """
        构建带时间间隔特征的柔性序列

        允许部分数据缺失，使用插值填充，并添加时间间隔特征。

        Args:
            code: 股票代码
            history_dates: 历史日期列表
            all_factors: 所有因子数据

        Returns:
            序列数组 [seq_len, n_features(+1)] 或 None
        """
        sequence = []
        time_gaps = []
        valid_count = 0

        prev_date = None
        for date in history_dates:
            # 计算时间间隔（天）
            if prev_date:
                gap = (date - prev_date).days
            else:
                gap = 0
            time_gaps.append(gap)
            prev_date = date

            # 获取因子数据
            if date in all_factors:
                df = all_factors[date]
                code_row = df[df['code'] == code]
                if len(code_row) > 0:
                    features = code_row[self.feature_cols].values[0]
                    sequence.append(features)
                    valid_count += 1
                else:
                    sequence.append(None)  # 标记缺失
            else:
                sequence.append(None)

        # 检查覆盖率
        coverage = valid_count / len(sequence) if len(sequence) > 0 else 0
        if coverage < self.min_coverage:
            raise RuntimeError(f"coverage {coverage:.2f} below minimum {self.min_coverage}")

        # 柔性序列对齐到固定长度，避免 batch 堆叠尺寸不一致
        if len(sequence) < self.seq_len:
            pad_len = self.seq_len - len(sequence)
            sequence = [None] * pad_len + sequence
            time_gaps = [0] * pad_len + time_gaps
        elif len(sequence) > self.seq_len:
            sequence = sequence[-self.seq_len:]
            time_gaps = time_gaps[-self.seq_len:]

        if any(item is None for item in sequence):
            raise RuntimeError(f"missing factors for {code} in flexible sequence")
        sequence = np.array(sequence, dtype=np.float32)

        # 添加时间间隔特征
        if self.add_time_gap_feature:
            time_gaps = np.array(time_gaps, dtype=np.float32).reshape(-1, 1) / 30.0  # 归一化到月
            sequence = np.hstack([sequence, time_gaps])

        if np.isnan(sequence).any() or np.isinf(sequence).any():
            raise RuntimeError(f"sequence contains NaN/inf for {code}")
        return sequence.astype(np.float32)

    def _interpolate_missing(self, sequence: List) -> np.ndarray:
        """
        使用线性插值填充缺失值

        Args:
            sequence: 特征序列列表，缺失值为 None

        Returns:
            填充后的数组 [seq_len, n_features]
        """
        n_features = len(self.feature_cols)
        result = np.zeros((len(sequence), n_features), dtype=np.float32)

        for i, feat in enumerate(sequence):
            if feat is not None:
                result[i] = feat
            else:
                # 前向查找
                prev_idx = i - 1
                while prev_idx >= 0 and sequence[prev_idx] is None:
                    prev_idx -= 1

                # 后向查找
                next_idx = i + 1
                while next_idx < len(sequence) and sequence[next_idx] is None:
                    next_idx += 1

                # 插值
                if prev_idx >= 0 and next_idx < len(sequence):
                    # 线性插值
                    alpha = (i - prev_idx) / (next_idx - prev_idx)
                    result[i] = (1 - alpha) * np.array(sequence[prev_idx]) + alpha * np.array(sequence[next_idx])
                elif prev_idx >= 0:
                    result[i] = sequence[prev_idx]  # 前向填充
                elif next_idx < len(sequence):
                    result[i] = sequence[next_idx]  # 后向填充
                # else: 保持0

        return result

    def _save_cache(self, cache_path: Path):
        """保存缓存（包含 CacheMeta）"""
        # 收集所有股票代码用于 universe_id
        all_codes = sorted(set(s['code'] for s in self.samples))
        self.universe_id = generate_universe_id(
            all_codes,
            self.end_date.strftime('%Y%m%d')
        )

        # 创建元信息
        self.meta = CacheMeta(
            universe_id=self.universe_id,
            config_hash=self.config_hash,
            adjust=self.adjust,
            n_samples=len(self.samples),
            date_range=(
                self.start_date.strftime('%Y%m%d'),
                self.end_date.strftime('%Y%m%d')
            ),
            feature_cols=self.feature_cols,
        )

        # 保存数据
        torch.save({
            'samples': self.samples,
            'feature_cols': self.feature_cols,
            'seq_len': self.seq_len,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'universe_id': self.universe_id,
        }, cache_path)

        # 保存元信息
        self.meta.save(cache_path)
        logger.info(f"缓存已保存: {cache_path} (universe_id={self.universe_id})")

    def _load_cache(self, cache_path: Path):
        """加载缓存（包含元信息验证）"""
        # PyTorch 2.6+ 需要 weights_only=False 以加载包含 pandas 对象的缓存
        data = torch.load(cache_path, weights_only=False)
        self.samples = data['samples']
        self.feature_cols = data['feature_cols']
        self.universe_id = data.get('universe_id', '')

        # 加载元信息
        self.meta = CacheMeta.load(cache_path)

        # Normalize cached sequence length to seq_len
        padded = 0
        for sample in self.samples:
            seq = sample.get('sequence')
            if seq is None:
                continue
            seq_arr = np.asarray(seq)
            if seq_arr.ndim != 2:
                continue
            if seq_arr.shape[0] != self.seq_len:
                pad_len = self.seq_len - seq_arr.shape[0]
                if pad_len > 0:
                    if seq_arr.shape[0] == 0:
                        pad_block = np.zeros((self.seq_len, seq_arr.shape[1]), dtype=np.float32)
                        sample['sequence'] = pad_block
                    else:
                        pad_row = seq_arr[0]
                        pad_block = np.repeat(pad_row[None, :], pad_len, axis=0)
                        sample['sequence'] = np.vstack([pad_block, seq_arr]).astype(np.float32)
                else:
                    sample['sequence'] = seq_arr[-self.seq_len:].astype(np.float32)
                padded += 1
                seq_arr = np.asarray(sample['sequence'])

            sample['sequence'] = _sanitize_sequence_values(seq_arr)

        if padded > 0:
            logger.warning(f"Detected {padded} samples with non-matching seq_len, padded to {self.seq_len}")

        logger.info(f"加载 {len(self.samples)} 个样本 (universe_id={self.universe_id})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取样本

        Returns:
            (sequence, label)
            - sequence: [seq_len, n_features]
            - label: 标量
        """
        sample = self.samples[idx]
        sequence = torch.tensor(sample['sequence'], dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.float32)
        return sequence, label

    def get_sample_info(self, idx: int) -> Dict:
        """获取样本元信息"""
        sample = self.samples[idx]
        return {
            'code': sample['code'],
            'date': sample['date'],
            'label': sample['label'],
        }

    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return len(self.feature_cols)

    def to_sequence_batch(self, indices: List[int]) -> SequenceBatch:
        """
        将指定索引的样本转换为 SequenceBatch 数据契约

        Args:
            indices: 样本索引列表

        Returns:
            SequenceBatch 对象
        """
        if len(indices) == 0:
            return SequenceBatch(
                sequences=torch.empty(0, self.seq_len, len(self.feature_cols)),
                labels=torch.empty(0),
                codes=[],
                dates=[],
                universe_id=self.universe_id,
                meta=self.meta,
            )

        sequences = []
        labels = []
        codes = []
        dates = []

        for idx in indices:
            sample = self.samples[idx]
            sequences.append(sample['sequence'])
            labels.append(sample['label'])
            codes.append(sample['code'])
            dates.append(sample['date'])

        return SequenceBatch(
            sequences=torch.tensor(np.array(sequences), dtype=torch.float32),
            labels=torch.tensor(labels, dtype=torch.float32),
            codes=codes,
            dates=dates,
            universe_id=self.universe_id,
            meta=self.meta,
        )

    def get_batch_as_sequence_batch(self, batch_indices: torch.Tensor,
                                     sequences: torch.Tensor,
                                     labels: torch.Tensor) -> SequenceBatch:
        """
        将 DataLoader 的 batch 转换为 SequenceBatch

        Args:
            batch_indices: batch 中样本的原始索引
            sequences: [batch_size, seq_len, n_features]
            labels: [batch_size]

        Returns:
            SequenceBatch 对象
        """
        codes = [self.samples[i]['code'] for i in batch_indices.tolist()]
        dates = [self.samples[i]['date'] for i in batch_indices.tolist()]

        return SequenceBatch(
            sequences=sequences,
            labels=labels,
            codes=codes,
            dates=dates,
            universe_id=self.universe_id,
            meta=self.meta,
        )


class FactorSequenceBuilder:
    """
    因子序列构建器

    用于推理时动态构建序列，不需要预先加载所有数据。
    支持柔性序列模式，与 FactorSequenceDataset 保持一致。
    """

    def __init__(self, config: dict, universe_id: str = None):
        self.config = config
        seq_config = config.get('sequence', {})
        self.seq_len = seq_config.get('seq_len', 60)
        self.processor = DataProcessor()
        self.factor_cache_dir = get_project_root() / 'data' / 'processed' / 'factors'
        self._universe_id_hint = universe_id  # Bug修复: 添加 universe_id 支持

        # 柔性序列配置（与 FactorSequenceDataset 保持一致）
        self.continuity_mode = seq_config.get('continuity_mode', 'strict')
        self.min_coverage = seq_config.get('min_coverage', 0.8)
        self.add_time_gap_feature = seq_config.get('add_time_gap_feature', True)
        self.max_lookback_multiplier = seq_config.get('max_lookback_multiplier', 2.0)

    def _load_factor_data(self, date_str: str) -> Optional[pd.DataFrame]:
        """
        加载因子数据，支持新/旧格式兼容 (Bug修复)

        Bug修复 R3: 使用确定性选择策略 (股票数量+文件名) 而非修改时间

        Args:
            date_str: 日期字符串 YYYYMMDD

        Returns:
            因子数据 DataFrame 或 None
        """
        if not self._universe_id_hint:
            raise RuntimeError("universe_id is required to load factor cache")
        exact_path = self.factor_cache_dir / f'factors_{date_str}_{self._universe_id_hint}.parquet'
        if not exact_path.exists():
            raise RuntimeError(f"factor cache not found for {date_str}: {exact_path}")
        return load_parquet(exact_path)

    def build_sequences_for_date(self,
                                  target_date: pd.Timestamp,
                                  codes: List[str] = None) -> Dict[str, np.ndarray]:
        """
        为指定日期构建所有股票的序列

        支持 strict/flexible 两种模式：
        - strict: 要求连续 seq_len 天数据
        - flexible: 允许非连续数据，使用插值填充

        Args:
            target_date: 目标日期
            codes: 股票代码列表（可选，默认所有可用）

        Returns:
            {code: sequence_array} 字典
        """
        # 根据模式确定最小需求
        if self.continuity_mode == 'flexible':
            min_required = int(self.seq_len * self.min_coverage)
        else:
            min_required = self.seq_len

        # 获取历史日期
        calendar = self.processor.trade_calendar
        target_loc = calendar.get_loc(target_date) if target_date in calendar else None

        if target_loc is None:
            # 找最近的交易日
            idx = calendar.searchsorted(target_date)
            target_loc = idx - 1 if idx > 0 else 0
            target_date = calendar[target_loc]

        # 获取历史日期范围
        max_lookback = int(self.seq_len * self.max_lookback_multiplier)
        start_loc = max(0, target_loc - max_lookback)
        history_dates = calendar[start_loc:target_loc]

        if len(history_dates) < min_required:
            logger.warning(f"历史数据不足: {len(history_dates)} < {min_required}")
            return {}

        # 加载因子数据 (Bug修复: 使用新方法)
        all_factors = {}
        for date in history_dates:
            date_str = date.strftime('%Y%m%d')
            df = self._load_factor_data(date_str)
            if df is not None:
                all_factors[date] = df

        if len(all_factors) < min_required:
            logger.warning(f"因子缓存不足: {len(all_factors)} 天")
            return {}

        # 确定特征列
        sample_df = list(all_factors.values())[0]
        exclude_cols = {'code', 'name', 'date', 'industry', 'label', 'forward_return', 'total_score'}
        feature_cols = [
            c for c in sample_df.columns
            if c not in exclude_cols
            and not c.endswith('_raw')
            and sample_df[c].dtype in [np.float64, np.float32, np.int64]
        ]
        self._feature_cols = feature_cols  # 保存供后续使用

        # 确定股票列表
        if codes is None:
            codes = set()
            for df in all_factors.values():
                codes.update(df['code'].unique())
            codes = sorted(codes)

        # 获取有效的因子日期（排序）
        sorted_dates = sorted(all_factors.keys())

        # 选取最近的 seq_len 个日期（柔性模式允许不足）
        if len(sorted_dates) >= self.seq_len:
            selected_dates = sorted_dates[-self.seq_len:]
        elif self.continuity_mode == 'flexible' and len(sorted_dates) >= min_required:
            selected_dates = sorted_dates
        else:
            logger.warning(f"可用因子日期不足: {len(sorted_dates)}")
            return {}

        # 构建序列
        sequences = {}

        for code in codes:
            if self.continuity_mode == 'flexible':
                seq = self._build_flexible_sequence(code, selected_dates, all_factors, feature_cols)
            else:
                seq = self._build_strict_sequence(code, selected_dates, all_factors, feature_cols)

            if seq is not None:
                sequences[code] = seq

        return sequences

    def _build_strict_sequence(self, code: str, dates: List[pd.Timestamp],
                                all_factors: Dict, feature_cols: List[str]) -> Optional[np.ndarray]:
        """构建严格模式序列（要求所有日期都有数据）"""
        seq = []
        for date in dates:
            if date not in all_factors:
                raise RuntimeError(f"missing factor data for {date.date()}")
            df = all_factors[date]
            code_row = df[df['code'] == code]
            if len(code_row) > 0:
                seq.append(code_row[feature_cols].values[0])
            else:
                raise RuntimeError(f"missing factor row for {code} on {date.date()}")

        seq_arr = np.array(seq, dtype=np.float32)
        return _sanitize_sequence_values(seq_arr)

    def _build_flexible_sequence(self, code: str, dates: List[pd.Timestamp],
                                   all_factors: Dict, feature_cols: List[str]) -> Optional[np.ndarray]:
        """构建柔性模式序列（允许部分缺失，使用插值填充）"""
        sequence = []
        time_gaps = []
        valid_count = 0

        prev_date = None
        for date in dates:
            # 计算时间间隔
            if prev_date:
                gap = (date - prev_date).days
            else:
                gap = 0
            time_gaps.append(gap)
            prev_date = date

            # 获取因子数据
            if date in all_factors:
                df = all_factors[date]
                code_row = df[df['code'] == code]
                if len(code_row) > 0:
                    features = code_row[feature_cols].values[0]
                    sequence.append(features)
                    valid_count += 1
                else:
                    sequence.append(None)
            else:
                sequence.append(None)

        # 检查覆盖率
        coverage = valid_count / len(sequence) if len(sequence) > 0 else 0
        if coverage < self.min_coverage:
            raise RuntimeError(f"coverage {coverage:.2f} below minimum {self.min_coverage}")

        # 柔性序列对齐到固定长度，避免 batch 堆叠尺寸不一致
        if len(sequence) < self.seq_len:
            pad_len = self.seq_len - len(sequence)
            sequence = [None] * pad_len + sequence
            time_gaps = [0] * pad_len + time_gaps
        elif len(sequence) > self.seq_len:
            sequence = sequence[-self.seq_len:]
            time_gaps = time_gaps[-self.seq_len:]

        if any(item is None for item in sequence):
            raise RuntimeError(f"missing factors for {code} in flexible sequence")
        result = np.array(sequence, dtype=np.float32)

        # 添加时间间隔特征
        if self.add_time_gap_feature:
            time_gaps = np.array(time_gaps, dtype=np.float32).reshape(-1, 1) / 30.0
            result = np.hstack([result, time_gaps])

        result = _sanitize_sequence_values(result)
        return result.astype(np.float32)


def create_dataloaders(config: dict,
                       train_start: str,
                       train_end: str,
                       val_start: str = None,
                       val_end: str = None,
                       batch_size: int = 256,
                       universe_id: str = None) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    创建训练和验证数据加载器

    Args:
        config: 配置字典
        train_start: 训练开始日期
        train_end: 训练结束日期
        val_start: 验证开始日期
        val_end: 验证结束日期
        batch_size: 批大小
        universe_id: 股票池标识 (Bug修复: 用于匹配因子缓存)

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = FactorSequenceDataset(
        config,
        start_date=train_start,
        end_date=train_end,
        mode='train',
        universe_id=universe_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows 兼容
        pin_memory=True,
    )

    val_loader = None
    if val_start and val_end:
        val_dataset = FactorSequenceDataset(
            config,
            start_date=val_start,
            end_date=val_end,
            mode='val',
            universe_id=universe_id,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    return train_loader, val_loader
