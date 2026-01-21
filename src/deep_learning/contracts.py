"""
数据契约模块 (Data Contracts)

定义统一的数据结构，确保模块间数据传递的一致性：
- FactorFrame: 因子横截面数据
- SequenceBatch: 序列数据批次
- GraphBundle: 图结构数据包
- CacheMeta: 缓存元信息

用法：
    from src.deep_learning.contracts import FactorFrame, SequenceBatch, GraphBundle

    # 创建因子数据
    factor_frame = FactorFrame.from_dataframe(df, date=date)

    # 验证数据完整性
    factor_frame.validate()

    # 获取缓存 key
    cache_key = factor_frame.get_cache_key()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger

logger = setup_logger(__name__)


# ===================== 缓存元信息 =====================

@dataclass
class CacheMeta:
    """
    缓存元信息 - 用于验证缓存一致性

    包含：
    - universe_id: 股票池标识
    - config_hash: 配置哈希
    - adjust: 复权口径
    - source_versions: 数据源版本
    - created_at: 创建时间
    """

    universe_id: str
    config_hash: str
    adjust: str
    source_versions: Dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # 额外元信息
    n_samples: int = 0
    date_range: Tuple[str, str] = ('', '')
    feature_cols: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'universe_id': self.universe_id,
            'config_hash': self.config_hash,
            'adjust': self.adjust,
            'source_versions': self.source_versions,
            'created_at': self.created_at,
            'n_samples': self.n_samples,
            'date_range': self.date_range,
            'feature_cols': self.feature_cols,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CacheMeta':
        """从字典创建"""
        return cls(
            universe_id=data.get('universe_id', ''),
            config_hash=data.get('config_hash', ''),
            adjust=data.get('adjust', ''),
            source_versions=data.get('source_versions', {}),
            created_at=data.get('created_at', ''),
            n_samples=data.get('n_samples', 0),
            date_range=tuple(data.get('date_range', ('', ''))),
            feature_cols=data.get('feature_cols', []),
        )

    def save(self, path: Path):
        """保存元信息"""
        meta_path = path.with_suffix('.meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional['CacheMeta']:
        """加载元信息"""
        meta_path = path.with_suffix('.meta.json')
        if not meta_path.exists():
            return None
        with open(meta_path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    def is_compatible(self, other: 'CacheMeta') -> Tuple[bool, List[str]]:
        """
        检查缓存兼容性

        Returns:
            (是否兼容, 不兼容原因列表)
        """
        issues = []

        if self.universe_id != other.universe_id:
            issues.append(f'universe_id mismatch: {self.universe_id} vs {other.universe_id}')

        if self.config_hash != other.config_hash:
            issues.append(f'config_hash mismatch: {self.config_hash} vs {other.config_hash}')

        if self.adjust != other.adjust:
            issues.append(f'adjust mismatch: {self.adjust} vs {other.adjust}')

        return len(issues) == 0, issues


# ===================== UniverseID 生成器 =====================

def generate_universe_id(codes: List[str],
                         date: str,
                         filter_rules: Dict[str, Any] = None) -> str:
    """
    生成股票池唯一标识

    Args:
        codes: 股票代码列表
        date: 日期
        filter_rules: 过滤规则（如 exclude_st, min_amount 等）

    Returns:
        universe_id 字符串
    """
    # 排序确保一致性
    sorted_codes = sorted(codes)

    # 构建哈希内容
    content = {
        'n_codes': len(sorted_codes),
        'codes_hash': hashlib.md5('|'.join(sorted_codes).encode()).hexdigest()[:8],
        'date': date,
        'filter_rules': filter_rules or {},
    }

    content_str = json.dumps(content, sort_keys=True)
    full_hash = hashlib.md5(content_str.encode()).hexdigest()

    # 格式: {date}_{n_codes}_{hash}
    return f"{date}_{len(codes)}_{full_hash[:8]}"


# ===================== FactorFrame =====================

@dataclass
class FactorFrame:
    """
    因子横截面数据容器

    包含：
    - 必须列: code, date
    - 因子列: 所有数值因子
    - 可选列: industry, name, market_state, tradeable
    - 元信息: CacheMeta
    """

    data: pd.DataFrame
    date: pd.Timestamp
    universe_id: str
    meta: CacheMeta

    # 必须列
    REQUIRED_COLS = {'code', 'date'}

    # 因子列前缀（用于自动识别）
    FACTOR_PREFIXES = ('ret_', 'vol_', 'mom_', 'val_', 'qual_', 'flow_', 'sent_')

    def __post_init__(self):
        """验证数据"""
        self.validate()

    @classmethod
    def from_dataframe(cls,
                       df: pd.DataFrame,
                       date: pd.Timestamp,
                       universe_id: str = None,
                       config_hash: str = '',
                       adjust: str = 'hfq') -> 'FactorFrame':
        """从 DataFrame 创建"""
        df = df.copy()

        # 确保日期列
        if 'date' not in df.columns:
            df['date'] = date

        # 生成 universe_id
        if universe_id is None:
            codes = df['code'].tolist()
            universe_id = generate_universe_id(codes, date.strftime('%Y%m%d'))

        # 创建元信息
        meta = CacheMeta(
            universe_id=universe_id,
            config_hash=config_hash,
            adjust=adjust,
            n_samples=len(df),
            date_range=(date.strftime('%Y%m%d'), date.strftime('%Y%m%d')),
            feature_cols=cls._identify_factor_cols(df),
        )

        return cls(data=df, date=date, universe_id=universe_id, meta=meta)

    @staticmethod
    def _identify_factor_cols(df: pd.DataFrame) -> List[str]:
        """识别因子列"""
        exclude = {'code', 'name', 'date', 'industry', 'tradeable', 'suspended',
                   'label', 'forward_return', 'total_score', 'rank'}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude]

    def validate(self) -> bool:
        """验证数据完整性"""
        # 检查必须列
        missing = self.REQUIRED_COLS - set(self.data.columns)
        if missing:
            raise ValueError(f"缺少必须列: {missing}")

        # 检查空数据
        if len(self.data) == 0:
            logger.warning("FactorFrame 数据为空")

        # 检查重复
        if self.data['code'].duplicated().any():
            logger.warning("存在重复的股票代码")

        return True

    def get_factor_cols(self) -> List[str]:
        """获取因子列名"""
        return self.meta.feature_cols

    def get_codes(self) -> List[str]:
        """获取股票代码列表"""
        return self.data['code'].tolist()

    def get_cache_key(self) -> str:
        """获取缓存 key"""
        return f"factors_{self.date.strftime('%Y%m%d')}_{self.universe_id}"

    def filter_tradeable(self) -> 'FactorFrame':
        """过滤可交易股票"""
        if 'tradeable' in self.data.columns:
            filtered = self.data[self.data['tradeable'] == True].copy()
        else:
            filtered = self.data.copy()

        return FactorFrame(
            data=filtered,
            date=self.date,
            universe_id=self.universe_id + '_tradeable',
            meta=self.meta,
        )


# ===================== SequenceBatch =====================

@dataclass
class SequenceBatch:
    """
    序列数据批次容器

    包含：
    - sequences: 序列张量 [batch, seq_len, n_features]
    - labels: 标签张量 [batch]
    - codes: 股票代码列表
    - dates: 日期列表
    - tradeable_mask: 可交易掩码
    - universe_id: 股票池标识
    """

    sequences: Any  # torch.Tensor or np.ndarray
    labels: Any
    codes: List[str]
    dates: List[pd.Timestamp]
    tradeable_mask: Optional[Any] = None
    universe_id: str = ''
    meta: Optional[CacheMeta] = None

    @property
    def batch_size(self) -> int:
        """批次大小"""
        return len(self.codes)

    @property
    def seq_len(self) -> int:
        """序列长度"""
        if hasattr(self.sequences, 'shape'):
            return self.sequences.shape[1]
        return 0

    @property
    def n_features(self) -> int:
        """特征维度"""
        if hasattr(self.sequences, 'shape') and len(self.sequences.shape) >= 3:
            return self.sequences.shape[2]
        return 0

    def to_device(self, device) -> 'SequenceBatch':
        """移动到指定设备（GPU/CPU）"""
        if not TORCH_AVAILABLE:
            return self

        import torch

        new_sequences = self.sequences.to(device) if isinstance(self.sequences, torch.Tensor) else self.sequences
        new_labels = self.labels.to(device) if isinstance(self.labels, torch.Tensor) else self.labels
        new_mask = None
        if self.tradeable_mask is not None:
            new_mask = self.tradeable_mask.to(device) if isinstance(self.tradeable_mask, torch.Tensor) else self.tradeable_mask

        return SequenceBatch(
            sequences=new_sequences,
            labels=new_labels,
            codes=self.codes,
            dates=self.dates,
            tradeable_mask=new_mask,
            universe_id=self.universe_id,
            meta=self.meta,
        )

    def get_cache_key(self) -> str:
        """获取缓存 key"""
        date_str = self.dates[0].strftime('%Y%m%d') if self.dates else 'unknown'
        return f"seq_{date_str}_L{self.seq_len}_{self.universe_id}"


# ===================== GraphBundle =====================

@dataclass
class GraphBundle:
    """
    图结构数据包

    包含：
    - node_features: 节点特征 [n_nodes, n_features]
    - edge_index: 边索引 [2, n_edges]
    - edge_weight: 边权重 [n_edges]
    - node_codes: 节点对应的股票代码
    - graph_type: 图类型 (industry/corr/hybrid)
    - date: 图版本日期
    - universe_id: 股票池标识
    """

    node_features: Any  # torch.Tensor or np.ndarray
    edge_index: Any
    edge_weight: Optional[Any] = None
    node_codes: List[str] = field(default_factory=list)
    graph_type: str = 'hybrid'
    date: Optional[pd.Timestamp] = None
    universe_id: str = ''
    meta: Optional[CacheMeta] = None

    @property
    def n_nodes(self) -> int:
        """节点数量"""
        return len(self.node_codes)

    @property
    def n_edges(self) -> int:
        """边数量"""
        if hasattr(self.edge_index, 'shape'):
            return self.edge_index.shape[1]
        return 0

    @property
    def avg_degree(self) -> float:
        """平均度数"""
        if self.n_nodes == 0:
            return 0.0
        return self.n_edges / self.n_nodes

    def to_pyg_data(self):
        """转换为 PyTorch Geometric Data 对象"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch 未安装")

        try:
            from torch_geometric.data import Data
            import torch

            data = Data(
                x=torch.tensor(self.node_features, dtype=torch.float32) if not isinstance(self.node_features, torch.Tensor) else self.node_features,
                edge_index=torch.tensor(self.edge_index, dtype=torch.long) if not isinstance(self.edge_index, torch.Tensor) else self.edge_index,
            )

            if self.edge_weight is not None:
                data.edge_attr = torch.tensor(self.edge_weight, dtype=torch.float32) if not isinstance(self.edge_weight, torch.Tensor) else self.edge_weight

            return data

        except ImportError:
            raise ImportError("PyTorch Geometric 未安装")

    def get_cache_key(self) -> str:
        """获取缓存 key"""
        date_str = self.date.strftime('%Y%m%d') if self.date else 'static'
        return f"graph_{date_str}_{self.graph_type}_{self.universe_id}"

    def get_subgraph(self, node_indices: List[int]) -> 'GraphBundle':
        """获取子图"""
        # 筛选节点
        new_codes = [self.node_codes[i] for i in node_indices]
        node_map = {old: new for new, old in enumerate(node_indices)}

        # 筛选边
        if TORCH_AVAILABLE and hasattr(self.edge_index, 'numpy'):
            edge_idx_np = self.edge_index.numpy()
        else:
            edge_idx_np = np.array(self.edge_index)

        mask = np.isin(edge_idx_np[0], node_indices) & np.isin(edge_idx_np[1], node_indices)

        new_edge_index = edge_idx_np[:, mask]
        new_edge_index = np.array([[node_map[e] for e in new_edge_index[0]],
                                    [node_map[e] for e in new_edge_index[1]]])

        new_features = self.node_features[node_indices] if hasattr(self.node_features, '__getitem__') else None
        new_weight = self.edge_weight[mask] if self.edge_weight is not None else None

        return GraphBundle(
            node_features=new_features,
            edge_index=new_edge_index,
            edge_weight=new_weight,
            node_codes=new_codes,
            graph_type=self.graph_type,
            date=self.date,
            universe_id=self.universe_id + '_sub',
            meta=self.meta,
        )


# ===================== 验证函数 =====================

def validate_cache_compatibility(cache_path: Path, expected_meta: CacheMeta) -> Tuple[bool, str]:
    """
    验证缓存兼容性

    Args:
        cache_path: 缓存文件路径
        expected_meta: 期望的元信息

    Returns:
        (是否兼容, 不兼容原因)
    """
    existing_meta = CacheMeta.load(cache_path)

    if existing_meta is None:
        return False, "元信息文件不存在"

    compatible, issues = existing_meta.is_compatible(expected_meta)

    if not compatible:
        return False, "; ".join(issues)

    return True, "OK"


# ===================== Config Hash 机制 =====================

# 影响数据契约的配置字段
# 这些字段变更时，缓存应该失效重算
# Bug修复 R6: 补充缺失的关键口径字段
CONFIG_HASH_KEYS = [
    # 数据获取
    'data_fetch.adjust',
    'data_fetch.start_date',

    # 执行策略 (Bug修复: 补充标签口径)
    'execution.trade_at_close',
    'execution.execution_price',
    'execution.label_entry',           # R6 新增
    'execution.label_exit',            # R6 新增
    'execution.limit_check_price',     # R6 新增

    # 回测设置
    'backtest.rebalance_freq',
    'backtest.top_n',

    # 股票池过滤 (Bug修复: 补充可交易性相关)
    'stock_pool.exclude_st',
    'stock_pool.exclude_new_stocks',
    'stock_pool.min_avg_amount',
    'stock_pool.min_avg_turnover',
    'stock_pool.min_list_days',        # R6 新增
    'stock_pool.exclude_bj',           # R6 新增
    'stock_pool.exclude_delisted',     # R6 新增

    # 流动性配置 (Bug修复: 新增)
    'liquidity.min_amount_threshold',  # R6 新增
    'liquidity.min_turnover_pct',      # R6 新增

    # 因子设置 (Bug修复: 修正键名以匹配 config.yaml)
    'factors.momentum_windows',
    'factors.volatility_window',
    'factors.momentum_skip_days',
    'factors.flow_windows',

    # 图设置 (Bug修复: 补充行业来源)
    'graph.edge_type',
    'graph.corr_window',
    'graph.top_k',
    'graph.corr_threshold',
    'graph.industry_source',           # R6 新增

    # 序列设置
    'sequence.seq_len',
    'sequence.stride',
]


def _get_nested_value(config: dict, key: str) -> Any:
    """
    获取嵌套字典的值

    Args:
        config: 配置字典
        key: 点分隔的 key (如 'data_fetch.adjust')

    Returns:
        对应的值，不存在返回 None
    """
    keys = key.split('.')
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    return value


def compute_config_hash(config: dict,
                        hash_keys: List[str] = None,
                        include_all: bool = False) -> str:
    """
    计算配置哈希

    仅针对影响数据契约的配置字段计算哈希，
    用于缓存一致性校验。

    Args:
        config: 完整配置字典
        hash_keys: 要包含的配置 key 列表（点分隔格式）
        include_all: 是否包含所有配置（忽略 hash_keys）

    Returns:
        16 位哈希字符串
    """
    if include_all:
        # 使用所有配置
        filtered = config
    else:
        # 仅提取影响数据契约的字段
        hash_keys = hash_keys or CONFIG_HASH_KEYS
        filtered = {}

        for key in hash_keys:
            value = _get_nested_value(config, key)
            if value is not None:
                # 保持嵌套结构用于哈希
                keys = key.split('.')
                d = filtered
                for k in keys[:-1]:
                    if k not in d:
                        d[k] = {}
                    d = d[k]
                d[keys[-1]] = value

    # 序列化并计算哈希
    try:
        content = json.dumps(filtered, sort_keys=True, default=str)
        full_hash = hashlib.sha256(content.encode()).hexdigest()
        return full_hash[:16]
    except Exception as e:
        logger.warning(f"计算配置哈希失败: {e}")
        return "unknown"


def get_cache_config_hash(config: dict) -> str:
    """
    获取用于缓存的配置哈希（便捷函数）

    Args:
        config: 配置字典

    Returns:
        配置哈希
    """
    return compute_config_hash(config, CONFIG_HASH_KEYS)


def check_cache_validity(cache_meta: CacheMeta,
                         current_config: dict,
                         current_adjust: str = None) -> Tuple[bool, List[str]]:
    """
    检查缓存是否仍然有效

    Args:
        cache_meta: 缓存的元信息
        current_config: 当前配置
        current_adjust: 当前复权类型

    Returns:
        (是否有效, 无效原因列表)
    """
    issues = []

    # 检查配置哈希
    current_hash = compute_config_hash(current_config)
    if cache_meta.config_hash and cache_meta.config_hash != current_hash:
        issues.append(f"config_hash mismatch: cache={cache_meta.config_hash}, current={current_hash}")

    # 检查复权类型
    if current_adjust is not None:
        cache_adjust = cache_meta.adjust or ''
        if cache_adjust != current_adjust:
            issues.append(f"adjust mismatch: cache={cache_adjust}, current={current_adjust}")

    return len(issues) == 0, issues


def create_cache_meta(config: dict,
                      codes: List[str],
                      date: str,
                      adjust: str = None,
                      n_samples: int = 0,
                      feature_cols: List[str] = None) -> CacheMeta:
    """
    创建缓存元信息（便捷函数）

    Args:
        config: 配置字典
        codes: 股票代码列表
        date: 日期
        adjust: 复权类型
        n_samples: 样本数量
        feature_cols: 特征列名

    Returns:
        CacheMeta 对象
    """
    # 生成 universe_id
    filter_rules = {
        'exclude_st': config.get('stock_pool', {}).get('exclude_st', True),
        'min_amount': config.get('stock_pool', {}).get('min_avg_amount', 0),
    }
    universe_id = generate_universe_id(codes, date, filter_rules)

    # 计算配置哈希
    config_hash = compute_config_hash(config)

    # 获取复权类型
    if adjust is None:
        adjust = config.get('data_fetch', {}).get('adjust', 'qfq')

    return CacheMeta(
        universe_id=universe_id,
        config_hash=config_hash,
        adjust=adjust,
        n_samples=n_samples,
        date_range=(date, date),
        feature_cols=feature_cols or [],
    )

