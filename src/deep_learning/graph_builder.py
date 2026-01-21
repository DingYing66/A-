"""
图结构构建器模块 (Phase 2)

构建股票关系图，用于 GNN 模型：
- 行业图：同行业股票相连
- 相关性图：收益率相关性高的股票相连
- 混合图：行业 + 相关性融合

强约束：
- 相关性计算只使用 t 日之前的数据，禁止跨期统计
- 缓存使用 universe_id + adjust 验证一致性

用法：
    from src.deep_learning.graph_builder import StockGraphBuilder

    builder = StockGraphBuilder(config)
    graph = builder.build_graph(codes, date, factor_df)
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime

import numpy as np
import pandas as pd

# 导入数据契约
from .contracts import GraphBundle, CacheMeta, generate_universe_id

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger, get_project_root, ensure_dir, load_parquet

logger = setup_logger(__name__)

# 检查 PyTorch 和 PyG 是否可用
TORCH_AVAILABLE = False
PYG_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import torch_geometric
    PYG_AVAILABLE = True
except ImportError:
    pass


class StockGraphBuilder:
    """
    股票关系图构建器

    支持三种图类型：
    - industry: 行业邻接图（同行业全连接）
    - correlation: 相关性图（Top-K 相关）
    - hybrid: 混合图（行业 + 相关性加权融合）

    支持动态边更新频率：
    - daily: 每日更新
    - weekly: 每周更新
    - monthly: 每月更新
    """

    def __init__(self, config: dict = None):
        """
        Args:
            config: 配置字典
        """
        self.config = config or {}
        graph_config = self.config.get('graph', {})

        # 图类型
        self.edge_type = graph_config.get('edge_type', 'hybrid')

        # 刷新频率 (daily/weekly/monthly)
        self.refresh_freq = graph_config.get('refresh_freq', 'monthly')
        if self.refresh_freq not in ('daily', 'weekly', 'monthly'):
            logger.warning(f"无效的 refresh_freq: {self.refresh_freq}，使用默认值 'monthly'")
            self.refresh_freq = 'monthly'

        # 相关性参数
        self.corr_window = graph_config.get('corr_window', 60)  # 相关性计算窗口
        self.corr_top_k = graph_config.get('top_k', 10)  # 每只股票保留 Top-K 相关
        self.corr_threshold = graph_config.get('corr_threshold', 0.3)  # 相关性阈值

        # 行业参数
        self.industry_source = graph_config.get('industry_source', 'ths')  # ths/em

        # 混合图权重
        self.industry_weight = graph_config.get('industry_weight', 0.5)
        self.corr_weight = graph_config.get('corr_weight', 0.5)

        # 自环
        self.add_self_loops = graph_config.get('add_self_loops', True)

        # 缓存
        self.cache_dir = Path(get_project_root()) / 'data' / 'processed' / 'graphs'
        ensure_dir(self.cache_dir)

        # 日K缓存目录
        self.price_cache_dir = Path(get_project_root()) / 'data' / 'raw' / 'daily_price'

        # 复权口径
        data_config = self.config.get('data_fetch', self.config.get('data', {}))
        self.adjust = data_config.get('adjust', 'hfq')

        # 上次构建日期记录 (用于判断是否需要重建)
        self._last_build_dates: Dict[str, pd.Timestamp] = {}  # {universe_id: last_build_date}

        logger.info(f"初始化图构建器: type={self.edge_type}, refresh={self.refresh_freq}, "
                    f"corr_window={self.corr_window}, top_k={self.corr_top_k}")

    def should_rebuild(self, current_date: pd.Timestamp, universe_id: str) -> bool:
        """
        判断是否需要重建图

        根据 refresh_freq 配置决定是否需要重新构建图：
        - daily: 每日重建
        - weekly: 每周重建（距上次构建 >= 5 个交易日）
        - monthly: 每月重建（月份变化时）

        Args:
            current_date: 当前日期
            universe_id: 股票池标识

        Returns:
            是否需要重建
        """
        if universe_id not in self._last_build_dates:
            return True

        last_build_date = self._last_build_dates[universe_id]

        if self.refresh_freq == 'daily':
            # 每日更新
            return True
        elif self.refresh_freq == 'weekly':
            # 每周更新：距上次构建 >= 5 个交易日
            days_diff = (current_date - last_build_date).days
            return days_diff >= 5
        else:  # monthly
            # 每月更新：月份变化时
            return current_date.month != last_build_date.month or \
                   current_date.year != last_build_date.year

    def _get_period_key(self, date: pd.Timestamp) -> str:
        """
        获取周期键（用于缓存）

        根据 refresh_freq 返回对应的周期标识：
        - daily: YYYYMMDD
        - weekly: YYYYWNN (年份+周数)
        - monthly: YYYYMM

        Args:
            date: 日期

        Returns:
            周期键字符串
        """
        if self.refresh_freq == 'daily':
            return date.strftime('%Y%m%d')
        elif self.refresh_freq == 'weekly':
            # ISO 周数
            return f"{date.year}W{date.isocalendar()[1]:02d}"
        else:  # monthly
            return date.strftime('%Y%m')

    def _get_period_cache_path(self, date: pd.Timestamp, edge_type: str, universe_id: str) -> Path:
        """
        获取基于周期的缓存路径

        Args:
            date: 日期
            edge_type: 图类型
            universe_id: 股票池标识

        Returns:
            缓存文件路径
        """
        period_key = self._get_period_key(date)
        return self.cache_dir / f'graph_{edge_type}_{period_key}_{universe_id}.npz'

    def _find_valid_period_cache(self, date: pd.Timestamp, edge_type: str, universe_id: str) -> Optional[Path]:
        """
        查找当前周期内有效的缓存

        对于 weekly/monthly 模式，查找同一周期内已有的缓存

        Args:
            date: 日期
            edge_type: 图类型
            universe_id: 股票池标识

        Returns:
            缓存路径（如果存在）
        """
        period_cache_path = self._get_period_cache_path(date, edge_type, universe_id)

        if period_cache_path.exists():
            return period_cache_path

        # 对于非 daily 模式，检查是否有同期缓存
        if self.refresh_freq != 'daily':
            period_key = self._get_period_key(date)
            # 搜索同周期的缓存文件
            pattern = f'graph_{edge_type}_{period_key}*.npz'
            matching_files = list(self.cache_dir.glob(pattern))
            if matching_files:
                # 返回最新的缓存
                return max(matching_files, key=lambda p: p.stat().st_mtime)

        return None

    def clear_period_cache(self, date: pd.Timestamp = None, edge_type: str = None, universe_id: str = None):
        """
        清除周期缓存

        Args:
            date: 指定日期（清除该周期的缓存）
            edge_type: 图类型（None 表示所有类型）
            universe_id: 股票池标识（None 表示所有股票池）
        """
        pattern_parts = ['graph']

        if edge_type:
            pattern_parts.append(edge_type)
        else:
            pattern_parts.append('*')

        if date:
            period_key = self._get_period_key(date)
            pattern_parts.append(period_key + '*')
        else:
            pattern_parts.append('*')

        if universe_id:
            pattern_parts.append(f'{universe_id}.npz')
        else:
            pattern_parts.append('*.npz')

        pattern = '_'.join(pattern_parts)
        matching_files = list(self.cache_dir.glob(pattern))

        for f in matching_files:
            try:
                f.unlink()
                # 同时删除元信息文件
                meta_file = f.with_suffix('.meta.json')
                if meta_file.exists():
                    meta_file.unlink()
                logger.debug(f"已删除缓存: {f.name}")
            except Exception as e:
                logger.warning(f"删除缓存失败 {f.name}: {e}")

        if matching_files:
            logger.info(f"已清除 {len(matching_files)} 个图缓存文件")

        # 清除内存中的构建日期记录
        if universe_id and universe_id in self._last_build_dates:
            del self._last_build_dates[universe_id]
        elif universe_id is None:
            self._last_build_dates.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            缓存统计字典
        """
        cache_files = list(self.cache_dir.glob('graph_*.npz'))

        stats = {
            'total_files': len(cache_files),
            'total_size_mb': sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
            'by_edge_type': {},
            'by_period': {},
        }

        for f in cache_files:
            parts = f.stem.split('_')
            if len(parts) >= 3:
                edge_type = parts[1]
                period = parts[2]

                # 按图类型统计
                if edge_type not in stats['by_edge_type']:
                    stats['by_edge_type'][edge_type] = 0
                stats['by_edge_type'][edge_type] += 1

                # 按周期统计
                if period not in stats['by_period']:
                    stats['by_period'][period] = 0
                stats['by_period'][period] += 1

        return stats

    def build_graph(self,
                    codes: List[str],
                    date: pd.Timestamp,
                    factor_df: pd.DataFrame = None,
                    industry_map: Dict[str, str] = None,
                    force_rebuild: bool = False) -> GraphBundle:
        """
        构建股票关系图

        根据 refresh_freq 配置决定是否使用缓存：
        - daily: 每日重新构建
        - weekly: 每周重新构建（同周内复用缓存）
        - monthly: 每月重新构建（同月内复用缓存）

        Args:
            codes: 股票代码列表
            date: 构建日期（相关性只用此日期之前的数据）
            factor_df: 因子数据（用于节点特征）
            industry_map: 行业映射 {code: industry}
            force_rebuild: 强制重建（忽略 refresh_freq）

        Returns:
            GraphBundle 对象
        """
        # 生成 universe_id
        universe_id = generate_universe_id(codes, date.strftime('%Y%m%d'))

        # 检查是否需要重建
        need_rebuild = force_rebuild or self.should_rebuild(date, universe_id)

        # 尝试加载周期缓存（如果不需要重建）
        if not need_rebuild:
            cache_path = self._find_valid_period_cache(date, self.edge_type, universe_id)
            if cache_path is not None:
                cached = self._load_cache(cache_path)
                if cached is not None:
                    logger.info(f"复用 {self.refresh_freq} 周期图缓存: {cache_path.name}")
                    return cached

        # 尝试加载当前日期的精确缓存
        period_cache_path = self._get_period_cache_path(date, self.edge_type, universe_id)
        if period_cache_path.exists():
            cached = self._load_cache(period_cache_path)
            if cached is not None:
                logger.info(f"加载图缓存: {period_cache_path.name}")
                # 更新构建日期记录
                self._last_build_dates[universe_id] = date
                return cached

        logger.info(f"构建 {self.edge_type} 图 (refresh={self.refresh_freq}): "
                    f"{len(codes)} 节点, date={date.date()}")

        # 构建节点索引映射
        code_to_idx = {code: i for i, code in enumerate(codes)}

        # 根据图类型构建边
        if self.edge_type == 'industry':
            edge_index, edge_weight = self._build_industry_edges(
                codes, code_to_idx, industry_map
            )
        elif self.edge_type == 'correlation':
            edge_index, edge_weight = self._build_correlation_edges(
                codes, code_to_idx, date
            )
        elif self.edge_type == 'hybrid':
            edge_index, edge_weight = self._build_hybrid_edges(
                codes, code_to_idx, date, industry_map
            )
        else:
            raise ValueError(f"未知图类型: {self.edge_type}")

        # 添加自环
        if self.add_self_loops:
            edge_index, edge_weight = self._add_self_loops(
                edge_index, edge_weight, len(codes)
            )

        # 构建节点特征
        node_features = self._build_node_features(codes, factor_df)

        # 创建 GraphBundle
        graph = GraphBundle(
            node_features=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            node_codes=codes,
            graph_type=self.edge_type,
            date=date,
            universe_id=universe_id,
            meta=CacheMeta(
                universe_id=universe_id,
                config_hash=self._compute_config_hash(),
                adjust=self.adjust,
                n_samples=len(codes),
                date_range=(date.strftime('%Y%m%d'), date.strftime('%Y%m%d')),
            )
        )

        # 保存缓存（使用周期路径）
        save_cache_path = self._get_period_cache_path(date, self.edge_type, universe_id)
        self._save_cache(save_cache_path, graph)

        # 更新构建日期记录
        self._last_build_dates[universe_id] = date

        logger.info(f"图构建完成: {graph.n_nodes} 节点, {graph.n_edges} 边, "
                    f"平均度数={graph.avg_degree:.2f}, 周期={self._get_period_key(date)}")

        return graph

    def _build_industry_edges(self,
                               codes: List[str],
                               code_to_idx: Dict[str, int],
                               industry_map: Dict[str, str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建行业邻接边

        同行业股票全连接，但限制最大行业规模防止边爆炸

        Bug修复 R2: 添加空映射检测和组大小限制
        """
        if industry_map is None:
            industry_map = self._load_industry_map(codes)

        # R2 修复: 检查空映射
        if not industry_map:
            logger.warning("行业映射为空，跳过行业边构建。请先运行 IndustryFetcher 获取数据。")
            return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

        # R2 修复: 最大行业组大小限制 (防止 'unknown' 全连接爆炸)
        MAX_GROUP_SIZE = 100

        # 按行业分组
        industry_groups: Dict[str, List[str]] = {}
        unknown_count = 0
        for code in codes:
            industry = industry_map.get(code, 'unknown')
            if industry == 'unknown':
                unknown_count += 1
            if industry not in industry_groups:
                industry_groups[industry] = []
            industry_groups[industry].append(code)

        # R2 修复: 过滤过大的 'unknown' 组
        if 'unknown' in industry_groups and len(industry_groups['unknown']) > MAX_GROUP_SIZE:
            logger.warning(f"'unknown' 行业组过大 ({len(industry_groups['unknown'])} 只股票)，"
                          f"超过阈值 {MAX_GROUP_SIZE}，跳过以防止边爆炸。请补充行业映射数据。")
            del industry_groups['unknown']

        # 统计行业覆盖率
        if unknown_count > 0:
            coverage = 100 * (len(codes) - unknown_count) / len(codes)
            logger.info(f"行业映射覆盖率: {len(codes) - unknown_count}/{len(codes)} ({coverage:.1f}%)")

        # 构建边
        src_nodes = []
        dst_nodes = []
        weights = []

        for industry, group_codes in industry_groups.items():
            if len(group_codes) < 2:
                continue

            # R2 修复: 对过大的行业组进行采样
            if len(group_codes) > MAX_GROUP_SIZE:
                logger.warning(f"行业 '{industry}' 组过大 ({len(group_codes)} 只)，采样到 {MAX_GROUP_SIZE}")
                group_codes = group_codes[:MAX_GROUP_SIZE]

            # 同行业全连接
            for i, code_i in enumerate(group_codes):
                for j, code_j in enumerate(group_codes):
                    if i != j:
                        src_nodes.append(code_to_idx[code_i])
                        dst_nodes.append(code_to_idx[code_j])
                        weights.append(1.0)

        if len(src_nodes) == 0:
            # 没有边时返回空数组
            return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

        edge_index = np.array([src_nodes, dst_nodes], dtype=np.int64)
        edge_weight = np.array(weights, dtype=np.float32)

        return edge_index, edge_weight

    def _build_correlation_edges(self,
                                   codes: List[str],
                                   code_to_idx: Dict[str, int],
                                   date: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建相关性边

        只使用 date 之前的历史数据计算相关性
        """
        # 加载历史收益率
        returns_df = self._load_returns_history(codes, date)

        if returns_df is None or len(returns_df) < 20:
            logger.warning("收益率数据不足，返回空图")
            return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

        # C7 修复: 处理缺失值，避免无效相关性
        nan_count = returns_df.isna().sum().sum()
        if nan_count > 0:
            logger.debug(f"收益率数据包含 {nan_count} 个 NaN，将填充为 0")
            returns_df = returns_df.fillna(0)

        # 计算相关性矩阵
        corr_matrix = returns_df.corr()

        # C7 修复: 过滤相关性矩阵中的 NaN
        corr_nan_count = corr_matrix.isna().sum().sum()
        if corr_nan_count > 0:
            logger.debug(f"相关性矩阵包含 {corr_nan_count} 个 NaN，将填充为 0")
            corr_matrix = corr_matrix.fillna(0)

        # 构建 Top-K 边
        src_nodes = []
        dst_nodes = []
        weights = []

        for code_i in codes:
            if code_i not in corr_matrix.columns:
                continue

            idx_i = code_to_idx[code_i]

            # 获取与其他股票的相关性
            corrs = corr_matrix[code_i].drop(code_i, errors='ignore')

            # C7 修复: 移除 NaN 相关性
            corrs = corrs.dropna()

            # 过滤阈值
            valid_corrs = corrs[corrs >= self.corr_threshold]

            # 取 Top-K
            top_k_codes = valid_corrs.nlargest(self.corr_top_k).index.tolist()

            for code_j in top_k_codes:
                if code_j in code_to_idx:
                    idx_j = code_to_idx[code_j]
                    src_nodes.append(idx_i)
                    dst_nodes.append(idx_j)
                    weights.append(float(corr_matrix.loc[code_i, code_j]))

        if len(src_nodes) == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

        edge_index = np.array([src_nodes, dst_nodes], dtype=np.int64)
        edge_weight = np.array(weights, dtype=np.float32)

        return edge_index, edge_weight

    def _build_hybrid_edges(self,
                             codes: List[str],
                             code_to_idx: Dict[str, int],
                             date: pd.Timestamp,
                             industry_map: Dict[str, str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建混合图（行业 + 相关性）

        边权重 = industry_weight * industry_edge + corr_weight * corr_edge
        """
        # 构建行业边
        ind_edge_index, ind_edge_weight = self._build_industry_edges(
            codes, code_to_idx, industry_map
        )

        # 构建相关性边
        corr_edge_index, corr_edge_weight = self._build_correlation_edges(
            codes, code_to_idx, date
        )

        # 合并边（使用边集合去重）
        edge_dict: Dict[Tuple[int, int], float] = {}

        # 添加行业边
        for i in range(ind_edge_index.shape[1]):
            src, dst = int(ind_edge_index[0, i]), int(ind_edge_index[1, i])
            key = (src, dst)
            edge_dict[key] = self.industry_weight * ind_edge_weight[i]

        # 添加相关性边（累加权重）
        for i in range(corr_edge_index.shape[1]):
            src, dst = int(corr_edge_index[0, i]), int(corr_edge_index[1, i])
            key = (src, dst)
            if key in edge_dict:
                edge_dict[key] += self.corr_weight * corr_edge_weight[i]
            else:
                edge_dict[key] = self.corr_weight * corr_edge_weight[i]

        if len(edge_dict) == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

        # 转换为数组
        src_nodes = [k[0] for k in edge_dict.keys()]
        dst_nodes = [k[1] for k in edge_dict.keys()]
        weights = list(edge_dict.values())

        edge_index = np.array([src_nodes, dst_nodes], dtype=np.int64)
        edge_weight = np.array(weights, dtype=np.float32)

        return edge_index, edge_weight

    def _add_self_loops(self,
                        edge_index: np.ndarray,
                        edge_weight: np.ndarray,
                        n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
        """添加自环"""
        self_loops = np.arange(n_nodes)
        self_loop_index = np.stack([self_loops, self_loops], axis=0)
        self_loop_weight = np.ones(n_nodes, dtype=np.float32)

        new_edge_index = np.concatenate([edge_index, self_loop_index], axis=1)
        new_edge_weight = np.concatenate([edge_weight, self_loop_weight], axis=0)

        return new_edge_index, new_edge_weight

    def _build_node_features(self,
                              codes: List[str],
                              factor_df: pd.DataFrame = None) -> np.ndarray:
        """构建节点特征"""
        if factor_df is None or len(factor_df) == 0:
            # 返回单位特征
            return np.ones((len(codes), 1), dtype=np.float32)

        # 确定特征列
        exclude_cols = {'code', 'name', 'date', 'industry', 'label', 'forward_return', 'total_score'}
        feature_cols = [c for c in factor_df.columns
                       if c not in exclude_cols and factor_df[c].dtype in [np.float64, np.float32, np.int64]]

        if len(feature_cols) == 0:
            raise RuntimeError("no numeric features available for graph nodes")

        # 按代码对齐
        features = []
        for code in codes:
            code_row = factor_df[factor_df['code'] == code]
            if len(code_row) == 0:
                raise RuntimeError(f"missing features for code {code}")
            feat = code_row[feature_cols].values[0]
            features.append(feat)

        features = np.array(features, dtype=np.float32)

        if np.isnan(features).any():
            raise RuntimeError("node features contain NaN")
        if np.isinf(features).any():
            raise RuntimeError("node features contain inf")

        return features

    def _load_industry_map(self, codes: List[str]) -> Dict[str, str]:
        """
        加载行业映射

        Bug修复: 支持多种来源，优先 processed 目录 (规范化格式)
        """
        root = Path(get_project_root())

        processed_path = root / 'data' / 'processed' / 'industry_map.parquet'
        if not processed_path.exists():
            raise RuntimeError("industry_map.parquet not found in processed data")
        df = load_parquet(processed_path)
        if df is None or 'code' not in df.columns or 'industry' not in df.columns:
            raise RuntimeError("industry_map.parquet missing required columns")
        logger.debug(f"加载行业映射 (processed): {len(df)} 条记录")
        return dict(zip(df['code'], df['industry']))

    def _load_returns_history(self,
                               codes: List[str],
                               end_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        加载历史收益率数据（防泄露：只使用 end_date 之前的数据）

        Bug修复: 使用 adjust 分区路径
        """
        returns_dict = {}

        for code in codes:
            # P1-3: 严格模式 - 只使用 adjust 分区路径
            # 路径: data/raw/daily_price/{adjust}/{code}.parquet
            price_path = self.price_cache_dir / self.adjust / f'{code}.parquet'

            if not price_path.exists():
                raise RuntimeError(f"missing price data for {code}: {price_path}")

            df = load_parquet(price_path)
            if df is None or len(df) == 0:
                raise RuntimeError(f"empty price data for {code}")

            if 'date' not in df.columns:
                raise RuntimeError(f"missing date column for {code}")
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] < end_date]

            # 取最近 corr_window 天
            df = df.tail(self.corr_window)

            if len(df) < 20:
                raise RuntimeError(f"insufficient history for {code}")

            if 'close' not in df.columns:
                raise RuntimeError(f"missing close column for {code}")
            returns = df['close'].pct_change().dropna()
            if len(returns) == 0:
                raise RuntimeError(f"empty returns for {code}")
            returns_dict[code] = returns.values

        # C7 修复: 验证数据充足性
        if len(returns_dict) < 10:
            raise RuntimeError(f"insufficient valid stocks ({len(returns_dict)} < 10)")

        # C7 修复: 检查并记录长度差异
        lengths = [len(r) for r in returns_dict.values()]
        min_len = min(lengths)
        max_len = max(lengths)

        # 最小重叠天数要求
        MIN_OVERLAP_DAYS = 20
        if min_len < MIN_OVERLAP_DAYS:
            raise RuntimeError(f"min returns length {min_len} < {MIN_OVERLAP_DAYS}")

        if max_len - min_len > 20:
            logger.debug(f"股票收益率长度差异较大: min={min_len}, max={max_len}")

        # 对齐长度
        returns_df = pd.DataFrame({
            code: r[-min_len:] for code, r in returns_dict.items()
        })

        return returns_df

    def _compute_config_hash(self) -> str:
        """计算配置哈希"""
        config_str = json.dumps({
            'edge_type': self.edge_type,
            'refresh_freq': self.refresh_freq,
            'corr_window': self.corr_window,
            'corr_top_k': self.corr_top_k,
            'corr_threshold': self.corr_threshold,
            'industry_weight': self.industry_weight,
            'corr_weight': self.corr_weight,
            'adjust': self.adjust,
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _get_cache_path(self, date: pd.Timestamp, edge_type: str, universe_id: str) -> Path:
        """获取缓存路径"""
        date_str = date.strftime('%Y%m%d')
        return self.cache_dir / f'graph_{edge_type}_{date_str}_{universe_id}.npz'

    def _save_cache(self, cache_path: Path, graph: GraphBundle):
        """保存图缓存 (Fix 3: 安全格式，避免 pickle)"""
        # 将字符串数据编码为字节数组，避免使用 pickle
        node_codes_bytes = np.array([c.encode('utf-8') for c in graph.node_codes], dtype='S10')
        graph_type_bytes = np.array([graph.graph_type.encode('utf-8')], dtype='S32')
        universe_id_bytes = np.array([graph.universe_id.encode('utf-8')], dtype='S32')

        np.savez_compressed(
            cache_path,
            node_features=graph.node_features,
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight if graph.edge_weight is not None else np.array([]),
            node_codes=node_codes_bytes,
            graph_type=graph_type_bytes,
            universe_id=universe_id_bytes,
        )

        # 保存元信息
        if graph.meta:
            graph.meta.save(cache_path)

        logger.debug(f"图缓存已保存: {cache_path}")

    def _load_cache(self, cache_path: Path) -> Optional[GraphBundle]:
        """加载图缓存 (Fix 3: 禁用 pickle，使用安全格式，兼容旧格式)"""
        try:
            # Fix 3: 禁用 pickle 以避免反序列化执行风险
            data = np.load(cache_path, allow_pickle=False)

            edge_weight = data['edge_weight']
            if len(edge_weight) == 0:
                edge_weight = None

            # 解码字节数组为字符串 (兼容新旧格式)
            node_codes_raw = data['node_codes']
            if node_codes_raw.dtype.kind == 'S':  # 字节数组 (新格式)
                node_codes = [c.decode('utf-8') for c in node_codes_raw]
            else:  # 字符串数组 (旧格式，可能是 Unicode)
                node_codes = [str(c) for c in node_codes_raw]

            # graph_type 兼容处理
            graph_type_raw = data['graph_type']
            if isinstance(graph_type_raw, np.ndarray):
                if graph_type_raw.dtype.kind == 'S':  # 字节数组
                    graph_type = graph_type_raw[0].decode('utf-8')
                else:
                    graph_type = str(graph_type_raw[0]) if len(graph_type_raw) > 0 else str(graph_type_raw)
            else:
                graph_type = str(graph_type_raw)

            # universe_id 兼容处理
            universe_id_raw = data['universe_id']
            if isinstance(universe_id_raw, np.ndarray):
                if universe_id_raw.dtype.kind == 'S':  # 字节数组
                    universe_id = universe_id_raw[0].decode('utf-8')
                else:
                    universe_id = str(universe_id_raw[0]) if len(universe_id_raw) > 0 else str(universe_id_raw)
            else:
                universe_id = str(universe_id_raw)

            graph = GraphBundle(
                node_features=data['node_features'],
                edge_index=data['edge_index'],
                edge_weight=edge_weight,
                node_codes=node_codes,
                graph_type=graph_type,
                universe_id=universe_id,
                meta=CacheMeta.load(cache_path),
            )

            return graph

        except Exception as e:
            # 旧格式或损坏的缓存，删除并返回 None 触发重建
            logger.warning(f"图缓存加载失败，将删除并重建: {cache_path}, 错误: {e}")
            try:
                cache_path.unlink()
            except Exception:
                pass
            return None


def build_stock_graph(codes: List[str],
                      date: pd.Timestamp,
                      factor_df: pd.DataFrame = None,
                      edge_type: str = 'hybrid',
                      refresh_freq: str = 'monthly',
                      force_rebuild: bool = False,
                      **kwargs) -> GraphBundle:
    """
    便捷函数：构建股票图

    Args:
        codes: 股票代码列表
        date: 日期
        factor_df: 因子数据
        edge_type: 图类型 ('industry', 'correlation', 'hybrid')
        refresh_freq: 刷新频率 ('daily', 'weekly', 'monthly')
        force_rebuild: 强制重建（忽略 refresh_freq）
        **kwargs: 传递给 StockGraphBuilder (corr_window, top_k, corr_threshold 等)

    Returns:
        GraphBundle
    """
    config = {'graph': {'edge_type': edge_type, 'refresh_freq': refresh_freq, **kwargs}}
    builder = StockGraphBuilder(config)
    return builder.build_graph(codes, date, factor_df, force_rebuild=force_rebuild)
