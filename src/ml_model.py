"""
机器学习模型模块
实现XGBoost/LightGBM等模型的训练、预测和评估
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

from .utils import (
    load_config, setup_logger, save_parquet, load_parquet,
    get_project_root, ensure_dir, get_rebalance_dates,
    get_execution_config
)
from .data_processor import DataProcessor
from .factor_engine import FactorEngine

logger = setup_logger(__name__)


class PurgedGroupTimeSeriesSplit:
    """
    时间序列交叉验证 + Purge (P0-3修复版)

    参考: López de Prado, Advances in Financial Machine Learning (2018)

    关键概念:
    - Group: 按日期分组，同一天的样本必须在同一组
    - Purge: 删除训练集末尾与验证集标签重叠的样本

    设计说明 (P0-3修复):
    - 原版Embargo是"限制训练集末尾"，而非"砍掉验证集开头"
    - 对于严格的walk-forward回测，Purge已足够防止标签泄露
    - 因此简化为只使用Purge，移除Embargo (避免概念混淆)
    """

    def __init__(self, n_splits: int = 5,
                 purge_days: int = 20,
                 trade_calendar: pd.DatetimeIndex = None):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.trade_calendar = trade_calendar

    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: pd.Series = None):
        """
        生成训练/验证索引

        Args:
            X: 特征DataFrame (必须包含'date'列)
            y: 标签
            groups: 分组列 (默认用X['date'])

        Yields:
            (train_idx, val_idx) 元组
        """
        if groups is None:
            if 'date' in X.columns:
                groups = X['date']
            else:
                raise ValueError("X must contain 'date' column or provide groups")

        groups = pd.to_datetime(groups)
        unique_dates = np.sort(groups.unique())
        n_dates = len(unique_dates)

        # 计算每个fold的验证集大小
        val_size = n_dates // (self.n_splits + 1)

        for i in range(self.n_splits):
            # 验证集日期范围 (从后往前)
            val_end_idx = n_dates - i * val_size
            val_start_idx = val_end_idx - val_size

            if val_start_idx < 0 or val_end_idx <= val_start_idx:
                continue

            val_start_date = unique_dates[val_start_idx]
            val_end_date = unique_dates[min(val_end_idx - 1, n_dates - 1)]

            # Purge: 训练集不能包含 val_start - purge_days 之后的日期
            # 这确保训练集的标签不会与验证集日期重叠
            if self.trade_calendar is None or len(self.trade_calendar) == 0:
                raise RuntimeError("trade_calendar required for purged split")
            # 用交易日索引计算
            cal_list = self.trade_calendar.tolist()
            if val_start_date not in cal_list:
                raise RuntimeError("val_start_date not found in trade_calendar")
            val_start_loc = cal_list.index(val_start_date)
            purge_end_loc = max(0, val_start_loc - self.purge_days)
            purge_end_date = self.trade_calendar[purge_end_loc]

            # P0-3修复: 移除Embargo，验证集直接从val_start_date开始
            # 原因: walk-forward场景下purge已足够，Embargo会错误地砍掉验证集数据
            # 生成索引
            train_mask = groups < purge_end_date
            val_mask = (groups >= val_start_date) & (groups <= val_end_date)

            train_idx = X.index[train_mask].tolist()
            val_idx = X.index[val_mask].tolist()

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx


def evaluate_with_proper_metrics(X_val: pd.DataFrame, y_val: pd.Series,
                                  predictions: np.ndarray,
                                  baseline_predictions: np.ndarray = None,
                                  turnover_config: Dict = None,
                                  tradeable_mask: np.ndarray = None) -> Dict:
    """
    正确的评估指标 (按日期分组) - P2-1增强版

    不用全局R2/IC，改用:
    1. 日度Rank-IC及其IR
    2. Top-20未来超额收益
    3. 换手成本后收益 (P2-1: 使用实际换手率)
    4. 相对基线的提升
    5. P2-1新增: 可交易失败率统计
    6. P2-1新增: 滑点损耗估算
    7. P2-1新增: 流动性加权成本

    Args:
        X_val: 验证集特征 (必须包含'date'列)
        y_val: 验证集标签
        predictions: 模型预测值
        baseline_predictions: 基线预测值 (可选，用于计算相对提升)
        turnover_config: 换手配置 {'turnover_rate': float, 'commission': float, 'slippage': float}
        tradeable_mask: 可交易掩码数组 (True=可交易)

    Returns:
        评估指标字典
    """
    results = {}

    # 确保有日期列
    if 'date' not in X_val.columns:
        logger.warning("X_val缺少date列，无法计算日度指标")
        return results

    # P2-1: 默认成本参数，可通过turnover_config覆盖
    default_config = {
        'turnover_rate': 0.5,           # 双周换手率
        'commission': 0.0003,           # 券商佣金 (万三)
        'stamp_duty': 0.001,            # 印花税 (千一, 仅卖出)
        'slippage': 0.001,              # 滑点 (千一)
        'impact_coefficient': 0.1,      # 市场冲击系数
    }
    if turnover_config:
        default_config.update(turnover_config)
    config = default_config

    # 辅助函数: 计算日度Rank-IC
    def _compute_daily_rank_ic(preds: np.ndarray) -> list:
        daily_ics = []
        for date in X_val['date'].unique():
            mask = X_val['date'] == date
            if mask.sum() < 20:
                continue

            pred_vals = preds[mask.values] if isinstance(mask, pd.Series) else preds[mask]
            true_vals = y_val[mask].values if isinstance(y_val, pd.Series) else y_val[mask]

            # 计算排名
            pred_rank = np.argsort(np.argsort(pred_vals))
            true_rank = np.argsort(np.argsort(true_vals))

            # Rank-IC (Spearman相关)
            if len(pred_rank) > 1:
                ic = np.corrcoef(pred_rank, true_rank)[0, 1]
                if not np.isnan(ic):
                    daily_ics.append(ic)
        return daily_ics

    # 1. 模型日度Rank-IC
    daily_ics = _compute_daily_rank_ic(predictions)

    if len(daily_ics) > 0:
        results['mean_rank_ic'] = np.mean(daily_ics)
        results['ic_std'] = np.std(daily_ics)
        results['ic_ir'] = results['mean_rank_ic'] / (results['ic_std'] + 1e-10)
        results['ic_positive_ratio'] = np.mean([ic > 0 for ic in daily_ics])
    else:
        results['mean_rank_ic'] = 0
        results['ic_std'] = 0
        results['ic_ir'] = 0
        results['ic_positive_ratio'] = 0

    # 2. Top-20平均未来收益 + P2-1: 换手追踪
    top20_excess = []
    turnover_rates = []
    prev_top20_codes = set()

    # P2-1新增: 可交易失败统计
    total_selected = 0
    tradeable_failures = 0

    for date in sorted(X_val['date'].unique()):
        mask = X_val['date'] == date
        if mask.sum() < 20:
            continue

        # 选Top-20
        pred_vals = predictions[mask.values] if isinstance(mask, pd.Series) else predictions[mask]
        true_vals = y_val[mask].values if isinstance(y_val, pd.Series) else y_val[mask]

        top20_idx = np.argsort(pred_vals)[-20:]
        total_selected += len(top20_idx)

        # P2-1: 计算可交易失败率
        if tradeable_mask is not None:
            mask_vals = tradeable_mask[mask.values] if isinstance(mask, pd.Series) else tradeable_mask[mask]
            failed = np.sum(~mask_vals[top20_idx])
            tradeable_failures += failed

        # Top-20的实际超额收益
        top20_return = true_vals[top20_idx].mean()
        benchmark_return = true_vals.mean()
        top20_excess.append(top20_return - benchmark_return)

        # P2-1: 计算实际换手率
        if 'code' in X_val.columns:
            mask_df = X_val[mask]
            current_top20_codes = set(mask_df.iloc[top20_idx]['code'].values)

            if len(prev_top20_codes) > 0:
                # 换手 = (卖出数 + 买入数) / (2 * 持仓数)
                turnover = len(prev_top20_codes - current_top20_codes) / len(prev_top20_codes)
                turnover_rates.append(turnover)

            prev_top20_codes = current_top20_codes

    if len(top20_excess) > 0:
        results['top20_excess_mean'] = np.mean(top20_excess)
        results['top20_excess_std'] = np.std(top20_excess)
        results['top20_sharpe'] = results['top20_excess_mean'] / (results['top20_excess_std'] + 1e-10)
    else:
        results['top20_excess_mean'] = 0
        results['top20_excess_std'] = 0
        results['top20_sharpe'] = 0

    # P2-1: 实际换手率统计
    if len(turnover_rates) > 0:
        results['actual_turnover_mean'] = np.mean(turnover_rates)
        results['actual_turnover_std'] = np.std(turnover_rates)
        actual_turnover = results['actual_turnover_mean']
    else:
        results['actual_turnover_mean'] = config['turnover_rate']  # 使用默认值
        results['actual_turnover_std'] = 0
        actual_turnover = config['turnover_rate']

    # P2-1: 可交易失败率
    if total_selected > 0:
        results['tradeable_failure_rate'] = tradeable_failures / total_selected
    else:
        results['tradeable_failure_rate'] = 0

    # 3. P2-1增强: 精细化成本估算
    # 单边成本 = 佣金 + 滑点 + 市场冲击
    single_side_cost = (
        config['commission'] +          # 佣金
        config['slippage'] +             # 滑点
        config['impact_coefficient'] * actual_turnover  # 市场冲击(与换手相关)
    )

    # 卖出额外成本 = 印花税
    sell_extra_cost = config['stamp_duty']

    # 双边成本 = 买入成本 + 卖出成本
    round_trip_cost = single_side_cost + (single_side_cost + sell_extra_cost)

    # 换手成本 = 换手率 * 双边成本
    cost_drag = actual_turnover * round_trip_cost

    results['estimated_cost_drag'] = cost_drag
    results['cost_adjusted_return'] = results['top20_excess_mean'] - cost_drag

    # P2-1: 滑点损耗详细拆解
    results['slippage_loss'] = actual_turnover * config['slippage'] * 2  # 双边
    results['commission_cost'] = actual_turnover * config['commission'] * 2  # 双边
    results['stamp_duty_cost'] = actual_turnover * config['stamp_duty']  # 仅卖出
    results['market_impact'] = actual_turnover * config['impact_coefficient'] * actual_turnover

    # P2-1: 净收益率(年化估算，假设双周调仓)
    periods_per_year = 26  # 双周
    results['annual_excess_return'] = results['top20_excess_mean'] * periods_per_year
    results['annual_cost'] = cost_drag * periods_per_year
    results['annual_net_return'] = results['cost_adjusted_return'] * periods_per_year

    # 4. 相对基线的提升
    if baseline_predictions is not None:
        baseline_ics = _compute_daily_rank_ic(baseline_predictions)

        if len(baseline_ics) > 0:
            baseline_mean_ic = np.mean(baseline_ics)
            baseline_ic_std = np.std(baseline_ics)
            baseline_ic_ir = baseline_mean_ic / (baseline_ic_std + 1e-10)

            results['baseline_mean_ic'] = baseline_mean_ic
            results['baseline_ic_ir'] = baseline_ic_ir

            # 绝对提升
            results['ic_improvement'] = results['mean_rank_ic'] - baseline_mean_ic
            results['ic_ir_improvement'] = results['ic_ir'] - baseline_ic_ir

            # 相对提升 (百分比)
            if abs(baseline_mean_ic) > 1e-10:
                results['relative_ic_improvement'] = (
                    (results['mean_rank_ic'] - baseline_mean_ic) / abs(baseline_mean_ic)
                )
            else:
                results['relative_ic_improvement'] = float('inf') if results['mean_rank_ic'] > 0 else 0

            logger.info(f"基线对比: IC提升 {results['ic_improvement']:.4f} "
                       f"({results['relative_ic_improvement']:.1%})")

    # P2-1: 综合评估日志
    logger.info(f"成本分析: 实际换手={actual_turnover:.1%}, "
               f"成本拖累={cost_drag:.4%}, "
               f"可交易失败率={results['tradeable_failure_rate']:.1%}")
    logger.info(f"年化估算: 超额收益={results['annual_excess_return']:.1%}, "
               f"交易成本={results['annual_cost']:.1%}, "
               f"净收益={results['annual_net_return']:.1%}")

    return results


class MLModel:
    """机器学习选股模型"""

    def __init__(self, config: dict = None):
        """
        初始化ML模型

        Args:
            config: 配置字典
        """
        self.config = config or load_config()
        self.ml_config = self.config['ml']
        self.root = get_project_root()
        self.processor = DataProcessor(self.config)
        self.factor_engine = FactorEngine(self.config)

        # P5: 加载执行配置，确保标签与回测对齐
        self.exec_config = get_execution_config(self.config)
        self.label_entry_price = self.exec_config['label_entry']   # 'close' or 'open'
        self.label_exit_price = self.exec_config['label_exit']     # 'close' or 'open'
        self.trade_at_close = self.exec_config['trade_at_close']   # 是否同日收盘执行
        self.purge_safety_buffer = self.exec_config.get('purge_safety_buffer', 5)

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def _get_model(self, model_type: str = 'xgboost', task: str = 'regression'):
        """
        获取模型实例

        Args:
            model_type: 模型类型 ('xgboost', 'lightgbm', 'linear')
            task: 任务类型 ('regression', 'classification')

        Returns:
            模型实例
        """
        if model_type == 'xgboost':
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise RuntimeError("XGBoost is required for model_type='xgboost'") from exc
            params = self.ml_config.get('xgboost', {})
            if task == 'regression':
                return xgb.XGBRegressor(**params, random_state=42)
            else:
                return xgb.XGBClassifier(**params, random_state=42)

        if model_type == 'lightgbm':
            try:
                import lightgbm as lgb
            except ImportError as exc:
                raise RuntimeError("LightGBM is required for model_type='lightgbm'") from exc
            params = self.ml_config.get('lightgbm', {})
            if task == 'regression':
                return lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
            else:
                return lgb.LGBMClassifier(**params, random_state=42, verbose=-1)

        if model_type == 'linear':
            from sklearn.linear_model import Ridge, LogisticRegression
            if task == 'regression':
                return Ridge(alpha=1.0)
            else:
                return LogisticRegression(max_iter=1000)

        if model_type == 'sklearn':
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            if task == 'regression':
                return RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            else:
                return RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

        raise ValueError(f"unsupported model_type: {model_type}")

    def prepare_features(self, factor_df: pd.DataFrame,
                         exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        准备特征矩阵

        Args:
            factor_df: 因子数据
            exclude_cols: 排除的列

        Returns:
            (特征矩阵, 特征名列表)
        """
        exclude_cols = exclude_cols or ['code', 'date', 'trade_date']

        # 选择数值型列作为特征
        feature_cols = []
        for col in factor_df.columns:
            if col in exclude_cols:
                continue
            if factor_df[col].dtype in [np.float64, np.int64, float, int]:
                feature_cols.append(col)

        if len(feature_cols) == 0:
            raise RuntimeError("no numeric features available for ML")

        X = factor_df[feature_cols].copy()

        if X.isna().any().any():
            raise RuntimeError("feature matrix contains NaN")
        if np.isinf(X.to_numpy()).any():
            raise RuntimeError("feature matrix contains inf")

        return X, feature_cols

    def compute_forward_returns(self, codes: List[str],
                                 start_date: pd.Timestamp,
                                 forward_days: int = 20,
                                 benchmark: str = 'sh000300',
                                 use_open_entry: bool = None,
                                 end_date: pd.Timestamp = None) -> pd.Series:
        """
        计算未来收益率（标签）

        P5改造: 标签入场/出场价格由配置控制，与回测执行对齐
        - label_entry='close': 使用T日收盘价入场（收盘换仓模式）
        - label_entry='open': 使用T+1开盘价入场（T+1执行模式）
        - label_exit: 出场价格，通常为'close'

        参数use_open_entry已废弃，仅保留向后兼容。

        时间线说明（收盘换仓模式）:
        - 选股决策: rb_date收盘后
        - 执行入场: rb_date收盘（同日）
        - 持有期: 到end_date
        - 退出: end_date收盘

        Args:
            codes: 股票代码列表
            start_date: 选股决策日期（rb_date）
            forward_days: 持有天数（当end_date未指定时使用）
            benchmark: 基准指数
            use_open_entry: [废弃] 改用self.label_entry_price配置
            end_date: P4-2新增，下次调仓日期（动态标签长度）

        Returns:
            超额收益序列
        """
        calendar = self.processor.trade_calendar
        future_dates = calendar[calendar > start_date]

        if len(future_dates) < 2:
            raise RuntimeError("insufficient future dates to compute labels")

        if use_open_entry is not None:
            raise RuntimeError("use_open_entry is deprecated; use execution config")
        entry_price_type = self.label_entry_price
        exit_price_type = self.label_exit_price
        if entry_price_type not in ['open', 'close']:
            raise RuntimeError(f"invalid label_entry price type: {entry_price_type}")
        if exit_price_type not in ['open', 'close']:
            raise RuntimeError(f"invalid label_exit price type: {exit_price_type}")

        # P5: 入场日期确定
        if entry_price_type == 'open':
            # 开盘入场: T+1日
            entry_date = future_dates[0]
        else:
            # 收盘入场: T日（决策日当天收盘）
            entry_date = start_date

        # 出场日期确定（动态或固定）
        if end_date is not None:
            exit_date = end_date
            actual_days = len(calendar[(calendar > start_date) & (calendar <= end_date)])
            logger.debug(f"P5动态标签: {start_date.date()} -> {end_date.date()}, 实际{actual_days}个交易日")
        else:
            if entry_price_type == 'open':
                exit_date = future_dates[forward_days] if len(future_dates) > forward_days else future_dates[-1]
            else:
                # 收盘入场时，从start_date开始算forward_days
                exit_date = future_dates[forward_days - 1] if len(future_dates) >= forward_days else future_dates[-1]

        returns = {}
        for code in codes:
            df = self.processor.load_daily_price(code)

            df['date'] = pd.to_datetime(df['date'])

            # P5: 统一的价格获取逻辑
            entry_row = df[df['date'] == entry_date]
            exit_row = df[df['date'] == exit_date]

            if len(entry_row) == 0 or len(exit_row) == 0:
                raise RuntimeError(f"missing price rows for {code} on {entry_date} or {exit_date}")

            # 入场价格
            if entry_price_type == 'open':
                entry_price = entry_row['open'].iloc[0]
            else:
                entry_price = entry_row['close'].iloc[0]

            # 出场价格
            if exit_price_type == 'open':
                exit_price = exit_row['open'].iloc[0]
            else:
                exit_price = exit_row['close'].iloc[0]

            if entry_price <= 0:
                raise RuntimeError(f"invalid entry price for {code} on {entry_date}")
            returns[code] = (exit_price - entry_price) / entry_price

        stock_returns = pd.Series(returns)

        # P5: 基准收益也使用相同的入场/出场价格类型
        index_df = self.processor.load_index_data(benchmark)
        index_df['date'] = pd.to_datetime(index_df['date'])

        idx_entry = index_df[index_df['date'] == entry_date]
        idx_exit = index_df[index_df['date'] == exit_date]

        if len(idx_entry) == 0 or len(idx_exit) == 0:
            raise RuntimeError("missing benchmark price rows for label dates")

        # 入场价格
        if entry_price_type == 'open':
            idx_entry_price = idx_entry['open'].iloc[0]
        else:
            idx_entry_price = idx_entry['close'].iloc[0]

        # 出场价格
        if exit_price_type == 'open':
            idx_exit_price = idx_exit['open'].iloc[0]
        else:
            idx_exit_price = idx_exit['close'].iloc[0]

        if idx_entry_price <= 0:
            raise RuntimeError("invalid benchmark entry price")
        benchmark_return = (idx_exit_price - idx_entry_price) / idx_entry_price
        stock_returns = stock_returns - benchmark_return

        return stock_returns

    def build_training_samples(self, start_date: str, end_date: str,
                                freq: str = 'biweekly',
                                use_dynamic_label: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        构建训练样本

        P5改造: 动态标签的end_date与执行模式对齐
        - trade_at_close=True: end_date = next_rb_date（收盘换仓）
        - trade_at_close=False: end_date = next_exec_date（T+1执行）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            freq: 采样频率
            use_dynamic_label: 是否使用动态标签长度

        Returns:
            (特征DataFrame, 标签Series)
        """
        logger.info(f"构建训练样本: {start_date} ~ {end_date}")
        exec_mode = '收盘换仓' if self.trade_at_close else 'T+1开盘'
        if use_dynamic_label:
            logger.info(f"P5: 使用动态标签长度，执行模式: {exec_mode}")
        else:
            logger.info(f"使用固定标签长度: {self.ml_config['forward_days']}天")

        calendar = self.processor.trade_calendar
        rebalance_dates = get_rebalance_dates(start_date, end_date, calendar, freq)

        forward_days = self.ml_config['forward_days']

        all_X = []
        all_y = []

        # P5: 根据配置确定动态标签的结束日期
        def get_label_end_date(next_rb_date: pd.Timestamp) -> pd.Timestamp:
            """获取标签的结束日期"""
            if self.trade_at_close:
                # 收盘换仓: 标签到下次调仓日收盘
                return next_rb_date
            else:
                # T+1执行: 标签到下次执行日
                future_dates = calendar[calendar > next_rb_date]
                if len(future_dates) > 0:
                    return future_dates[0]
                raise RuntimeError("no future dates available for label end date")

        for i, date in enumerate(rebalance_dates[:-1]):  # 最后一个日期没有未来收益
            # 优先从缓存加载因子，没有则计算并保存
            date_str = date.strftime('%Y%m%d')
            factor_df = self.factor_engine.load_factor_data(date_str)

            if len(factor_df) == 0:
                # 缓存不存在，计算因子
                factor_df = self.factor_engine.compute_factor_cross_section(date)
                if len(factor_df) > 0:
                    factor_df = self.factor_engine.standardize_factors(factor_df)
                    self.factor_engine.save_factor_data(factor_df, date_str)

            if len(factor_df) == 0:
                raise RuntimeError(f"factor data empty for {date.date()}")

            # 准备特征
            X, feature_names = self.prepare_features(factor_df)
            self.feature_names = feature_names

            # P5: 计算未来收益（动态或固定标签长度）
            codes = factor_df['code'].tolist()

            if use_dynamic_label and i + 1 < len(rebalance_dates):
                # 动态标签: 使用下次调仓日期
                next_rb_date = rebalance_dates[i + 1]
                label_end_date = get_label_end_date(next_rb_date)

                returns = self.compute_forward_returns(
                    codes, date, forward_days=forward_days,
                    end_date=label_end_date
                )
                logger.debug(f"P5动态标签: {date.date()} -> {label_end_date.date()}")
            else:
                # 固定标签长度
                returns = self.compute_forward_returns(codes, date, forward_days)

            # 对齐
            common_codes = set(codes) & set(returns.index)
            if len(common_codes) < 10:
                raise RuntimeError("insufficient labels for training sample")

            mask = factor_df['code'].isin(common_codes)
            X_aligned = X[mask].copy()
            X_aligned['code'] = factor_df.loc[mask, 'code'].values
            X_aligned['date'] = date

            y_aligned = returns[factor_df.loc[mask, 'code'].values]

            all_X.append(X_aligned)
            all_y.append(y_aligned)

        if len(all_X) == 0:
            raise RuntimeError("no training samples generated")

        X_all = pd.concat(all_X, ignore_index=True)
        y_all = pd.concat(all_y, ignore_index=True)

        logger.info(f"样本数: {len(X_all)}, 特征数: {len(self.feature_names)}")

        return X_all, y_all

    def train(self, X: pd.DataFrame, y: pd.Series,
              model_type: str = 'xgboost',
              task: str = 'regression',
              use_purged_cv: bool = True) -> Dict:
        """
        训练模型 (P0-3修复: 使用PurgedGroupTimeSeriesSplit)

        Args:
            X: 特征数据 (必须包含'date'列)
            y: 标签
            model_type: 模型类型
            task: 任务类型
            use_purged_cv: 是否使用Purged CV

        Returns:
            训练结果字典
        """
        logger.info(f"训练 {model_type} 模型 ({task})...")

        # 提取纯特征（不含code, date）
        feature_cols = [c for c in X.columns if c not in ['code', 'date']]
        self.feature_names = feature_cols
        if len(feature_cols) == 0:
            raise RuntimeError("no feature columns available for training")

        # 按时间排序 (确保时间序列正确)
        if 'date' in X.columns:
            X = X.copy()
            X['date'] = pd.to_datetime(X['date'])
            sort_idx = X['date'].argsort()
            X = X.iloc[sort_idx].reset_index(drop=True)
            y = y.iloc[sort_idx].reset_index(drop=True)

        if use_purged_cv:
            if 'date' not in X.columns:
                raise RuntimeError("Purged CV requires 'date' column in X")
            # P5: purge_days = max(forward_days, 最大调仓间隔) + safety_buffer
            base_forward_days = self.ml_config.get('forward_days', 20)

            # 计算最大调仓间隔（基于rebalance_freq）
            rebalance_freq = self.config.get('backtest', {}).get('rebalance_freq', 'biweekly')
            freq_to_days = {'weekly': 7, 'biweekly': 14, 'monthly': 22}
            max_rebalance_interval = freq_to_days.get(rebalance_freq, 14)

            # purge_days = max(固定前视天数, 最大调仓间隔) + 安全缓冲
            purge_days = max(base_forward_days, max_rebalance_interval) + self.purge_safety_buffer

            logger.info(f"P5 Purge配置: purge_days={purge_days} "
                       f"(forward={base_forward_days}, interval={max_rebalance_interval}, buffer={self.purge_safety_buffer})")

            purged_cv = PurgedGroupTimeSeriesSplit(
                n_splits=1,  # 只用最后一折作为验证
                purge_days=purge_days,
                trade_calendar=self.processor.trade_calendar
            )

            # 获取训练/验证索引
            splits = list(purged_cv.split(X))
            if len(splits) == 0:
                raise RuntimeError("Purged CV split failed")
            train_idx, val_idx = splits[0]
            X_train_df = X.iloc[train_idx]
            X_val_df = X.iloc[val_idx]
            y_train_raw = y.iloc[train_idx]
            y_val_raw = y.iloc[val_idx]

            logger.info(f"Purged CV: 训练 {len(train_idx)} 样本, 验证 {len(val_idx)} 样本")

            # 获取日期范围
            train_dates = X_train_df['date']
            val_dates = X_val_df['date']
            logger.info(f"训练集: {train_dates.min().date()} ~ {train_dates.max().date()}")
            logger.info(f"验证集: {val_dates.min().date()} ~ {val_dates.max().date()}")
        else:
            # 简单时间序列分割
            split_idx = int(len(X) * 0.8)
            X_train_df = X.iloc[:split_idx]
            X_val_df = X.iloc[split_idx:]
            y_train_raw = y.iloc[:split_idx]
            y_val_raw = y.iloc[split_idx:]

        # 提取特征矩阵
        X_train_data = X_train_df[feature_cols].values
        X_val_data = X_val_df[feature_cols].values

        # 标准化
        X_train = self.scaler.fit_transform(X_train_data)
        X_val = self.scaler.transform(X_val_data)

        # 处理标签
        if task == 'classification':
            threshold = y_train_raw.quantile(0.8)
            y_train = (y_train_raw >= threshold).astype(int).values
            y_val = (y_val_raw >= threshold).astype(int).values
        else:
            y_train = y_train_raw.values
            y_val = y_val_raw.values

        # 获取模型
        self.model = self._get_model(model_type, task)

        # P1-2修复: 从配置获取早停参数
        early_stopping_rounds = self.ml_config.get(model_type, {}).get('early_stopping_rounds', 20)

        # 训练 (使用早停)
        try:
            if model_type in ['xgboost', 'lightgbm']:
                eval_set = [(X_val, y_val)]
                # P1-2修复: 启用早停
                fit_params = {
                    'eval_set': eval_set,
                    'verbose': False
                }
                # 尝试添加早停参数
                try:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )
                    logger.info(f"启用早停: early_stopping_rounds={early_stopping_rounds}")
                except TypeError:
                    # 某些版本不支持early_stopping_rounds参数
                    self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            else:
                self.model.fit(X_train, y_train)
        except TypeError:
            self.model.fit(X_train, y_train)

        # 评估 (使用验证集)
        y_pred_val = self.model.predict(X_val)

        # 基础指标
        if task == 'regression':
            mse = mean_squared_error(y_val, y_pred_val)
            r2 = r2_score(y_val, y_pred_val)
            ic = np.corrcoef(y_val, y_pred_val)[0, 1] if len(y_val) > 1 else 0
            result = {'mse': mse, 'r2': r2, 'ic': ic}
            logger.info(f"验证集 MSE: {mse:.6f}, R2: {r2:.4f}, IC: {ic:.4f}")
        else:
            acc = accuracy_score(y_val, y_pred_val)
            result = {'accuracy': acc}
            logger.info(f"验证集准确率: {acc:.4f}")

        # P0-3: 使用正确的评估指标
        if 'date' in X_val_df.columns:
            proper_metrics = evaluate_with_proper_metrics(
                X_val_df, y_val_raw, y_pred_val
            )
            result.update(proper_metrics)
            logger.info(f"Rank-IC: {proper_metrics.get('mean_rank_ic', 0):.4f}, "
                       f"IC-IR: {proper_metrics.get('ic_ir', 0):.4f}, "
                       f"Top-20超额: {proper_metrics.get('top20_excess_mean', 0):.4%}")

        # 记录特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.Series(
                self.model.feature_importances_,
                index=feature_cols
            ).sort_values(ascending=False)
            result['feature_importance'] = importance
            logger.info(f"Top 5 重要因子: {importance.head().to_dict()}")

        return result

    def predict(self, factor_df: pd.DataFrame) -> pd.Series:
        """
        使用模型预测

        Args:
            factor_df: 因子数据

        Returns:
            预测得分
        """
        if self.model is None:
            raise ValueError("模型未训练")

        X, _ = self.prepare_features(factor_df)

        # 确保特征对齐
        if self.feature_names:
            missing_cols = set(self.feature_names) - set(X.columns)
            if missing_cols:
                raise RuntimeError(f"missing features for prediction: {sorted(missing_cols)}")
            X = X[self.feature_names]

        X_scaled = self.scaler.transform(X.values)
        predictions = self.model.predict(X_scaled)

        return pd.Series(predictions, index=factor_df.index)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       n_splits: int = 5,
                       model_type: str = 'xgboost') -> Dict:
        """
        时间序列交叉验证

        Args:
            X: 特征数据
            y: 标签
            n_splits: 折数
            model_type: 模型类型

        Returns:
            交叉验证结果
        """
        logger.info(f"进行 {n_splits} 折时间序列交叉验证...")

        # 按日期排序
        if 'date' in X.columns:
            sort_idx = X['date'].argsort()
            X = X.iloc[sort_idx].reset_index(drop=True)
            y = y.iloc[sort_idx].reset_index(drop=True)

        feature_cols = [c for c in X.columns if c not in ['code', 'date']]
        X_values = X[feature_cols].values

        tscv = TimeSeriesSplit(n_splits=n_splits)

        scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_values)):
            X_train, X_val = X_values[train_idx], X_values[val_idx]
            y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values

            # 标准化
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            # 训练
            model = self._get_model(model_type, 'regression')
            model.fit(X_train, y_train)

            # 预测
            y_pred = model.predict(X_val)

            # 评估
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            # 计算选股IC（相关性）
            ic = np.corrcoef(y_val, y_pred)[0, 1] if len(y_val) > 1 else 0

            scores.append({'fold': fold + 1, 'mse': mse, 'r2': r2, 'ic': ic})
            logger.info(f"Fold {fold + 1}: MSE={mse:.6f}, R2={r2:.4f}, IC={ic:.4f}")

        results = pd.DataFrame(scores)
        logger.info(f"\n平均 IC: {results['ic'].mean():.4f} ± {results['ic'].std():.4f}")

        return {
            'fold_results': results,
            'mean_ic': results['ic'].mean(),
            'std_ic': results['ic'].std(),
            'mean_r2': results['r2'].mean()
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性

        Returns:
            特征重要性DataFrame
        """
        if self.model is None or self.feature_names is None:
            return pd.DataFrame()

        # 尝试获取特征重要性
        importance = None

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)

        if importance is None:
            return pd.DataFrame()

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        df = df.sort_values('importance', ascending=False)

        return df

    def save_model(self, path: str = None) -> None:
        """
        保存模型

        Args:
            path: 保存路径
        """
        if self.model is None:
            logger.warning("模型未训练，无法保存")
            return

        if path is None:
            model_dir = ensure_dir(self.root / 'output' / 'models')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = model_dir / f"model_{timestamp}.joblib"

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.ml_config
        }

        joblib.dump(model_data, path)
        logger.info(f"模型已保存: {path}")

    def load_model(self, path: str) -> None:
        """
        加载模型

        Args:
            path: 模型路径
        """
        model_data = joblib.load(path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']

        logger.info(f"模型已加载: {path}")

    def select_stocks_ml(self, factor_df: pd.DataFrame,
                          n: int = 50) -> pd.DataFrame:
        """
        使用ML模型选股

        Args:
            factor_df: 因子数据
            n: 选股数量

        Returns:
            选股结果
        """
        if self.model is None:
            raise ValueError("模型未训练")

        df = factor_df.copy()

        # 预测得分
        df['ml_score'] = self.predict(df)

        # 排序选股
        df = df.sort_values('ml_score', ascending=False)
        selected = df.head(n).copy()
        selected['rank'] = range(1, len(selected) + 1)

        return selected


class EnsembleModel:
    """集成模型：组合多因子打分和ML模型"""

    def __init__(self, config: dict = None):
        """
        初始化集成模型

        Args:
            config: 配置字典
        """
        self.config = config or load_config()
        self.ml_model = MLModel(self.config)

        from .scorer import FactorScorer
        self.scorer = FactorScorer(self.config)

    def predict_ensemble(self, factor_df: pd.DataFrame,
                          ml_weight: float = 0.5) -> pd.Series:
        """
        集成预测

        Args:
            factor_df: 因子数据
            ml_weight: ML模型权重 (0-1)

        Returns:
            集成得分
        """
        # 多因子打分
        factor_score = self.scorer.compute_total_score(factor_df)

        # ML预测
        if self.ml_model.model is None:
            raise RuntimeError("ML model not trained for ensemble scoring")
        ml_score = self.ml_model.predict(factor_df)
        # 标准化ML得分
        ml_score = (ml_score - ml_score.mean()) / ml_score.std()
        ml_score = ml_score.clip(-3, 3)  # 截断极端值
        ml_score = (ml_score + 3) / 6  # 归一化到 [0, 1]

        # 集成
        ensemble_score = (1 - ml_weight) * factor_score + ml_weight * ml_score

        return ensemble_score

    def select_stocks_ensemble(self, factor_df: pd.DataFrame,
                                n: int = 50,
                                ml_weight: float = 0.5) -> pd.DataFrame:
        """
        使用集成模型选股

        Args:
            factor_df: 因子数据
            n: 选股数量
            ml_weight: ML权重

        Returns:
            选股结果
        """
        df = factor_df.copy()
        df['ensemble_score'] = self.predict_ensemble(df, ml_weight)

        df = df.sort_values('ensemble_score', ascending=False)
        selected = df.head(n).copy()
        selected['rank'] = range(1, len(selected) + 1)

        return selected


class LGBMRankerModel:
    """
    P1-1: LightGBM Ranker 模型

    优势:
    - 直接优化NDCG@20，与选股目标一致
    - 比回归/分类更稳定
    - 支持按日期分组

    使用Learning-to-Rank方法，直接优化排序质量而非预测精度
    """

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.ml_config = self.config.get('ml', {})
        self.root = get_project_root()
        self.processor = DataProcessor(self.config)

        self.ranker = None
        self.feature_names = None
        self.scaler = StandardScaler()

    def prepare_ranking_data(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        准备排序学习数据

        Args:
            X: 特征DataFrame (必须包含'date'列)
            y: 标签 (未来收益率)

        Returns:
            (X_array, y_relevance, group_sizes)
        """
        if 'date' not in X.columns:
            raise ValueError("X must contain 'date' column for ranking")

        # 按日期排序
        X_sorted = X.sort_values('date').copy()
        y_sorted = y.loc[X_sorted.index]

        # 计算每组大小 (每个日期是一个query group)
        group_sizes = X_sorted.groupby('date').size().values

        # 特征矩阵
        self.feature_names = [c for c in X_sorted.columns if c not in ['code', 'date']]
        X_array = X_sorted[self.feature_names].values

        # 标签转为相关度得分 (0-4)
        # 每日内部排名，分5档
        y_relevance = []
        for date in X_sorted['date'].unique():
            mask = X_sorted['date'] == date
            y_daily = y_sorted[mask]

            # 分5档: 0=最差20%, 4=最好20%
            try:
                ranks = pd.qcut(y_daily, q=5, labels=[0, 1, 2, 3, 4], duplicates='drop')
            except ValueError:
                # 数据太少或重复值太多，用cut
                try:
                    ranks = pd.cut(y_daily, bins=5, labels=[0, 1, 2, 3, 4])
                except:
                    # 最后回退: 简单按排名分档
                    n = len(y_daily)
                    percentile_ranks = y_daily.rank(pct=True)
                    ranks = (percentile_ranks * 4).round().astype(int)

            y_relevance.extend(ranks.astype(int).tolist())

        return X_array, np.array(y_relevance), group_sizes

    def train(self, X: pd.DataFrame, y: pd.Series,
              use_purged_cv: bool = True) -> dict:
        """
        训练Ranker

        Args:
            X: 特征数据 (必须包含'date'列)
            y: 标签
            use_purged_cv: 是否使用Purged CV

        Returns:
            训练结果字典
        """
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("LightGBM is required for LGBMRankerModel") from exc

        logger.info("训练 LGBMRanker 模型...")

        # 按时间排序
        X = X.copy()
        X['date'] = pd.to_datetime(X['date'])
        sort_idx = X['date'].argsort()
        X = X.iloc[sort_idx].reset_index(drop=True)
        y = y.iloc[sort_idx].reset_index(drop=True)

        # 使用Purged CV分割
        if use_purged_cv:
            if 'date' not in X.columns:
                raise RuntimeError("Purged CV requires 'date' column in X")
            forward_days = self.ml_config.get('forward_days', 20)
            purged_cv = PurgedGroupTimeSeriesSplit(
                n_splits=1,
                purge_days=forward_days,
                trade_calendar=self.processor.trade_calendar
            )

            splits = list(purged_cv.split(X))
            if len(splits) == 0:
                raise RuntimeError("Purged CV split failed")
            train_idx, val_idx = splits[0]
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
            logger.info(f"Purged CV: 训练 {len(train_idx)} 样本, 验证 {len(val_idx)} 样本")
        else:
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # 准备训练数据
        X_train_array, y_train_rel, train_groups = self.prepare_ranking_data(X_train, y_train)

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train_array)

        # 创建LightGBM Dataset
        train_data = lgb.Dataset(X_train_scaled, label=y_train_rel, group=train_groups)

        # 模型参数
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [10, 20, 50],
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'verbose': -1
        }

        # P0-5修复: 准备验证集数据 (用于真正的early stopping)
        valid_sets = [train_data]
        valid_names = ['train']
        callbacks = [lgb.log_evaluation(50)]

        if len(X_val) > 0:
            X_val_array, y_val_rel, val_groups = self.prepare_ranking_data(X_val, y_val)
            if len(val_groups) > 0:
                X_val_scaled = self.scaler.transform(X_val_array)
                val_data = lgb.Dataset(X_val_scaled, label=y_val_rel, group=val_groups, reference=train_data)
                valid_sets = [train_data, val_data]
                valid_names = ['train', 'valid']
                # P0-5修复: 添加早停，监控验证集
                callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=True))
                logger.info(f"启用验证集早停，验证集: {len(X_val)} 样本, {len(val_groups)} 组")

        # 训练
        self.ranker = lgb.train(
            params,
            train_data,
            num_boost_round=500,  # 增大上限，依赖早停
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        # 评估
        result = {}

        # 特征重要性
        importance = pd.Series(
            self.ranker.feature_importance(importance_type='gain'),
            index=self.feature_names
        ).sort_values(ascending=False)
        result['feature_importance'] = importance
        logger.info(f"Top 10 重要特征: {importance.head(10).to_dict()}")

        # 验证集评估
        if len(X_val) > 0:
            X_val_array = X_val[self.feature_names].values
            X_val_scaled = self.scaler.transform(X_val_array)
            predictions = self.ranker.predict(X_val_scaled)

            # 使用正确的评估指标
            proper_metrics = evaluate_with_proper_metrics(X_val, y_val, predictions)
            result.update(proper_metrics)

            logger.info(f"Rank-IC: {proper_metrics.get('mean_rank_ic', 0):.4f}, "
                       f"IC-IR: {proper_metrics.get('ic_ir', 0):.4f}, "
                       f"Top-20超额: {proper_metrics.get('top20_excess_mean', 0):.4%}")

        return result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测排序得分

        Args:
            X: 特征数据

        Returns:
            排序得分数组
        """
        if self.ranker is None:
            raise ValueError("模型未训练")

        X_array = X[self.feature_names].values
        X_scaled = self.scaler.transform(X_array)
        return self.ranker.predict(X_scaled)

    def select_top_n(self, X: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """
        选择Top-N股票

        Args:
            X: 特征数据 (必须包含'code'列)
            n: 选股数量

        Returns:
            选股结果DataFrame
        """
        if self.ranker is None:
            raise ValueError("模型未训练")

        df = X.copy()
        scores = self.predict(X)
        df['rank_score'] = scores

        # 按得分降序，取前N
        df = df.sort_values('rank_score', ascending=False)
        top_n = df.head(n).copy()
        top_n['rank'] = range(1, len(top_n) + 1)

        return top_n

    def save_model(self, path: str = None) -> None:
        """保存模型"""
        if self.ranker is None:
            logger.warning("模型未训练，无法保存")
            return

        if path is None:
            model_dir = ensure_dir(self.root / 'output' / 'models')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = model_dir / f"ranker_{timestamp}.joblib"

        model_data = {
            'ranker': self.ranker,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.ml_config
        }

        joblib.dump(model_data, path)
        logger.info(f"Ranker模型已保存: {path}")

    def load_model(self, path: str) -> None:
        """加载模型"""
        model_data = joblib.load(path)

        self.ranker = model_data['ranker']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']

        logger.info(f"Ranker模型已加载: {path}")


class SoftGatedModel:
    """
    P2-1: 市场状态软门控模型

    替代硬切bull/bear/neutral:
    - 用市场因子作为gating输入
    - 输出连续权重，平滑切换
    - 动态调整动量/质量/价值因子权重

    优势:
    - 避免状态切换时的"跳变"
    - 更平滑的策略表现
    - 更好的鲁棒性
    """

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.root = get_project_root()
        self.processor = DataProcessor(self.config)

        # 三个基础模型 (按因子类别)
        self.base_models = {
            'momentum': None,
            'quality': None,
            'value': None
        }

        # 因子到类别的映射
        self.factor_to_category = {
            # 动量因子
            'ret_20d': 'momentum', 'ret_60d': 'momentum', 'ret_120d': 'momentum',
            'relative_strength_20d': 'momentum', 'relative_strength_60d': 'momentum',
            'dist_to_high_20d': 'momentum', 'dist_to_high_60d': 'momentum',
            'ma20_slope': 'momentum', 'trend_strength': 'momentum',
            'consecutive_up_days': 'momentum', 'relative_strength': 'momentum',
            # 质量因子
            'roe': 'quality', 'roa': 'quality', 'gross_margin': 'quality',
            'net_margin': 'quality', 'ocf_ratio': 'quality',
            # 价值因子
            'pe_ttm': 'value', 'pb': 'value', 'ps_ttm': 'value',
            'div_yield': 'value', 'pe_relative': 'value', 'pb_relative': 'value',
            # 波动/防守因子 (归入质量)
            'vol_20d': 'quality', 'max_drawdown_20d': 'quality',
            'current_drawdown': 'quality', 'atr_contraction': 'quality',
        }

        self.scaler = StandardScaler()
        self.feature_names = None

    def compute_gating_weights(self, market_features: Dict[str, float]) -> Dict[str, float]:
        """
        计算各类因子的动态权重

        基于市场环境自适应调整:
        - 趋势上升 + 广度好 → 动量权重高
        - 高波动 + 趋势下降 → 质量/防守权重高
        - 低估值环境 → 价值权重适当提升

        Args:
            market_features: 包含 market_trend, market_breadth, market_volatility

        Returns:
            各类别权重 {'momentum': w1, 'quality': w2, 'value': w3}
        """
        trend = market_features.get('market_trend', 0)
        breadth = market_features.get('market_breadth', 0.5)
        vol = market_features.get('market_volatility', 0.2)

        # === 动量权重计算 ===
        # 趋势向上 + 赚钱效应好 → 动量有效
        # tanh 将 trend 映射到 [-1, 1]
        momentum_signal = np.tanh(trend * 10) * max(0, (breadth - 0.3) / 0.4)
        momentum_weight = np.clip(0.35 + momentum_signal * 0.25, 0.10, 0.55)

        # === 质量权重计算 ===
        # 高波动 → 防守因子更重要
        # 趋势下降 → 也需要更多质量因子
        defensive_signal = np.tanh((vol - 0.2) * 5) - np.tanh(trend * 5)
        quality_weight = np.clip(0.30 + defensive_signal * 0.15, 0.15, 0.50)

        # === 价值权重计算 ===
        # 低估值环境 (breadth低但vol不高) → 价值因子
        # 作为剩余权重，并做clip
        value_weight = 1.0 - momentum_weight - quality_weight
        value_weight = np.clip(value_weight, 0.10, 0.40)

        # === 归一化确保和为1 ===
        total = momentum_weight + quality_weight + value_weight

        weights = {
            'momentum': momentum_weight / total,
            'quality': quality_weight / total,
            'value': value_weight / total
        }

        return weights

    def _split_features_by_category(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        将特征按类别分组

        Args:
            X: 完整特征DataFrame

        Returns:
            按类别分组的特征字典
        """
        feature_cols = [c for c in X.columns if c not in ['code', 'date']]

        category_features = {'momentum': [], 'quality': [], 'value': []}

        for col in feature_cols:
            cat = self.factor_to_category.get(col, 'quality')  # 默认归入质量
            category_features[cat].append(col)

        result = {}
        for cat, cols in category_features.items():
            if cols:
                result[cat] = X[cols].copy()
            else:
                result[cat] = pd.DataFrame(index=X.index)

        return result

    def train(self, X: pd.DataFrame, y: pd.Series,
              model_type: str = 'xgboost') -> Dict:
        """
        训练三个基础模型 (每个因子类别一个)

        Args:
            X: 特征数据 (必须包含'date'列)
            y: 标签
            model_type: 基础模型类型

        Returns:
            训练结果字典
        """
        logger.info("训练 SoftGatedModel (3个基础模型)...")

        self.feature_names = [c for c in X.columns if c not in ['code', 'date']]
        if len(self.feature_names) == 0:
            raise RuntimeError("no features available for SoftGatedModel")

        # 按时间排序
        if 'date' in X.columns:
            X = X.copy()
            X['date'] = pd.to_datetime(X['date'])
            sort_idx = X['date'].argsort()
            X = X.iloc[sort_idx].reset_index(drop=True)
            y = y.iloc[sort_idx].reset_index(drop=True)

        # 简单时间分割
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # 分类别训练
        results = {}
        category_features = self._split_features_by_category(X_train)

        for cat in ['momentum', 'quality', 'value']:
            cat_X = category_features[cat]

            if len(cat_X.columns) == 0:
                raise RuntimeError(f"no features for category: {cat}")

            # 标准化
            cat_X_train = self.scaler.fit_transform(cat_X.values)
            cat_X_val_df = self._split_features_by_category(X_val)[cat]
            cat_X_val = self.scaler.transform(cat_X_val_df.values)

            # 创建模型
            if model_type == 'xgboost':
                try:
                    import xgboost as xgb
                except ImportError as exc:
                    raise RuntimeError("XGBoost is required for SoftGatedModel") from exc
                model = xgb.XGBRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.7, colsample_bytree=0.7, random_state=42
                )
            elif model_type == 'sklearn':
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            else:
                raise ValueError(f"unsupported model_type: {model_type}")

            # 训练
            model.fit(cat_X_train, y_train.values)
            self.base_models[cat] = model

            # 验证
            pred_val = model.predict(cat_X_val)
            ic = np.corrcoef(y_val.values, pred_val)[0, 1] if len(y_val) > 1 else 0
            results[f'{cat}_ic'] = ic
            logger.info(f"  {cat} 模型 IC: {ic:.4f}")

        return results

    def predict(self, X: pd.DataFrame, market_features: Dict[str, float] = None) -> np.ndarray:
        """
        软门控预测

        根据市场环境动态加权组合各基础模型的预测

        Args:
            X: 特征数据
            market_features: 市场环境因子，None则从X提取

        Returns:
            加权组合的预测得分
        """
        if all(m is None for m in self.base_models.values()):
            raise ValueError("模型未训练")

        # 提取市场特征 (如果没有传入)
        if market_features is None:
            required_cols = ['market_trend', 'market_breadth', 'market_volatility']
            missing = [col for col in required_cols if col not in X.columns]
            if missing:
                raise RuntimeError(f"missing market features: {missing}")
            market_features = {
                'market_trend': X['market_trend'].iloc[0],
                'market_breadth': X['market_breadth'].iloc[0],
                'market_volatility': X['market_volatility'].iloc[0]
            }

        # 计算门控权重
        weights = self.compute_gating_weights(market_features)
        logger.info(f"门控权重: 动量={weights['momentum']:.2f}, "
                   f"质量={weights['quality']:.2f}, 价值={weights['value']:.2f}")

        # 各模型预测
        category_features = self._split_features_by_category(X)
        predictions = {}

        for cat in ['momentum', 'quality', 'value']:
            if self.base_models[cat] is None:
                predictions[cat] = np.zeros(len(X))
                continue

            cat_X = category_features[cat]
            if len(cat_X.columns) == 0:
                predictions[cat] = np.zeros(len(X))
                continue

            cat_X_scaled = self.scaler.transform(cat_X.values)
            predictions[cat] = self.base_models[cat].predict(cat_X_scaled)

        # 加权组合
        ensemble = (weights['momentum'] * predictions['momentum'] +
                    weights['quality'] * predictions['quality'] +
                    weights['value'] * predictions['value'])

        return ensemble

    def select_stocks(self, X: pd.DataFrame, n: int = 20,
                      market_features: Dict[str, float] = None) -> pd.DataFrame:
        """
        使用软门控模型选股

        Args:
            X: 因子数据
            n: 选股数量
            market_features: 市场环境因子

        Returns:
            选股结果DataFrame
        """
        df = X.copy()
        df['gated_score'] = self.predict(X, market_features)

        # 按得分排序选股
        df = df.sort_values('gated_score', ascending=False)
        selected = df.head(n).copy()
        selected['rank'] = range(1, len(selected) + 1)

        return selected

    def save_model(self, path: str = None) -> None:
        """保存模型"""
        if all(m is None for m in self.base_models.values()):
            logger.warning("模型未训练，无法保存")
            return

        if path is None:
            model_dir = ensure_dir(self.root / 'output' / 'models')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = model_dir / f"soft_gated_{timestamp}.joblib"

        model_data = {
            'base_models': self.base_models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'factor_to_category': self.factor_to_category
        }

        joblib.dump(model_data, path)
        logger.info(f"SoftGatedModel已保存: {path}")

    def load_model(self, path: str) -> None:
        """加载模型"""
        model_data = joblib.load(path)

        self.base_models = model_data['base_models']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.factor_to_category = model_data.get('factor_to_category', self.factor_to_category)

        logger.info(f"SoftGatedModel已加载: {path}")


class TripleBarrierLabeler:
    """
    P1-2: Triple-Barrier 标签生成器 (波动率自适应版)

    参考: López de Prado, Advances in Financial Machine Learning (2018)

    优势:
    - 高波动票和低波动票使用不同的阈值 (ATR自适应)
    - 避免"小票全触发、大票不触发"的问题
    - 更合理地定义"上涨/下跌/中性"

    标签含义:
    - 1: 先触及止盈 (看涨)
    - 0: 先触及止损 (看跌)
    - -1: 未触及任何边界，到期 (中性，可不参与训练)
    """

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.processor = DataProcessor(self.config)

        # 默认参数
        self.forward_days = self.config.get('ml', {}).get('forward_days', 20)
        self.tp_multiplier = 2.0   # 止盈 = k * ATR
        self.sl_multiplier = 1.5   # 止损 = k * ATR
        self.neutral_threshold = 0.02  # 到期时判断中性的阈值

    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR (Average True Range)"""
        if len(df) < period + 1:
            return df['high'].iloc[-period:].mean() - df['low'].iloc[-period:].mean()

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr_list = []
        for i in range(1, min(period + 1, len(df))):
            tr = max(
                high[-period + i] - low[-period + i],
                abs(high[-period + i] - close[-period + i - 1]),
                abs(low[-period + i] - close[-period + i - 1])
            )
            tr_list.append(tr)

        return np.mean(tr_list) if tr_list else 0

    def compute_label(self, code: str, start_date: pd.Timestamp,
                      tp_mult: float = None, sl_mult: float = None) -> int:
        """
        计算单只股票的Triple-Barrier标签

        Args:
            code: 股票代码
            start_date: 入场日期 (收盘决定，T+1开盘入场)
            tp_mult: 止盈乘数
            sl_mult: 止损乘数

        Returns:
            1 (止盈), 0 (止损), -1 (中性/无效)
        """
        tp_mult = tp_mult or self.tp_multiplier
        sl_mult = sl_mult or self.sl_multiplier

        df = self.processor.load_daily_price(code)
        if df is None or len(df) < 30:
            return -1

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # 获取入场前的数据计算ATR
        pre_entry = df[df['date'] <= start_date].tail(20)
        if len(pre_entry) < 14:
            return -1

        # 计算ATR
        atr = self.compute_atr(pre_entry)
        if atr <= 0:
            return -1

        # 自适应阈值
        entry_price = pre_entry['close'].iloc[-1]
        take_profit = (atr * tp_mult) / entry_price
        stop_loss = (atr * sl_mult) / entry_price

        # 获取未来数据
        future = df[df['date'] > start_date].head(self.forward_days)
        if len(future) < self.forward_days * 0.8:
            return -1  # 未来数据不足

        # T+1开盘入场
        actual_entry = future['open'].iloc[0]

        # 逐日检查触发
        for i in range(len(future)):
            max_up = (future['high'].iloc[i] - actual_entry) / actual_entry
            max_down = (actual_entry - future['low'].iloc[i]) / actual_entry

            if max_up >= take_profit:
                return 1  # 止盈触发
            if max_down >= stop_loss:
                return 0  # 止损触发

        # 时间边界到期，用最终收益判断
        final_return = (future['close'].iloc[-1] - actual_entry) / actual_entry

        # 获取基准收益
        benchmark_return = self._get_market_return(start_date)
        excess = final_return - benchmark_return

        if excess > self.neutral_threshold:
            return 1
        elif excess < -self.neutral_threshold:
            return 0
        else:
            return -1  # 中性，不参与训练

    def _get_market_return(self, date: pd.Timestamp) -> float:
        """获取市场基准收益"""
        index_df = self.processor.load_index_data('sh000300')
        if len(index_df) == 0:
            return 0.0

        index_df['date'] = pd.to_datetime(index_df['date'])
        index_df = index_df.sort_values('date')

        future = index_df[index_df['date'] > date].head(self.forward_days)
        if len(future) < 2:
            return 0.0

        entry = future['open'].iloc[0]
        final = future['close'].iloc[-1]
        return (final - entry) / entry if entry > 0 else 0.0

    def generate_labels(self, codes: List[str], date: pd.Timestamp) -> pd.Series:
        """
        批量生成标签

        Args:
            codes: 股票代码列表
            date: 入场日期

        Returns:
            标签Series {code: label}
        """
        labels = {}
        for code in codes:
            label = self.compute_label(code, date)
            if label != -1:  # 排除中性样本
                labels[code] = label

        return pd.Series(labels)

    def generate_training_labels(self, X: pd.DataFrame, y: pd.Series = None) -> pd.Series:
        """
        为训练集生成Triple-Barrier标签

        替代原有的连续收益率标签

        Args:
            X: 特征DataFrame (必须包含 'code' 和 'date' 列)
            y: 原始标签 (可选，用于对比)

        Returns:
            Triple-Barrier标签Series
        """
        if 'code' not in X.columns or 'date' not in X.columns:
            raise ValueError("X必须包含'code'和'date'列")

        labels = []
        valid_mask = []

        for idx, row in X.iterrows():
            code = row['code']
            date = pd.to_datetime(row['date'])

            label = self.compute_label(code, date)

            if label != -1:
                labels.append(label)
                valid_mask.append(True)
            else:
                labels.append(0)  # 占位
                valid_mask.append(False)

        result = pd.Series(labels, index=X.index)
        result_mask = pd.Series(valid_mask, index=X.index)

        logger.info(f"Triple-Barrier标签: 有效样本 {sum(valid_mask)}/{len(X)}, "
                   f"看涨 {sum(l==1 for l in labels if valid_mask[labels.index(l)])}, "
                   f"看跌 {sum(l==0 for l in labels if valid_mask[labels.index(l)])}")

        return result, result_mask


if __name__ == "__main__":
    # 测试代码
    ml = MLModel()

    # 创建测试数据
    np.random.seed(42)
    n_samples = 500

    test_X = pd.DataFrame({
        'code': [f'{i:06d}' for i in range(n_samples)],
        'date': pd.Timestamp('2024-01-15'),
        'roe': np.random.randn(n_samples),
        'pe_ttm': np.random.randn(n_samples),
        'ret_20d': np.random.randn(n_samples),
        'net_inflow_5d': np.random.randn(n_samples),
    })

    # 模拟标签（与roe正相关）
    test_y = 0.3 * test_X['roe'] + 0.2 * test_X['ret_20d'] + 0.1 * np.random.randn(n_samples)
    test_y = pd.Series(test_y)

    # 训练
    result = ml.train(test_X, test_y, model_type='xgboost')
    print(f"\n训练结果: {result}")

    # 特征重要性
    importance = ml.get_feature_importance()
    print("\n特征重要性:")
    print(importance)

    # 预测
    predictions = ml.predict(test_X)
    print(f"\n预测分数范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
