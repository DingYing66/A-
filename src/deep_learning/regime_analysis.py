"""
市场状态分层分析模块 (Phase 3.2)

按市场状态 (Regime) 分层评估模型表现:
- bull/bear: 牛市/熊市
- high_vol/low_vol: 高波动/低波动
- 组合状态: bull_high_vol, bull_low_vol, bear_high_vol, bear_low_vol

用法:
    from src.deep_learning.regime_analysis import RegimeAnalyzer

    analyzer = RegimeAnalyzer()
    regimes = analyzer.classify_regime(index_returns)
    results = analyzer.evaluate_by_regime(predictions, returns, dates, regimes)
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger

logger = setup_logger(__name__)


class RegimeAnalyzer:
    """
    市场状态分层分析器

    基于指数收益和波动率将市场划分为不同状态，
    并分别评估模型在各状态下的表现。
    """

    def __init__(self,
                 trend_window: int = 60,
                 vol_window: int = 20,
                 trend_threshold: float = 0.0,
                 vol_percentile: float = 50.0):
        """
        Args:
            trend_window: 趋势计算窗口 (交易日)
            vol_window: 波动率计算窗口
            trend_threshold: 趋势判断阈值 (默认 0)
            vol_percentile: 波动率分位数阈值 (默认 50，即中位数)
        """
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.trend_threshold = trend_threshold
        self.vol_percentile = vol_percentile

    def classify_regime(self,
                        index_returns: Union[pd.Series, np.ndarray],
                        dates: pd.DatetimeIndex = None) -> pd.Series:
        """
        分类市场状态

        状态定义:
        - bull: 滚动收益 > threshold
        - bear: 滚动收益 <= threshold
        - high_vol: 滚动波动率 > 中位数
        - low_vol: 滚动波动率 <= 中位数

        组合状态: bull_high_vol, bull_low_vol, bear_high_vol, bear_low_vol

        Args:
            index_returns: 指数日收益率序列
            dates: 日期索引 (可选)

        Returns:
            市场状态 Series
        """
        if isinstance(index_returns, np.ndarray):
            if dates is None:
                dates = pd.RangeIndex(len(index_returns))
            index_returns = pd.Series(index_returns, index=dates)

        # 计算滚动指标
        rolling_ret = index_returns.rolling(self.trend_window).mean()
        rolling_vol = index_returns.rolling(self.vol_window).std()

        # 计算波动率中位数 (全样本)
        vol_median = rolling_vol.median()

        # 分类
        regimes = pd.Series(index=index_returns.index, dtype=str)

        # 趋势分类
        is_bull = rolling_ret > self.trend_threshold
        is_high_vol = rolling_vol > vol_median

        # 组合状态
        regimes[is_bull & is_high_vol] = 'bull_high_vol'
        regimes[is_bull & ~is_high_vol] = 'bull_low_vol'
        regimes[~is_bull & is_high_vol] = 'bear_high_vol'
        regimes[~is_bull & ~is_high_vol] = 'bear_low_vol'

        # 填充早期缺失值
        regimes = regimes.fillna('unknown')

        return regimes

    def classify_regime_simple(self,
                                index_returns: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        简化的市场状态分类 (仅牛熊)

        Args:
            index_returns: 指数日收益率

        Returns:
            简化状态 Series (bull/bear)
        """
        if isinstance(index_returns, np.ndarray):
            index_returns = pd.Series(index_returns)

        rolling_ret = index_returns.rolling(self.trend_window).mean()

        regimes = pd.Series(index=index_returns.index, dtype=str)
        regimes[rolling_ret > self.trend_threshold] = 'bull'
        regimes[rolling_ret <= self.trend_threshold] = 'bear'
        regimes = regimes.fillna('unknown')

        return regimes

    def evaluate_by_regime(self,
                            predictions: np.ndarray,
                            returns: np.ndarray,
                            dates: np.ndarray,
                            regimes: pd.Series,
                            n: int = 20) -> Dict[str, Dict[str, float]]:
        """
        按 Regime 分层评估

        Args:
            predictions: 预测分数
            returns: 实际收益
            dates: 日期
            regimes: 市场状态 Series
            n: 选股数量

        Returns:
            分层评估结果 {regime: {metric: value}}
        """
        from .metrics import compute_rank_ic, compute_top_n_excess

        # 转换日期
        if not isinstance(dates[0], (pd.Timestamp, np.datetime64)):
            dates = pd.to_datetime(dates)

        # 移除无效值
        valid_mask = ~(np.isnan(predictions) | np.isnan(returns))
        predictions = predictions[valid_mask]
        returns = returns[valid_mask]
        dates = dates[valid_mask]

        results = {}
        unique_regimes = regimes.dropna().unique()
        unique_regimes = [r for r in unique_regimes if r != 'unknown']

        for regime in unique_regimes:
            # 获取该 regime 的日期
            regime_dates = regimes[regimes == regime].index

            # 过滤数据
            regime_mask = pd.Series(dates).isin(regime_dates).values
            if regime_mask.sum() < 100:  # 样本量不足
                continue

            pred_regime = predictions[regime_mask]
            ret_regime = returns[regime_mask]
            dates_regime = dates[regime_mask]

            # 计算指标
            ic_metrics = compute_rank_ic(pred_regime, ret_regime, dates_regime)
            excess_metrics = compute_top_n_excess(pred_regime, ret_regime, dates_regime, n)

            results[regime] = {
                'n_samples': regime_mask.sum(),
                'n_periods': len(np.unique(dates_regime)),
                'rank_ic': ic_metrics['rank_ic'],
                'mean_daily_ic': ic_metrics['mean_daily_ic'],
                'ic_ir': ic_metrics['ic_ir'],
                'excess_return': excess_metrics['excess_return'],
                'win_rate': excess_metrics['win_rate'],
                'avg_period_excess': excess_metrics['avg_period_excess'],
            }

        return results

    def compute_regime_stability(self,
                                  predictions: np.ndarray,
                                  returns: np.ndarray,
                                  dates: np.ndarray,
                                  regimes: pd.Series,
                                  n: int = 20) -> Dict[str, float]:
        """
        计算模型在不同 Regime 下的稳定性

        稳定性指标:
        - IC 跨 Regime 标准差
        - 超额收益跨 Regime 标准差
        - 最差 Regime 表现

        Args:
            predictions: 预测分数
            returns: 实际收益
            dates: 日期
            regimes: 市场状态
            n: 选股数量

        Returns:
            稳定性指标
        """
        regime_results = self.evaluate_by_regime(
            predictions, returns, dates, regimes, n
        )

        if len(regime_results) == 0:
            return {
                'ic_std_across_regimes': np.nan,
                'excess_std_across_regimes': np.nan,
                'worst_regime_ic': np.nan,
                'best_regime_ic': np.nan,
            }

        ics = [r['rank_ic'] for r in regime_results.values() if not np.isnan(r['rank_ic'])]
        excesses = [r['excess_return'] for r in regime_results.values()
                   if not np.isnan(r['excess_return'])]

        return {
            'ic_std_across_regimes': np.std(ics) if len(ics) > 1 else np.nan,
            'ic_mean_across_regimes': np.mean(ics) if len(ics) > 0 else np.nan,
            'excess_std_across_regimes': np.std(excesses) if len(excesses) > 1 else np.nan,
            'excess_mean_across_regimes': np.mean(excesses) if len(excesses) > 0 else np.nan,
            'worst_regime_ic': min(ics) if len(ics) > 0 else np.nan,
            'best_regime_ic': max(ics) if len(ics) > 0 else np.nan,
            'n_regimes_evaluated': len(regime_results),
        }


def get_index_returns(index_code: str = '000300',
                      start_date: str = None,
                      end_date: str = None) -> pd.Series:
    """
    获取指数收益率

    Args:
        index_code: 指数代码 (默认沪深300)
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        指数日收益率 Series
    """
    try:
        import akshare as ak

        # 获取指数行情
        if index_code.startswith('000') or index_code.startswith('399'):
            df = ak.index_zh_a_hist(symbol=index_code)
        else:
            df = ak.index_zh_a_hist(symbol=index_code)

        if df is None or df.empty:
            logger.warning(f"无法获取指数 {index_code} 数据")
            return pd.Series()

        # 标准化
        df['date'] = pd.to_datetime(df['日期'])
        df = df.set_index('date').sort_index()

        # 计算日收益率
        if '收盘' in df.columns:
            returns = df['收盘'].pct_change()
        elif 'close' in df.columns:
            returns = df['close'].pct_change()
        else:
            logger.warning("无法识别收盘价列")
            return pd.Series()

        # 日期过滤
        if start_date:
            returns = returns[returns.index >= pd.to_datetime(start_date)]
        if end_date:
            returns = returns[returns.index <= pd.to_datetime(end_date)]

        return returns.dropna()

    except ImportError:
        logger.warning("akshare 未安装")
        return pd.Series()
    except Exception as e:
        logger.warning(f"获取指数数据失败: {e}")
        return pd.Series()


def evaluate_with_regime(predictions: np.ndarray,
                          returns: np.ndarray,
                          dates: np.ndarray,
                          index_returns: pd.Series = None,
                          n: int = 20) -> Dict[str, any]:
    """
    带市场状态分层的综合评估

    Args:
        predictions: 预测分数
        returns: 实际收益
        dates: 日期
        index_returns: 指数收益 (可选，如不提供则尝试获取)
        n: 选股数量

    Returns:
        包含分层结果的评估字典
    """
    from .metrics import evaluate_model_comprehensive

    # 基础评估
    results = evaluate_model_comprehensive(predictions, returns, dates, n=n)

    # 获取指数收益 (如未提供)
    if index_returns is None or len(index_returns) == 0:
        # 转换日期格式
        if isinstance(dates[0], (pd.Timestamp, np.datetime64)):
            start_date = pd.Timestamp(min(dates)).strftime('%Y%m%d')
            end_date = pd.Timestamp(max(dates)).strftime('%Y%m%d')
        else:
            start_date = str(min(dates))[:8]
            end_date = str(max(dates))[:8]

        index_returns = get_index_returns('000300', start_date, end_date)

    if len(index_returns) > 0:
        # 分类市场状态
        analyzer = RegimeAnalyzer()
        regimes = analyzer.classify_regime(index_returns)

        # 分层评估
        regime_results = analyzer.evaluate_by_regime(
            predictions, returns, dates, regimes, n
        )
        results['regime_analysis'] = regime_results

        # 稳定性指标
        stability = analyzer.compute_regime_stability(
            predictions, returns, dates, regimes, n
        )
        results.update({f'stability_{k}': v for k, v in stability.items()})

    return results


def print_regime_report(regime_results: Dict[str, Dict[str, float]],
                         model_name: str = 'Model'):
    """
    打印 Regime 分层报告

    Args:
        regime_results: 分层评估结果
        model_name: 模型名称
    """
    print("\n" + "=" * 70)
    print(f"              {model_name} 市场状态分层报告")
    print("=" * 70)

    if len(regime_results) == 0:
        print("  无分层评估结果")
        print("=" * 70)
        return

    # 表头
    print(f"\n{'Regime':<20}{'样本数':<10}{'期数':<8}{'Rank-IC':<10}{'IC-IR':<10}{'超额收益':<12}{'胜率':<8}")
    print("-" * 70)

    # 排序输出
    for regime in sorted(regime_results.keys()):
        r = regime_results[regime]
        print(f"{regime:<20}{r['n_samples']:<10}{r['n_periods']:<8}"
              f"{r['rank_ic']:<10.4f}{r['ic_ir']:<10.4f}"
              f"{r['excess_return']:<12.4f}{r['win_rate']:<8.2%}")

    print("=" * 70)

    # 汇总统计
    ics = [r['rank_ic'] for r in regime_results.values() if not np.isnan(r['rank_ic'])]
    excesses = [r['excess_return'] for r in regime_results.values()
               if not np.isnan(r['excess_return'])]

    if len(ics) > 0:
        print(f"\n【跨 Regime 汇总】")
        print(f"  IC 均值:        {np.mean(ics):.4f}")
        print(f"  IC 标准差:      {np.std(ics):.4f}")
        print(f"  IC 最大值:      {max(ics):.4f}")
        print(f"  IC 最小值:      {min(ics):.4f}")
        print(f"  超额收益均值:   {np.mean(excesses):.4f}")
        print("=" * 70)
