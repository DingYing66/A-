"""
统一仓位控制策略模块

解决问题: main.py 和 backtester.py 中仓位控制逻辑不一致
- 阈值不同 (market_breadth: 0.3 vs 0.35)
- 调整系数不同

此模块提供统一的仓位计算逻辑，确保信号生成和回测使用相同的策略。
"""

from typing import Tuple
from .utils import setup_logger

logger = setup_logger(__name__)


def compute_position_ratio(market_trend: float,
                           market_breadth: float,
                           market_volatility: float) -> Tuple[float, str]:
    """
    统一的仓位计算逻辑

    基于"赚钱效应"的核心逻辑:
    - 市场广度高 (多数股票上涨) → 高仓位
    - 市场趋势向下 + 高波动 → 低仓位

    Args:
        market_trend: 市场趋势 (20日涨跌幅)
        market_breadth: 市场广度 (上涨股票比例, 0-1)
        market_volatility: 市场波动率 (年化)

    Returns:
        (position_ratio, risk_level): 仓位比例 (0.3-1.0) 和风险等级
    """
    position_ratio = 1.0
    risk_level = "正常"

    # 1. 根据市场趋势调整
    if market_trend < -0.05:  # 市场20日跌超5%
        position_ratio *= 0.6
        risk_level = "高风险"
    elif market_trend < 0:
        position_ratio *= 0.8
        if risk_level == "正常":
            risk_level = "中风险"

    # 2. 根据市场广度调整 (赚钱效应核心)
    # 统一使用三档阈值: 0.35, 0.45, 0.60
    if market_breadth < 0.35:  # 少于35%股票上涨
        position_ratio *= 0.5
        risk_level = "高风险"
    elif market_breadth < 0.45:
        position_ratio *= 0.7
        if risk_level == "正常":
            risk_level = "中风险"
    elif market_breadth > 0.60:  # 超过60%股票上涨
        position_ratio *= 1.1  # 可以稍微加仓

    # 3. 根据市场波动率调整
    # 统一使用两档阈值: 0.25, 0.35
    if market_volatility > 0.35:  # 年化波动超35%
        position_ratio *= 0.7
        if risk_level == "正常":
            risk_level = "中风险"
    elif market_volatility > 0.25:
        position_ratio *= 0.85

    # 限制在合理范围 [0.3, 1.0]
    position_ratio = max(0.3, min(1.0, position_ratio))

    return position_ratio, risk_level


def get_market_factors_from_df(factor_df, default_trend: float = 0,
                                default_breadth: float = 0.5,
                                default_volatility: float = 0.2) -> Tuple[float, float, float]:
    """
    从因子DataFrame中提取市场环境因子

    Args:
        factor_df: 因子数据DataFrame
        default_trend: 默认趋势值
        default_breadth: 默认广度值
        default_volatility: 默认波动率值

    Returns:
        (market_trend, market_breadth, market_volatility)
    """
    market_trend = default_trend
    market_breadth = default_breadth
    market_volatility = default_volatility

    if 'market_trend' in factor_df.columns:
        market_trend = factor_df['market_trend'].iloc[0]
    if 'market_breadth' in factor_df.columns:
        market_breadth = factor_df['market_breadth'].iloc[0]
    if 'market_volatility' in factor_df.columns:
        market_volatility = factor_df['market_volatility'].iloc[0]

    return market_trend, market_breadth, market_volatility
