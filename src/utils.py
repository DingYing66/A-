"""
工具函数模块
包含配置加载、日志、通用工具等
"""

import os
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Union, List, Optional

import pandas as pd
import numpy as np


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent


def load_config(config_path: str = None) -> dict:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径，默认为 config/config.yaml

    Returns:
        配置字典
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def get_execution_config(config: dict) -> dict:
    """
    Get execution config (strict).
    """
    execution_config = config.get('execution')
    if execution_config is None:
        raise RuntimeError("missing execution config")
    required_keys = [
        'execution_price',
        'trade_at_close',
        'limit_check_price',
        'label_entry',
        'label_exit',
        'purge_safety_buffer',
    ]
    missing = [key for key in required_keys if key not in execution_config]
    if missing:
        raise RuntimeError(f"missing execution config keys: {missing}")
    return execution_config


def get_liquidity_config(config: dict) -> dict:
    """
    P5: 获取流动性配置，提供默认值

    Args:
        config: 主配置字典

    Returns:
        流动性配置字典，包含:
        - avg_amount_window: 成交额计算窗口
        - avg_turnover_window: 换手率计算窗口
        - min_amount_threshold: 最小日均成交额(元)
        - min_turnover_pct: 最小日均换手率(%)
        - fallback_percentile: 百分位回退阈值
    """
    default_liquidity = {
        'avg_amount_window': 20,
        'avg_turnover_window': 20,
        'min_amount_threshold': 10_000_000,  # 1000万元
        'min_turnover_pct': 0.5,             # 0.5%
        'fallback_percentile': 10            # 10%分位
    }

    liquidity_config = config.get('liquidity', {})
    return {**default_liquidity, **liquidity_config}


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志名称
        level: 日志级别

    Returns:
        日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，不存在则创建

    Args:
        path: 目录路径

    Returns:
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def standardize_datetime_column(series: pd.Series) -> pd.Series:
    """
    标准化日期时间列为 datetime64 格式

    支持的输入格式:
    - Unix timestamp (毫秒): 1577923200000 -> 2020-01-02
    - Unix timestamp (秒): 1577923200 -> 2020-01-02
    - 字符串: '2020-01-02', '2020/01/02', '20200102'
    - datetime对象

    输出格式: datetime64[ns]

    Args:
        series: 日期列

    Returns:
        标准化后的日期列（datetime64 类型）
    """
    if series.empty:
        return series

    # 如果已经是 datetime64 类型，直接返回
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    # 如果是数值类型，可能是 timestamp
    if pd.api.types.is_numeric_dtype(series):
        sample = series.dropna().iloc[0] if not series.dropna().empty else 0
        if sample > 1e12:
            # 毫秒级 timestamp
            return pd.to_datetime(series, unit='ms', errors='coerce')
        elif sample > 1e9:
            # 秒级 timestamp
            return pd.to_datetime(series, unit='s', errors='coerce')
        else:
            # 可能是 YYYYMMDD 格式的整数
            return pd.to_datetime(series, format='%Y%m%d', errors='coerce')

    # 字符串类型，尝试自动解析
    return pd.to_datetime(series, errors='coerce')


def save_parquet(df: pd.DataFrame, path: Union[str, Path], standardize_dates: bool = True, **kwargs) -> None:
    """
    保存DataFrame为Parquet格式，自动标准化日期列

    Args:
        df: 数据框
        path: 保存路径
        standardize_dates: 是否标准化日期列（默认True）
        **kwargs: 传递给to_parquet的参数
    """
    path = Path(path)
    ensure_dir(path.parent)

    df_clean = df.copy()

    # 标准化日期列
    if standardize_dates:
        date_keywords = ['date', 'time', 'datetime']
        for col in df_clean.columns:
            col_lower = col.lower()
            # 识别日期列
            if any(keyword in col_lower for keyword in date_keywords):
                try:
                    df_clean[col] = standardize_datetime_column(df_clean[col])
                except Exception:
                    pass  # 如果标准化失败，保持原样
        if isinstance(df_clean.index, pd.DatetimeIndex):
            df_clean.index = pd.Index(
                df_clean.index.strftime('%Y-%m-%d'),
                name=df_clean.index.name
            )

    # 处理混合类型列，避免pyarrow类型转换错误
    for col in df_clean.columns:
        # 检查是否有混合类型 (object类型且包含非字符串值)
        if df_clean[col].dtype == 'object':
            # 将所有值转换为字符串
            df_clean[col] = df_clean[col].apply(
                lambda x: str(x) if pd.notna(x) and not isinstance(x, str) else x
            )

    df_clean.to_parquet(path, **kwargs)


def load_parquet(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    加载Parquet文件

    Args:
        path: 文件路径
        **kwargs: 传递给read_parquet的参数

    Returns:
        DataFrame
    """
    return pd.read_parquet(path, **kwargs)


def get_trade_dates(start_date: str, end_date: str,
                    calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    获取指定范围内的交易日

    Args:
        start_date: 开始日期
        end_date: 结束日期
        calendar: 交易日历

    Returns:
        交易日列表
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    return calendar[(calendar >= start) & (calendar <= end)]


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Winsorize处理（分位数截断）

    Args:
        series: 输入序列
        lower: 下分位数
        upper: 上分位数

    Returns:
        处理后的序列
    """
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def zscore(series: pd.Series) -> pd.Series:
    """
    Z-score标准化

    Args:
        series: 输入序列

    Returns:
        标准化后的序列
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0, index=series.index)
    return (series - mean) / std


def standardize(series: pd.Series, winsor_lower: float = 0.01,
                winsor_upper: float = 0.99) -> pd.Series:
    """
    因子标准化：先Winsorize再Z-score

    Args:
        series: 输入序列
        winsor_lower: Winsorize下分位数
        winsor_upper: Winsorize上分位数

    Returns:
        标准化后的序列
    """
    winsorized = winsorize(series, winsor_lower, winsor_upper)
    return zscore(winsorized)


def calc_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    计算收益率

    Args:
        prices: 价格序列
        periods: 计算周期

    Returns:
        收益率序列
    """
    return prices.pct_change(periods)


def calc_rolling_std(returns: pd.Series, window: int) -> pd.Series:
    """
    计算滚动波动率

    Args:
        returns: 收益率序列
        window: 窗口大小

    Returns:
        波动率序列
    """
    return returns.rolling(window=window, min_periods=window//2).std()


def calc_max_drawdown(nav: pd.Series) -> float:
    """
    计算最大回撤

    Args:
        nav: 净值序列

    Returns:
        最大回撤值
    """
    peak = nav.expanding().max()
    drawdown = (nav - peak) / peak
    return drawdown.min()


def calc_max_drawdown_duration(nav: pd.Series) -> int:
    """
    计算最大回撤持续天数

    Args:
        nav: 净值序列

    Returns:
        最大回撤持续天数
    """
    peak = nav.expanding().max()
    drawdown = (nav - peak) / peak

    # 找到回撤开始和结束的位置
    is_drawdown = drawdown < 0

    if not is_drawdown.any():
        return 0

    # 计算每段回撤的持续时间
    drawdown_groups = (is_drawdown != is_drawdown.shift()).cumsum()
    drawdown_durations = is_drawdown.groupby(drawdown_groups).sum()

    return int(drawdown_durations.max())


def is_trading_day(date: Union[str, datetime], calendar: pd.DatetimeIndex) -> bool:
    """
    判断是否为交易日

    Args:
        date: 日期
        calendar: 交易日历

    Returns:
        是否为交易日
    """
    date = pd.to_datetime(date)
    return date in calendar


def get_next_trading_day(date: Union[str, datetime],
                         calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """
    获取下一个交易日

    Args:
        date: 当前日期
        calendar: 交易日历

    Returns:
        下一个交易日
    """
    date = pd.to_datetime(date)
    future_dates = calendar[calendar > date]
    if len(future_dates) == 0:
        return None
    return future_dates[0]


def get_prev_trading_day(date: Union[str, datetime],
                         calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """
    获取上一个交易日

    Args:
        date: 当前日期
        calendar: 交易日历

    Returns:
        上一个交易日
    """
    date = pd.to_datetime(date)
    past_dates = calendar[calendar < date]
    if len(past_dates) == 0:
        return None
    return past_dates[-1]


def get_rebalance_dates(start_date: str, end_date: str,
                        calendar: pd.DatetimeIndex,
                        freq: str = 'biweekly') -> List[pd.Timestamp]:
    """
    生成调仓日期列表

    Args:
        start_date: 开始日期
        end_date: 结束日期
        calendar: 交易日历
        freq: 调仓频率 ('daily', 'weekly', 'biweekly', 'monthly')

    Returns:
        调仓日期列表
    """
    trade_dates = get_trade_dates(start_date, end_date, calendar)

    if freq == 'daily':
        # 每个交易日都计算 (用于深度学习训练)
        rebalance_dates = trade_dates

    elif freq == 'weekly':
        # 每周五或当周最后一个交易日
        df = pd.DataFrame({'date': trade_dates})
        df['week'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        rebalance_dates = df.groupby(['year', 'week'])['date'].last().tolist()

    elif freq == 'biweekly':
        # 每两周一次
        df = pd.DataFrame({'date': trade_dates})
        df['week'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        weekly = df.groupby(['year', 'week'])['date'].last().reset_index(drop=True)
        rebalance_dates = weekly.iloc[::2].tolist()

    elif freq == 'monthly':
        # 每月最后一个交易日
        df = pd.DataFrame({'date': trade_dates})
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        rebalance_dates = df.groupby(['year', 'month'])['date'].last().tolist()

    else:
        raise ValueError(f"Unknown frequency: {freq}")

    return rebalance_dates


def format_pct(value: float, decimals: int = 2) -> str:
    """
    格式化百分比

    Args:
        value: 数值
        decimals: 小数位数

    Returns:
        格式化后的字符串
    """
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """
    格式化数字

    Args:
        value: 数值
        decimals: 小数位数

    Returns:
        格式化后的字符串
    """
    return f"{value:.{decimals}f}"
