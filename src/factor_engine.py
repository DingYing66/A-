"""
因子计算引擎
负责计算质量、估值、动量、资金流等因子

支持 universe_id 缓存机制，确保不同股票池使用独立缓存
"""

from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import pandas as pd
import numpy as np
from tqdm import tqdm

from .utils import (
    load_config, setup_logger, load_parquet, save_parquet,
    get_project_root, standardize, winsorize, zscore, ensure_dir
)
from .data_processor import DataProcessor, HistoricalUniverse

# 导入 universe_id 生成函数
try:
    from .deep_learning.contracts import generate_universe_id
except ImportError:
    # 如果 deep_learning 模块不可用，提供简单实现
    import hashlib
    def generate_universe_id(codes: List[str], date: str = None) -> str:
        """生成股票池唯一标识"""
        sorted_codes = sorted(codes)
        content = ','.join(sorted_codes)
        if date:
            content = f"{date}:{content}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

logger = setup_logger(__name__)


class FactorEngine:
    """因子计算引擎"""

    def __init__(self, config: dict = None, use_universe: bool = True):
        """
        初始化因子引擎

        Args:
            config: 配置字典
            use_universe: 是否使用HistoricalUniverse消除生存者偏差（P0-4）
        """
        self.config = config or load_config()
        self.root = get_project_root()
        self.processor = DataProcessor(self.config)
        self.factor_config = self.config['factors']

        # P0-4修复: 使用HistoricalUniverse消除生存者偏差
        self.use_universe = use_universe
        if use_universe:
            self.universe = HistoricalUniverse(self.config)
        else:
            self.universe = None

        # 因子分类映射
        self.factor_categories = {
            'quality': ['roe', 'roa', 'gross_margin', 'net_margin', 'ocf_ratio'],
            'valuation': ['pe_ttm', 'pb', 'ps_ttm', 'div_yield',
                          'pe_relative', 'pb_relative'],
            'momentum': ['ret_20d', 'ret_60d', 'ret_120d',
                         'new_high', 'consecutive_up'],
            'volatility': ['vol_20d', 'max_drawdown_20d'],
            'flow': ['net_inflow_5d', 'net_inflow_20d', 'north_change'],
            # 赚钱效应因子 (新增 new_high_20d/60d, dist_to_breakout)
            'sentiment': ['relative_strength_20d', 'relative_strength_60d',
                          'reversal_5d', 'reversal_10d', 'price_position',
                          'new_high_20d', 'new_high_60d', 'dist_to_breakout'],
            # 市场环境因子 (用于动态调整)
            'market_regime': ['market_trend', 'market_breadth', 'market_volatility'],
            # P0-2: 趋势质量因子 (可复现)
            # 注意: 删除了与其他类别重复的因子:
            # - relative_strength: 与sentiment类别的relative_strength_20d/60d功能重复
            # - max_drawdown_20d: 保留在volatility类别中
            'trend_quality': ['dist_to_high_20d', 'dist_to_high_60d', 'ma20_bias',
                              'ma20_slope', 'trend_strength', 'atr_contraction',
                              'volume_ratio', 'up_down_volume_ratio',
                              'current_drawdown', 'consecutive_up_days'],
            # P2-3: 机构持股变化因子 (使用变化型特征避免规模偏置)
            'institution': ['inst_count_change', 'inst_ratio_change', 'new_inst_count',
                            'north_hold_change_20d', 'north_hold_change_5d']
        }

        # 因子方向 (1: 越大越好, -1: 越小越好)
        self.factor_direction = {
            # 质量因子
            'roe': 1, 'roa': 1, 'gross_margin': 1,
            'net_margin': 1, 'ocf_ratio': 1,
            # 估值因子 (低估值好)
            'pe_ttm': -1, 'pb': -1, 'ps_ttm': -1,
            'div_yield': 1,  # 高股息好
            'pe_relative': -1, 'pb_relative': -1,
            # 动量因子
            'ret_20d': 1, 'ret_60d': 1, 'ret_120d': 1,
            'new_high': 1, 'consecutive_up': 1,
            # 波动因子 (低波动好)
            'vol_20d': -1, 'max_drawdown_20d': -1,
            # 资金因子
            'net_inflow_5d': 1, 'net_inflow_20d': 1, 'north_change': 1,
            # 赚钱效应因子
            'relative_strength_20d': 1,  # 相对强度，越强越好
            'relative_strength_60d': 1,
            'reversal_5d': -1,   # 短期反转，近期跌的反而好(均值回归)
            'reversal_10d': -1,
            'price_position': 1,  # 价格位置 (0-1)，高位代表强势
            'new_high_20d': 1,    # 创新高因子，接近1表示接近新高
            'new_high_60d': 1,    # 创新高因子，接近1表示接近新高
            'dist_to_breakout': -1,  # 距突破位距离，越小越接近突破
            # 市场环境因子
            'market_trend': 1,    # 市场趋势，上涨为正
            'market_breadth': 1,  # 市场广度，上涨股票比例
            'market_volatility': -1,  # 市场波动率，低波动好
            # P0-2: 趋势质量因子
            'dist_to_high_20d': 1,  # 距高点比例，越高越好
            'dist_to_high_60d': 1,
            'ma20_bias': 1,         # 乖离率，正向好
            'ma20_slope': 1,        # MA斜率，上升好
            'trend_strength': 1,    # 趋势强度，越强越好
            'atr_contraction': 1,   # ATR收敛，波动收敛后更容易突破
            'volume_ratio': 1,      # 量能放大，确认趋势
            'up_down_volume_ratio': 1,  # 量价配合，上涨放量
            'current_drawdown': -1,     # 回撤，越小越好
            'consecutive_up_days': 1,   # 连涨天数
            # P2-3: 机构持股变化因子
            'inst_count_change': 1,     # 机构数量增加，利好
            'inst_ratio_change': 1,     # 持股比例增加，利好
            'new_inst_count': 1,        # 新进机构数，利好
            'north_hold_change_20d': 1, # 北向增持，利好
            'north_hold_change_5d': 1   # 北向短期增持，利好
        }

        # 缓存市场环境数据
        self._market_regime_cache = {}

    def compute_momentum_factors(self, code: str, date: pd.Timestamp,
                                  lookback_days: int = 150) -> Dict[str, float]:
        """
        计算动量因子 (部分计算模式: 能算多少算多少)

        Args:
            code: 股票代码
            date: 计算日期
            lookback_days: 回看天数

        Returns:
            动量因子字典，数据不足的窗口返回NaN
        """
        df = self.processor.load_daily_price(code)
        # 最少需要2天数据才能计算任何收益率
        if df is None or len(df) < 2:
            windows = list(self.factor_config['momentum_windows'])
            return {f'ret_{w}d': np.nan for w in windows}

        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= date].tail(lookback_days)

        if len(df) < 2:
            windows = list(self.factor_config['momentum_windows'])
            return {f'ret_{w}d': np.nan for w in windows}

        df = df.sort_values('date')
        if 'close' not in df.columns:
            raise RuntimeError("factor data missing close prices")
        close = df['close'].to_numpy()
        skip_days = int(self.factor_config['momentum_skip_days'])
        if skip_days < 0:
            raise RuntimeError("momentum_skip_days must be non-negative")

        factors = {}
        windows = list(self.factor_config['momentum_windows'])
        if not windows:
            raise RuntimeError("momentum_windows is empty")

        # 部分计算模式: 对每个窗口分别判断
        for window in windows:
            required = window + skip_days + 1
            if len(close) >= required:
                start_idx = -(window + skip_days + 1)
                end_idx = -(skip_days + 1) if skip_days > 0 else -1
                recent = close[start_idx:]
                # 检查数据有效性
                if np.all(np.isfinite(recent)):
                    start_price = close[start_idx]
                    end_price = close[end_idx]
                    if start_price > 0 and end_price > 0:
                        ret = (end_price - start_price) / start_price
                        factors[f'ret_{window}d'] = ret
                    else:
                        factors[f'ret_{window}d'] = np.nan
                else:
                    factors[f'ret_{window}d'] = np.nan
            else:
                # 数据不足该窗口，返回NaN
                factors[f'ret_{window}d'] = np.nan

        return factors

    def compute_volatility_factors(self, code: str, date: pd.Timestamp,
                                    lookback_days: int = 60) -> Dict[str, float]:
        """
        计算波动率因子 (部分计算模式)

        Args:
            code: 股票代码
            date: 计算日期
            lookback_days: 回看天数

        Returns:
            波动率因子字典，数据不足返回NaN
        """
        factors = {'vol_20d': np.nan, 'max_drawdown_20d': np.nan}

        df = self.processor.load_daily_price(code)
        if df is None or len(df) == 0:
            return factors

        window = int(self.factor_config['volatility_window'])
        if window <= 1:
            raise RuntimeError("volatility_window must be greater than 1")

        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= date].tail(lookback_days)

        if len(df) < window + 1:
            return factors  # 数据不足，返回NaN

        df = df.sort_values('date')
        if 'close' not in df.columns:
            raise RuntimeError("factor data missing close prices")

        close_window = df['close'].tail(window + 1).to_numpy()
        # 数据质量检查，不通过则返回NaN
        if not np.all(np.isfinite(close_window)) or np.any(close_window <= 0):
            return factors

        returns = np.diff(close_window) / close_window[:-1]
        if not np.all(np.isfinite(returns)):
            return factors

        factors['vol_20d'] = np.std(returns, ddof=1) * np.sqrt(252)

        prices = close_window[1:]
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        factors['max_drawdown_20d'] = abs(drawdown.min())

        return factors

    def compute_quality_factors(self, code: str, date: pd.Timestamp) -> Dict[str, float]:
        """
        计算质量因子

        Args:
            code: 股票代码
            date: 计算日期

        Returns:
            质量因子字典
        """
        fin_df = self.processor.load_financial_data(code)
        if fin_df is None or len(fin_df) == 0:
            return {}  # 数据不足，跳过

        # 对齐到计算日期
        aligned = self.processor.align_financial_by_announce_date(
            fin_df, pd.DatetimeIndex([date])
        )
        if len(aligned) == 0:
            return {}  # 数据不足，跳过

        latest = aligned.iloc[0]
        factors = {}

        def parse_ratio(value: object, name: str) -> float:
            if pd.isna(value):
                raise RuntimeError(f"quality factor {name} is missing")
            if isinstance(value, str):
                raw = value.strip()
                if raw.endswith('%'):
                    raw = raw.replace('%', '')
                    value = float(raw) / 100
                else:
                    value = float(raw)
            else:
                value = float(value)
            if not np.isfinite(value):
                raise RuntimeError(f"quality factor {name} is not finite")
            return float(value)

        standard_cols = ['roe', 'roa', 'gross_margin', 'net_margin']
        for col in standard_cols:
            if col not in latest.index:
                raise RuntimeError(f"quality factor {col} is missing")
            factors[col] = parse_ratio(latest[col], col)

        return factors

    def compute_valuation_factors(self, code: str, date: pd.Timestamp) -> Dict[str, float]:
        """
        计算估值因子 (修复: 使用 EPS/BPS 计算 PE/PB)

        Args:
            code: 股票代码
            date: 计算日期

        Returns:
            估值因子字典
        """
        factors = {}

        # 从日K数据获取当日价格
        price_data = self.processor.get_price_on_date(code, date)
        if price_data is None:
            return {}  # 数据不足，跳过

        close_price = price_data.get('close')
        if close_price is None or close_price <= 0:
            return {}  # 数据不足，跳过

        # 从财务数据获取 EPS, BPS 等
        fin_df = self.processor.load_financial_data(code)
        if fin_df is not None and len(fin_df) > 0:
            aligned = self.processor.align_financial_by_announce_date(
                fin_df, pd.DatetimeIndex([date])
            )
            if len(aligned) > 0:
                latest = aligned.iloc[0]

                # 方案1: 直接使用财务数据中的 pe/pb (如果存在)
                if 'pe' in latest.index and pd.notna(latest['pe']):
                    factors['pe_ttm'] = float(latest['pe'])
                if 'pb' in latest.index and pd.notna(latest['pb']):
                    factors['pb'] = float(latest['pb'])

                # 方案2: 使用 EPS/BPS 计算 (如果 pe/pb 不存在)
                if 'pe_ttm' not in factors and 'eps' in latest.index:
                    eps = latest['eps']
                    if pd.notna(eps):
                        try:
                            eps_val = float(eps)
                            if eps_val > 0:  # 只计算正 EPS 的 PE
                                factors['pe_ttm'] = close_price / eps_val
                        except (ValueError, TypeError):
                            pass  # 无法转换，跳过

                if 'pb' not in factors and 'bps' in latest.index:
                    bps = latest['bps']
                    if pd.notna(bps):
                        try:
                            bps_val = float(bps)
                            if bps_val > 0:  # 只计算正 BPS 的 PB
                                factors['pb'] = close_price / bps_val
                        except (ValueError, TypeError):
                            pass  # 无法转换，跳过

        return factors

    def compute_flow_factors(self, code: str, date: pd.Timestamp) -> Dict[str, float]:
        """
        计算资金流因子

        Args:
            code: 股票代码
            date: 计算日期

        Returns:
            资金流因子字典
        """
        factors = {}

        flow_df = self.processor.load_moneyflow(code)
        if flow_df is None or len(flow_df) == 0:
            return {}  # 数据不足，跳过

        if 'date' not in flow_df.columns:
            return {}  # 数据不足，跳过

        flow_df['date'] = pd.to_datetime(flow_df['date'])
        flow_df = flow_df[flow_df['date'] <= date].sort_values('date')
        if len(flow_df) == 0:
            return {}  # 数据不足，跳过

        if 'net_inflow' not in flow_df.columns:
            return {}  # 数据不足，跳过

        flow_windows = list(self.factor_config['flow_windows'])
        if not flow_windows:
            raise RuntimeError("flow_windows is empty")
        required = max(flow_windows)
        if len(flow_df) < required:
            return {}  # 数据不足，跳过

        for window in flow_windows:
            recent = flow_df.tail(window)
            inflow = pd.to_numeric(recent['net_inflow'], errors='raise')
            if inflow.isna().any():
                raise RuntimeError("net_inflow contains NaN")
            factors[f'net_inflow_{window}d'] = float(inflow.sum())

        return factors

    def compute_liquidity_factors(self, code: str, date: pd.Timestamp) -> Dict[str, float]:
        """
        P5: 计算流动性因子

        用于流动性约束检查，确保选股结果可执行。

        Args:
            code: 股票代码
            date: 计算日期

        Returns:
            流动性因子字典，包含:
            - avg_amount_20d: 20日日均成交额(元)
            - avg_turnover_20d: 20日日均换手率(%)
            - liquidity_score: 综合流动性得分
        """
        factors = {}

        liq_config = self.config.get('liquidity', {})
        amount_window = int(liq_config.get('avg_amount_window', 20))
        turnover_window = int(liq_config.get('avg_turnover_window', 20))
        if amount_window <= 0 or turnover_window <= 0:
            raise RuntimeError("liquidity windows must be positive")

        df = self.processor.load_daily_price(code)
        if df is None or len(df) == 0:
            return {}  # 数据不足，跳过

        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= date].sort_values('date')

        if 'amount' not in df.columns:
            raise RuntimeError("factor data missing amount")
        if 'turnover' not in df.columns:
            raise RuntimeError("factor data missing turnover")

        required = max(amount_window, turnover_window)
        if len(df) < required:
            return {}  # 数据不足，跳过

        recent_amount = df.tail(amount_window)
        if recent_amount['amount'].isna().any():
            raise RuntimeError("amount contains NaN")
        avg_amount = float(recent_amount['amount'].mean())
        if not np.isfinite(avg_amount) or avg_amount <= 0:
            raise RuntimeError("invalid avg_amount_20d")
        factors['avg_amount_20d'] = avg_amount

        recent_turnover = df.tail(turnover_window)
        if recent_turnover['turnover'].isna().any():
            raise RuntimeError("turnover contains NaN")
        avg_turnover = float(recent_turnover['turnover'].mean())
        if not np.isfinite(avg_turnover) or avg_turnover <= 0:
            raise RuntimeError("invalid avg_turnover_20d")
        factors['avg_turnover_20d'] = avg_turnover

        factors['liquidity_score'] = avg_amount

        return factors

    def compute_sentiment_factors(self, code: str, date: pd.Timestamp,
                                   lookback_days: int = 150) -> Dict[str, float]:
        """
        计算赚钱效应/情绪因子

        包含:
        - 相对强度: 个股收益 vs 市场平均收益
        - 短期反转: 捕捉均值回归机会
        - 价格位置: 当前价格在N日高低点中的位置
        - 创新高因子: 当前价格距离N日高点的接近程度

        Args:
            code: 股票代码
            date: 计算日期
            lookback_days: 回看天数

        Returns:
            情绪因子字典
        """
        factors = {}

        df = self.processor.load_daily_price(code)
        if df is None or len(df) == 0:
            return {}  # 数据不足，跳过

        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= date].tail(lookback_days)

        if len(df) < 61:
            return {}  # 数据不足，跳过

        df = df.sort_values('date')
        for col in ('close', 'high', 'low'):
            if col not in df.columns:
                raise RuntimeError(f"factor data missing {col} prices")
        close = df['close'].to_numpy()
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        if not np.all(np.isfinite(close[-61:])):
            raise RuntimeError('factor data contains non-finite close prices')
        if not np.all(np.isfinite(high[-60:])) or not np.all(np.isfinite(low[-60:])):
            raise RuntimeError('factor data contains non-finite high/low prices')

        # 1. 相对强度因子 (个股收益率 - 市场平均收益率)
        # 需要获取市场基准收益
        market_ret = self._get_market_return(date, 20)
        market_ret_60 = self._get_market_return(date, 60)

        start_20 = close[-21]
        start_60 = close[-61]
        if start_20 <= 0 or start_60 <= 0:
            raise RuntimeError('invalid close price for sentiment factors')
        stock_ret_20 = (close[-1] - start_20) / start_20
        stock_ret_60 = (close[-1] - start_60) / start_60
        factors['relative_strength_20d'] = stock_ret_20 - market_ret
        factors['relative_strength_60d'] = stock_ret_60 - market_ret_60

        # 2. 短期反转因子 (近期跌的股票可能反弹)
        start_5 = close[-6]
        start_10 = close[-11]
        if start_5 <= 0 or start_10 <= 0:
            raise RuntimeError('invalid close price for sentiment factors')
        factors['reversal_5d'] = (close[-1] - start_5) / start_5
        factors['reversal_10d'] = (close[-1] - start_10) / start_10

        # 3. 价格位置因子 (当前价格在60日高低点中的位置, 0-1)
        high_60 = np.max(high[-60:])
        low_60 = np.min(low[-60:])
        if high_60 <= low_60:
            raise RuntimeError('invalid 60d high/low for price_position')
        factors['price_position'] = (close[-1] - low_60) / (high_60 - low_60)

        # 4. 创新高因子 (当前收盘价 / N日最高价, 1.0表示创新高)
        # 使用最高价而非收盘价来判断是否创新高
        high_20 = np.max(high[-20:])
        if high_20 <= 0:
            raise RuntimeError('invalid 20d high for new_high_20d')
        factors['new_high_20d'] = close[-1] / high_20

        if high_60 <= 0:
            raise RuntimeError('invalid 60d high for new_high_60d')
        factors['new_high_60d'] = close[-1] / high_60

        # 5. 距突破位距离 (当前价格距离20日高点的百分比距离)
        factors['dist_to_breakout'] = (high_20 - close[-1]) / high_20

        return factors

    def compute_market_regime_factors(self, date: pd.Timestamp) -> Dict[str, float]:
        """Compute market regime factors (strict)."""
        date_str = date.strftime('%Y%m%d')
        if date_str in self._market_regime_cache:
            return self._market_regime_cache[date_str]
        window = 20
        index_df = self.processor.load_index_data('sh000300')
        index_df['date'] = pd.to_datetime(index_df['date'])
        index_df = index_df[index_df['date'] <= date].sort_values('date')
        if len(index_df) < window + 1:
            raise RuntimeError('insufficient index data for market regime')
        if 'close' not in index_df.columns:
            raise RuntimeError('index data missing close prices')
        close_window = index_df['close'].tail(window + 1).to_numpy()
        if not np.all(np.isfinite(close_window)):
            raise RuntimeError('index data contains non-finite close prices')
        if np.any(close_window <= 0):
            raise RuntimeError('invalid index close price')
        market_trend = (close_window[-1] - close_window[0]) / close_window[0]
        returns = np.diff(close_window) / close_window[:-1]
        if len(returns) == 0 or not np.all(np.isfinite(returns)):
            raise RuntimeError('insufficient returns for market volatility')
        market_volatility = np.std(returns, ddof=1) * np.sqrt(252)
        market_breadth = self._compute_market_breadth(date)
        factors = {
            'market_trend': market_trend,
            'market_breadth': market_breadth,
            'market_volatility': market_volatility,
        }
        self._market_regime_cache[date_str] = factors
        return factors

    def _get_market_return(self, date: pd.Timestamp, days: int) -> float:
        """Get market benchmark return (strict)."""
        index_df = self.processor.load_index_data('sh000300')
        index_df['date'] = pd.to_datetime(index_df['date'])
        index_df = index_df[index_df['date'] <= date].sort_values('date')
        if len(index_df) < days + 1:
            raise RuntimeError('insufficient market index data')
        if 'close' not in index_df.columns:
            raise RuntimeError('index data missing close prices')
        close_window = index_df['close'].tail(days + 1).to_numpy()
        if not np.all(np.isfinite(close_window)):
            raise RuntimeError('market index contains non-finite close prices')
        if np.any(close_window <= 0):
            raise RuntimeError('invalid market index price')
        return (close_window[-1] - close_window[0]) / close_window[0]

    def _compute_market_breadth(self, date: pd.Timestamp, sample_size: int = 500) -> float:
        """Compute market breadth (strict)."""
        if not hasattr(self, '_market_breadth_cache'):
            self._market_breadth_cache = {}
        date_str = date.strftime('%Y%m%d')
        if date_str in self._market_breadth_cache:
            return self._market_breadth_cache[date_str]
        if self.use_universe and self.universe is not None:
            codes = self.universe.get_tradeable_codes_fast(date)
        else:
            codes = self.processor.get_available_codes(date)
        if len(codes) == 0:
            raise RuntimeError('no available codes for market breadth')
        if len(codes) > sample_size:
            np.random.seed(42)
            codes = list(np.random.choice(codes, sample_size, replace=False))
        up_count = 0
        total_count = 0
        for code in codes:
            df = self.processor.load_daily_price(code)
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] <= date].tail(5)
            if len(df) >= 2:
                ret = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
                if ret > 0:
                    up_count += 1
                total_count += 1
        if total_count == 0:
            raise RuntimeError('market breadth computation has no valid samples')
        breadth = up_count / total_count
        self._market_breadth_cache[date_str] = breadth
        return breadth

    def compute_trend_quality_factors(self, code: str, date: pd.Timestamp,
                                        lookback_days: int = 120) -> Dict[str, float]:
        """
        P0-2: 趋势质量因子 (可复现，连续取值)

        设计原则:
        1. 只用历史K线计算，不依赖外部榜单
        2. 连续取值而非0/1，减少过拟合
        3. 少而精，8个核心因子

        因子列表:
        - dist_to_high_20d/60d: 距N日高点比例 (0-1, 1=创新高)
        - ma20_bias: MA20乖离率
        - ma20_slope: MA20斜率
        - trend_strength: 趋势强度 (ADX代理, 0-1)
        - atr_contraction: ATR压缩比 (>1 波动收敛)
        - volume_ratio: 量能比
        - up_down_volume_ratio: 量价配合度
        - current_drawdown: 当前回撤

        Args:
            code: 股票代码
            date: 计算日期
            lookback_days: 回看天数

        Returns:
            趋势质量因子字典
        """
        factors = {}

        df = self.processor.load_daily_price(code)
        if df is None or len(df) == 0:
            return {}  # 数据不足，跳过

        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= date].tail(lookback_days)
        if len(df) < 60:
            return {}  # 数据不足，跳过

        df = df.sort_values('date')
        for col in ('close', 'high', 'low', 'volume'):
            if col not in df.columns:
                raise RuntimeError(f"factor data missing {col} prices")
        close = df['close'].to_numpy()
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        volume = df['volume'].to_numpy()

        if not np.all(np.isfinite(close[-60:])):
            raise RuntimeError('factor data contains non-finite close prices')
        if not np.all(np.isfinite(high[-60:])) or not np.all(np.isfinite(low[-60:])):
            raise RuntimeError('factor data contains non-finite high/low prices')
        if np.any(close[-60:] <= 0) or np.any(high[-60:] <= 0) or np.any(low[-60:] <= 0):
            raise RuntimeError('invalid price values for trend quality factors')
        if np.any(high[-60:] < low[-60:]):
            raise RuntimeError('high/low price relationship invalid')
        if not np.all(np.isfinite(volume[-20:])) or np.any(volume[-20:] <= 0):
            raise RuntimeError('volume data missing or insufficient')

        high_20 = np.max(high[-20:])
        high_60 = np.max(high[-60:])
        if high_20 <= 0 or high_60 <= 0:
            raise RuntimeError('invalid high prices for trend quality factors')
        factors['dist_to_high_20d'] = close[-1] / high_20
        factors['dist_to_high_60d'] = close[-1] / high_60

        ma20 = np.mean(close[-20:])
        if ma20 <= 0:
            raise RuntimeError('invalid MA20 for trend quality factors')
        factors['ma20_bias'] = (close[-1] - ma20) / ma20

        if len(close) < 25:
            return {}  # 数据不足，跳过
        ma20_5d_ago = np.mean(close[-25:-5])
        if ma20_5d_ago <= 0:
            raise RuntimeError('invalid MA20_5d_ago for trend quality factors')
        factors['ma20_slope'] = (ma20 - ma20_5d_ago) / ma20_5d_ago

        period = 14
        if len(close) < period + 1:
            return {}  # 数据不足，跳过
        high_n = high[-period:]
        low_n = low[-period:]
        prev_close = close[-period-1:-1]
        tr = np.maximum.reduce([
            high_n - low_n,
            np.abs(high_n - prev_close),
            np.abs(low_n - prev_close),
        ])
        if not np.all(np.isfinite(tr)):
            raise RuntimeError('trend strength TR contains non-finite values')
        atr = tr.mean()
        if atr <= 0:
            factors['trend_strength'] = 0.0
        else:
            up_move = high_n - high[-period-1:-1]
            down_move = low[-period-1:-1] - low_n
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
            di_plus = plus_dm.mean() / atr
            di_minus = minus_dm.mean() / atr
            denom = di_plus + di_minus
            factors['trend_strength'] = 0.0 if denom == 0 else abs(di_plus - di_minus) / denom

        def calc_atr(n: int) -> float:
            if len(close) < n + 1:
                return {}  # 数据不足，跳过
            h = high[-n:]
            l = low[-n:]
            prev = close[-n-1:-1]
            tr_local = np.maximum.reduce([
                h - l,
                np.abs(h - prev),
                np.abs(l - prev),
            ])
            if not np.all(np.isfinite(tr_local)):
                raise RuntimeError('ATR contains non-finite values')
            atr_val = tr_local.mean()
            if atr_val <= 0:
                raise RuntimeError('ATR must be positive')
            return float(atr_val)

        atr_10 = calc_atr(10)
        atr_30 = calc_atr(30)
        factors['atr_contraction'] = atr_30 / atr_10

        vol_5 = np.mean(volume[-5:])
        vol_20 = np.mean(volume[-20:])
        if vol_20 <= 0:
            raise RuntimeError('invalid volume for volume_ratio')
        factors['volume_ratio'] = vol_5 / vol_20

        returns = np.diff(close[-20:])
        if not np.all(np.isfinite(returns)):
            raise RuntimeError('returns contain non-finite values')
        vol_recent = volume[-19:]
        up_mask = returns > 0
        down_mask = returns < 0
        if not np.any(up_mask) or not np.any(down_mask):
            raise RuntimeError('up/down volume ratio undefined')
        up_vol = vol_recent[up_mask].sum()
        down_vol = vol_recent[down_mask].sum()
        if down_vol <= 0:
            raise RuntimeError('down volume is zero')
        factors['up_down_volume_ratio'] = up_vol / down_vol

        prices = close[-20:]
        peak = np.maximum.accumulate(prices)
        if np.any(peak <= 0):
            raise RuntimeError('invalid peak for drawdown')
        drawdown = (peak - prices) / peak
        factors['current_drawdown'] = float(drawdown[-1])

        consecutive = 0
        for i in range(1, min(20, len(close))):
            if close[-i] > close[-i-1]:
                consecutive += 1
            else:
                break
        factors['consecutive_up_days'] = consecutive

        return factors

    def compute_institution_factors_aligned(self, code: str, date: pd.Timestamp) -> Dict[str, float]:
        """
        P2-3: 机构持股因子 (披露日对齐版)

        关键点:
        1. 只用已公告的数据 (防止前视偏差)
        2. 使用变化型特征而非绝对水平 (避免规模偏置)
        3. 缺失值不强行填充
        4. P2-3新增: 数据缺失时记录警告

        因子列表:
        - inst_count_change: 机构数量变化
        - inst_ratio_change: 持股比例变化
        - new_inst_count: 新进机构数
        - north_hold_change_20d: 北向20日增持变化
        - north_hold_change_5d: 北向5日增持变化

        Args:
            code: 股票代码
            date: 计算日期

        Returns:
            机构因子字典
        """
        factors = {}

        inst_df = self.processor.load_institution_holding(code)
        if inst_df is None or len(inst_df) == 0:
            raise RuntimeError('institution data missing or insufficient')

        aligned = self.processor.align_slow_freq_data_by_announce_date(inst_df, date)
        if len(aligned) == 0:
            raise RuntimeError('institution data missing or insufficient')
        latest = aligned.iloc[-1]

        prev_date = date - pd.Timedelta(days=100)
        prev_aligned = self.processor.align_slow_freq_data_by_announce_date(inst_df, prev_date)
        if len(prev_aligned) == 0:
            raise RuntimeError('institution history insufficient for change')
        prev = prev_aligned.iloc[-1]

        for col in ('inst_count', 'inst_ratio', 'new_inst'):
            if col not in aligned.columns:
                raise RuntimeError(f"institution data missing {col}")

        curr_count = float(latest['inst_count'])
        prev_count = float(prev['inst_count'])
        if not np.isfinite(curr_count) or not np.isfinite(prev_count):
            raise RuntimeError('inst_count contains non-finite values')
        factors['inst_count_change'] = curr_count - prev_count

        curr_ratio = float(latest['inst_ratio'])
        prev_ratio = float(prev['inst_ratio'])
        if not np.isfinite(curr_ratio) or not np.isfinite(prev_ratio):
            raise RuntimeError('inst_ratio contains non-finite values')
        factors['inst_ratio_change'] = curr_ratio - prev_ratio

        new_inst = float(latest['new_inst'])
        if not np.isfinite(new_inst):
            raise RuntimeError('new_inst contains non-finite values')
        factors['new_inst_count'] = new_inst

        factors['north_hold_change_20d'] = self.processor.get_north_holding_change(
            code, date, days=20
        )
        factors['north_hold_change_5d'] = self.processor.get_north_holding_change(
            code, date, days=5
        )

        return factors

    def check_institution_data_availability(self) -> Dict[str, float]:
        """
        P2-3: 检查机构持股数据可用性

        用于诊断数据缺失情况

        Returns:
            数据可用性统计
        """
        inst_dir = self.root / self.config['paths']['raw_data'] / 'institution'

        stats = {
            'inst_dir_exists': inst_dir.exists(),
            'inst_file_count': 0,
            'fund_holding_summary_exists': False,
            'institution_research_exists': False,
        }

        if inst_dir.exists():
            parquet_files = list(inst_dir.glob('*.parquet'))
            stats['inst_file_count'] = len(parquet_files)

            # 检查汇总文件
            if (inst_dir / 'fund_holding_summary.parquet').exists():
                stats['fund_holding_summary_exists'] = True
            if (inst_dir / 'institution_research.parquet').exists():
                stats['institution_research_exists'] = True

        # 输出警告
        if not stats['inst_dir_exists']:
            logger.warning("P2-3警告: 机构持股目录不存在 (data/raw/institution)")
            logger.warning("  建议运行: fetcher.fetch_fund_holding_stocks()")
        elif stats['inst_file_count'] == 0:
            logger.warning("P2-3警告: 机构持股目录为空")
            logger.warning("  机构因子 (inst_count_change, inst_ratio_change, new_inst_count) 将不可用")
        else:
            logger.info(f"P2-3: 机构持股数据可用, {stats['inst_file_count']} 个文件")

        return stats

    def compute_tech_signal_factors(self, code: str, date: pd.Timestamp) -> Dict[str, float]:
        """
        计算技术信号因子 (P0-3修复: 改为K线可复现版本)

        原版本依赖榜单快照数据(new_high/consecutive_up)，历史可能为空，
        线上可能不为空��会造成分布漂移。

        P0-3修复: 改为从K线直接计算，确保可复现:
        - new_high_20d: 是否创20日新高
        - new_high_60d: 是否创60日新高
        - consecutive_up: 已在trend_quality中计算，此处不再重复

        Args:
            code: 股票代码
            date: 计算日期

        Returns:
            技术信号因子字典
        """
        factors = {}

        # 加载K线数据
        df = self.processor.load_daily_price(code)
        if df is None or len(df) == 0:
            return {}  # 数据不足，跳过

        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= date].sort_values('date')
        if len(df) < 60:
            return {}  # 数据不足，跳过

        if len(df) < 5:  # 数据太少
            return {}  # 数据不足，跳过

        current_close = df.iloc[-1]['close']
        current_high = df.iloc[-1]['high']

        # P0-3修复: 从K线计算创新高
        # 20日新高: 当前收盘价 >= 过去20日最高价
        if len(df) >= 20:
            high_20d = df.tail(20)['high'].max()
            factors['new_high_20d'] = 1 if current_high >= high_20d else 0
        else:
            factors['new_high_20d'] = 0

        # 60日新高
        if len(df) >= 60:
            high_60d = df.tail(60)['high'].max()
            factors['new_high_60d'] = 1 if current_high >= high_60d else 0
        else:
            factors['new_high_60d'] = 0

        # 突破前期高点比例 (距离20日高点的距离)
        if len(df) >= 20:
            high_20d = df.tail(20)['high'].max()
            if high_20d > 0:
                factors['dist_to_breakout'] = (current_close - high_20d) / high_20d

        return factors

    def compute_all_factors(self, code: str, date: pd.Timestamp) -> Dict[str, float]:
        """
        计算单只股票的所有因子 (部分计算模式: 单个因子失败不影响其他)

        Args:
            code: 股票代码
            date: 计算日期

        Returns:
            所有因子的字典
        """
        factors = {'code': code, 'date': date}

        # 动量因子
        try:
            momentum = self.compute_momentum_factors(code, date)
            factors.update(momentum)
        except Exception:
            pass  # 失败时跳过，保持已有因子

        # 波动率因子
        try:
            volatility = self.compute_volatility_factors(code, date)
            factors.update(volatility)
        except Exception:
            pass

        # 质量因子
        try:
            quality = self.compute_quality_factors(code, date)
            factors.update(quality)
        except Exception:
            pass

        # 估值因子
        try:
            valuation = self.compute_valuation_factors(code, date)
            factors.update(valuation)
        except Exception:
            pass

        # 资金流因子
        try:
            flow = self.compute_flow_factors(code, date)
            factors.update(flow)
        except Exception:
            pass

        # P5: 流动性因子
        try:
            liquidity = self.compute_liquidity_factors(code, date)
            factors.update(liquidity)
        except Exception:
            pass

        # 技术信号因子
        try:
            tech = self.compute_tech_signal_factors(code, date)
            factors.update(tech)
        except Exception:
            pass

        # 新增: 赚钱效应/情绪因子
        try:
            sentiment = self.compute_sentiment_factors(code, date)
            factors.update(sentiment)
        except Exception:
            pass

        # P0-2: 趋势质量因子 (可复现)
        try:
            trend_quality = self.compute_trend_quality_factors(code, date)
            factors.update(trend_quality)
        except Exception:
            pass

        # P2-3: 机构持股因子 (披露日对齐)
        try:
            institution = self.compute_institution_factors_aligned(code, date)
            factors.update(institution)
        except Exception:
            pass

        # 新增: 市场环境因子 (全市场统一值)
        try:
            market_regime = self.compute_market_regime_factors(date)
            factors.update(market_regime)
        except Exception:
            pass

        return factors

    def compute_factor_cross_section(self, date: pd.Timestamp,
                                      codes: List[str] = None,
                                      exclude_st: bool = True,
                                      add_forward_return: bool = True,
                                      holding_period: int = None,
                                      parallel: Optional[bool] = None,
                                      max_workers: Optional[int] = None) -> pd.DataFrame:
        """
        计算指定日期的因子横截面 (P0-4修复: 使用Universe消除生存者偏差)

        Args:
            date: 计算日期
            codes: 股票代码列表，默认使用当日可交易股票
            exclude_st: 是否排除ST股票（P1-1）
            add_forward_return: 是否添加前瞻收益率标签 (Bug修复: 默认True)
            holding_period: 持仓周期(交易日)，默认从配置读取

        Returns:
            因子横截面DataFrame (包含 forward_return 列用于深度学习训练)
        """
        if codes is None:
            # P0-4修复: 优先使用HistoricalUniverse
            if self.use_universe and self.universe is not None:
                codes = self.universe.get_tradeable_codes_fast(date)
                logger.debug(f"使用HistoricalUniverse获取 {len(codes)} 只股票")
            else:
                codes = self.processor.get_available_codes(date)

        if len(codes) == 0:
            logger.warning(f"{date} 没有可用股票")
            raise RuntimeError("factor data missing or cache not found")

        # P1-1: 排除ST股票
        if exclude_st:
            original_count = len(codes)
            codes = [c for c in codes if not self.processor.is_st_stock(c, date)]
            excluded = original_count - len(codes)
            if excluded > 0:
                logger.info(f"排除 {excluded} 只ST股票")

        logger.info(f"计算 {date.date()} 的因子横截面, 股票数: {len(codes)}")

        parallel_enabled = self.factor_config.get('parallel', False)
        parallel_flag = parallel_enabled if parallel is None else parallel
        worker_count = self.factor_config.get('parallel_workers', 4) if max_workers is None else max_workers
        if worker_count is None:
            worker_count = 1
        worker_count = int(worker_count)
        if worker_count < 1:
            raise RuntimeError("parallel_workers must be positive")

        factor_list = []
        if parallel_flag and worker_count > 1:
            compute = partial(self.compute_all_factors, date=date)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for factors in tqdm(
                    executor.map(compute, codes),
                    total=len(codes),
                    desc=f"计算因子 {date.date()}",
                ):
                    factor_list.append(factors)
        else:
            for code in tqdm(codes, desc=f"计算因子 {date.date()}"):
                factors = self.compute_all_factors(code, date)
                factor_list.append(factors)

        df = pd.DataFrame(factor_list)

        # Bug修复: 添加前瞻收益率标签 (双轨策略)
        # 缓存保存原始 close-to-close 收益率
        # 深度管道训练时可按 ExecutionPolicy 调整
        if add_forward_return and len(df) > 0:
            if holding_period is None:
                holding_period = self.config.get('backtest', {}).get('holding_period', 10)
            df = self._add_forward_return(df, date, holding_period)

        return df

    def _add_forward_return(self, df: pd.DataFrame, date: pd.Timestamp,
                            holding_period: int = 10) -> pd.DataFrame:
        """Add forward_return column (部分计算模式: 数据缺失返回NaN)."""
        if 'code' not in df.columns:
            raise RuntimeError('factor data missing code column')
        exec_config = self.config.get('execution', {})
        label_entry = exec_config.get('label_entry')
        label_exit = exec_config.get('label_exit')
        if label_entry is None or label_exit is None:
            raise RuntimeError('execution.label_entry/label_exit required')
        future_date = self._get_future_trading_date(date, holding_period)
        if label_entry == 'open':
            next_date = self._get_future_trading_date(date, 1)
        else:
            next_date = None
        df = df.copy()
        df['forward_return'] = np.nan
        for idx, row in df.iterrows():
            code = row['code']
            try:
                price_data = self.processor.load_daily_price(code)
                if price_data is None or len(price_data) == 0:
                    continue  # 无数据，保持NaN
                price_data['date'] = pd.to_datetime(price_data['date'])
                current_row = price_data[price_data['date'] == date]
                future_row = price_data[price_data['date'] == future_date]
                if len(current_row) == 0 or len(future_row) == 0:
                    continue  # 数据缺失，保持NaN
                if label_entry == 'close':
                    entry_price = current_row['close'].values[0]
                else:
                    next_row = price_data[price_data['date'] == next_date]
                    if len(next_row) == 0 or 'open' not in next_row.columns:
                        continue  # 数据缺失，保持NaN
                    entry_price = next_row['open'].values[0]
                if label_exit == 'close':
                    exit_price = future_row['close'].values[0]
                else:
                    if 'open' not in future_row.columns:
                        continue  # 数据缺失，保持NaN
                    exit_price = future_row['open'].values[0]
                if entry_price <= 0 or exit_price <= 0:
                    continue  # 无效价格，保持NaN
                df.at[idx, 'forward_return'] = (exit_price - entry_price) / entry_price
            except Exception:
                continue  # 任何异常，保持NaN
        return df

    def _get_future_trading_date(self, current_date: pd.Timestamp,
                                  days_ahead: int) -> Optional[pd.Timestamp]:
        """Get future trading date (strict)."""
        calendar = self.processor.trade_calendar
        if calendar is None or len(calendar) == 0:
            raise RuntimeError('trade calendar is empty')
        calendar_ts = [pd.Timestamp(d) for d in sorted(calendar)]
        if current_date not in calendar_ts:
            raise RuntimeError(f"date not in trade calendar: {current_date}")
        current_idx = calendar_ts.index(current_date)
        future_idx = current_idx + days_ahead
        if future_idx >= len(calendar_ts):
            raise RuntimeError('future trading date out of range')
        return calendar_ts[future_idx]

    def standardize_factors(self, factor_df: pd.DataFrame,
                            exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        因子标准化 (Winsorize + Z-score)

        P0-3修复: 保留风控约束所需因子的原始值(*_raw列)
        - 打分/模型使用标准化后的值
        - 风控约束(波动率、流动性等)使用原始值

        Args:
            factor_df: 因子数据
            exclude_cols: 不需要标准化的列

        Returns:
            标准化后的因子数据
        """
        # Bug修复: forward_return 是标签，不应被标准化
        default_exclude = ['code', 'date', 'forward_return', 'label']
        exclude_cols = exclude_cols or default_exclude
        # 合并用户指定的排除列
        exclude_cols = list(set(exclude_cols) | set(default_exclude))
        df = factor_df.copy()

        # P0-3修复: 需要保留原始值的因子（用于风控约束）
        # 这些因子有明确的物理含义，约束阈值需要用原始尺度
        # P5: 新增流动性因子
        raw_factors = ['vol_20d', 'volume', 'turnover', 'amount',
                       'max_drawdown_20d', 'current_drawdown',
                       'turnover_rate', 'avg_turnover',
                       'avg_amount_20d', 'avg_turnover_20d', 'liquidity_score']  # P5新增

        factor_cols = [c for c in df.columns if c not in exclude_cols]

        for col in factor_cols:
            if df[col].dtype in [np.float64, np.int64, float, int]:
                # 跳过全为空或只有一个值的列
                non_null = df[col].dropna()
                if len(non_null) < 2 or non_null.nunique() < 2:
                    continue

                # P0-3修复: 先保留原始值（在标准化之前）
                if col in raw_factors:
                    df[f'{col}_raw'] = df[col].copy()

                df[col] = standardize(df[col])

        return df

    def neutralize_industry(self, factor_df: pd.DataFrame,
                            factor_col: str,
                            industry_col: str) -> pd.Series:
        """
        行业中性化

        Args:
            factor_df: 因子数据
            factor_col: 因子列名
            industry_col: 行业列名

        Returns:
            中性化后的因子
        """
        if industry_col not in factor_df.columns:
            return factor_df[factor_col]

        # 行业内排名或回归残差
        df = factor_df.copy()
        df['neutralized'] = df.groupby(industry_col)[factor_col].transform(
            lambda x: zscore(x)
        )
        return df['neutralized']

    def compute_and_save_factors(self, start_date: str, end_date: str,
                                  freq: str = 'biweekly',
                                  parallel_dates: bool = None,
                                  date_workers: int = None,
                                  skip_existing: bool = True) -> None:
        """
        计算并保存因子数据 (支持 universe_id 缓存 + 日期级并行)

        Args:
            start_date: 开始日期
            end_date: 结束日期
            freq: 计算频率 ('daily', 'weekly', 'biweekly', 'monthly')
            parallel_dates: 是否启用日期级并行（默认从配置读取）
            date_workers: 日期并行线程数（默认从配置读取）
            skip_existing: 是否跳过已存在的因子缓存
        """
        from .utils import get_rebalance_dates

        calendar = self.processor.trade_calendar
        rebalance_dates = get_rebalance_dates(start_date, end_date, calendar, freq)

        # 从配置读取并行参数
        if parallel_dates is None:
            parallel_dates = self.factor_config.get('parallel_dates', False)
        if date_workers is None:
            date_workers = self.factor_config.get('date_workers', 4)

        if skip_existing:
            factors_dir = self.root / self.config['paths']['processed_data'] / 'factors'
            cached_dates = set()
            if factors_dir.exists():
                for path in factors_dir.glob("factors_*.parquet"):
                    stem = path.stem
                    if not stem.startswith("factors_"):
                        continue
                    parts = stem.split("_")
                    if len(parts) >= 2 and len(parts[1]) == 8 and parts[1].isdigit():
                        cached_dates.add(parts[1])

            if cached_dates:
                original_count = len(rebalance_dates)
                rebalance_dates = [
                    d for d in rebalance_dates
                    if d.strftime('%Y%m%d') not in cached_dates
                ]
                skipped = original_count - len(rebalance_dates)
                if skipped > 0:
                    logger.info(f"跳过已有因子日期: {skipped}")

        logger.info(f"计算因子: {start_date} ~ {end_date}, "
                    f"共 {len(rebalance_dates)} 个日期, "
                    f"日期并行: {parallel_dates}, workers: {date_workers}")

        if len(rebalance_dates) == 0:
            logger.info("因子缓存已齐全，无需重复计算")
            return

        if parallel_dates and date_workers > 1 and len(rebalance_dates) > 1:
            # 日期级并行模式
            # 注意: 并行时关闭内部股票级并行，避免线程爆炸
            self._compute_dates_parallel(rebalance_dates, date_workers)
        else:
            # 串行模式 (保持原有逻辑)
            for date in tqdm(rebalance_dates, desc="计算因子"):
                self._compute_single_date(date)

        logger.info("因子计算完成")

    def _compute_single_date(self, date: pd.Timestamp) -> bool:
        """
        计算单个日期的因子 (内部方法)

        Args:
            date: 计算日期

        Returns:
            是否成功
        """
        try:
            factor_df = self.compute_factor_cross_section(date)

            if len(factor_df) > 0:
                # 标准化
                factor_df = self.standardize_factors(factor_df)

                # 保存 (使用 universe_id)
                date_str = date.strftime('%Y%m%d')
                if 'code' in factor_df.columns:
                    codes = factor_df['code'].tolist()
                    universe_id = generate_universe_id(codes, date_str)
                    self.save_factor_data(factor_df, date_str, universe_id)
                else:
                    self.save_factor_data(factor_df, date_str)
                return True
            return False
        except Exception as e:
            logger.error(f"日期 {date.date()} 因子计算失败: {e}")
            return False

    def _compute_dates_parallel(self, dates: List[pd.Timestamp], workers: int) -> None:
        """
        并行计算多个日期的因子

        Args:
            dates: 日期列表
            workers: 并行线程数
        """
        success_count = 0
        fail_count = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交所有任务
            future_to_date = {
                executor.submit(self._compute_single_date, date): date
                for date in dates
            }

            # 使用 tqdm 显示进度
            for future in tqdm(as_completed(future_to_date), total=len(dates), desc="计算因子(并行)"):
                date = future_to_date[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    logger.error(f"日期 {date.date()} 异常: {e}")
                    fail_count += 1

        logger.info(f"并行计算完成: 成功 {success_count}, 失败 {fail_count}")

    def load_factor_data(self, date: str, universe_id: str = None) -> pd.DataFrame:
        """
        加载已保存的因子数据 (严格模式: 只支持新格式)

        Args:
            date: 日期字符串 YYYYMMDD
            universe_id: 股票池标识（推荐提供，确保可复现性）

        Returns:
            因子数据，如不存在返回空DataFrame
        """
        factors_dir = self.root / self.config['paths']['processed_data'] / 'factors'

        # 严格模式: 只支持带 universe_id 的新格式
        if universe_id:
            path = factors_dir / f"factors_{date}_{universe_id}.parquet"
            if path.exists():
                logger.debug(f"从缓存加载因子: {path.name}")
                return load_parquet(path)
            else:
                # 缓存不存在是正常场景，返回空DF让调用方计算
                logger.debug(f"因子缓存不存在: {path.name}，将重新计算")
                return pd.DataFrame()

        # 无 universe_id 时，查找该日期的新格式文件（单一匹配）
        pattern = f"factors_{date}_*.parquet"
        matching_files = list(factors_dir.glob(pattern))
        if len(matching_files) == 1:
            path = matching_files[0]
            logger.debug(f"从缓存加载因子: {path.name}")
            return load_parquet(path)
        elif len(matching_files) > 1:
            # 严格模式: 多个匹配时报错，要求明确指定 universe_id
            logger.warning(
                f"日期 {date} 有 {len(matching_files)} 个因子缓存文件，"
                f"请明确指定 universe_id 参数以确保可复现性"
            )
            raise RuntimeError("multiple factor cache files found, please specify universe_id")

        # 无匹配文件，返回空DF让调用方计算
        logger.debug(f"日期 {date} 无因子缓存，将重新计算")
        return pd.DataFrame()

    def save_factor_data(self, factor_df: pd.DataFrame, date: str,
                         universe_id: str = None) -> None:
        """
        保存因子数据到缓存

        Args:
            factor_df: 因子数据DataFrame
            date: 日期字符串 YYYYMMDD
            universe_id: 股票池标识（可选）
        """
        save_dir = ensure_dir(self.root / self.config['paths']['processed_data'] / 'factors')

        if universe_id:
            save_path = save_dir / f"factors_{date}_{universe_id}.parquet"
        else:
            # 自动生成 universe_id
            if 'code' in factor_df.columns:
                codes = factor_df['code'].tolist()
                universe_id = generate_universe_id(codes, date)
                save_path = save_dir / f"factors_{date}_{universe_id}.parquet"
            else:
                save_path = save_dir / f"factors_{date}.parquet"

        save_parquet(factor_df, save_path)
        logger.debug(f"因子数据已缓存: {save_path}")

    def get_factor_names(self, category: str = None) -> List[str]:
        """
        获取因子名称列表

        Args:
            category: 因子类别，None表示全部

        Returns:
            因子名称列表
        """
        if category is None:
            all_factors = []
            for factors in self.factor_categories.values():
                all_factors.extend(factors)
            return all_factors

        return self.factor_categories.get(category, [])


if __name__ == "__main__":
    # 测试代码
    engine = FactorEngine()

    # 测试单只股票因子计算
    test_date = pd.Timestamp('2024-01-15')
    test_code = '000001'

    factors = engine.compute_all_factors(test_code, test_date)
    print(f"\n{test_code} 在 {test_date.date()} 的因子:")
    for k, v in factors.items():
        print(f"  {k}: {v}")
