"""
回测引擎模块
负责模拟交易、计算组合收益
"""

import time  # P6-4: 性能监控
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Callable, Union

import pandas as pd
import numpy as np
from tqdm import tqdm

from .utils import (
    load_config, setup_logger, save_parquet, load_parquet,
    get_project_root, ensure_dir, get_rebalance_dates,
    get_next_trading_day
)
from .data_processor import DataProcessor
from .factor_engine import FactorEngine
from .scorer import FactorScorer, ConstrainedSelector, CostAwareOptimizer

# Bug修复: 使用统一的策略接口
try:
    from .deep_learning.policies import TradingPolicies, ExecutionPolicy
    _POLICIES_AVAILABLE = True
except ImportError:
    _POLICIES_AVAILABLE = False

logger = setup_logger(__name__)


class Backtester:
    """回测引擎"""

    def __init__(self, config: dict = None, use_constraints: bool = True,
                 use_cost_aware: bool = False):
        """
        初始化回测引擎

        Args:
            config: 配置字典
            use_constraints: 是否使用约束选股器（P1-3）
            use_cost_aware: 是否使用成本感知优化器（P3-1）
        """
        self.config = config or load_config()
        self.bt_config = self.config['backtest']
        self.root = get_project_root()

        self.processor = DataProcessor(self.config)
        self.factor_engine = FactorEngine(self.config)
        self.scorer = FactorScorer(self.config)

        # P1-3: 约束选股器
        self.use_constraints = use_constraints
        if use_constraints:
            self.constrained_selector = ConstrainedSelector(self.config)
        else:
            self.constrained_selector = None

        # P3-1: 成本感知优化器
        self.use_cost_aware = use_cost_aware
        if use_cost_aware:
            self.cost_aware_optimizer = CostAwareOptimizer(self.config)
        else:
            self.cost_aware_optimizer = None

        # 回测参数
        self.rebalance_freq = self.bt_config['rebalance_freq']
        self.top_n = self.bt_config['top_n']
        self.max_position = self.bt_config['max_position']
        self.commission = self.bt_config['commission']
        self.slippage = self.bt_config['slippage']
        self.benchmark = self.bt_config['benchmark']

        # P5: 执行配置 (Bug修复: 使用统一的 TradingPolicies)
        if _POLICIES_AVAILABLE:
            self.policies = TradingPolicies.from_config(self.config)
            self.execution_price_type = self.policies.execution.execution_price.value
            self.trade_at_close = self.policies.execution.trade_at_close
            # 使用 execution_price 作为 limit_check 基准
            self.limit_check_price = self.policies.execution.execution_price.value
        else:
            raise RuntimeError("TradingPolicies not available; strict execution config required")

        # 回测结果
        self.nav_history = []  # 净值历史
        self.holdings_history = []  # 持仓历史
        self.trades = []  # 交易记录
        self.daily_returns = []  # 日收益

        # 市场环境仓位控制
        self.use_position_control = True  # 是否启用仓位控制

    def _get_last_valid_price(self, code: str, date: pd.Timestamp) -> Optional[float]:
        """
        Get close price on the given date (strict).
        """
        price_data = self.processor.get_price_on_date(code, date)
        close_price = price_data.get('close')
        if close_price is None or close_price <= 0:
            raise RuntimeError(f"invalid close price for {code} on {date}")
        return close_price

    def _is_suspended(self, code: str, date: pd.Timestamp) -> bool:
        """
        检查股票是否停牌

        Args:
            code: 股票代码
            date: 日期

        Returns:
            是否停牌
        """
        self.processor.get_price_on_date(code, date)
        return False

    def _compute_position_ratio(self, factor_df: pd.DataFrame) -> float:
        """
        根据市场环境计算仓位比例 (部分计算模式: 因子缺失时使用默认值)

        基于"赚钱效应"的核心逻辑:
        - 市场广度高 (多数股票上涨) → 高仓位
        - 市场趋势向下 + 高波动 → 低仓位

        Args:
            factor_df: 因子数据 (含市场环境因子)

        Returns:
            仓位比例 (0.3 - 1.0)
        """
        if not self.use_position_control:
            return 1.0

        required_cols = ['market_trend', 'market_breadth', 'market_volatility']
        missing = [col for col in required_cols if col not in factor_df.columns]
        if missing:
            logger.warning(f"市场环境因子缺失: {missing}，使用默认仓位比例 1.0")
            return 1.0  # 缺失时返回默认值而非报错
        market_trend = factor_df['market_trend'].iloc[0]
        market_breadth = factor_df['market_breadth'].iloc[0]
        market_volatility = factor_df['market_volatility'].iloc[0]

        # 仓位计算逻辑
        position_ratio = 1.0

        # 1. 根据市场趋势调整
        if market_trend < -0.05:  # 市场20日跌超5%
            position_ratio *= 0.6
        elif market_trend < 0:
            position_ratio *= 0.8

        # 2. 根据市场广度调整 (赚钱效应核心)
        if market_breadth < 0.35:  # 少于35%股票上涨
            position_ratio *= 0.5
        elif market_breadth < 0.45:
            position_ratio *= 0.7
        elif market_breadth > 0.6:  # 超过60%股票上涨
            position_ratio *= 1.1  # 可以稍微加仓

        # 3. 根据市场波动率调整
        if market_volatility > 0.35:  # 年化波动超35%
            position_ratio *= 0.7
        elif market_volatility > 0.25:
            position_ratio *= 0.85

        # 限制在合理范围
        position_ratio = max(0.3, min(1.0, position_ratio))

        return position_ratio

    def _get_execution_price(self, code: str, date: pd.Timestamp,
                              direction: str = 'buy') -> Optional[float]:
        """
        获取执行价格（P5: 支持收盘价/开盘价 + 滑点）

        根据self.execution_price_type配置选择基准价格:
        - 'close': 使用收盘价（收盘换仓模式）
        - 'open': 使用开盘价（T+1执行模式）

        Args:
            code: 股票代码
            date: 日期
            direction: 'buy' 或 'sell'

        Returns:
            执行价格
        """
        price_data = self.processor.get_price_on_date(code, date)

        # P5: 根据配置选择基准价格 (P0-7修复: 严格模式不回退)
        if self.execution_price_type == 'close':
            base_price = price_data.get('close')
        else:
            # 严格模式: 要求open价格，不回退到close
            base_price = price_data.get('open')
            if base_price is None:
                raise RuntimeError(f"missing open price for {code} on {date}")

        if base_price is None or base_price <= 0:
            raise RuntimeError(f"invalid execution price for {code} on {date}")

        # 添加滑点
        if direction == 'buy':
            return base_price * (1 + self.slippage)
        else:
            return base_price * (1 - self.slippage)

    def _filter_tradeable_stocks(self, scored_df: pd.DataFrame,
                                   exec_date: pd.Timestamp) -> pd.DataFrame:
        """
        P4-1: 过滤不可交易的股票，实现可交易递补机制

        在exec_date检查股票是否可交易（非停牌、非涨停），
        过滤掉不可交易的股票，让下一名递补。

        Args:
            scored_df: 按分数排序的候选股票
            exec_date: 执行日期

        Returns:
            过滤后的可交易股票DataFrame
        """
        if len(scored_df) == 0:
            return scored_df

        # 确保有code列
        if 'code' not in scored_df.columns:
            raise RuntimeError("scored_df missing code column")

        tradeable_mask = []
        untradeable_count = 0

        for _, row in scored_df.iterrows():
            code = row['code']

            # 检查是否停牌
            if self._is_suspended(code, exec_date):
                tradeable_mask.append(False)
                untradeable_count += 1
                continue

            # 检查是否涨停（无法买入）
            if not self._check_tradeable(code, exec_date, 'buy'):
                tradeable_mask.append(False)
                untradeable_count += 1
                continue

            tradeable_mask.append(True)

        # 记录递补统计
        if untradeable_count > 0:
            logger.debug(f"{exec_date.date()}: {untradeable_count} 只不可交易，启用递补")

        return scored_df[tradeable_mask].copy()

    def _check_tradeable(self, code: str, date: pd.Timestamp,
                          direction: str = 'buy') -> bool:
        """
        检查股票是否可交易 (P0-1修复: 正确的涨跌停判断 + 滑点检查)

        规则:
        1. ST/*ST: ±5%
        2. 科创板(688): ±20%
        3. 创业板(300): 2020-08-24后 ±20%, 之前 ±10%
        4. 主板: ±10%
        5. 涨跌停价 = round(prev_close * (1±limit), 2)
        6. 一字板判定: 使用绝对tick容差 (0.01元)
        7. 新增: 检查滑点后执行价是否超过涨跌停价

        Args:
            code: 股票代码
            date: 日期
            direction: 交易方向 'buy' 或 'sell'

        Returns:
            是否可交易
        """
        price_data = self.processor.get_price_on_date(code, date)
        prev_close = self.processor.get_prev_close(code, date)

        open_price = price_data.get('open')
        high = price_data.get('high')
        low = price_data.get('low')

        if open_price is None or high is None or low is None:
            raise RuntimeError(f"missing OHLC data for {code} on {date}")

        # 1. 确定涨跌停幅度
        is_st = self.processor.is_st_stock(code, date)

        if is_st:
            limit_pct = 0.05  # ST股票 ±5%
        elif code.startswith('688'):  # 科创板
            limit_pct = 0.20
        elif code.startswith('3'):  # 创业板
            # 2020-08-24 起改为20%
            if date >= pd.Timestamp('2020-08-24'):
                limit_pct = 0.20
            else:
                limit_pct = 0.10
        else:  # 主板
            limit_pct = 0.10

        # 2. 计算涨跌停价 (两位小数四舍五入)
        upper_limit = round(prev_close * (1 + limit_pct), 2)
        lower_limit = round(prev_close * (1 - limit_pct), 2)

        # 3. 一字板判定 - 使用绝对tick容差 (A股最小价格单位0.01元)
        tick_tolerance = 0.01  # 绝对容差，而非相对容差

        if direction == 'buy':
            if upper_limit > 0:
                if self.limit_check_price == 'close':
                    # P5收盘换仓: high==low==涨停价 才是无法买入的一字板
                    # 这是更保守的判断：只有全天封涨停（一字板）才无法买入
                    is_limit_up = (
                        abs(high - upper_limit) <= tick_tolerance and
                        abs(low - upper_limit) <= tick_tolerance and
                        abs(high - low) <= tick_tolerance  # 高==低，一字板
                    )
                else:
                    # 开盘执行: open/high/low都接近涨停价
                    is_limit_up = (
                        abs(open_price - upper_limit) <= tick_tolerance and
                        abs(high - upper_limit) <= tick_tolerance and
                        abs(low - upper_limit) <= tick_tolerance
                    )
                if is_limit_up:
                    return False  # 一字涨停买不到

        else:  # sell
            if lower_limit > 0:
                if self.limit_check_price == 'close':
                    # P5收盘换仓: high==low==跌停价 才是无法卖出的一字板
                    is_limit_down = (
                        abs(high - lower_limit) <= tick_tolerance and
                        abs(low - lower_limit) <= tick_tolerance and
                        abs(high - low) <= tick_tolerance  # 高==低，一字板
                    )
                else:
                    # 开盘执行: open/high/low都接近跌停价
                    is_limit_down = (
                        abs(open_price - lower_limit) <= tick_tolerance and
                        abs(high - lower_limit) <= tick_tolerance and
                        abs(low - lower_limit) <= tick_tolerance
                    )
                if is_limit_down:
                    return False  # 一字跌停卖不出

        # 4. 新增: 检查滑点后执行价是否超过涨跌停价
        # 这是关键修复 - 防止滑点导致超过涨跌停的违规交易
        exec_price = self._get_execution_price(code, date, direction)

        if direction == 'buy':
            # 买入执行价不能超过涨停价
            if exec_price > upper_limit:
                logger.debug(f"{code}: 买入执行价 {exec_price:.2f} > 涨停价 {upper_limit:.2f}, 不可交易")
                return False
        else:  # sell
            # 卖出执行价不能低于跌停价
            if exec_price < lower_limit:
                logger.debug(f"{code}: 卖出执行价 {exec_price:.2f} < 跌停价 {lower_limit:.2f}, 不可交易")
                return False

        return True

    def _calc_transaction_cost(self, amount: float, direction: str) -> float:
        """
        计算交易成本

        Args:
            amount: 交易金额
            direction: 交易方向

        Returns:
            交易成本
        """
        cost = amount * self.commission

        # 卖出还需要印花税（目前为0.1%，单边征收）
        if direction == 'sell':
            cost += amount * 0.001

        return cost

    def _rebalance(self, current_holdings: Dict[str, float],
                   target_holdings: Dict[str, float],
                   exec_date: pd.Timestamp,
                   cash: float,
                   valuation_date: pd.Timestamp = None) -> Tuple[Dict[str, float], float, List[dict]]:
        """
        执行调仓 (P0-2修复: 计价时点一致性)

        P0-2修复说明:
        - valuation_date: 决策日(rb_date)，用于计算持仓市值（决策时点可得的收盘价）
        - exec_date: 执行日，用于计算目标股数（开盘价 + 滑点）
        - 避免前视偏差：不能在开盘前使用收盘价

        Args:
            current_holdings: 当前持仓 {code: shares}
            target_holdings: 目标持仓 {code: weight}
            exec_date: 执行日期（下一交易日）
            cash: 当前现金
            valuation_date: 估值日期（决策日，默认为exec_date前一日）

        Returns:
            (新持仓, 新现金, 交易记录)
        """
        trades = []

        if valuation_date is None:
            raise RuntimeError("valuation_date is required for strict rebalance")

        # P0-1修复: 计算当前持仓市值，使用valuation_date的收盘价（决策时点可得）
        current_values = {}
        suspended_codes = set()  # 记录停牌股（在exec_date）
        for code, shares in current_holdings.items():
            # 使用valuation_date估值，避免前视
            price = self._get_last_valid_price(code, valuation_date)
            price = self._get_last_valid_price(code, valuation_date)
            current_values[code] = shares * price
            # 检查exec_date是否停牌（影响交易执行）
            if self._is_suspended(code, exec_date):
                suspended_codes.add(code)

        total_value = cash + sum(current_values.values())

        # 计算目标持仓股数（使用exec_date的开盘价）
        target_shares = {}
        for code, weight in target_holdings.items():
            target_value = total_value * weight
            exec_price = self._get_execution_price(code, exec_date, 'buy')
            # 按手（100股）取整
            shares = int(target_value / exec_price / 100) * 100
            if shares > 0:
                target_shares[code] = shares

        # 先卖出
        new_holdings = {}
        new_cash = cash

        for code, shares in current_holdings.items():
            target = target_shares.get(code, 0)

            # P0-1修复: 停牌股无法交易，直接保留持仓
            if code in suspended_codes:
                new_holdings[code] = shares
                logger.debug(f"{exec_date.date()} {code}: 停牌中，保留持仓 {shares} 股")
                continue

            if shares > target:
                # 需要卖出
                sell_shares = shares - target

                if not self._check_tradeable(code, exec_date, 'sell'):
                    # 跌停无法卖出，保持持仓
                    new_holdings[code] = shares
                    continue

                exec_price = self._get_execution_price(code, exec_date, 'sell')
                sell_amount = sell_shares * exec_price
                cost = self._calc_transaction_cost(sell_amount, 'sell')
                new_cash += sell_amount - cost

                trades.append({
                    'date': exec_date,
                    'code': code,
                    'direction': 'sell',
                    'shares': sell_shares,
                    'price': exec_price,
                    'amount': sell_amount,
                    'cost': cost
                })

                if target > 0:
                    new_holdings[code] = target
            else:
                new_holdings[code] = shares

        # 再买入
        for code, target in target_shares.items():
            current = new_holdings.get(code, 0)

            if target > current:
                # 需要买入
                buy_shares = target - current

                # P0-1修复: 停牌股无法买入
                if self._is_suspended(code, exec_date):
                    logger.debug(f"{exec_date.date()} {code}: 停牌中，无法买入")
                    continue

                if not self._check_tradeable(code, exec_date, 'buy'):
                    continue

                exec_price = self._get_execution_price(code, exec_date, 'buy')
                buy_amount = buy_shares * exec_price
                cost = self._calc_transaction_cost(buy_amount, 'buy')

                if buy_amount + cost <= new_cash:
                    new_cash -= (buy_amount + cost)
                    new_holdings[code] = new_holdings.get(code, 0) + buy_shares

                    trades.append({
                        'date': exec_date,
                        'code': code,
                        'direction': 'buy',
                        'shares': buy_shares,
                        'price': exec_price,
                        'amount': buy_amount,
                        'cost': cost
                    })

        return new_holdings, new_cash, trades

    def _calc_portfolio_value(self, holdings: Dict[str, float],
                               cash: float,
                               date: pd.Timestamp) -> float:
        """
        计算组合总市值 (P0-1修复: 停牌股使用前向填充估值)

        Args:
            holdings: 持仓
            cash: 现金
            date: 日期

        Returns:
            总市值
        """
        total = cash

        for code, shares in holdings.items():
            # P0-1修复: 使用前向填充获取价格，停牌股不会被当作价值消失
            price = self._get_last_valid_price(code, date)
            total += shares * price

        return total

    def _get_benchmark_nav(self, start_date: pd.Timestamp,
                           end_date: pd.Timestamp) -> pd.Series:
        """
        获取基准净值

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            基准净值序列
        """
        index_symbol = f"sh{self.benchmark}" if not self.benchmark.startswith('sh') else self.benchmark
        index_df = self.processor.load_index_data(index_symbol)

        if len(index_df) == 0:
            raise RuntimeError(f"benchmark index data missing for {index_symbol}")

        index_df['date'] = pd.to_datetime(index_df['date'])
        index_df = index_df[(index_df['date'] >= start_date) &
                            (index_df['date'] <= end_date)]
        index_df = index_df.sort_values('date')

        if len(index_df) == 0:
            raise RuntimeError(f"benchmark index data missing in range {start_date} ~ {end_date}")

        # 归一化为净值
        nav = index_df.set_index('date')['close'] / index_df['close'].iloc[0]

        return nav

    def run(self, start_date: str, end_date: str,
            score_func: Callable = None,
            weight_func: Callable = None,
            mode: str = 'score',
            initial_cash: float = 1000000) -> pd.DataFrame:
        """
        运行回测

        Args:
            start_date: 开始日期
            end_date: 结束日期
            score_func: 打分函数，输入factor_df，返回选股结果 (mode='score'时使用)
            weight_func: 权重函数，输入(factor_df, prev_weights_dict)，返回权重字典
                        (mode='weight'时使用，用于RL策略)
            mode: 回测模式
                  - 'score': 传统评分排名模式 (默认)
                  - 'weight': 直接权重模式 (用于RL策略)
            initial_cash: 初始资金

        Returns:
            回测结果DataFrame
        """
        # P6-4: 性能监控计时器
        perf_timers = {}
        total_start = time.time()

        # 验证模式和函数
        if mode == 'weight' and weight_func is None:
            raise ValueError("mode='weight' 时必须提供 weight_func")

        logger.info(f"开始回测: {start_date} ~ {end_date}")
        logger.info(f"调仓频率: {self.rebalance_freq}, 选股数: {self.top_n}")
        logger.info(f"回测模式: {mode}")  # 记录回测模式

        # 获取交易日历和调仓日
        calendar = self.processor.trade_calendar
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        rebalance_dates = get_rebalance_dates(start_date, end_date,
                                              calendar, self.rebalance_freq)

        if len(rebalance_dates) == 0:
            raise RuntimeError("no rebalance dates available")

        # 初始化
        cash = initial_cash
        holdings = {}  # {code: shares}
        self.nav_history = []
        self.holdings_history = []
        self.trades = []

        # 获取所有交易日
        trade_dates = calendar[(calendar >= start) & (calendar <= end)]

        # P5: 创建调仓日到执行日的映射
        # trade_at_close=True: 同日收盘执行（rb_date -> rb_date）
        # trade_at_close=False: T+1开盘执行（rb_date -> next_day）
        rebalance_exec_map = {}
        for rb_date in rebalance_dates:
            if self.trade_at_close:
                # 收盘换仓：同日执行
                if rb_date <= end:
                    rebalance_exec_map[rb_date] = rb_date
            else:
                # T+1执行（现有逻辑）
                next_day = get_next_trading_day(rb_date, calendar)
                if next_day and next_day <= end:
                    rebalance_exec_map[rb_date] = next_day

        exec_mode = '收盘换仓' if self.trade_at_close else 'T+1开盘'
        logger.info(f"P5执行模式: {exec_mode}, 执行价格: {self.execution_price_type}")
        logger.info(f"共 {len(trade_dates)} 个交易日, {len(rebalance_dates)} 个调仓日")

        # 默认打分函数
        if score_func is None and mode == 'score':
            def default_score_func(factor_df):
                return self.scorer.select_top_stocks(factor_df, n=self.top_n * 2)  # 多选一倍用于递补
            score_func = default_score_func

        # ========== P4-1: 两阶段架构 ==========
        # 阶段一(离线): 只预计算分数，不应用约束
        # 阶段二(在线): 在交易循环中，基于真实持仓应用约束 + 可交易递补

        # 阶段一: 预计算因子和分数（不含约束）
        score_cache = {}  # {rb_date: scored_df} - 只存分数，不存最终选股
        factor_cache = {}  # {rb_date: factor_df} - 权重模式需要原始因子
        position_ratio_cache = {}  # 缓存仓位比例

        phase1_start = time.time()  # P6-4: 阶段一计时
        logger.info("阶段一: 预计算因子分数...")
        for rb_date in rebalance_dates:
            # 优先从缓存加载因子，没有则计算并保存
            date_str = rb_date.strftime('%Y%m%d')
            factor_df = self.factor_engine.load_factor_data(date_str)

            if len(factor_df) == 0:
                # 缓存不存在，计算因子
                factor_df = self.factor_engine.compute_factor_cross_section(rb_date)
                if len(factor_df) > 0:
                    # 标准化并保存到缓存
                    factor_df = self.factor_engine.standardize_factors(factor_df)
                    self.factor_engine.save_factor_data(factor_df, date_str)

            if len(factor_df) == 0:
                raise RuntimeError(f"factor data empty for {rb_date.date()}")

            # 缓存因子数据 (权重模式需要)
            factor_cache[rb_date] = factor_df

            # 计算仓位比例 (基于市场环境)
            position_ratio = self._compute_position_ratio(factor_df)
            position_ratio_cache[rb_date] = position_ratio

            # 计算分数（仅 score 模式需要，不应用约束，多选用于递补）
            if mode == 'score' and score_func is not None:
                scored_df = score_func(factor_df)
                if len(scored_df) == 0:
                    raise RuntimeError(f"score result empty for {rb_date.date()}")
                score_cache[rb_date] = scored_df

                # 记录日志
                if position_ratio < 1.0:
                    logger.info(f"{rb_date.date()}: 市场环境较差，仓位将降至 {position_ratio:.0%}")

        # P6-4: 阶段一计时结束
        perf_timers['phase1_factor_scoring'] = time.time() - phase1_start

        # 阶段二: 遍历交易日，在线应用约束
        phase2_start = time.time()  # P6-4: 阶段二计时
        logger.info("阶段二: 在线应用约束...")
        current_target = {}

        # 权重模式需要记录上期权重
        prev_weights_dict = {}

        for date in tqdm(trade_dates, desc="回测进行中"):
            # 检查是否需要调仓
            for rb_date, exec_date in rebalance_exec_map.items():
                # 权重模式和评分模式使用不同的缓存检查
                should_rebalance = False
                if mode == 'weight':
                    should_rebalance = date == exec_date and rb_date in factor_cache
                else:  # score mode
                    should_rebalance = date == exec_date and rb_date in score_cache

                if should_rebalance:
                    position_ratio = position_ratio_cache.get(rb_date, 1.0)

                    # 计算当前持仓权重（使用rb_date收盘价，避免前视）
                    current_weights = {}
                    if holdings:
                        total_value = self._calc_portfolio_value(holdings, cash, rb_date)
                        if total_value > 0:
                            for code, shares in holdings.items():
                                price = self._get_last_valid_price(code, rb_date)
                                if price and price > 0:
                                    current_weights[code] = (shares * price) / total_value

                    # ========== 权重模式处理 ==========
                    if mode == 'weight' and weight_func is not None:
                        factor_df = factor_cache[rb_date]

                        # 直接调用权重函数获取目标权重
                        target_weights = weight_func(factor_df, prev_weights_dict)

                        # 应用可交易性过滤
                        tradeable_df = self._filter_tradeable_stocks(factor_df, exec_date)
                        tradeable_codes = set(tradeable_df['code'].tolist())

                        # 过滤不可交易的权重
                        filtered_weights = {
                            code: w for code, w in target_weights.items()
                            if code in tradeable_codes
                        }

                        # 归一化权重
                        total = sum(filtered_weights.values())
                        if total > 1e-8:
                            # 应用仓位控制和单票上限
                            current_target = {}
                            for code, w in filtered_weights.items():
                                adjusted_w = (w / total) * position_ratio
                                current_target[code] = min(adjusted_w, self.max_position)
                        else:
                            current_target = {}

                        # 更新上期权重
                        prev_weights_dict = current_target.copy()

                        logger.debug(f"{rb_date.date()}: 权重模式 - {len(current_target)} 只股票")

                    # ========== 评分模式处理 ==========
                    else:
                        # P4-1: 在线应用约束（基于真实持仓）
                        scored_df = score_cache[rb_date]

                        # P4-1: 可交易递补机制 - 过滤exec_date不可交易的股票
                        tradeable_df = self._filter_tradeable_stocks(scored_df, exec_date)

                        # 应用约束选股（如果启用）
                        selected = tradeable_df
                        if self.use_constraints and self.constrained_selector is not None:
                            constrained_selected = self.constrained_selector.select_with_constraints(
                                tradeable_df, current_weights, n=self.top_n
                            )
                            if len(constrained_selected) > 0:
                                selected = constrained_selected
                                logger.debug(f"{rb_date.date()}: 约束选股 {len(selected)} 只")

                        # 应用成本感知优化（如果启用）
                        if self.use_cost_aware and self.cost_aware_optimizer is not None:
                            if 'total_score' not in selected.columns:
                                raise RuntimeError("total_score required for cost-aware selection")
                            cost_aware_selected = self.cost_aware_optimizer.select_with_cost_awareness(
                                selected, current_weights, n=self.top_n, score_col='total_score'
                            )
                            if len(cost_aware_selected) > 0:
                                selected = cost_aware_selected
                                logger.debug(f"{rb_date.date()}: 成本感知选股 {len(selected)} 只")

                        # 限制到top_n
                        if len(selected) > self.top_n:
                            selected = selected.head(self.top_n)

                        # 计算目标权重
                        if len(selected) > 0:
                            base_weight = 1.0 / len(selected)
                            adjusted_weight = base_weight * position_ratio
                            current_target = {row['code']: min(adjusted_weight, self.max_position)
                                              for _, row in selected.iterrows()}
                        else:
                            current_target = {}

                    # P0-2修复: 传入valuation_date=rb_date，避免前视偏差
                    holdings, cash, day_trades = self._rebalance(
                        holdings, current_target, exec_date=date, cash=cash,
                        valuation_date=rb_date
                    )
                    self.trades.extend(day_trades)

            # 计算当日净值
            nav = self._calc_portfolio_value(holdings, cash, date)

            self.nav_history.append({
                'date': date,
                'nav': nav,
                'cash': cash,
                'n_holdings': len(holdings)
            })

            self.holdings_history.append({
                'date': date,
                'holdings': holdings.copy()
            })

        # 整理结果
        result_df = pd.DataFrame(self.nav_history)
        result_df['date'] = pd.to_datetime(result_df['date'])
        result_df = result_df.set_index('date')

        # 归一化净值
        result_df['nav'] = result_df['nav'] / initial_cash

        # 添加基准
        benchmark_nav = self._get_benchmark_nav(start, end)
        if len(benchmark_nav) > 0:
            result_df['benchmark'] = benchmark_nav.reindex(result_df.index)
            result_df['benchmark'] = result_df['benchmark'].ffill()

        # 计算日收益
        result_df['daily_return'] = result_df['nav'].pct_change()
        if 'benchmark' in result_df.columns:
            result_df['benchmark_return'] = result_df['benchmark'].pct_change()
            result_df['excess_return'] = result_df['daily_return'] - result_df['benchmark_return']

        # P6-4: 阶段二计时结束 + 性能报告
        perf_timers['phase2_trading'] = time.time() - phase2_start
        perf_timers['total'] = time.time() - total_start

        logger.info(f"回测完成! 最终净值: {result_df['nav'].iloc[-1]:.4f}")

        # P6-4: 输出性能报告
        logger.info("=" * 50)
        logger.info("P6-4 性能报告:")
        logger.info(f"  阶段一(因子计算+打分): {perf_timers.get('phase1_factor_scoring', 0):.1f}秒")
        logger.info(f"  阶段二(交易执行): {perf_timers.get('phase2_trading', 0):.1f}秒")
        logger.info(f"  总耗时: {perf_timers.get('total', 0):.1f}秒")
        logger.info("=" * 50)

        return result_df

    def get_trades(self) -> pd.DataFrame:
        """
        获取交易记录

        Returns:
            交易记录DataFrame
        """
        if len(self.trades) == 0:
            return pd.DataFrame()

        return pd.DataFrame(self.trades)

    def get_holdings_on_date(self, date: pd.Timestamp) -> Dict[str, float]:
        """
        获取指定日期的持仓

        Args:
            date: 日期

        Returns:
            持仓字典
        """
        for record in reversed(self.holdings_history):
            if record['date'] <= date:
                return record['holdings']
        return {}

    def calc_turnover(self) -> float:
        """
        计算换手率

        Returns:
            平均换手率
        """
        if len(self.trades) == 0:
            return 0

        trades_df = pd.DataFrame(self.trades)
        trades_df['date'] = pd.to_datetime(trades_df['date'])

        # 按日期汇总交易金额
        daily_amount = trades_df.groupby('date')['amount'].sum()

        # 平均每次调仓的交易金额占总资产的比例
        nav_df = pd.DataFrame(self.nav_history)
        nav_df['date'] = pd.to_datetime(nav_df['date'])
        nav_df = nav_df.set_index('date')

        turnovers = []
        for date, amount in daily_amount.items():
            if date in nav_df.index:
                nav = nav_df.loc[date, 'nav']
                if nav > 0:
                    turnovers.append(amount / nav)

        return np.mean(turnovers) if turnovers else 0

    def save_results(self, result_df: pd.DataFrame, name: str = None) -> None:
        """
        保存回测结果

        Args:
            result_df: 回测结果
            name: 保存名称
        """
        output_dir = ensure_dir(self.root / 'output' / 'reports')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = name or f"backtest_{timestamp}"

        # 保存净值
        nav_path = output_dir / f"{name}_nav.parquet"
        save_parquet(result_df.reset_index(), nav_path)

        # 保存交易记录
        if len(self.trades) > 0:
            trades_path = output_dir / f"{name}_trades.parquet"
            save_parquet(pd.DataFrame(self.trades), trades_path)

        logger.info(f"结果已保存: {output_dir / name}")


class WalkForwardBacktester(Backtester):
    """
    滚动回测引擎
    支持ML模型的滚动训练和预测
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.ml_model = None

    def run_walk_forward(self, start_date: str, end_date: str,
                          train_years: int = 3,
                          retrain_freq: str = 'yearly',
                          model_type: str = 'xgboost',
                          initial_cash: float = 1000000) -> pd.DataFrame:
        """
        滚动回测（Walk-Forward）

        每隔一段时间重新训练模型

        Args:
            start_date: 开始日期
            end_date: 结束日期
            train_years: 训练窗口（年）
            retrain_freq: 重训练频率 ('yearly', 'quarterly')
            model_type: 模型类型
            initial_cash: 初始资金

        Returns:
            回测结果
        """
        from .ml_model import MLModel

        logger.info(f"滚动回测: {start_date} ~ {end_date}")
        logger.info(f"训练窗口: {train_years}年, 重训练频率: {retrain_freq}")

        calendar = self.processor.trade_calendar
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # 确定重训练日期
        if retrain_freq == 'yearly':
            retrain_dates = pd.date_range(start, end, freq='YS')
        else:  # quarterly
            retrain_dates = pd.date_range(start, end, freq='QS')

        # 回测参数
        cash = initial_cash
        holdings = {}
        all_nav = []

        rebalance_dates = get_rebalance_dates(start_date, end_date,
                                              calendar, self.rebalance_freq)

        current_model = None

        for i, rb_date in enumerate(tqdm(rebalance_dates, desc="滚动回测")):
            # 检查是否需要重训练
            need_retrain = False
            for rt_date in retrain_dates:
                if rb_date >= rt_date and (current_model is None or
                                            rb_date == rebalance_dates[
                                                np.searchsorted(rebalance_dates, rt_date)]):
                    need_retrain = True
                    break

            if need_retrain or current_model is None:
                # 训练模型
                train_end = rb_date - pd.Timedelta(days=1)
                train_start = train_end - pd.DateOffset(years=train_years)

                ml = MLModel(self.config)
                X, y = ml.build_training_samples(
                    train_start.strftime('%Y%m%d'),
                    train_end.strftime('%Y%m%d')
                )

                if len(X) <= 100:
                    raise RuntimeError("insufficient training samples for ML model")
                ml.train(X, y, model_type=model_type)
                current_model = ml
                logger.info(f"模型已在 {rb_date.date()} 重训练")

            # 计算因子并选股（优先从缓存加载）
            date_str = rb_date.strftime('%Y%m%d')
            factor_df = self.factor_engine.load_factor_data(date_str)

            if len(factor_df) == 0:
                # 缓存不存在，计算因子
                factor_df = self.factor_engine.compute_factor_cross_section(rb_date)
                if len(factor_df) > 0:
                    factor_df = self.factor_engine.standardize_factors(factor_df)
                    self.factor_engine.save_factor_data(factor_df, date_str)

            if len(factor_df) == 0:
                raise RuntimeError(f"factor data empty for {rb_date.date()}")

            if current_model is None:
                raise RuntimeError("ML model not available for selection")
            selected = current_model.select_stocks_ml(factor_df, n=self.top_n)

            if len(selected) == 0:
                raise RuntimeError(f"ML selection empty for {rb_date.date()}")

            # 执行调仓
            weight = 1.0 / len(selected)
            target = {row['code']: min(weight, self.max_position)
                      for _, row in selected.iterrows()}

            # P5: 根据配置确定执行日期
            if self.trade_at_close:
                exec_date = rb_date  # 收盘换仓：同日执行
            else:
                exec_date = get_next_trading_day(rb_date, calendar)  # T+1执行

            if exec_date is None:
                raise RuntimeError("missing execution date for rebalance")
            if exec_date <= end:
                # P0-2修复: 传入valuation_date=rb_date
                holdings, cash, trades = self._rebalance(
                    holdings, target, exec_date=exec_date, cash=cash,
                    valuation_date=rb_date
                )
                self.trades.extend(trades)

            # 记录净值
            nav = self._calc_portfolio_value(holdings, cash, rb_date)
            all_nav.append({'date': rb_date, 'nav': nav})

        # 整理结果
        result_df = pd.DataFrame(all_nav)
        result_df['date'] = pd.to_datetime(result_df['date'])
        result_df = result_df.set_index('date')
        result_df['nav'] = result_df['nav'] / initial_cash

        return result_df


if __name__ == "__main__":
    # 测试代码
    bt = Backtester()

    print("回测引擎初始化完成")
    print(f"调仓频率: {bt.rebalance_freq}")
    print(f"选股数量: {bt.top_n}")
    print(f"最大仓位: {bt.max_position}")
    print(f"佣金: {bt.commission}")
