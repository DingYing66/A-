"""
单一事实来源组件 (Single Source of Truth)

提供统一的执行、可交易性与成本规则，确保：
- 训练/回测/部署口径一致
- Backtester 与 RL 环境共用同一实现

组件列表：
- ExecutionPolicy: 决策时点/执行时点/成交价口径
- TradeabilityPolicy: 停牌/涨跌停/ST/退市/流动性判定
- CostModel: 佣金/印花税/滑点/冲击成本

用法：
    from src.deep_learning.policies import ExecutionPolicy, TradeabilityPolicy, CostModel

    exec_policy = ExecutionPolicy.from_config(config)
    trade_policy = TradeabilityPolicy.from_config(config)
    cost_model = CostModel.from_config(config)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, time
from enum import Enum
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger

logger = setup_logger(__name__)


# ===================== 枚举定义 =====================

class ExecutionTiming(Enum):
    """执行时点"""
    T_CLOSE = "t_close"       # T日收盘价执行
    T1_OPEN = "t1_open"       # T+1日开盘价执行
    T1_VWAP = "t1_vwap"       # T+1日VWAP执行


class PriceType(Enum):
    """价格类型"""
    CLOSE = "close"
    OPEN = "open"
    VWAP = "vwap"
    TWAP = "twap"


# ===================== ExecutionPolicy =====================

@dataclass
class ExecutionPolicy:
    """
    执行策略 - 单一事实来源 (Bug修复: 默认值与 utils.py 保持一致)

    定义决策时点、执行时点、成交价口径，确保训练/回测/部署一致。

    默认行为 (与 utils.get_execution_config 一致):
    - trade_at_close=False: T+1开盘执行
    - execution_price='open': 使用开盘价执行
    - label_entry_price='open': 标签入场价用开盘价
    - label_exit_price='close': 标签出场价用收盘价
    """

    # 决策与执行时点 (Bug修复: 默认 T+1 开盘执行，与 utils.py 一致)
    trade_at_close: bool = False          # False: T+1开盘执行 (默认)
    execution_price: PriceType = PriceType.OPEN
    label_entry_price: PriceType = PriceType.OPEN
    label_exit_price: PriceType = PriceType.CLOSE

    # 复权口径（全局唯一）
    adjust: str = "hfq"  # hfq/qfq/None

    # 交易日推进规则
    skip_non_trading_days: bool = True

    # 信号生效延迟（交易日）
    signal_delay: int = 0  # 0 = 当日生效

    def __post_init__(self):
        """验证配置一致性"""
        if self.trade_at_close:
            # T日收盘执行时，入场价必须是收盘价
            if self.execution_price != PriceType.CLOSE:
                logger.warning(f"trade_at_close=True 时建议 execution_price=close, 当前: {self.execution_price}")
            if self.label_entry_price != self.execution_price:
                raise ValueError(f"label_entry_price ({self.label_entry_price}) 必须与 execution_price ({self.execution_price}) 一致")
        else:
            # T+1开盘执行
            if self.execution_price != PriceType.OPEN:
                logger.warning(f"trade_at_close=False 时建议 execution_price=open, 当前: {self.execution_price}")

    @classmethod
    def from_config(cls, config: dict) -> 'ExecutionPolicy':
        """从配置字典创建 (P1-5: 严格模式统一配置键名)"""
        exec_cfg = config.get('execution', {})
        data_cfg = config.get('data_fetch', {})
        required_keys = ['trade_at_close', 'execution_price', 'label_entry', 'label_exit']
        missing = [key for key in required_keys if key not in exec_cfg]
        if missing:
            raise RuntimeError(f"missing execution config keys: {missing}")
        if 'adjust' not in data_cfg:
            raise RuntimeError("missing data_fetch.adjust in config")

        # P1-5: 统一使用 label_entry/label_exit，不再支持 label_entry_price 等旧键名
        return cls(
            trade_at_close=exec_cfg['trade_at_close'],
            execution_price=PriceType(exec_cfg['execution_price']),
            label_entry_price=PriceType(exec_cfg['label_entry']),
            label_exit_price=PriceType(exec_cfg['label_exit']),
            adjust=data_cfg['adjust'],
            skip_non_trading_days=exec_cfg.get('skip_non_trading_days', True),
            signal_delay=exec_cfg.get('signal_delay', 0),
        )

    def get_execution_price(self, row: pd.Series) -> float:
        """获取执行价格"""
        price_col = self.execution_price.value
        if price_col in row:
            return row[price_col]
        raise ValueError(f"无法获取执行价格，缺少 {price_col} 列")

    def get_label_entry_price(self, row: pd.Series) -> float:
        """获取标签入场价格 (P1-5: 严格模式不回退)"""
        price_col = self.label_entry_price.value
        if price_col in row:
            return row[price_col]
        raise ValueError(f"无法获取标签入场价格，缺少 {price_col} 列")

    def get_label_exit_price(self, row: pd.Series) -> float:
        """获取标签出场价格 (P1-5: 严格模式不回退)"""
        price_col = self.label_exit_price.value
        if price_col in row:
            return row[price_col]
        raise ValueError(f"无法获取标签出场价格，缺少 {price_col} 列")

    def to_dict(self) -> dict:
        """转换为字典（用于缓存 key）"""
        return {
            'trade_at_close': self.trade_at_close,
            'execution_price': self.execution_price.value,
            'label_entry_price': self.label_entry_price.value,
            'label_exit_price': self.label_exit_price.value,
            'adjust': self.adjust,
            'signal_delay': self.signal_delay,
        }


# ===================== TradeabilityPolicy =====================

@dataclass
class TradeabilityPolicy:
    """
    可交易性策略 - 单一事实来源

    定义停牌、涨跌停、ST/退市、流动性等可交易性判定规则。

    强约束：
    - 所有可交易性判定必须在样本构造阶段统一处理
    - 训练/回测/部署使用相同的判定规则
    """

    # ST/退市过滤
    exclude_st: bool = True
    exclude_delisted: bool = True

    # 涨跌停处理
    check_limit_up: bool = True    # 涨停不可买入
    check_limit_down: bool = True  # 跌停不可卖出
    limit_threshold: float = 0.098  # 涨跌停阈值（考虑精度）

    # 停牌处理
    exclude_suspended: bool = True

    # 流动性约束
    min_amount: float = 5000 * 10000      # 最小成交额（元）
    min_turnover: float = 0.5             # 最小换手率（%）
    min_market_cap: float = 0.0           # 最小市值（元）

    # 上市天数约束
    min_list_days: int = 60

    # 北交所过滤
    exclude_bj: bool = True

    @classmethod
    def from_config(cls, config: dict) -> 'TradeabilityPolicy':
        """从配置字典创建"""
        pool_cfg = config.get('stock_pool', {})
        backtest_cfg = config.get('backtest', {})

        return cls(
            exclude_st=pool_cfg.get('exclude_st', True),
            exclude_delisted=pool_cfg.get('exclude_delisted', True),
            check_limit_up=backtest_cfg.get('check_limit_up', True),
            check_limit_down=backtest_cfg.get('check_limit_down', True),
            limit_threshold=backtest_cfg.get('limit_threshold', 0.098),
            exclude_suspended=backtest_cfg.get('exclude_suspended', True),
            min_amount=pool_cfg.get('min_avg_amount', 5000) * 10000,
            min_turnover=pool_cfg.get('min_avg_turnover', 0.5),
            min_market_cap=pool_cfg.get('min_market_cap', 0.0),
            min_list_days=pool_cfg.get('min_list_days', 60),
            exclude_bj=pool_cfg.get('exclude_bj', True),
        )

    def is_tradeable(self, row: pd.Series, action: str = 'buy') -> Tuple[bool, str]:
        """
        判断是否可交易

        Args:
            row: 股票行情/属性数据
            action: 'buy' 或 'sell'

        Returns:
            (可交易性, 不可交易原因)
        """
        code = row.get('code', '')
        name = row.get('name', '')

        # ST/退市
        if self.exclude_st:
            if 'ST' in str(name) or '*ST' in str(name):
                return False, 'ST'
            if '退' in str(name):
                return False, 'Delisted'

        # 北交所
        if self.exclude_bj:
            if str(code).startswith(('8', '4', '92')):
                return False, 'BJ_Exchange'

        # 停牌
        if self.exclude_suspended:
            if row.get('suspended', False) or row.get('is_suspended', False):
                return False, 'Suspended'
            # 成交量为0也视为停牌
            if row.get('volume', 1) == 0 or row.get('amount', 1) == 0:
                return False, 'No_Volume'

        # 涨跌停检查
        pct_change = row.get('pct_change', 0) / 100 if row.get('pct_change', 0) > 1 else row.get('pct_change', 0)

        if action == 'buy' and self.check_limit_up:
            if pct_change >= self.limit_threshold:
                return False, 'Limit_Up'

        if action == 'sell' and self.check_limit_down:
            if pct_change <= -self.limit_threshold:
                return False, 'Limit_Down'

        # 流动性检查
        amount = row.get('amount', float('inf'))
        if amount < self.min_amount:
            return False, 'Low_Amount'

        turnover = row.get('turnover', float('inf'))
        if turnover < self.min_turnover:
            return False, 'Low_Turnover'

        return True, 'OK'

    def filter_tradeable(self, df: pd.DataFrame, action: str = 'buy') -> pd.DataFrame:
        """
        过滤可交易股票

        Args:
            df: 股票DataFrame
            action: 'buy' 或 'sell'

        Returns:
            过滤后的DataFrame
        """
        mask = df.apply(lambda row: self.is_tradeable(row, action)[0], axis=1)
        return df[mask].copy()

    def to_dict(self) -> dict:
        """转换为字典（用于缓存 key）"""
        return {
            'exclude_st': self.exclude_st,
            'exclude_delisted': self.exclude_delisted,
            'check_limit_up': self.check_limit_up,
            'check_limit_down': self.check_limit_down,
            'limit_threshold': self.limit_threshold,
            'min_amount': self.min_amount,
            'min_turnover': self.min_turnover,
            'min_list_days': self.min_list_days,
            'exclude_bj': self.exclude_bj,
        }


# ===================== CostModel =====================

@dataclass
class CostModel:
    """
    成本模型 - 单一事实来源

    定义佣金/印花税/滑点/冲击成本计算规则。

    用途：
    - 回测成本扣减
    - RL 奖励中的成本惩罚
    - 成本调整后收益计算
    """

    # 佣金（双向）
    commission: float = 0.0003        # 万三

    # 印花税（仅卖出）
    stamp_duty: float = 0.001         # 千一

    # 滑点
    slippage: float = 0.001           # 千一

    # 市场冲击（可选，按成交额比例）
    impact_coefficient: float = 0.0   # 冲击系数
    impact_power: float = 0.5         # 冲击指数

    # 最小佣金（元）
    min_commission: float = 5.0

    @classmethod
    def from_config(cls, config: dict) -> 'CostModel':
        """从配置字典创建"""
        backtest_cfg = config.get('backtest', {})
        cost_cfg = config.get('cost', backtest_cfg)

        return cls(
            commission=cost_cfg.get('commission', 0.0003),
            stamp_duty=cost_cfg.get('stamp_duty', 0.001),
            slippage=cost_cfg.get('slippage', 0.001),
            impact_coefficient=cost_cfg.get('impact_coefficient', 0.0),
            impact_power=cost_cfg.get('impact_power', 0.5),
            min_commission=cost_cfg.get('min_commission', 5.0),
        )

    def compute_trade_cost(self,
                           trade_value: float,
                           action: str = 'buy',
                           market_volume: float = None) -> float:
        """
        计算单笔交易成本

        Args:
            trade_value: 交易金额（元）
            action: 'buy' 或 'sell'
            market_volume: 市场成交额（用于冲击成本，可选）

        Returns:
            总成本（元）
        """
        # 佣金
        commission = max(trade_value * self.commission, self.min_commission)

        # 印花税（仅卖出）
        stamp = trade_value * self.stamp_duty if action == 'sell' else 0

        # 滑点
        slippage = trade_value * self.slippage

        # 市场冲击（可选）
        impact = 0
        if self.impact_coefficient > 0 and market_volume and market_volume > 0:
            participation_rate = trade_value / market_volume
            impact = trade_value * self.impact_coefficient * (participation_rate ** self.impact_power)

        return commission + stamp + slippage + impact

    def compute_round_trip_cost(self,
                                 trade_value: float,
                                 market_volume: float = None) -> float:
        """
        计算往返成本（买入+卖出）

        Args:
            trade_value: 交易金额
            market_volume: 市场成交额

        Returns:
            往返总成本
        """
        buy_cost = self.compute_trade_cost(trade_value, 'buy', market_volume)
        sell_cost = self.compute_trade_cost(trade_value, 'sell', market_volume)
        return buy_cost + sell_cost

    def compute_turnover_cost(self,
                               turnover_rate: float,
                               portfolio_value: float) -> float:
        """
        计算换手成本

        Args:
            turnover_rate: 换手率（0~1）
            portfolio_value: 组合总市值

        Returns:
            换手成本
        """
        trade_value = portfolio_value * turnover_rate
        return self.compute_round_trip_cost(trade_value)

    def cost_rate_per_trade(self, action: str = 'buy') -> float:
        """
        获取单次交易成本率

        Returns:
            成本占交易额的比例
        """
        if action == 'buy':
            return self.commission + self.slippage
        else:
            return self.commission + self.stamp_duty + self.slippage

    def round_trip_cost_rate(self) -> float:
        """
        获取往返成本率

        Returns:
            往返成本占交易额的比例
        """
        return self.cost_rate_per_trade('buy') + self.cost_rate_per_trade('sell')

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'commission': self.commission,
            'stamp_duty': self.stamp_duty,
            'slippage': self.slippage,
            'impact_coefficient': self.impact_coefficient,
            'min_commission': self.min_commission,
        }


# ===================== 统一策略容器 =====================

@dataclass
class TradingPolicies:
    """
    交易策略容器 - 统一管理所有策略

    确保训练/回测/部署使用相同的策略实例
    """

    execution: ExecutionPolicy
    tradeability: TradeabilityPolicy
    cost: CostModel

    @classmethod
    def from_config(cls, config: dict) -> 'TradingPolicies':
        """从配置创建所有策略"""
        return cls(
            execution=ExecutionPolicy.from_config(config),
            tradeability=TradeabilityPolicy.from_config(config),
            cost=CostModel.from_config(config),
        )

    def get_config_hash(self) -> str:
        """获取配置哈希（用于缓存校验）"""
        import hashlib
        import json

        config_dict = {
            'execution': self.execution.to_dict(),
            'tradeability': self.tradeability.to_dict(),
            'cost': self.cost.to_dict(),
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def validate_consistency(self, other: 'TradingPolicies') -> Tuple[bool, List[str]]:
        """
        验证两个策略配置的一致性

        用于校验缓存是否可复用
        """
        differences = []

        # 检查执行策略
        if self.execution.to_dict() != other.execution.to_dict():
            differences.append('execution_policy')

        # 检查可交易性策略
        if self.tradeability.to_dict() != other.tradeability.to_dict():
            differences.append('tradeability_policy')

        # 检查成本模型
        if self.cost.to_dict() != other.cost.to_dict():
            differences.append('cost_model')

        return len(differences) == 0, differences


# ===================== 便捷函数 =====================

def get_default_policies() -> TradingPolicies:
    """
    获取默认交易策略

    Returns:
        使用默认配置的 TradingPolicies 实例
    """
    return TradingPolicies(
        execution=ExecutionPolicy(),
        tradeability=TradeabilityPolicy(),
        cost=CostModel(),
    )


def load_policies_from_config(config: dict) -> TradingPolicies:
    """
    从配置加载交易策略

    Args:
        config: 配置字典

    Returns:
        TradingPolicies 实例
    """
    return TradingPolicies.from_config(config)
