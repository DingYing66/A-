"""
强化学习环境模块 (Phase 5)

提供基于 Gymnasium 的投资组合优化环境:
- PortfolioEnv: 组合权重调整环境
- 支持连续/离散动作空间
- 多种奖励函数 (Sharpe/Sortino/Return)
- 交易成本和换手率惩罚

用法:
    from src.deep_learning.rl_env import PortfolioEnv, create_portfolio_env

    env = create_portfolio_env(factor_df, config)
    obs, info = env.reset()
    action = agent.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from collections import Counter
import copy

import numpy as np
import pandas as pd

# Gymnasium 导入
GYM_AVAILABLE = False
gym = None
spaces = None

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        pass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger

# R8 修复: 导入统一策略
try:
    from .policies import TradingPolicies, CostModel
    POLICIES_AVAILABLE = True
except ImportError:
    try:
        from src.deep_learning.policies import TradingPolicies, CostModel
        POLICIES_AVAILABLE = True
    except ImportError:
        POLICIES_AVAILABLE = False
        TradingPolicies = None
        CostModel = None

logger = setup_logger(__name__)


# ===================== 奖励函数 =====================

class RewardCalculator:
    """奖励函数计算器"""

    def __init__(self,
                 reward_type: str = 'sharpe',
                 cost_penalty: float = 0.002,
                 turnover_penalty: float = 0.001,
                 risk_free_rate: float = 0.03,
                 lookback: int = 20):
        self.reward_type = reward_type
        self.cost_penalty = cost_penalty
        self.turnover_penalty = turnover_penalty
        self.risk_free_rate = risk_free_rate / 252
        self.lookback = lookback
        self.return_history: List[float] = []

    def reset(self):
        self.return_history = []

    def calculate(self, portfolio_return: float, turnover: float,
                  prev_weights: np.ndarray, new_weights: np.ndarray) -> Tuple[float, Dict]:
        self.return_history.append(portfolio_return)

        if self.reward_type == 'return':
            base_reward = portfolio_return * 100
        elif self.reward_type == 'sharpe':
            base_reward = self._calculate_sharpe()
        elif self.reward_type == 'sortino':
            base_reward = self._calculate_sortino()
        else:
            base_reward = portfolio_return * 100

        cost_penalty = self.cost_penalty * turnover
        turnover_penalty = self.turnover_penalty * turnover
        reward = base_reward - cost_penalty - turnover_penalty

        info = {
            'base_reward': base_reward,
            'portfolio_return': portfolio_return,
            'turnover': turnover,
            'cost_penalty': cost_penalty,
            'turnover_penalty': turnover_penalty,
        }
        return reward, info

    def _calculate_sharpe(self) -> float:
        if len(self.return_history) < 2:
            return 0.0
        returns = np.array(self.return_history[-self.lookback:])
        if np.std(returns) < 1e-8:
            return 0.0
        excess = returns - self.risk_free_rate
        return np.mean(excess) / np.std(returns) * np.sqrt(252)

    def _calculate_sortino(self) -> float:
        if len(self.return_history) < 2:
            return 0.0
        returns = np.array(self.return_history[-self.lookback:])
        excess = returns - self.risk_free_rate
        downside = returns[returns < 0]
        downside_std = np.std(downside) if len(downside) >= 2 else 1e-8
        if downside_std < 1e-8:
            return 0.0
        return np.mean(excess) / downside_std * np.sqrt(252)


# ===================== 环境类占位符 =====================

# 当 gym 不可用时，提供占位符类
PortfolioEnv = None
SimplePortfolioEnv = None


# ===================== 条件定义环境类 =====================

if GYM_AVAILABLE:

    class _PortfolioEnv(gym.Env):
        """投资组合优化环境"""

        metadata = {'render_modes': ['human']}

        def __init__(self, factor_data: pd.DataFrame, config: dict = None,
                     max_stocks: int = 50, action_type: str = 'continuous',
                     reward_type: str = 'sharpe', episode_length: int = 252,
                     lookback_window: int = 20, trading_policies=None):
            super().__init__()

            self.config = config or {}
            rl_config = self.config.get('rl', {})

            # R6修复: 优先使用传入参数 max_stocks（来自 rl_adapter），
            # 仅当未传入时才读取 rl.action.n_stocks
            if max_stocks != 50:  # 50是默认值，说明调用者显式传入了值
                self.max_stocks = max_stocks
            else:
                self.max_stocks = rl_config.get('action', {}).get('n_stocks', max_stocks)
            self.action_type = rl_config.get('action', {}).get('type', action_type)
            self.episode_length = episode_length
            self.lookback_window = lookback_window

            # R6修复: 统一 max_position 读取逻辑 (与 rl_adapter.py 一致)
            # 优先 rl_adapter.max_position，回退 backtest.max_position
            rl_adapter_cfg = self.config.get('rl_adapter', {})
            bt_config = self.config.get('backtest', {})
            self.max_position = rl_adapter_cfg.get('max_position',
                                bt_config.get('max_position', 0.08))

            # Q4修复: 从 rl_adapter 配置读取 extended_features
            self.use_extended_features = rl_adapter_cfg.get('extended_features', False)

            # R5修复: 股票池配置 (避免前视偏差)
            # stock_pool_mode: 'static' = 全数据前N天 / 'dynamic' = 按episode动态调整
            self.stock_pool_mode = rl_config.get('stock_pool_mode', 'dynamic')
            self.stock_pool_lookback = rl_config.get('stock_pool_lookback', 30)

            # R8 修复: 使用统一策略
            if trading_policies is not None and POLICIES_AVAILABLE:
                self.trading_policies = trading_policies
            elif POLICIES_AVAILABLE and TradingPolicies is not None:
                self.trading_policies = TradingPolicies.from_config(self.config)
            else:
                self.trading_policies = None

            self._prepare_data(factor_data)

            # R8 修复: 从 TradingPolicies 获取成本参数
            reward_config = rl_config.get('reward', {})
            if self.trading_policies is not None and hasattr(self.trading_policies, 'cost'):
                cost_model = self.trading_policies.cost
                cost_penalty = cost_model.commission + cost_model.slippage
                turnover_penalty = reward_config.get('turnover_penalty', 0.001)
            else:
                cost_penalty = reward_config.get('cost_penalty', 0.002)
                turnover_penalty = reward_config.get('turnover_penalty', 0.001)

            self.reward_calculator = RewardCalculator(
                reward_type=reward_config.get('type', reward_type),
                cost_penalty=cost_penalty,
                turnover_penalty=turnover_penalty,
            )

            if self.action_type == 'continuous':
                self.action_space = spaces.Box(low=0.0, high=1.0,
                                               shape=(self.n_stocks,), dtype=np.float32)
            else:
                self.action_space = spaces.Discrete(self.n_stocks)

            # Q4修复: 观测维度根据 extended_features 调整
            feature_dim = self.n_features * 4 if self.use_extended_features else self.n_features
            obs_dim = feature_dim + self.n_stocks + 4
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(obs_dim,), dtype=np.float32)

            self.current_step = 0
            self.current_date_idx = 0
            self.current_weights = np.zeros(self.n_stocks, dtype=np.float32)
            self.portfolio_value = 1.0

            logger.info(f"初始化 PortfolioEnv: {self.n_dates} 天, {self.n_stocks} 只股票")

        def _prepare_data(self, factor_data: pd.DataFrame):
            if 'date' not in factor_data.columns:
                raise ValueError("因子数据必须包含 'date' 列")

            self.dates = sorted(factor_data['date'].unique())
            self.n_dates = len(self.dates)

            exclude_cols = {'code', 'name', 'date', 'industry', 'label', 'forward_return', 'total_score'}
            self.feature_cols = [c for c in factor_data.columns
                                if c not in exclude_cols
                                and factor_data[c].dtype in [np.float64, np.float32, np.int64]]
            self.n_features = len(self.feature_cols)

            # P5修复: 验证奖励列是连续值，优先使用 forward_return
            self.return_col = None
            for col in ['forward_return', 'ret_forward']:  # 移除 'label'
                if col in factor_data.columns:
                    self.return_col = col
                    break

            if self.return_col is None:
                raise ValueError("未找到有效的收益列 (forward_return/ret_forward)")

            self.data_by_date = {}
            for date in self.dates:
                self.data_by_date[date] = factor_data[factor_data['date'] == date].reset_index(drop=True)

            # R5修复: 根据 stock_pool_mode 选择股票池策略
            if self.stock_pool_mode == 'static':
                # 静态模式: 使用全数据前N天 (旧行为，可用于对比实验)
                lookback_dates = sorted(self.dates)[:min(self.stock_pool_lookback, len(self.dates))]
                self._build_stock_pool_from_dates(lookback_dates)
                logger.info(f"股票池: {self.n_stocks} 只 (静态模式，基于前 {len(lookback_dates)} 天)")
            else:
                # 动态模式: 延迟到 reset() 时根据 episode 起始位置动态选择
                # 先用全部股票初始化，reset() 时会重新计算
                all_codes = set()
                for date in self.dates:
                    for code in self.data_by_date[date]['code']:
                        all_codes.add(code)
                self.stock_codes = list(all_codes)[:self.max_stocks]
                self.n_stocks = len(self.stock_codes)
                self.code_to_idx = {code: idx for idx, code in enumerate(self.stock_codes)}
                logger.info(f"股票池: {self.n_stocks} 只 (动态模式，将在每个episode根据起始位置调整)")

        def _build_stock_pool_from_dates(self, lookback_dates: list):
            """
            R5修复: 基于指定日期范围构建股票池

            Args:
                lookback_dates: 用于统计股票频率的日期列表
            """
            # 统计指定日期内每只股票出现的天数
            code_freq = Counter()
            for date in lookback_dates:
                if date in self.data_by_date:
                    for code in self.data_by_date[date]['code']:
                        code_freq[code] += 1

            all_codes_count = len(code_freq)

            # 按频率降序排列，保留最活跃的 max_stocks 只
            self.stock_codes = [c for c, _ in code_freq.most_common(self.max_stocks)]
            self.n_stocks = len(self.stock_codes)
            self.code_to_idx = {code: idx for idx, code in enumerate(self.stock_codes)}

            # Q3修复: 股票池被截断时警告
            if all_codes_count > self.max_stocks:
                logger.warning(
                    f"股票池 ({all_codes_count}) 超过 max_stocks ({self.max_stocks})，"
                    f"已按出现频率保留最活跃的 {self.n_stocks} 只股票。"
                    f"部分低频股票无法分配权重，建议增大 max_stocks 或使用稳定 universe"
                )

        def reset(self, seed: int = None, options: dict = None):
            super().reset(seed=seed)
            self.current_step = 0
            self.current_date_idx = np.random.randint(0, max(1, self.n_dates - self.episode_length))

            # R5修复: 动态模式下，根据 episode 起始位置更新股票池
            if self.stock_pool_mode == 'dynamic':
                self._update_stock_pool_dynamic(self.current_date_idx)

            self.current_weights = np.zeros(self.n_stocks, dtype=np.float32)
            self.portfolio_value = 1.0
            self.reward_calculator.reset()
            obs = self._get_observation()
            return obs, {'date': self.dates[self.current_date_idx], 'portfolio_value': self.portfolio_value}

        def _update_stock_pool_dynamic(self, current_date_idx: int):
            """
            R5修复: 动态更新股票池 (避免前视偏差)

            基于 current_date_idx 之前的 N 天数据重新计算股票池，
            确保不使用"未来"数据来选择股票。

            Args:
                current_date_idx: 当前 episode 的起始日期索引
            """
            # 计算 lookback 窗口: 从 current_date_idx 往前取 stock_pool_lookback 天
            # 但不能超过数据起始位置
            start_idx = max(0, current_date_idx - self.stock_pool_lookback)
            end_idx = current_date_idx  # 不包含当前日期，严格避免前视

            if start_idx >= end_idx:
                # 如果没有足够的历史数据，使用从0到当前位置的所有数据
                start_idx = 0
                end_idx = max(1, current_date_idx)

            lookback_dates = self.dates[start_idx:end_idx]

            if len(lookback_dates) == 0:
                # 极端情况: 使用第一天的数据
                lookback_dates = [self.dates[0]]

            old_n_stocks = self.n_stocks
            self._build_stock_pool_from_dates(lookback_dates)

            # 如果股票池大小变化，需要更新 action_space 和 observation_space
            if self.n_stocks != old_n_stocks:
                if self.action_type == 'continuous':
                    self.action_space = spaces.Box(low=0.0, high=1.0,
                                                   shape=(self.n_stocks,), dtype=np.float32)
                else:
                    self.action_space = spaces.Discrete(self.n_stocks)

                feature_dim = self.n_features * 4 if self.use_extended_features else self.n_features
                obs_dim = feature_dim + self.n_stocks + 4
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                                    shape=(obs_dim,), dtype=np.float32)

                logger.debug(f"动态股票池更新: {old_n_stocks} -> {self.n_stocks} 只 "
                            f"(基于 {self.dates[start_idx]} 至 {self.dates[end_idx-1]} 的 {len(lookback_dates)} 天)")

        def step(self, action):
            if self.action_type == 'continuous':
                new_weights = self._normalize_weights(action)
            else:
                new_weights = self._discrete_to_weights(action)

            current_date = self.dates[self.current_date_idx]
            current_df = self.data_by_date.get(current_date, pd.DataFrame())

            # C6 修复: 应用 TradeabilityPolicy 过滤不可交易股票
            tradeable_codes = set()
            if self.trading_policies is not None and len(current_df) > 0:
                for _, row in current_df.iterrows():
                    is_tradeable, _ = self.trading_policies.tradeability.is_tradeable(
                        row, action='buy'
                    )
                    if is_tradeable:
                        tradeable_codes.add(row.get('code', ''))

                # 过滤权重: 只保留可交易股票的权重
                if len(tradeable_codes) > 0:
                    filtered_weights = np.zeros_like(new_weights)
                    for i, code in enumerate(self.stock_codes):
                        if code in tradeable_codes:
                            filtered_weights[i] = new_weights[i]
                    # 重新归一化
                    if filtered_weights.sum() > 1e-8:
                        filtered_weights = filtered_weights / filtered_weights.sum()
                        new_weights = filtered_weights.astype(np.float32)
                    else:
                        logger.debug(f"{current_date}: 无可交易股票，保持原权重")

            portfolio_return = self._calculate_portfolio_return(new_weights, current_df)
            turnover = np.sum(np.abs(new_weights - self.current_weights))

            reward, reward_info = self.reward_calculator.calculate(
                portfolio_return, turnover, self.current_weights, new_weights)

            self.portfolio_value *= (1 + portfolio_return)
            self.current_weights = new_weights
            self.current_step += 1
            self.current_date_idx += 1

            terminated = self.current_date_idx >= self.n_dates - 1
            truncated = self.current_step >= self.episode_length

            obs = self._get_observation() if not (terminated or truncated) else np.zeros(self.observation_space.shape, dtype=np.float32)

            info = {'date': current_date, 'portfolio_return': portfolio_return,
                    'portfolio_value': self.portfolio_value, 'turnover': turnover, **reward_info}

            return obs, reward, terminated, truncated, info

        def _get_observation(self):
            current_date = self.dates[self.current_date_idx]
            current_df = self.data_by_date.get(current_date, pd.DataFrame())

            if len(current_df) > 0 and len(self.feature_cols) > 0:
                if self.use_extended_features:
                    # Q4修复: 与 RLAdapter 保持一致的扩展特征 (mean + std + q25 + q75)
                    n_cols = len(self.feature_cols)
                    factor_features = np.zeros(n_cols * 4, dtype=np.float32)
                    for i, col in enumerate(self.feature_cols):
                        col_data = current_df[col].dropna()
                        if len(col_data) > 0:
                            factor_features[i] = col_data.mean()
                            factor_features[n_cols + i] = col_data.std()
                            factor_features[2 * n_cols + i] = col_data.quantile(0.25)
                            factor_features[3 * n_cols + i] = col_data.quantile(0.75)
                else:
                    # 原始: 仅横截面均值
                    factor_features = current_df[self.feature_cols].mean().values.astype(np.float32)
                if np.isnan(factor_features).any() or np.isinf(factor_features).any():
                    raise RuntimeError("factor features contain NaN/inf")
            else:
                raise RuntimeError("empty factor data for state")

            market_state = self._get_market_state(current_df)
            return np.concatenate([factor_features, self.current_weights, market_state]).astype(np.float32)

        def _get_market_state(self, df):
            state = np.zeros(4, dtype=np.float32)
            if len(df) == 0 or self.return_col is None:
                raise RuntimeError("missing data for market state")
            returns = df[self.return_col].dropna()
            if len(returns) > 0:
                state[0] = np.clip(returns.mean() * 10, -1, 1)
                state[1] = np.clip(returns.std() * 10, 0, 1)
                state[2] = (returns > 0).mean() * 2 - 1
                state[3] = np.clip(returns.quantile(0.9) - returns.quantile(0.1), 0, 1)
            return state

        def _normalize_weights(self, weights):
            weights = np.clip(weights, 0, 1)

            # Q2修复: 添加单票上限约束 (与回测一致)
            weights = np.clip(weights, 0, self.max_position)

            # max_stocks 约束
            if np.sum(weights > 0) > self.max_stocks:
                threshold = np.sort(weights)[-self.max_stocks]
                weights = np.where(weights >= threshold, weights, 0)

            # 归一化
            total = np.sum(weights)
            if total > 1e-8:
                weights = weights / total
            else:
                weights = np.ones(self.n_stocks) / self.n_stocks
            return weights.astype(np.float32)

        def _discrete_to_weights(self, action):
            weights = np.zeros(self.n_stocks, dtype=np.float32)
            n_select = min(self.max_stocks, self.n_stocks)
            start_idx = action % max(1, self.n_stocks - n_select + 1)
            weights[start_idx:start_idx + n_select] = 1.0 / n_select
            return weights

        def _calculate_portfolio_return(self, weights, df):
            """
            计算组合收益 (使用统一执行口径)

            当 trading_policies 可用时，使用 ExecutionPolicy 中配置的
            入场/出场价格 (label_entry/label_exit) 计算收益，确保
            RL 训练与回测使用相同的收益计算口径。
            """
            if len(df) == 0 or self.return_col is None:
                raise RuntimeError("missing return data for portfolio return")

            portfolio_return = 0.0

            for code, weight in zip(self.stock_codes, weights):
                if weight < 1e-8:
                    continue

                stock_data = df[df['code'] == code]
                if len(stock_data) == 0:
                    raise RuntimeError(f"missing factor row for {code}")

                row = stock_data.iloc[0]

                # 使用 ExecutionPolicy 的入场/出场价格计算收益
                if self.trading_policies is None or not hasattr(self.trading_policies, 'execution'):
                    raise RuntimeError("trading_policies.execution is required")
                exec_policy = self.trading_policies.execution

                entry_price = exec_policy.get_label_entry_price(row)
                exit_price = exec_policy.get_label_exit_price(row)
                if entry_price <= 0:
                    raise RuntimeError(f"invalid entry price for {code}")
                ret = (exit_price - entry_price) / entry_price

                if not np.isnan(ret):
                    portfolio_return += weight * ret

            return portfolio_return

        def render(self, mode='human'):
            print(f"Step: {self.current_step}, Value: {self.portfolio_value:.4f}")

        def close(self):
            pass


    class _SimplePortfolioEnv(gym.Env):
        """简化版投资组合环境"""

        metadata = {'render_modes': ['human']}

        def __init__(self, returns_matrix: np.ndarray, features_matrix: np.ndarray = None,
                     n_stocks: int = 50, episode_length: int = 252, transaction_cost: float = 0.002):
            super().__init__()
            self.returns = returns_matrix
            self.features = features_matrix
            self.n_days, self.n_assets = returns_matrix.shape
            self.n_stocks = min(n_stocks, self.n_assets)
            self.episode_length = episode_length
            self.transaction_cost = transaction_cost
            self.n_features = features_matrix.shape[2] if features_matrix is not None else 0

            self.action_space = spaces.MultiDiscrete([self.n_assets] * self.n_stocks)
            obs_dim = self.n_features * self.n_assets + self.n_assets + 4
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

            self.current_step = 0
            self.start_day = 0
            self.current_weights = np.zeros(self.n_assets, dtype=np.float32)
            self.portfolio_value = 1.0
            self.return_history = []

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_step = 0
            self.start_day = np.random.randint(0, max(1, self.n_days - self.episode_length))
            self.current_weights = np.zeros(self.n_assets, dtype=np.float32)
            self.portfolio_value = 1.0
            self.return_history = []
            return self._get_observation(), {'day': self.start_day}

        def step(self, action):
            new_weights = np.zeros(self.n_assets, dtype=np.float32)
            unique_actions = np.unique(action)
            for idx in unique_actions:
                if 0 <= idx < self.n_assets:
                    new_weights[idx] = 1.0 / len(unique_actions)

            current_day = self.start_day + self.current_step
            day_returns = self.returns[current_day]
            portfolio_return = np.sum(new_weights * day_returns)
            turnover = np.sum(np.abs(new_weights - self.current_weights))
            net_return = portfolio_return - self.transaction_cost * turnover

            self.return_history.append(net_return)
            if len(self.return_history) >= 5:
                mean_ret = np.mean(self.return_history[-20:])
                std_ret = np.std(self.return_history[-20:]) + 1e-8
                reward = mean_ret / std_ret * np.sqrt(252)
            else:
                reward = net_return * 100

            self.portfolio_value *= (1 + net_return)
            self.current_weights = new_weights
            self.current_step += 1

            terminated = (self.start_day + self.current_step) >= self.n_days - 1
            truncated = self.current_step >= self.episode_length
            obs = self._get_observation() if not (terminated or truncated) else np.zeros(self.observation_space.shape)

            return obs, reward, terminated, truncated, {
                'portfolio_return': portfolio_return, 'net_return': net_return,
                'turnover': turnover, 'portfolio_value': self.portfolio_value}

        def _get_observation(self):
            current_day = self.start_day + self.current_step
            obs_parts = []
            if self.features is not None and current_day < len(self.features):
                obs_parts.append(self.features[current_day].flatten())
            elif self.n_features > 0:
                obs_parts.append(np.zeros(self.n_features * self.n_assets))
            obs_parts.append(self.current_weights)
            if current_day < len(self.returns):
                day_returns = self.returns[current_day]
                obs_parts.append(np.array([np.mean(day_returns), np.std(day_returns),
                                           (day_returns > 0).mean(), np.max(day_returns) - np.min(day_returns)]))
            else:
                obs_parts.append(np.zeros(4))
            return np.nan_to_num(np.concatenate(obs_parts).astype(np.float32), nan=0.0)

        def render(self, mode='human'):
            print(f"Step: {self.current_step}, Value: {self.portfolio_value:.4f}")

    # 赋值给全局变量
    PortfolioEnv = _PortfolioEnv
    SimplePortfolioEnv = _SimplePortfolioEnv


# ===================== 工厂函数 =====================

def create_portfolio_env(factor_data: pd.DataFrame, config: dict = None, **kwargs):
    """创建投资组合环境"""
    if not GYM_AVAILABLE:
        raise ImportError("需要安装 gymnasium: pip install gymnasium")
    return PortfolioEnv(factor_data, config, **kwargs)


def create_simple_env(returns_matrix: np.ndarray, features_matrix: np.ndarray = None, **kwargs):
    """创建简化版环境"""
    if not GYM_AVAILABLE:
        raise ImportError("需要安装 gymnasium: pip install gymnasium")
    return SimplePortfolioEnv(returns_matrix, features_matrix, **kwargs)


def check_gym_available() -> bool:
    """检查 Gymnasium 是否可用"""
    return GYM_AVAILABLE
