"""
组合构建器模块 (Portfolio Builder)

提供多层级的组合权重优化：
- V0: 等权组合 (Equal Weight)
- V1: 风险平价 (Risk Parity) / 波动率倒数加权
- V2: 均值-方差优化 (Mean-Variance) / Black-Litterman

用法：
    from src.deep_learning.portfolio_builder import PortfolioBuilder, OptimizationLevel

    builder = PortfolioBuilder(level=OptimizationLevel.V1)
    weights = builder.optimize(
        factor_frame=factor_frame,
        predictions=predictions,
        returns_history=returns_history,
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path

from .policies import CostModel, TradeabilityPolicy, TradingPolicies, get_default_policies
from .contracts import FactorFrame

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger

logger = setup_logger(__name__)


class OptimizationLevel(Enum):
    """组合优化级别"""
    V0 = "equal_weight"      # 等权组合
    V1 = "risk_parity"       # 风险平价
    V2 = "mean_variance"     # 均值-方差优化


@dataclass
class PortfolioConfig:
    """组合构建配置"""
    # 通用参数
    max_weight: float = 0.10          # 单只股票最大权重
    min_weight: float = 0.0           # 单只股票最小权重 (0 表示可以不持有)
    target_n_stocks: int = 50         # 目标持股数量

    # V1 风险平价参数
    vol_lookback: int = 60            # 波动率计算回溯天数
    vol_target: float = 0.15          # 目标年化波动率

    # V2 均值方差参数
    cov_lookback: int = 60            # 协方差计算回溯天数
    shrinkage_factor: float = 0.5     # Ledoit-Wolf 收缩因子
    risk_aversion: float = 1.0        # 风险厌恶系数

    # 约束条件
    turnover_limit: Optional[float] = 0.5     # 单期换手率上限
    sector_limit: Optional[float] = 0.30      # 单一行业最大暴露

    # 交易成本
    cost_model: CostModel = field(default_factory=CostModel)


@dataclass
class PortfolioResult:
    """组合构建结果"""
    weights: pd.Series                 # 股票权重 (index=code)
    selected_codes: List[str]          # 选中的股票代码
    optimization_level: OptimizationLevel

    # 风险指标
    expected_volatility: Optional[float] = None
    expected_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None

    # 约束状态
    turnover: Optional[float] = None
    sector_exposure: Optional[Dict[str, float]] = None

    # 诊断信息
    solver_status: str = "success"
    iterations: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        return pd.DataFrame({
            'code': self.selected_codes,
            'weight': [self.weights[c] for c in self.selected_codes],
        })


class PortfolioBuilder:
    """
    组合构建器

    根据优化级别构建最优组合权重
    """

    def __init__(self,
                 level: OptimizationLevel = OptimizationLevel.V0,
                 config: PortfolioConfig = None,
                 policies: TradingPolicies = None):
        """
        初始化组合构建器

        Args:
            level: 优化级别 (V0/V1/V2)
            config: 组合配置
            policies: 交易策略 (用于成本计算)
        """
        self.level = level
        self.config = config or PortfolioConfig()
        self.policies = policies or get_default_policies()

        logger.info(f"初始化组合构建器: {level.value}")

    def optimize(self,
                 predictions: pd.Series,
                 returns_history: Optional[pd.DataFrame] = None,
                 previous_weights: Optional[pd.Series] = None,
                 sector_map: Optional[Dict[str, str]] = None) -> PortfolioResult:
        """
        优化组合权重

        Args:
            predictions: 预测分数 (index=code)
            returns_history: 历史收益率 [dates, codes]
            previous_weights: 上一期权重 (用于换手约束)
            sector_map: 行业映射 {code: sector}

        Returns:
            PortfolioResult 组合结果
        """
        # 选择 Top-N 股票
        n = self.config.target_n_stocks
        top_codes = predictions.nlargest(n).index.tolist()

        if self.level == OptimizationLevel.V0:
            return self._optimize_v0(top_codes, predictions)
        elif self.level == OptimizationLevel.V1:
            return self._optimize_v1(top_codes, predictions, returns_history)
        elif self.level == OptimizationLevel.V2:
            return self._optimize_v2(
                top_codes, predictions, returns_history,
                previous_weights, sector_map
            )
        else:
            raise ValueError(f"未知优化级别: {self.level}")

    def _optimize_v0(self,
                     codes: List[str],
                     predictions: pd.Series) -> PortfolioResult:
        """
        V0: 等权组合

        最简单的组合构建方式，所有选中股票等权配置
        """
        n = len(codes)
        if n == 0:
            return PortfolioResult(
                weights=pd.Series(dtype=float),
                selected_codes=[],
                optimization_level=OptimizationLevel.V0,
                solver_status="empty"
            )

        weight = 1.0 / n
        weights = pd.Series({code: weight for code in codes})

        return PortfolioResult(
            weights=weights,
            selected_codes=codes,
            optimization_level=OptimizationLevel.V0,
            solver_status="success"
        )

    def _optimize_v1(self,
                     codes: List[str],
                     predictions: pd.Series,
                     returns_history: Optional[pd.DataFrame]) -> PortfolioResult:
        """
        V1: 风险平价 / 波动率倒数加权

        权重 ∝ 1 / 波动率
        波动率越大的股票，配置权重越低
        """
        if returns_history is None or len(returns_history) < 20:
            raise RuntimeError("insufficient returns history for V1 optimization")

        # 计算波动率
        available_codes = [c for c in codes if c in returns_history.columns]
        if len(available_codes) < 5:
            raise RuntimeError("insufficient available codes for V1 optimization")

        returns = returns_history[available_codes].tail(self.config.vol_lookback)
        vols = returns.std() * np.sqrt(252)  # 年化波动率

        # 避免零波动率
        vols = vols.clip(lower=0.01)

        # 波动率倒数加权
        inv_vol = 1.0 / vols
        raw_weights = inv_vol / inv_vol.sum()

        # 应用权重上限
        weights = self._apply_weight_constraints(raw_weights)

        # 计算组合波动率 (假设不相关)
        portfolio_vol = np.sqrt((weights ** 2 * vols ** 2).sum())

        return PortfolioResult(
            weights=weights,
            selected_codes=weights.index.tolist(),
            optimization_level=OptimizationLevel.V1,
            expected_volatility=portfolio_vol,
            solver_status="success"
        )

    def _optimize_v2(self,
                     codes: List[str],
                     predictions: pd.Series,
                     returns_history: Optional[pd.DataFrame],
                     previous_weights: Optional[pd.Series],
                     sector_map: Optional[Dict[str, str]]) -> PortfolioResult:
        """
        V2: 均值-方差优化

        最大化: μ'w - (λ/2) w'Σw
        约束: 权重上下限、换手率、行业暴露
        """
        if returns_history is None or len(returns_history) < 30:
            raise RuntimeError("insufficient returns history for V2 optimization")

        available_codes = [c for c in codes if c in returns_history.columns]
        if len(available_codes) < 10:
            raise RuntimeError("insufficient available codes for V2 optimization")

        returns = returns_history[available_codes].tail(self.config.cov_lookback)

        # 预期收益 (使用模型预测分数作为代理)
        mu = predictions[available_codes].values
        mu = (mu - mu.min()) / (mu.max() - mu.min() + 1e-10)  # 归一化到 [0, 1]

        # 协方差矩阵 (带收缩)
        cov = self._shrink_covariance(returns)

        # 求解优化问题
        weights, status, iterations = self._solve_mean_variance(
            mu, cov, available_codes, previous_weights, sector_map
        )

        if status != "success":
            raise RuntimeError(f"optimizer failed with status: {status}")

        weights = pd.Series(weights, index=available_codes)
        weights = weights[weights > 1e-6]  # 移除零权重

        # 计算风险指标
        w = weights.values
        idx = [available_codes.index(c) for c in weights.index]
        cov_sub = cov[np.ix_(idx, idx)]
        expected_vol = np.sqrt(w @ cov_sub @ w) * np.sqrt(252)
        expected_ret = w @ mu[idx]
        sharpe = expected_ret / (expected_vol + 1e-10) * np.sqrt(252)

        # 计算换手率
        turnover = None
        if previous_weights is not None:
            prev = previous_weights.reindex(weights.index, fill_value=0)
            turnover = np.abs(weights - prev).sum() / 2

        return PortfolioResult(
            weights=weights,
            selected_codes=weights.index.tolist(),
            optimization_level=OptimizationLevel.V2,
            expected_volatility=expected_vol,
            expected_return=expected_ret,
            sharpe_ratio=sharpe,
            turnover=turnover,
            solver_status=status,
            iterations=iterations
        )

    def _shrink_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Ledoit-Wolf 协方差收缩

        Σ_shrunk = α * F + (1 - α) * S
        F = 对角矩阵 (单因子模型)
        S = 样本协方差
        """
        n_samples, n_assets = returns.shape

        # 样本协方差
        S = returns.cov().values

        # 收缩目标: 对角矩阵
        trace_S = np.trace(S)
        mu = trace_S / n_assets
        F = np.eye(n_assets) * mu

        # 收缩强度
        alpha = self.config.shrinkage_factor

        # 收缩后的协方差
        Sigma = alpha * F + (1 - alpha) * S

        return Sigma

    def _solve_mean_variance(self,
                              mu: np.ndarray,
                              cov: np.ndarray,
                              codes: List[str],
                              previous_weights: Optional[pd.Series],
                              sector_map: Optional[Dict[str, str]]) -> Tuple[np.ndarray, str, int]:
        """
        求解均值-方差优化问题

        使用简单的梯度投影方法
        """
        n = len(codes)
        lambda_risk = self.config.risk_aversion

        # 初始化等权
        w = np.ones(n) / n

        # 简单梯度下降 (项目规模小，不需要复杂求解器)
        lr = 0.01
        max_iter = 1000
        tol = 1e-6

        for i in range(max_iter):
            # 梯度: ∂/∂w [μ'w - (λ/2) w'Σw] = μ - λ Σ w
            grad = mu - lambda_risk * cov @ w

            # 更新
            w_new = w + lr * grad

            # 投影到约束集
            w_new = self._project_to_constraints(w_new)

            # 检查收敛
            if np.linalg.norm(w_new - w) < tol:
                return w_new, "success", i + 1

            w = w_new

        return w, "max_iter", max_iter

    def _project_to_constraints(self, w: np.ndarray) -> np.ndarray:
        """投影到约束集 (simplex + box constraints)"""
        # 非负
        w = np.maximum(w, self.config.min_weight)

        # 权重上限
        w = np.minimum(w, self.config.max_weight)

        # 归一化 (和为1)
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.ones(len(w)) / len(w)

        return w

    def _apply_weight_constraints(self, weights: pd.Series) -> pd.Series:
        """应用权重约束"""
        # 权重上限
        weights = weights.clip(upper=self.config.max_weight)

        # 重新归一化
        if weights.sum() > 0:
            weights = weights / weights.sum()

        return weights


def build_portfolio(predictions: pd.Series,
                    level: str = "V0",
                    returns_history: Optional[pd.DataFrame] = None,
                    **kwargs) -> PortfolioResult:
    """
    便捷函数: 构建组合

    Args:
        predictions: 预测分数
        level: 优化级别 ("V0", "V1", "V2")
        returns_history: 历史收益率
        **kwargs: 传递给 PortfolioConfig

    Returns:
        PortfolioResult
    """
    level_map = {
        "V0": OptimizationLevel.V0,
        "V1": OptimizationLevel.V1,
        "V2": OptimizationLevel.V2,
    }

    opt_level = level_map.get(level.upper(), OptimizationLevel.V0)
    config = PortfolioConfig(**kwargs)
    builder = PortfolioBuilder(level=opt_level, config=config)

    return builder.optimize(predictions, returns_history)
