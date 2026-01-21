"""
多因子打分模块
负责计算因子得分、综合排名和选股
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from .utils import load_config, setup_logger, standardize, zscore
from .factor_engine import FactorEngine
from .adaptive_weights import AdaptiveWeightManager

logger = setup_logger(__name__)


class FactorScorer:
    """多因子打分器"""

    def __init__(self, config: dict = None):
        """
        初始化打分器

        Args:
            config: 配置字典
        """
        self.config = config or load_config()
        self.weights = self.config['weights']
        self.factor_engine = FactorEngine(self.config)

        # 自适应权重管理器
        self.adaptive_weight_manager = AdaptiveWeightManager(self.config)

        # P2-4修复: 各类别包含的因子（与FactorEngine保持一致）
        # 将trend_quality和institution因子合并到现有类别
        self.category_factors = {
            'quality': ['roe', 'roa', 'gross_margin', 'net_margin', 'ocf_ratio'],
            'valuation': ['pe_ttm', 'pb', 'ps_ttm', 'div_yield',
                          'pe_relative', 'pb_relative'],  # 添加相对估值
            'momentum': ['ret_20d', 'ret_60d', 'ret_120d', 'new_high', 'consecutive_up',
                         # P2-4: 从trend_quality合并的因子
                         'consecutive_up_days', 'dist_to_high_20d', 'dist_to_high_60d',
                         'ma20_bias', 'ma20_slope', 'trend_strength', 'price_position'],
            'flow': ['net_inflow_5d', 'net_inflow_20d', 'north_change',
                     # P2-4: 从institution合并的因子
                     'north_hold_change_20d', 'north_hold_change_5d'],
            # 赚钱效应/情绪因子
            'sentiment': ['relative_strength_20d', 'relative_strength_60d',
                          'reversal_5d', 'reversal_10d',
                          # P2-4: 从trend_quality合并的因子
                          'volume_ratio', 'up_down_volume_ratio', 'atr_contraction']
        }

        # 因子方向 (1: 值越大得分越高, -1: 值越小得分越高)
        self.factor_direction = self.factor_engine.factor_direction

    def compute_factor_score(self, factor_df: pd.DataFrame,
                             factor_name: str) -> pd.Series:
        """
        计算单个因子的得分（排序转换为百分位）

        Args:
            factor_df: 因子数据
            factor_name: 因子名称

        Returns:
            因子得分 (0-1之间)
        """
        if factor_name not in factor_df.columns:
            return pd.Series(np.nan, index=factor_df.index)

        values = factor_df[factor_name].copy()

        # 处理缺失值
        valid_mask = values.notna()
        if valid_mask.sum() < 2:
            return pd.Series(np.nan, index=factor_df.index)

        # 获取因子方向
        direction = self.factor_direction.get(factor_name, 1)

        # 排序转百分位得分
        scores = pd.Series(np.nan, index=factor_df.index)
        ranks = values[valid_mask].rank(ascending=(direction == 1))
        scores[valid_mask] = (ranks - 1) / (len(ranks) - 1)  # 归一化到 [0,1]

        return scores

    def compute_category_score(self, factor_df: pd.DataFrame,
                               category: str) -> pd.Series:
        """
        计算某类因子的综合得分

        Args:
            factor_df: 因子数据
            category: 因子类别

        Returns:
            类别综合得分
        """
        if category not in self.category_factors:
            raise ValueError(f"未知的因子类别: {category}")

        factors = self.category_factors[category]

        # 计算各因子得分
        factor_scores = []
        for factor in factors:
            score = self.compute_factor_score(factor_df, factor)
            if score.notna().any():
                factor_scores.append(score)

        if len(factor_scores) == 0:
            return pd.Series(np.nan, index=factor_df.index)

        # 类内等权平均
        scores_df = pd.concat(factor_scores, axis=1)
        category_score = scores_df.mean(axis=1)

        return category_score

    def compute_total_score(self, factor_df: pd.DataFrame,
                            weights: Dict[str, float] = None,
                            use_adaptive: bool = False) -> pd.Series:
        """
        计算总得分

        总分 = Σ(类别权重 × 类别得分)

        Args:
            factor_df: 因子数据
            weights: 类别权重，默认使用配置中的权重
            use_adaptive: 是否使用自适应权重（从factor_df提取市场环境）

        Returns:
            总得分
        """
        # 确定使用的权重
        if weights is not None:
            # 外部传入权重优先
            final_weights = weights
        elif use_adaptive:
            # 使用自适应权重
            final_weights, status = self.adaptive_weight_manager.get_weights_from_factor_df(factor_df)
            logger.debug(f"自适应权重 - {status}: {final_weights}")
        else:
            # 使用默认权重
            final_weights = self.weights

        category_scores = {}
        for category in final_weights.keys():
            score = self.compute_category_score(factor_df, category)
            category_scores[category] = score

        # 按股票动态归一化：只用每只股票实际有效的权重作为分母
        total_score = pd.Series(0.0, index=factor_df.index)
        valid_weight = pd.Series(0.0, index=factor_df.index)  # 按股票跟踪有效权重

        for category, weight in final_weights.items():
            if category in category_scores:
                score = category_scores[category]
                valid_mask = score.notna()
                total_score[valid_mask] += weight * score[valid_mask]
                valid_weight[valid_mask] += weight  # 按股票累加有效权重

        # 按股票归一化（避免除零）
        valid_weight = valid_weight.replace(0, np.nan)
        total_score = total_score / valid_weight

        return total_score

    def select_top_stocks(self, factor_df: pd.DataFrame,
                          n: int = 50,
                          weights: Dict[str, float] = None,
                          filters: Dict[str, any] = None,
                          use_adaptive: bool = False) -> pd.DataFrame:
        """
        选择得分最高的N只股票

        Args:
            factor_df: 因子数据
            n: 选股数量
            weights: 因子权重
            filters: 过滤条件
            use_adaptive: 是否使用自适应权重

        Returns:
            选股结果DataFrame
        """
        df = factor_df.copy()

        # 应用过滤条件
        if filters:
            for col, condition in filters.items():
                if col in df.columns:
                    if callable(condition):
                        df = df[condition(df[col])]
                    else:
                        df = df[df[col] == condition]

        if len(df) == 0:
            logger.warning("过滤后无股票")
            return pd.DataFrame()

        # 计算总得分（支持自适应权重）
        df['total_score'] = self.compute_total_score(df, weights, use_adaptive=use_adaptive)

        # 排除得分为空的
        df = df[df['total_score'].notna()]

        # 排序选股
        df = df.sort_values('total_score', ascending=False)
        selected = df.head(n).copy()

        # 添加排名
        selected['rank'] = range(1, len(selected) + 1)

        logger.info(f"选出 {len(selected)} 只股票")

        return selected

    def compute_equal_weight(self, selected_df: pd.DataFrame,
                              max_weight: float = None) -> pd.DataFrame:
        """
        计算等权权重

        Args:
            selected_df: 选股结果
            max_weight: 单票最大权重

        Returns:
            带权重的选股结果
        """
        n = len(selected_df)
        if n == 0:
            return selected_df

        weight = 1.0 / n

        if max_weight and weight > max_weight:
            weight = max_weight

        selected_df = selected_df.copy()
        selected_df['weight'] = weight

        return selected_df

    def compute_score_weight(self, selected_df: pd.DataFrame,
                              score_col: str = 'total_score',
                              max_weight: float = None) -> pd.DataFrame:
        """
        按得分计算权重（得分越高权重越大）

        Args:
            selected_df: 选股结果
            score_col: 得分列名
            max_weight: 单票最大权重

        Returns:
            带权重的选股结果
        """
        if len(selected_df) == 0:
            return selected_df

        selected_df = selected_df.copy()

        # 得分归一化为权重
        scores = selected_df[score_col]
        min_score = scores.min()
        max_score = scores.max()

        if max_score > min_score:
            normalized = (scores - min_score) / (max_score - min_score) + 0.5
        else:
            normalized = pd.Series(1.0, index=selected_df.index)

        weights = normalized / normalized.sum()

        # 限制最大权重
        if max_weight:
            weights = weights.clip(upper=max_weight)
            weights = weights / weights.sum()  # 重新归一化

        selected_df['weight'] = weights

        return selected_df

    def get_category_breakdown(self, factor_df: pd.DataFrame,
                               code: str) -> Dict[str, float]:
        """
        获取单只股票的分类得分明细

        Args:
            factor_df: 因子数据
            code: 股票代码

        Returns:
            各类别得分字典
        """
        if 'code' not in factor_df.columns:
            return {}

        stock_idx = factor_df[factor_df['code'] == code].index
        if len(stock_idx) == 0:
            return {}

        breakdown = {}
        for category in self.category_factors.keys():
            score = self.compute_category_score(factor_df, category)
            breakdown[category] = score.loc[stock_idx[0]]

        breakdown['total'] = self.compute_total_score(factor_df).loc[stock_idx[0]]

        return breakdown

    def analyze_factor_contribution(self, factor_df: pd.DataFrame,
                                    selected_df: pd.DataFrame) -> pd.DataFrame:
        """
        分析各因子对选股结果的贡献

        Args:
            factor_df: 全部因子数据
            selected_df: 选股结果

        Returns:
            因子贡献分析
        """
        contributions = []

        for category, factors in self.category_factors.items():
            for factor in factors:
                if factor not in factor_df.columns:
                    continue

                # 选中股票的因子平均值
                selected_codes = selected_df['code'].tolist()
                selected_mask = factor_df['code'].isin(selected_codes)

                selected_mean = factor_df.loc[selected_mask, factor].mean()
                all_mean = factor_df[factor].mean()
                all_std = factor_df[factor].std()

                # IC (因子与选中状态的相关性)
                if all_std > 0:
                    z_score = (selected_mean - all_mean) / all_std
                else:
                    z_score = 0

                contributions.append({
                    'category': category,
                    'factor': factor,
                    'selected_mean': selected_mean,
                    'all_mean': all_mean,
                    'z_score': z_score,
                    'weight': self.weights.get(category, 0) / len(factors)
                })

        return pd.DataFrame(contributions)

    def backtest_weights(self, factor_df: pd.DataFrame,
                         returns: pd.Series,
                         weight_options: List[Dict[str, float]]) -> pd.DataFrame:
        """
        回测不同权重配置的效果

        Args:
            factor_df: 因子数据
            returns: 各股票的未来收益
            weight_options: 权重配置列表

        Returns:
            各配置的表现
        """
        results = []

        for weights in weight_options:
            selected = self.select_top_stocks(factor_df, n=50, weights=weights)

            if len(selected) == 0:
                continue

            # 计算组合收益
            selected_codes = selected['code'].tolist()
            portfolio_return = returns[returns.index.isin(selected_codes)].mean()

            results.append({
                'weights': str(weights),
                'portfolio_return': portfolio_return,
                'n_stocks': len(selected),
                'avg_score': selected['total_score'].mean()
            })

        return pd.DataFrame(results)


def optimize_weights(factor_df: pd.DataFrame,
                     future_returns: pd.Series,
                     n_iter: int = 100) -> Dict[str, float]:
    """
    简单的权重优化（网格搜索）

    Args:
        factor_df: 因子数据
        future_returns: 未来收益
        n_iter: 迭代次数

    Returns:
        最优权重
    """
    best_weights = None
    best_return = -np.inf

    scorer = FactorScorer()

    for _ in range(n_iter):
        # 随机生成权重
        raw_weights = np.random.dirichlet([1, 1, 1, 1])
        weights = {
            'quality': raw_weights[0],
            'valuation': raw_weights[1],
            'momentum': raw_weights[2],
            'flow': raw_weights[3]
        }

        # 选股
        selected = scorer.select_top_stocks(factor_df, n=50, weights=weights)
        if len(selected) == 0:
            continue

        # 计算组合收益
        codes = selected['code'].tolist()
        code_mask = future_returns.index.isin(codes)
        if code_mask.sum() == 0:
            continue

        portfolio_return = future_returns[code_mask].mean()

        if portfolio_return > best_return:
            best_return = portfolio_return
            best_weights = weights

    return best_weights


class ConstrainedSelector:
    """
    P2-2: 带风控约束的选股器

    约束:
    1. 单票权重上限 - 防止集中风险
    2. 行业权重上限 - 行业分散
    3. 最大换手率 - 控制交易成本
    4. 最小流动性 - 确保可执行性
    5. 波动预算 - 控制组合风险

    选股逻辑: 贪心选取，按排名依次加入，跳过违反约束的
    """

    def __init__(self, config: dict = None):
        """
        初始化约束选股器

        Args:
            config: 配置字典
        """
        self.config = config or load_config()

        # 约束参数
        bt_config = self.config.get('backtest', {})
        self.max_single_weight = bt_config.get('max_position', 0.08)
        self.max_industry_weight = self.config.get('stock_pool', {}).get('max_industry_weight', 0.30)
        self.max_turnover = bt_config.get('max_turnover', 0.50)
        self.min_liquidity_percentile = self.config.get('stock_pool', {}).get('min_liquidity_pct', 0.30)
        self.max_portfolio_volatility = self.config.get('stock_pool', {}).get('max_portfolio_vol', 0.25)
        self.max_single_volatility = self.max_portfolio_volatility * 1.5

    def _check_liquidity(self, row: pd.Series, all_df: pd.DataFrame) -> Optional[bool]:
        """
        Strict liquidity check using unified columns only.
        """
        liq_config = self.config.get('liquidity', {})
        min_amount = liq_config.get('min_amount_threshold', 10_000_000)
        min_turnover = liq_config.get('min_turnover_pct', 0.005)

        amount_col = 'avg_amount_20d_raw' if 'avg_amount_20d_raw' in row.index else 'avg_amount_20d'
        turnover_col = 'avg_turnover_20d_raw' if 'avg_turnover_20d_raw' in row.index else 'avg_turnover_20d'

        amount_check = None
        if amount_col in row.index:
            val = row.get(amount_col)
            if pd.notna(val):
                amount_check = float(val) >= min_amount

        turnover_check = None
        if turnover_col in row.index:
            val = row.get(turnover_col)
            if pd.notna(val):
                turnover_val = float(val)
                # 自动转换单位：如果数据是百分比形式（>1），转换为小数
                if turnover_val > 1.0 and min_turnover <= 1.0:
                    turnover_val = turnover_val / 100.0
                turnover_check = turnover_val >= min_turnover

        if amount_check is None and turnover_check is None:
            # 流动性数据缺失时，默认通过检查（允许选入）
            # 这样可以在数据不完整时仍能运行回测
            return True

        if amount_check is not None and turnover_check is not None:
            return amount_check or turnover_check
        if amount_check is not None:
            return amount_check
        return turnover_check

    def _check_volatility(self, row: pd.Series) -> Optional[bool]:
        """
        Strict volatility check using raw annualized volatility.
        Returns False for stocks with missing volatility data to skip them.
        """
        col = 'vol_20d_raw' if 'vol_20d_raw' in row.index else 'vol_20d'
        if col not in row.index:
            code = row.get('code', 'unknown')
            logger.debug(f"missing volatility data for {code}, skipping")
            return False
        try:
            vol = float(row[col])
        except (ValueError, TypeError):
            code = row.get('code', 'unknown')
            logger.debug(f"invalid volatility value for {code}, skipping")
            return False
        if vol > 0.60:
            return False
        return True

    def _get_industry(self, row: pd.Series) -> str:
        """
        Get industry using unified column name.
        Returns "未知" for stocks with missing industry data to allow graceful degradation.
        """
        if 'industry' not in row.index or pd.isna(row.get('industry')):
            code = row.get('code', 'unknown')
            logger.debug(f"missing industry for {code}, using default '未知'")
            return "未知"
        return str(row.get('industry'))

    def calc_weight_turnover(self, new_holdings: Dict[str, float],
                              old_holdings: Dict[str, float]) -> float:
        """
        计算权重口径的换手率

        换手率 = 0.5 * Σ|w_new - w_old|

        Args:
            new_holdings: 新持仓权重 {code: weight}
            old_holdings: 旧持仓权重 {code: weight}

        Returns:
            换手率 (0-1)
        """
        all_codes = set(new_holdings.keys()) | set(old_holdings.keys())

        total_diff = 0
        for code in all_codes:
            new_w = new_holdings.get(code, 0)
            old_w = old_holdings.get(code, 0)
            total_diff += abs(new_w - old_w)

        return 0.5 * total_diff

    def select_with_constraints(self,
                                 ranked_df: pd.DataFrame,
                                 current_holdings: Dict[str, float] = None,
                                 n: int = 20) -> pd.DataFrame:
        """
        带约束的贪心选股

        按预测得分排名依次加入，跳过违反约束的股票

        Args:
            ranked_df: 已按得分降序排列的候选股票
            current_holdings: 当前持仓 {code: weight}，用于计算换手
            n: 目标选股数量

        Returns:
            满足约束的选股结果
        """
        current_holdings = current_holdings or {}

        selected = []
        industry_weights = {}
        base_weight = 1.0 / n

        # 确保有code列
        if 'code' not in ranked_df.columns:
            raise RuntimeError("ranked_df missing code column")

        for _, row in ranked_df.iterrows():
            if len(selected) >= n:
                break

            code = row['code']
            industry = self._get_industry(row)

            # === 约束1: 流动性 ===
            if not self._check_liquidity(row, ranked_df):
                logger.debug(f"{code}: 流动性不足，跳过")
                continue

            # === 约束2: 行业集中度 ===
            # P4-5: 行业数据不可用时优雅降级，跳过行业约束
            current_ind_weight = industry_weights.get(industry, 0)
            if current_ind_weight + base_weight > self.max_industry_weight:
                logger.debug(f"{code}: industry {industry} exceeds limit")
                continue

            # === 约束3: 单票波动率 ===
            if not self._check_volatility(row):
                logger.debug(f"{code}: 波动率过高，跳过")
                continue

            # 通过所有约束，加入选股池
            selected.append(row)
            industry_weights[industry] = industry_weights.get(industry, 0) + base_weight

        if len(selected) == 0:
            logger.warning("没有股票满足约束条件")
            return pd.DataFrame()

        selected_df = pd.DataFrame(selected)

        # === 约束4: 换手率控制 ===
        if len(current_holdings) > 0:
            # 计算预期换手
            new_holdings = {row['code']: base_weight for _, row in selected_df.iterrows()}
            expected_turnover = self.calc_weight_turnover(new_holdings, current_holdings)

            if expected_turnover > self.max_turnover:
                logger.info(f"换手率 {expected_turnover:.1%} 超过上限 {self.max_turnover:.1%}，执行换手控制")

                # 保留部分旧持仓
                old_codes = set(current_holdings.keys())
                new_codes = set(selected_df['code'].tolist())

                # 旧持仓中仍在新排名中的股票优先保留
                kept_from_old = selected_df[selected_df['code'].isin(old_codes)]
                new_only = selected_df[~selected_df['code'].isin(old_codes)]

                # 计算需要保留多少旧持仓
                target_turnover = self.max_turnover
                # 近似：保留 (1-target_turnover) 的旧持仓
                keep_count = max(1, int(n * (1 - target_turnover)))

                # 组合：优先保留旧持仓中排名靠前的
                final_kept = kept_from_old.head(keep_count)
                final_new = new_only.head(n - len(final_kept))

                selected_df = pd.concat([final_kept, final_new]).head(n)

        # 添加权重和排名
        selected_df = selected_df.copy()
        final_weight = 1.0 / len(selected_df)
        final_weight = min(final_weight, self.max_single_weight)
        selected_df['weight'] = final_weight
        selected_df['rank'] = range(1, len(selected_df) + 1)

        logger.info(f"约束选股完成: {len(selected_df)} 只股票, "
                   f"行业数: {len(industry_weights)}")

        return selected_df

    def compute_portfolio_risk(self, selected_df: pd.DataFrame) -> Dict[str, float]:
        """
        计算组合风险指标

        Args:
            selected_df: 选股结果

        Returns:
            风险指标字典
        """
        risk_metrics = {}

        # 1. 行业集中度 (HHI)
        if 'industry' in selected_df.columns or any(
            col in selected_df.columns for col in ['行业', 'sector']
        ):
            industry_col = next(
                (col for col in ['industry', '行业', 'sector'] if col in selected_df.columns),
                None
            )
            if industry_col:
                industry_weights = selected_df.groupby(industry_col).size() / len(selected_df)
                risk_metrics['industry_hhi'] = (industry_weights ** 2).sum()
                risk_metrics['max_industry_weight'] = industry_weights.max()
                risk_metrics['n_industries'] = len(industry_weights)

        # 2. 平均波动率
        vol_cols = ['vol_20d', 'volatility_20d', 'volatility']
        for col in vol_cols:
            if col in selected_df.columns:
                risk_metrics['avg_volatility'] = selected_df[col].mean()
                risk_metrics['max_volatility'] = selected_df[col].max()
                break

        # 3. 单票权重
        if 'weight' in selected_df.columns:
            risk_metrics['max_single_weight'] = selected_df['weight'].max()
            risk_metrics['weight_hhi'] = (selected_df['weight'] ** 2).sum()

        # 4. 股票数量
        risk_metrics['n_stocks'] = len(selected_df)

        return risk_metrics

    def get_constraint_summary(self) -> Dict[str, float]:
        """
        获取当前约束设置摘要

        Returns:
            约束参数字典
        """
        return {
            'max_single_weight': self.max_single_weight,
            'max_industry_weight': self.max_industry_weight,
            'max_turnover': self.max_turnover,
            'min_liquidity_percentile': self.min_liquidity_percentile,
            'max_portfolio_volatility': self.max_portfolio_volatility,
            'max_single_volatility': self.max_single_volatility
        }


class IndustryNeutralizer:
    """
    P2-4: 行业中性化处理器

    目的:
    - 消除行业因素对因子的影响
    - 使选股结果更加分散
    - 提高因子的纯净度
    """

    def __init__(self, config: dict = None):
        self.config = config or load_config()

    def neutralize_factor(self, factor_df: pd.DataFrame,
                          factor_col: str,
                          industry_col: str = 'industry') -> pd.Series:
        """
        对单个因子进行行业中性化

        方法: 行业内Z-score标准化

        Args:
            factor_df: 因子数据
            factor_col: 因子列名
            industry_col: 行业列名

        Returns:
            中性化后的因子值
        """
        if industry_col not in factor_df.columns:
            raise RuntimeError(f"missing industry column: {industry_col}")

        if factor_col not in factor_df.columns:
            return pd.Series(np.nan, index=factor_df.index)

        # 行业内标准化
        neutralized = factor_df.groupby(industry_col)[factor_col].transform(
            lambda x: zscore(x) if len(x) > 1 else x - x.mean()
        )

        return neutralized

    def neutralize_all_factors(self, factor_df: pd.DataFrame,
                                factor_cols: List[str] = None,
                                industry_col: str = 'industry') -> pd.DataFrame:
        """
        对多个因子进行行业中性化

        Args:
            factor_df: 因子数据
            factor_cols: 要中性化的因子列表，None表示所有数值列
            industry_col: 行业列名

        Returns:
            中性化后的因子DataFrame
        """
        df = factor_df.copy()

        if industry_col not in df.columns:
            raise RuntimeError(f"missing industry column: {industry_col}")

        # 确定要中性化的因子
        if factor_cols is None:
            exclude_cols = ['code', 'date', industry_col, 'name', '股票代码', '日期']
            factor_cols = [col for col in df.columns
                          if col not in exclude_cols
                          and df[col].dtype in [np.float64, np.int64, float, int]]

        # 逐因子中性化
        for col in factor_cols:
            if col in df.columns:
                df[f'{col}_neutral'] = self.neutralize_factor(df, col, industry_col)

        logger.info(f"行业中性化完成: {len(factor_cols)} 个因子")

        return df

    def select_industry_balanced(self, ranked_df: pd.DataFrame,
                                  n: int = 20,
                                  industry_col: str = 'industry',
                                  balance_method: str = 'proportional') -> pd.DataFrame:
        """
        行业均衡选股

        Args:
            ranked_df: 已按得分排序的候选股票
            n: 选股数量
            industry_col: 行业列名
            balance_method: 均衡方法
                - 'equal': 每个行业选相同数量
                - 'proportional': 按行业占比选择

        Returns:
            行业均衡的选股结果
        """
        if industry_col not in ranked_df.columns:
            raise RuntimeError(f"missing industry column: {industry_col}")

        industries = ranked_df[industry_col].unique()
        n_industries = len(industries)

        selected = []

        if balance_method == 'equal':
            # 每个行业选相同数量
            per_industry = max(1, n // n_industries)

            for industry in industries:
                industry_stocks = ranked_df[ranked_df[industry_col] == industry]
                selected.append(industry_stocks.head(per_industry))

        else:  # proportional
            # 按行业占比选择
            industry_counts = ranked_df[industry_col].value_counts()
            industry_ratios = industry_counts / len(ranked_df)

            for industry in industries:
                n_select = max(1, int(n * industry_ratios[industry]))
                industry_stocks = ranked_df[ranked_df[industry_col] == industry]
                selected.append(industry_stocks.head(n_select))

        selected_df = pd.concat(selected).head(n)
        selected_df = selected_df.copy()
        selected_df['rank'] = range(1, len(selected_df) + 1)

        return selected_df


class CostAwareOptimizer:
    """
    P3-1: 成本感知组合优化器

    核心思想:
    - 传统排序选股只看预期收益，忽略交易成本
    - 真实目标应该是: max(预期超额收益 - 交易成本)
    - 交易成本 = 换手率 × 单次交易成本

    优化目标:
    score_adjusted = alpha_score - lambda * turnover_cost

    其中:
    - alpha_score: 原始因子/模型得分
    - lambda: 成本惩罚系数
    - turnover_cost: 换手带来的成本

    使用场景:
    - 调仓时考虑是否值得换股
    - 避免频繁小幅调整
    - 提高实盘可执行性
    """

    def __init__(self, config: dict = None):
        """
        初始化成本感知优化器

        Args:
            config: 配置字典
        """
        self.config = config or load_config()

        # 成本参数
        bt_config = self.config.get('backtest', {})
        self.commission = bt_config.get('commission', 0.0003)  # 佣金 (万三)
        self.slippage = bt_config.get('slippage', 0.001)       # 滑点 (千一)
        self.stamp_duty = 0.001                                # 印花税 (千一, 仅卖出)

        # 单边交易成本 = 佣金 + 滑点
        self.single_side_cost = self.commission + self.slippage
        # 双边成本 = 买入成本 + 卖出成本(含印花税)
        self.round_trip_cost = self.single_side_cost + (self.single_side_cost + self.stamp_duty)

        # 成本惩罚系数 (可调参)
        # lambda = 1.0 表示 1%的换手成本需要1%的alpha提升来弥补
        self.cost_penalty_lambda = self.config.get('optimization', {}).get('cost_penalty_lambda', 1.0)

        # 最小换股阈值 (低于此得分差异不换股)
        self.min_swap_threshold = self.config.get('optimization', {}).get('min_swap_threshold', 0.02)

        logger.info(f"成本感知优化器初始化: 双边成本={self.round_trip_cost:.4%}, "
                   f"惩罚系数={self.cost_penalty_lambda}")

    def compute_turnover_cost(self, new_weights: Dict[str, float],
                               old_weights: Dict[str, float]) -> float:
        """
        计算换手成本

        Args:
            new_weights: 新权重 {code: weight}
            old_weights: 旧权重 {code: weight}

        Returns:
            换手成本 (占总资产的比例)
        """
        all_codes = set(new_weights.keys()) | set(old_weights.keys())

        total_turnover = 0
        for code in all_codes:
            new_w = new_weights.get(code, 0)
            old_w = old_weights.get(code, 0)
            total_turnover += abs(new_w - old_w)

        # 换手率 = 0.5 * Σ|w_new - w_old| (因为买卖抵消)
        turnover_rate = 0.5 * total_turnover

        # 换手成本 = 换手率 × 双边成本
        turnover_cost = turnover_rate * self.round_trip_cost

        return turnover_cost

    def compute_cost_adjusted_score(self, score: float,
                                     is_new_position: bool,
                                     current_weight: float = 0) -> float:
        """
        计算成本调整后的得分

        Args:
            score: 原始得分 (假设0-1标准化)
            is_new_position: 是否是新建仓
            current_weight: 当前权重

        Returns:
            成本调整后的得分
        """
        if is_new_position:
            # 新建仓需要扣除建仓成本
            cost_deduction = self.single_side_cost * self.cost_penalty_lambda
        else:
            # 已持有股票，无额外成本
            cost_deduction = 0

        adjusted_score = score - cost_deduction
        return adjusted_score

    def select_with_cost_awareness(self,
                                    ranked_df: pd.DataFrame,
                                    current_holdings: Dict[str, float] = None,
                                    n: int = 20,
                                    score_col: str = 'total_score') -> pd.DataFrame:
        """
        成本感知选股

        选股逻辑:
        1. 已持有��仍在候选池中的股票有"持仓优势"
        2. 新股票需要得分显著高于持仓股才值得换入
        3. 综合考虑alpha提升 vs 交易成本

        Args:
            ranked_df: 已按得分降序排列的候选股票
            current_holdings: 当前持仓权重 {code: weight}
            n: 目标选股数量
            score_col: 得分列名

        Returns:
            成本调整后的选股结果
        """
        current_holdings = current_holdings or {}

        if score_col not in ranked_df.columns:
            raise RuntimeError(f"missing score column: {score_col}")

        df = ranked_df.copy()

        # 标记是否为当前持仓
        df['is_held'] = df['code'].isin(current_holdings.keys())

        # 计算成本调整得分
        adjusted_scores = []
        for _, row in df.iterrows():
            code = row['code']
            original_score = row[score_col]

            if row['is_held']:
                # 已持有：无额外成本，但考虑"卖出成本"作为沉没优势
                # 如果卖出需要成本，则保留有隐性价值
                holding_bonus = self.single_side_cost * self.cost_penalty_lambda * 0.5
                adjusted = original_score + holding_bonus
            else:
                # 新建仓：需要支付建仓成本
                entry_cost = self.single_side_cost * self.cost_penalty_lambda
                adjusted = original_score - entry_cost

            adjusted_scores.append(adjusted)

        df['cost_adjusted_score'] = adjusted_scores

        # 按成本调整得分重新排序
        df = df.sort_values('cost_adjusted_score', ascending=False)

        # 选取前N只
        selected = df.head(n).copy()
        selected['rank'] = range(1, len(selected) + 1)

        # 统计换手情况
        old_codes = set(current_holdings.keys())
        new_codes = set(selected['code'].tolist())
        kept = len(old_codes & new_codes)
        entered = len(new_codes - old_codes)
        exited = len(old_codes - new_codes)

        logger.info(f"成本感知选股: 保留{kept}只, 新进{entered}只, 退出{exited}只")

        return selected

    def optimize_with_turnover_constraint(self,
                                           ranked_df: pd.DataFrame,
                                           current_holdings: Dict[str, float],
                                           n: int = 20,
                                           max_turnover: float = 0.5,
                                           score_col: str = 'total_score') -> pd.DataFrame:
        """
        带换手约束的优化选股

        在alpha最大化和换手控制之间取得平衡:
        - 优先保留仍在top排名的持仓股
        - 用高alpha新股替换低排名的持仓股
        - 控制总换手不超过上限

        Args:
            ranked_df: 已按得分降序排列的候选股票
            current_holdings: 当前持仓权重
            n: 目标选股数量
            max_turnover: 最大换手率
            score_col: 得分列名

        Returns:
            优化后的选股结果
        """
        if len(current_holdings) == 0:
            # 无持仓，直接选前N
            return ranked_df.head(n).copy()

        df = ranked_df.copy()
        old_codes = set(current_holdings.keys())

        # 将股票分为: 持仓中 vs 持仓外
        df['is_held'] = df['code'].isin(old_codes)
        held_stocks = df[df['is_held']].copy()
        new_stocks = df[~df['is_held']].copy()

        # 计算不换仓时的alpha损失
        # 思路: 比较 "完全按新排名选股" vs "保留所有旧持仓"
        top_n_codes = set(df.head(n)['code'].tolist())
        overlap = old_codes & top_n_codes

        # 计算最大可换股数量 (基于换手约束)
        # turnover = 换出数 / 持仓数 → 换出数 = turnover * 持仓数
        max_swap_out = int(max_turnover * len(current_holdings))
        max_swap_out = max(1, min(max_swap_out, len(old_codes)))

        # 贪心策略: 按alpha提升排序，优先换入alpha��升最大的
        # 同时确保换出的是当前持仓中排名最低的

        # 1. 持仓中仍在top-N的，必须保留
        must_keep = held_stocks[held_stocks['code'].isin(top_n_codes)]

        # 2. 持仓中不在top-N的，按得分排序，可能被换出
        may_exit = held_stocks[~held_stocks['code'].isin(top_n_codes)].sort_values(
            score_col, ascending=True  # 得分低的优先换出
        )

        # 3. 新股票按得分排序
        new_stocks = new_stocks.sort_values(score_col, ascending=False)

        # 4. 贪心匹配: 高alpha新股 替换 低alpha旧股
        selected_codes = set(must_keep['code'].tolist())
        swaps_done = 0

        # 添加新股票 (如果alpha提升足够大)
        for _, new_row in new_stocks.iterrows():
            if len(selected_codes) >= n:
                break

            new_score = new_row[score_col]

            # 检查是否值得换入
            if len(may_exit) > swaps_done:
                exit_candidate = may_exit.iloc[swaps_done]
                exit_score = exit_candidate[score_col]

                # 计算alpha提升是否足以覆盖交易成本
                alpha_gain = new_score - exit_score
                trade_cost = self.round_trip_cost  # 卖旧买新

                if alpha_gain > trade_cost * self.cost_penalty_lambda + self.min_swap_threshold:
                    # 值得换
                    selected_codes.add(new_row['code'])
                    swaps_done += 1
                else:
                    # alpha提升不足，保留旧股
                    selected_codes.add(exit_candidate['code'])
                    swaps_done += 1  # 标记已处理
            else:
                # 无更多可换出的旧股，直接添加新股
                selected_codes.add(new_row['code'])

        # 如果还不够n只，补充旧持仓
        remaining_held = [c for c in old_codes if c not in selected_codes]
        for code in remaining_held:
            if len(selected_codes) >= n:
                break
            selected_codes.add(code)

        # 构建结果
        selected_df = df[df['code'].isin(selected_codes)].copy()
        selected_df = selected_df.sort_values(score_col, ascending=False)
        selected_df['rank'] = range(1, len(selected_df) + 1)

        # 计算实际换手
        final_codes = set(selected_df['code'].tolist())
        actual_turnover = len(old_codes - final_codes) / len(old_codes) if len(old_codes) > 0 else 0

        logger.info(f"换手优化选股: 目标换手≤{max_turnover:.0%}, 实际换手={actual_turnover:.0%}")

        return selected_df

    def compute_expected_net_return(self,
                                     selected_df: pd.DataFrame,
                                     current_holdings: Dict[str, float],
                                     expected_return_col: str = 'expected_return') -> Dict[str, float]:
        """
        计算预期净收益 (扣除交易成本)

        Args:
            selected_df: 选股结果 (需包含预期收益列)
            current_holdings: 当前持仓
            expected_return_col: 预期收益列名

        Returns:
            净收益指标字典
        """
        metrics = {}

        # 计算换手成本
        new_weights = {row['code']: 1.0 / len(selected_df)
                      for _, row in selected_df.iterrows()}
        turnover_cost = self.compute_turnover_cost(new_weights, current_holdings)
        metrics['turnover_cost'] = turnover_cost

        # 计算预期收益
        if expected_return_col in selected_df.columns:
            gross_return = selected_df[expected_return_col].mean()
            net_return = gross_return - turnover_cost
            metrics['gross_return'] = gross_return
            metrics['net_return'] = net_return
            metrics['cost_ratio'] = turnover_cost / (gross_return + 1e-10)
        else:
            metrics['gross_return'] = None
            metrics['net_return'] = None

        # 换手率
        old_codes = set(current_holdings.keys())
        new_codes = set(selected_df['code'].tolist())
        if len(old_codes) > 0:
            metrics['turnover_rate'] = len(old_codes - new_codes) / len(old_codes)
        else:
            metrics['turnover_rate'] = 1.0

        return metrics

    def backtest_cost_sensitivity(self,
                                   ranked_df: pd.DataFrame,
                                   current_holdings: Dict[str, float],
                                   n: int = 20,
                                   lambda_range: List[float] = None) -> pd.DataFrame:
        """
        回测不同成本惩罚系数的效果

        用于调参: 找到最优的 cost_penalty_lambda

        Args:
            ranked_df: 候选股票
            current_holdings: 当前持仓
            n: 选股数量
            lambda_range: 测试的lambda值列表

        Returns:
            各lambda值的结果
        """
        if lambda_range is None:
            lambda_range = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

        results = []
        original_lambda = self.cost_penalty_lambda

        for lam in lambda_range:
            self.cost_penalty_lambda = lam

            selected = self.select_with_cost_awareness(
                ranked_df, current_holdings, n
            )

            if len(selected) == 0:
                continue

            # 计算指标
            new_weights = {row['code']: 1.0 / len(selected)
                          for _, row in selected.iterrows()}
            turnover_cost = self.compute_turnover_cost(new_weights, current_holdings)

            old_codes = set(current_holdings.keys())
            new_codes = set(selected['code'].tolist())
            turnover_rate = len(old_codes - new_codes) / max(1, len(old_codes))

            results.append({
                'lambda': lam,
                'turnover_rate': turnover_rate,
                'turnover_cost': turnover_cost,
                'avg_score': selected['total_score'].mean() if 'total_score' in selected.columns else None,
                'kept_count': len(old_codes & new_codes),
                'new_count': len(new_codes - old_codes)
            })

        # 恢复原始lambda
        self.cost_penalty_lambda = original_lambda

        return pd.DataFrame(results)


if __name__ == "__main__":
    # 测试代码
    scorer = FactorScorer()

    # 创建测试数据
    test_data = pd.DataFrame({
        'code': ['000001', '000002', '000003', '000004', '000005'],
        'roe': [0.15, 0.12, 0.18, 0.10, 0.20],
        'pe_ttm': [10, 15, 8, 20, 12],
        'ret_20d': [0.05, -0.02, 0.08, 0.01, 0.10],
        'net_inflow_5d': [1e8, -5e7, 2e8, 3e7, 1.5e8]
    })

    # 计算得分
    test_data['total_score'] = scorer.compute_total_score(test_data)
    print("测试数据得分:")
    print(test_data[['code', 'total_score']].sort_values('total_score', ascending=False))

    # 选股
    selected = scorer.select_top_stocks(test_data, n=3)
    print("\n选出的股票:")
    print(selected[['code', 'rank', 'total_score']])

    # P3-1: 测试成本感知优化器
    print("\n--- P3-1: 成本感知优化测试 ---")
    optimizer = CostAwareOptimizer()

    # 模拟当前持仓
    current_holdings = {'000001': 0.5, '000004': 0.5}

    # 成本感知选股
    cost_aware_selected = optimizer.select_with_cost_awareness(
        test_data.sort_values('total_score', ascending=False),
        current_holdings,
        n=3
    )
    print("\n成本感知选股结果:")
    print(cost_aware_selected[['code', 'rank', 'total_score', 'cost_adjusted_score']])
