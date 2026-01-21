"""
自适应因子权重管理器
根据市场环境（牛市/熊市/震荡市）动态调整因子权重
"""

from typing import Dict, Tuple, Optional
import pandas as pd

from .utils import load_config, setup_logger

logger = setup_logger(__name__)


class AdaptiveWeightManager:
    """根据市场环境动态调整因子权重"""

    def __init__(self, config: dict = None):
        """
        初始化自适应权重管理器

        Args:
            config: 配置字典
        """
        self.config = config or load_config()

        # 默认权重（均衡型）
        self.default_weights = self.config.get('weights', {
            'quality': 0.30,
            'valuation': 0.20,
            'momentum': 0.15,
            'flow': 0.20,
            'sentiment': 0.15
        })

        # 自适应权重配置
        adaptive_config = self.config.get('adaptive_weights', {})
        self.enabled = adaptive_config.get('enabled', False)

        # 牛市权重（进取型）
        self.bull_weights = adaptive_config.get('bull_weights', {
            'quality': 0.20,
            'valuation': 0.10,
            'momentum': 0.25,
            'flow': 0.20,
            'sentiment': 0.25
        })

        # 熊市权重（保守型）
        self.bear_weights = adaptive_config.get('bear_weights', {
            'quality': 0.35,
            'valuation': 0.25,
            'momentum': 0.10,
            'flow': 0.15,
            'sentiment': 0.15
        })

        # 阈值配置
        thresholds = adaptive_config.get('thresholds', {})
        self.bull_trend_threshold = thresholds.get('bull_trend', 0.02)
        self.bear_trend_threshold = thresholds.get('bear_trend', -0.02)
        self.bull_breadth_threshold = thresholds.get('bull_breadth', 0.50)
        self.bear_breadth_threshold = thresholds.get('bear_breadth', 0.40)

    def classify_market_regime(self, market_trend: float, market_breadth: float,
                                market_volatility: float = None) -> Tuple[str, bool]:
        """
        分类市场状态

        基于趋势和广度综合判断市场状态：
        - 牛市：趋势向上 + 广度高（赚钱效应好）
        - 熊市：趋势向下 + 广度低（赚钱效应差）
        - 震荡市：其他情况

        Args:
            market_trend: 市场趋势（20日涨跌幅）
            market_breadth: 市场广度（上涨股票占比）
            market_volatility: 市场波动率（可选，用于判断高波动）

        Returns:
            (市场状态, 是否高波动)
            市场状态: 'bull' / 'bear' / 'neutral'
        """
        # 判断高波动
        is_high_volatility = False
        if market_volatility is not None and market_volatility > 0.30:
            is_high_volatility = True

        # 牛市判定：趋势向上 + 广度高
        if market_trend > self.bull_trend_threshold and market_breadth > self.bull_breadth_threshold:
            return 'bull', is_high_volatility

        # 熊市判定：趋势向下 + 广度低
        if market_trend < self.bear_trend_threshold and market_breadth < self.bear_breadth_threshold:
            return 'bear', is_high_volatility

        # 其他情况为震荡市
        return 'neutral', is_high_volatility

    def get_adaptive_weights(self, market_trend: float, market_breadth: float,
                              market_volatility: float = None) -> Dict[str, float]:
        """
        根据市场环境返回调整后的权重

        Args:
            market_trend: 市场趋势（20日涨跌幅）
            market_breadth: 市场广度（上涨股票占比）
            market_volatility: 市场波动率（可选）

        Returns:
            调整后的权重字典
        """
        # 如果未启用自适应权重，返回默认权重
        if not self.enabled:
            return self.default_weights.copy()

        # 分类市场状态
        regime, is_high_vol = self.classify_market_regime(
            market_trend, market_breadth, market_volatility
        )

        # 根据市场状态选择权重
        if regime == 'bull':
            weights = self.bull_weights.copy()
            regime_name = '牛市'
        elif regime == 'bear':
            weights = self.bear_weights.copy()
            regime_name = '熊市'
        else:
            weights = self.default_weights.copy()
            regime_name = '震荡市'

        # 高波动时微调：降低动量权重，提高质量权重
        if is_high_vol and regime != 'bear':
            # 高波动环境下，动量因子噪音大，适当降低
            momentum_adj = weights.get('momentum', 0.15) * 0.8
            quality_adj = weights.get('quality', 0.30) * 1.1

            # 重新分配权重
            total_before = sum(weights.values())
            weights['momentum'] = momentum_adj
            weights['quality'] = quality_adj

            # 归一化确保权重和为1
            total_after = sum(weights.values())
            if total_after > 0:
                for k in weights:
                    weights[k] = weights[k] * total_before / total_after

        logger.info(f"市场状态: {regime_name}, 高波动: {is_high_vol}")
        logger.debug(f"使用权重: {weights}")

        return weights

    def get_weights_from_factor_df(self, factor_df: pd.DataFrame) -> Tuple[Dict[str, float], str]:
        """
        从因子DataFrame提取市场环境并返回权重

        Args:
            factor_df: 因子数据（需包含market_trend, market_breadth等列）

        Returns:
            (权重字典, 市场状态描述)
        """
        # 提取市场环境因子
        market_trend = 0.0
        market_breadth = 0.5
        market_volatility = 0.2

        if 'market_trend' in factor_df.columns:
            market_trend = factor_df['market_trend'].iloc[0]
        if 'market_breadth' in factor_df.columns:
            market_breadth = factor_df['market_breadth'].iloc[0]
        if 'market_volatility' in factor_df.columns:
            market_volatility = factor_df['market_volatility'].iloc[0]

        # 获取权重
        weights = self.get_adaptive_weights(market_trend, market_breadth, market_volatility)

        # 生成状态描述
        regime, is_high_vol = self.classify_market_regime(
            market_trend, market_breadth, market_volatility
        )

        regime_names = {'bull': '牛市', 'bear': '熊市', 'neutral': '震荡市'}
        status = regime_names.get(regime, '震荡市')
        if is_high_vol:
            status += '(高波动)'

        return weights, status

    def get_regime_info(self, market_trend: float, market_breadth: float,
                        market_volatility: float = None) -> Dict:
        """
        获取市场状态详细信息

        Args:
            market_trend: 市场趋势
            market_breadth: 市场广度
            market_volatility: 市场波动率

        Returns:
            包含市场状态信息的字典
        """
        regime, is_high_vol = self.classify_market_regime(
            market_trend, market_breadth, market_volatility
        )
        weights = self.get_adaptive_weights(market_trend, market_breadth, market_volatility)

        regime_names = {'bull': '牛市', 'bear': '熊市', 'neutral': '震荡市'}

        return {
            'regime': regime,
            'regime_name': regime_names.get(regime, '震荡市'),
            'is_high_volatility': is_high_vol,
            'market_trend': market_trend,
            'market_breadth': market_breadth,
            'market_volatility': market_volatility,
            'weights': weights,
            'adaptive_enabled': self.enabled
        }


if __name__ == "__main__":
    # 测试代码
    manager = AdaptiveWeightManager()

    # 测试不同市场环境
    test_cases = [
        (0.05, 0.60, 0.20, "牛市"),
        (-0.05, 0.35, 0.25, "熊市"),
        (0.01, 0.45, 0.20, "震荡市"),
        (0.03, 0.55, 0.35, "牛市高波动"),
    ]

    print("自适应权重测试:")
    print("=" * 60)

    for trend, breadth, vol, expected in test_cases:
        regime, is_high_vol = manager.classify_market_regime(trend, breadth, vol)
        weights = manager.get_adaptive_weights(trend, breadth, vol)

        print(f"\n趋势: {trend:+.2%}, 广度: {breadth:.1%}, 波动: {vol:.1%}")
        print(f"预期: {expected}, 实际: {regime}, 高波动: {is_high_vol}")
        print(f"权重: {weights}")
