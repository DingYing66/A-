"""
RL 策略适配器 - 桥接 RL 权重与现有回测系统

核心功能:
1. train(factor_data) - 训练并保存 RL 策略
2. predict_weights(factor_df, prev_weights) - 输出权重向量
3. weights_to_scores(weights) - 转为横截面 score (便于复用评分流程)
4. create_score_func() - 创建兼容 Backtester 的 score_func
5. create_weight_func() - 创建权重函数 (用于 Backtester 权重模式)

用法:
    from src.deep_learning.rl_adapter import RLAdapter

    # 训练模式
    adapter = RLAdapter()
    result = adapter.train(factor_data, save_path='output/models/rl/ppo_model.zip')

    # 回测模式 (score 模式)
    adapter = RLAdapter(model_path='output/models/rl/ppo_model.zip')
    score_func = adapter.create_score_func()
    backtester.run(start, end, score_func=score_func)

    # 回测模式 (weight 模式)
    weight_func = adapter.create_weight_func()
    backtester.run(start, end, weight_func=weight_func, mode='weight')
"""

from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Any, Union
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger, load_config, get_project_root, ensure_dir

logger = setup_logger(__name__)

# 导入策略模块
try:
    from .policies import TradingPolicies, CostModel, TradeabilityPolicy, ExecutionPolicy
    POLICIES_AVAILABLE = True
except ImportError:
    try:
        from src.deep_learning.policies import TradingPolicies, CostModel, TradeabilityPolicy, ExecutionPolicy
        POLICIES_AVAILABLE = True
    except ImportError:
        POLICIES_AVAILABLE = False
        TradingPolicies = None
        CostModel = None

# 导入 RL 模块
try:
    from .rl_env import PortfolioEnv, create_portfolio_env, check_gym_available
    from .rl_trainer import RLPortfolioTrainer, create_rl_trainer, check_sb3_available
    RL_AVAILABLE = True
except ImportError:
    try:
        from src.deep_learning.rl_env import PortfolioEnv, create_portfolio_env, check_gym_available
        from src.deep_learning.rl_trainer import RLPortfolioTrainer, create_rl_trainer, check_sb3_available
        RL_AVAILABLE = True
    except ImportError:
        RL_AVAILABLE = False


class RLAdapter:
    """
    RL 策略适配器

    桥接 RL 输出的权重向量与 Backtester 期望的 score_func/weight_func。
    确保 RL 训练、回测、部署使用统一的执行口径和成本模型。
    """

    def __init__(self,
                 config: dict = None,
                 model_path: str = None,
                 trading_policies: 'TradingPolicies' = None):
        """
        Args:
            config: 配置字典 (优先使用 config.yaml 的 rl_adapter 段)
            model_path: 预训练模型路径
            trading_policies: 预创建的 TradingPolicies (可选，否则自动创建)
        """
        # 加载配置
        self.config = config or load_config()

        # 创建统一策略
        if trading_policies is not None and POLICIES_AVAILABLE:
            self.policies = trading_policies
        elif POLICIES_AVAILABLE and TradingPolicies is not None:
            self.policies = TradingPolicies.from_config(self.config)
        else:
            self.policies = None
            logger.warning("TradingPolicies 不可用，将使用默认配置")

        # 从 CostModel 计算成本惩罚
        self.cost_penalty = self._compute_cost_penalty()

        # RL 适配器配置 (优先从 rl_adapter 段读取，回退到 rl 段)
        rl_adapter_cfg = self.config.get('rl_adapter', {})
        rl_cfg = self.config.get('rl', {})

        self.max_stocks = rl_adapter_cfg.get('max_stocks',
                          rl_cfg.get('action', {}).get('n_stocks', 50))
        self.max_position = rl_adapter_cfg.get('max_position',
                            self.config.get('backtest', {}).get('max_position', 0.08))
        self.action_type = rl_adapter_cfg.get('action_type',
                           rl_cfg.get('action', {}).get('type', 'continuous'))
        self.turnover_penalty = rl_adapter_cfg.get('turnover_penalty',
                                 rl_cfg.get('reward', {}).get('turnover_penalty', 0.001))

        # RL 训练参数
        self.algorithm = rl_adapter_cfg.get('algorithm', rl_cfg.get('algorithm', 'ppo'))
        self.total_timesteps = rl_adapter_cfg.get('total_timesteps',
                                rl_cfg.get('total_timesteps', 100000))
        self.learning_rate = rl_adapter_cfg.get('learning_rate',
                              rl_cfg.get('learning_rate', 0.0003))

        # 模型和股票代码
        self.trainer: Optional[RLPortfolioTrainer] = None
        self.model_path = model_path
        self.stock_codes: List[str] = []
        self.n_features: int = 0
        self.feature_cols: List[str] = []  # P2修复: 保存训练时的特征列名
        self.universe_id: Optional[str] = None  # P3修复: 股票池版本ID

        # 加载预训练模型
        if model_path and Path(model_path).exists():
            self._load_model(model_path)

        logger.info(f"初始化 RLAdapter: max_stocks={self.max_stocks}, "
                   f"max_position={self.max_position}, cost_penalty={self.cost_penalty:.4f}")

    def _compute_cost_penalty(self) -> float:
        """从 CostModel 计算成本惩罚 (单边成本)"""
        if self.policies is not None and hasattr(self.policies, 'cost'):
            cost_model = self.policies.cost
            # 单边成本 = 佣金 + 滑点
            return cost_model.commission + cost_model.slippage
        else:
            # 回退到配置
            backtest_cfg = self.config.get('backtest', {})
            return backtest_cfg.get('commission', 0.001) + backtest_cfg.get('slippage', 0.001)

    def train(self,
              factor_data: pd.DataFrame,
              total_timesteps: int = None,
              save_path: str = None,
              eval_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        训练 RL 策略

        Args:
            factor_data: 因子数据 (需包含 code, date, forward_return 或 label)
            total_timesteps: 训练步数 (默认从配置读取)
            save_path: 模型保存路径
            eval_data: 评估数据集 (可选)

        Returns:
            训练结果字典
        """
        if not RL_AVAILABLE:
            raise ImportError("需要安装 gymnasium 和 stable-baselines3")

        if not check_gym_available():
            raise ImportError("需要安装 gymnasium: pip install gymnasium")

        if not check_sb3_available():
            raise ImportError("需要安装 stable-baselines3: pip install stable-baselines3")

        logger.info("=" * 60)
        logger.info("开始 RL 策略训练")
        logger.info(f"数据范围: {factor_data['date'].min()} ~ {factor_data['date'].max()}")
        logger.info(f"股票数量: {factor_data['code'].nunique()}")
        logger.info("=" * 60)

        # 创建环境 (注入统一策略)
        env = create_portfolio_env(
            factor_data,
            config=self.config,
            max_stocks=self.max_stocks,
            action_type=self.action_type,
            trading_policies=self.policies,
        )

        # 记录股票代码和特征数
        self.stock_codes = list(env.stock_codes) if hasattr(env, 'stock_codes') else []
        self.n_features = env.n_features if hasattr(env, 'n_features') else 0
        self.feature_cols = list(env.feature_cols) if hasattr(env, 'feature_cols') else []  # P2修复

        # 创建训练器
        self.trainer = create_rl_trainer(
            env=env,
            config=self.config,
            algorithm=self.algorithm,
        )

        # 创建评估环境
        eval_env = None
        if eval_data is not None:
            eval_env = create_portfolio_env(
                eval_data,
                config=self.config,
                max_stocks=self.max_stocks,
                action_type=self.action_type,
                trading_policies=self.policies,
            )

        # 训练
        timesteps = total_timesteps or self.total_timesteps
        result = self.trainer.train(
            total_timesteps=timesteps,
            eval_env=eval_env,
        )

        # 保存模型
        if save_path:
            self.save(save_path)

        return result

    def predict_weights(self,
                        factor_df: pd.DataFrame,
                        prev_weights: np.ndarray = None) -> np.ndarray:
        """
        预测权重向量

        Args:
            factor_df: 当前因子数据 (单日横截面)
            prev_weights: 上一期权重 (可选)

        Returns:
            权重向量 np.ndarray (n_stocks,)，顺序与 factor_df['code'] 对齐
        """
        if self.trainer is None:
            raise ValueError("模型未加载，请先调用 train() 或 load()")

        # 获取当日股票代码
        current_codes = factor_df['code'].unique().tolist()

        # 如果没有训练时的 stock_codes，使用当前代码
        if not self.stock_codes:
            self.stock_codes = current_codes[:self.max_stocks]
            logger.warning("stock_codes 为空，使用当前代码初始化")

        # 构建观测 (使用训练时的 stock_codes 顺序)
        obs = self._build_observation(factor_df, prev_weights)

        # 模型输出权重 (长度 = len(self.stock_codes)，顺序对应 self.stock_codes)
        raw_weights = self.trainer.get_optimal_weights(obs)

        # M2修复: 按股票代码映射权重到当日可交易股票
        # 而不是简单截断/填充 (之前的实现会导致权重错位)
        weight_by_code = {}
        for i, code in enumerate(self.stock_codes):
            if i < len(raw_weights):
                weight_by_code[code] = raw_weights[i]

        # 提取当日可交易股票的权重 (顺序与 current_codes 对齐)
        weights = np.zeros(len(current_codes), dtype=np.float32)
        for i, code in enumerate(current_codes):
            weights[i] = weight_by_code.get(code, 0.0)

        # 应用可交易性过滤
        weights = self._apply_tradeability_filter(weights, factor_df)

        # 应用仓位约束
        weights = self._apply_position_constraints(weights)

        return weights

    def weights_to_scores(self,
                          weights: np.ndarray,
                          stock_codes: List[str] = None) -> pd.Series:
        """
        将权重转为横截面 score

        用于复用现有评分/排名流程

        Args:
            weights: 权重向量
            stock_codes: 股票代码列表

        Returns:
            score Series (index=code, values=score)
        """
        codes = stock_codes or self.stock_codes

        if len(weights) != len(codes):
            min_len = min(len(weights), len(codes))
            weights = weights[:min_len]
            codes = codes[:min_len]

        # 权重直接作为 score (权重越高，score 越高)
        # 乘以 100 使数值更易读
        scores = pd.Series(weights * 100, index=codes, name='total_score')

        return scores

    def create_score_func(self) -> Callable[[pd.DataFrame], pd.DataFrame]:
        """
        创建兼容 Backtester 的 score_func

        返回的函数签名: score_func(factor_df) -> DataFrame with 'total_score'

        Returns:
            score_func 函数
        """
        # 使用列表封装以支持闭包修改
        state = {
            'prev_weights': None,
            'prev_codes': None,
        }

        def score_func(factor_df: pd.DataFrame) -> pd.DataFrame:
            """评分函数 (兼容 Backtester)"""
            # 获取当前股票代码
            current_codes = factor_df['code'].tolist()

            # M2修复: 上期权重需要按 self.stock_codes 顺序对齐
            prev_weights = None
            if state['prev_weights'] is not None and state['prev_codes'] is not None and self.stock_codes:
                prev_weights = np.zeros(len(self.stock_codes), dtype=np.float32)
                code_to_weight = dict(zip(state['prev_codes'], state['prev_weights']))
                code_to_idx = {c: i for i, c in enumerate(self.stock_codes)}
                for code, w in code_to_weight.items():
                    if code in code_to_idx:
                        prev_weights[code_to_idx[code]] = w

            # 预测权重 (现在返回的权重已按 current_codes 顺序对齐)
            weights = self.predict_weights(factor_df, prev_weights)

            # 更新状态 (weights 已对齐 current_codes)
            state['prev_weights'] = weights.copy()
            state['prev_codes'] = current_codes

            # 转为 score DataFrame
            result = factor_df[['code']].copy()
            result['total_score'] = weights * 100  # 放大为更易读的 score

            # 过滤零权重
            result = result[result['total_score'] > 1e-6]

            return result.sort_values('total_score', ascending=False)

        return score_func

    def create_weight_func(self) -> Callable[[pd.DataFrame, Dict[str, float]], Dict[str, float]]:
        """
        创建权重函数 (用于 Backtester 权重模式)

        返回的函数签名: weight_func(factor_df, prev_weights) -> Dict[str, float]

        Returns:
            weight_func 函数
        """
        def weight_func(factor_df: pd.DataFrame,
                        prev_weights_dict: Dict[str, float] = None) -> Dict[str, float]:
            """权重函数 (用于 Backtester 权重模式)"""
            # 获取当前股票代码
            current_codes = factor_df['code'].tolist()

            # M2修复: 上期权重需要按 self.stock_codes 顺序对齐
            # (因为 _build_observation 使用 self.stock_codes 长度)
            prev_weights = None
            if prev_weights_dict and self.stock_codes:
                prev_weights = np.zeros(len(self.stock_codes), dtype=np.float32)
                code_to_idx = {c: i for i, c in enumerate(self.stock_codes)}
                for code, w in prev_weights_dict.items():
                    if code in code_to_idx:
                        prev_weights[code_to_idx[code]] = w

            # 预测权重 (现在已按 current_codes 顺序映射)
            weights = self.predict_weights(factor_df, prev_weights)

            # 转换为字典
            result = {}
            for i, code in enumerate(current_codes):
                if i < len(weights) and weights[i] > 1e-8:
                    result[code] = float(weights[i])

            return result

        return weight_func

    def _build_observation(self,
                           factor_df: pd.DataFrame,
                           prev_weights: np.ndarray = None) -> np.ndarray:
        """构建 RL 观测向量"""
        # P2修复: 优先使用训练时保存的特征列，确保维度一致
        if self.feature_cols:
            # 使用训练时的特征列 (缺失列填0)
            feature_cols = self.feature_cols
            available_cols = [c for c in feature_cols if c in factor_df.columns]
            if len(available_cols) < len(feature_cols):
                missing = set(feature_cols) - set(available_cols)
                logger.warning(f"缺失 {len(missing)} 个特征列，将填充0: {list(missing)[:5]}...")
        else:
            # 回退: 动态发现特征列
            exclude_cols = {'code', 'name', 'date', 'industry', 'label', 'forward_return',
                           'total_score', 'rank', 'weight'}
            feature_cols = [c for c in factor_df.columns
                           if c not in exclude_cols
                           and factor_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

        # P6修复: 扩展特征统计量 (mean + std + q25 + q75)
        # 通过配置项控制是否使用扩展特征
        use_extended_features = self.config.get('rl_adapter', {}).get('extended_features', False)

        if len(feature_cols) > 0:
            if use_extended_features:
                # 扩展特征: mean + std + q25 + q75 (4倍特征维度)
                n_cols = len(feature_cols)
                features = np.zeros(n_cols * 4, dtype=np.float32)
                for i, col in enumerate(feature_cols):
                    if col in factor_df.columns:
                        col_data = factor_df[col].dropna()
                        if len(col_data) > 0:
                            features[i] = col_data.mean()
                            features[n_cols + i] = col_data.std()
                            features[2 * n_cols + i] = col_data.quantile(0.25)
                            features[3 * n_cols + i] = col_data.quantile(0.75)
            else:
                # 原始: 仅横截面均值
                features = np.zeros(len(feature_cols), dtype=np.float32)
                for i, col in enumerate(feature_cols):
                    if col in factor_df.columns:
                        features[i] = factor_df[col].mean()
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            n_base = self.n_features or 10
            features = np.zeros(n_base * 4 if use_extended_features else n_base, dtype=np.float32)

        # P2修复: 上期权重按股票代码对齐 (使用 self.stock_codes)
        n_stocks = len(self.stock_codes) if self.stock_codes else len(factor_df)
        if prev_weights is None:
            prev_weights = np.zeros(n_stocks, dtype=np.float32)
        elif len(prev_weights) != n_stocks:
            # 调整长度
            if len(prev_weights) < n_stocks:
                prev_weights = np.pad(prev_weights, (0, n_stocks - len(prev_weights)))
            else:
                prev_weights = prev_weights[:n_stocks]

        # 市场状态
        market_state = self._get_market_state(factor_df)

        # 组合观测
        obs = np.concatenate([features, prev_weights.astype(np.float32), market_state])

        return obs.astype(np.float32)

    def _get_market_state(self, df: pd.DataFrame) -> np.ndarray:
        """获取市场状态向量"""
        state = np.zeros(4, dtype=np.float32)

        # 尝试从收益列获取市场状态
        return_col = None
        for col in ['forward_return', 'label', 'ret_forward', 'return']:
            if col in df.columns:
                return_col = col
                break

        if return_col is not None:
            returns = df[return_col].dropna()
            if len(returns) > 0:
                state[0] = np.clip(returns.mean() * 10, -1, 1)      # 平均收益
                state[1] = np.clip(returns.std() * 10, 0, 1)        # 收益波动
                state[2] = (returns > 0).mean() * 2 - 1              # 上涨比例
                state[3] = np.clip(returns.quantile(0.9) - returns.quantile(0.1), 0, 1)  # 分散度

        return state

    def _apply_tradeability_filter(self,
                                   weights: np.ndarray,
                                   factor_df: pd.DataFrame) -> np.ndarray:
        """应用可交易性过滤"""
        if self.policies is None or not hasattr(self.policies, 'tradeability'):
            return weights

        # 获取可交易股票
        tradeable_codes = set()
        for _, row in factor_df.iterrows():
            is_tradeable, reason = self.policies.tradeability.is_tradeable(row, action='buy')
            if is_tradeable:
                tradeable_codes.add(row.get('code', ''))

        if len(tradeable_codes) == 0:
            logger.warning("无可交易股票，保持原权重")
            return weights

        # 获取股票代码列表
        codes = factor_df['code'].tolist()

        # 过滤不可交易股票的权重
        filtered_weights = weights.copy()
        for i, code in enumerate(codes):
            if i < len(filtered_weights) and code not in tradeable_codes:
                filtered_weights[i] = 0.0

        # 重新归一化
        total = filtered_weights.sum()
        if total > 1e-8:
            filtered_weights = filtered_weights / total

        return filtered_weights.astype(np.float32)

    def _apply_position_constraints(self, weights: np.ndarray) -> np.ndarray:
        """应用仓位约束"""
        weights = weights.copy()

        # 单票上限
        weights = np.clip(weights, 0, self.max_position)

        # 最大持仓数
        n_positive = np.sum(weights > 1e-8)
        if n_positive > self.max_stocks:
            # 保留权重最大的 max_stocks 只股票
            threshold = np.sort(weights)[-self.max_stocks]
            weights = np.where(weights >= threshold, weights, 0)

        # 重新归一化
        total = weights.sum()
        if total > 1e-8:
            weights = weights / total
        else:
            # 等权
            n = min(self.max_stocks, len(weights))
            weights = np.zeros_like(weights)
            weights[:n] = 1.0 / n

        return weights.astype(np.float32)

    def save(self, path: str):
        """保存模型"""
        if self.trainer is None:
            raise ValueError("没有可保存的模型")

        # 确保目录存在
        save_path = Path(path)
        ensure_dir(save_path.parent)

        # M1修复: 直接保存模型到指定路径，不经过 trainer._save_model()
        # 避免路径前缀重复问题 (trainer._save_model 会额外添加算法前缀)
        model_path = save_path.with_suffix('.zip')
        self.trainer.model.save(str(model_path))
        self.model_path = str(model_path)

        # 保存元数据 (与模型文件在同一目录)
        import json
        meta_path = save_path.with_suffix('.meta.json')
        meta = {
            'stock_codes': self.stock_codes,
            'n_features': self.n_features,
            'feature_cols': self.feature_cols,  # P2修复: 保存特征列名
            'universe_id': self.universe_id,    # P3修复: 保存 universe_id
            'max_stocks': self.max_stocks,
            'max_position': self.max_position,
            'algorithm': self.algorithm,
            'cost_penalty': self.cost_penalty,
            'config_hash': self.policies.get_config_hash() if self.policies else '',
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"模型已保存: {model_path}")
        logger.info(f"元数据已保存: {meta_path}")

    def _load_model(self, path: str):
        """加载预训练模型"""
        if not RL_AVAILABLE:
            raise ImportError("需要安装 gymnasium 和 stable-baselines3")

        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")

        # 加载元数据
        meta_path = model_path.with_suffix('.meta.json')
        if meta_path.exists():
            import json
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self.stock_codes = meta.get('stock_codes', [])
            self.n_features = meta.get('n_features', 0)
            self.feature_cols = meta.get('feature_cols', [])  # P2修复: 加载特征列名
            self.universe_id = meta.get('universe_id')  # P3修复: 加载 universe_id
            logger.info(f"加载元数据: {len(self.stock_codes)} 只股票, {self.n_features} 个特征, "
                       f"{len(self.feature_cols)} 个特征列"
                       + (f", universe_id={self.universe_id}" if self.universe_id else ""))

        # 创建临时环境用于加载模型
        # 需要一个虚拟的因子数据
        n_stocks = len(self.stock_codes) or self.max_stocks
        dummy_data = pd.DataFrame({
            'code': self.stock_codes[:n_stocks] if self.stock_codes else [f'{i:06d}' for i in range(n_stocks)],
            'date': ['20240101'] * n_stocks,
            'forward_return': np.random.randn(n_stocks) * 0.01,
        })
        # P2修复: 使用训练时的特征列名 (如果保存了)
        if self.feature_cols:
            for col in self.feature_cols:
                dummy_data[col] = np.random.randn(n_stocks)
        else:
            # 回退: 使用通用特征名
            for i in range(self.n_features or 10):
                dummy_data[f'feature_{i}'] = np.random.randn(n_stocks)

        env = create_portfolio_env(
            dummy_data,
            config=self.config,
            max_stocks=self.max_stocks,
            action_type=self.action_type,
            trading_policies=self.policies,
        )

        # 创建训练器并加载模型
        self.trainer = create_rl_trainer(env, self.config, self.algorithm)
        self.trainer.load_model(str(model_path))

        self.model_path = str(model_path)
        logger.info(f"模型已加载: {path}")

    def load(self, path: str):
        """加载模型 (公开接口)"""
        self._load_model(path)

    @classmethod
    def from_model(cls,
                   model_path: str,
                   config: dict = None) -> 'RLAdapter':
        """从模型路径创建适配器"""
        return cls(config=config, model_path=model_path)


# ===================== 便捷函数 =====================

def create_rl_adapter(config: dict = None,
                      model_path: str = None) -> RLAdapter:
    """
    创建 RL 适配器

    Args:
        config: 配置字典
        model_path: 预训练模型路径

    Returns:
        RLAdapter 实例
    """
    return RLAdapter(config=config, model_path=model_path)


def train_rl_strategy(factor_data: pd.DataFrame,
                      config: dict = None,
                      save_path: str = None,
                      total_timesteps: int = None) -> Tuple[RLAdapter, Dict[str, Any]]:
    """
    训练 RL 策略

    Args:
        factor_data: 因子数据
        config: 配置字典
        save_path: 保存路径
        total_timesteps: 训练步数

    Returns:
        (RLAdapter, 训练结果)
    """
    adapter = RLAdapter(config=config)
    result = adapter.train(
        factor_data=factor_data,
        total_timesteps=total_timesteps,
        save_path=save_path,
    )
    return adapter, result
