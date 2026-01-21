"""
强化学习训练器模块 (Phase 5)

提供 RL 智能体训练和评估功能:
- RLPortfolioTrainer: 基于 Stable-Baselines3 的训练器
- 支持 PPO, A2C, SAC 算法
- 自定义回调和日志

用法:
    from src.deep_learning.rl_trainer import RLPortfolioTrainer, train_rl_agent

    trainer = RLPortfolioTrainer(env, config)
    trainer.train(total_timesteps=100000)
    weights = trainer.get_optimal_weights(obs)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime
import copy
import os

import numpy as np
import pandas as pd

try:
    from stable_baselines3 import PPO, A2C, SAC
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    # 占位符，用于类型注解
    BaseCallback = object
    PPO = None
    A2C = None
    SAC = None

from .rl_env import PortfolioEnv, SimplePortfolioEnv, create_portfolio_env, check_gym_available

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger, get_project_root, ensure_dir

logger = setup_logger(__name__)


# ===================== 自定义回调 =====================

if SB3_AVAILABLE:
    class PortfolioCallback(BaseCallback):
        """
        投资组合训练回调

        记录训练过程中的关键指标
        """

        def __init__(self, eval_freq: int = 1000, verbose: int = 0):
            super().__init__(verbose)
            self.eval_freq = eval_freq
            self.episode_rewards = []
            self.episode_values = []
            self.best_mean_reward = -np.inf

        def _on_step(self) -> bool:
            # 记录 episode 信息
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                if 'r' in ep_info:
                    self.episode_rewards.append(ep_info['r'])

            # 定期评估
            if self.n_calls % self.eval_freq == 0 and len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                if self.verbose > 0:
                    logger.info(f"Step {self.n_calls}: Mean Reward = {mean_reward:.4f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward

            return True

        def _on_training_end(self) -> None:
            if self.verbose > 0:
                logger.info(f"训练结束: 最佳平均奖励 = {self.best_mean_reward:.4f}")


    class TensorboardCallback(BaseCallback):
        """
        TensorBoard 日志回调
        """

        def __init__(self, verbose: int = 0):
            super().__init__(verbose)

        def _on_step(self) -> bool:
            # 记录额外指标到 TensorBoard
            if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                env = self.training_env.envs[0]
                if hasattr(env, 'portfolio_value'):
                    self.logger.record('portfolio/value', env.portfolio_value)
                if hasattr(env, 'current_step'):
                    self.logger.record('portfolio/step', env.current_step)

            return True


# ===================== RL 训练器 =====================

class RLPortfolioTrainer:
    """
    强化学习投资组合训练器

    支持多种 RL 算法:
    - PPO: Proximal Policy Optimization
    - A2C: Advantage Actor-Critic
    - SAC: Soft Actor-Critic (连续动作)
    """

    def __init__(self,
                 env: Union[PortfolioEnv, SimplePortfolioEnv],
                 config: dict = None,
                 algorithm: str = 'ppo',
                 model_dir: str = None):
        """
        Args:
            env: 训练环境
            config: 配置字典
            algorithm: RL 算法 (ppo/a2c/sac)
            model_dir: 模型保存目录
        """
        if not SB3_AVAILABLE:
            raise ImportError("需要安装 stable-baselines3: pip install stable-baselines3")

        self.env = env
        self.config = config or {}
        self.algorithm = algorithm.lower()
        self.model_dir = model_dir or str(ensure_dir(get_project_root() / 'output' / 'models' / 'rl'))

        # N3修复: RL 配置 - 合并 rl_adapter 和 rl (rl_adapter 优先)
        rl_config = self.config.get('rl', {})
        rl_adapter_config = self.config.get('rl_adapter', {})
        merged_rl_config = {**rl_config, **rl_adapter_config}  # rl_adapter 覆盖 rl

        self.total_timesteps = merged_rl_config.get('total_timesteps', 100000)
        self.learning_rate = merged_rl_config.get('learning_rate', 3e-4)
        self.n_steps = merged_rl_config.get('n_steps', 2048)
        self.batch_size = merged_rl_config.get('batch_size', 64)
        self.gamma = merged_rl_config.get('gamma', 0.99)
        self.clip_range = merged_rl_config.get('clip_range', 0.2)

        # 创建向量化环境
        self.vec_env = DummyVecEnv([lambda: Monitor(env)])

        # 创建模型
        self.model = self._create_model()

        # 训练历史
        self.training_history = {
            'rewards': [],
            'portfolio_values': [],
            'sharpe_ratios': [],
        }

        logger.info(f"初始化 RL 训练器: 算法={self.algorithm}, 步数={self.total_timesteps}")

    def _create_model(self):
        """创建 RL 模型"""
        common_params = {
            'policy': 'MlpPolicy',
            'env': self.vec_env,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'verbose': 1,
            'tensorboard_log': os.path.join(self.model_dir, 'tb_logs'),
        }

        if self.algorithm == 'ppo':
            model = PPO(
                **common_params,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                clip_range=self.clip_range,
                ent_coef=0.01,
            )
        elif self.algorithm == 'a2c':
            model = A2C(
                **common_params,
                n_steps=min(self.n_steps, 5),  # A2C 通常使用较小的 n_steps
                ent_coef=0.01,
            )
        elif self.algorithm == 'sac':
            # SAC 只支持连续动作空间
            model = SAC(
                **common_params,
                batch_size=self.batch_size,
                buffer_size=100000,
                learning_starts=1000,
            )
        else:
            raise ValueError(f"未知算法: {self.algorithm}")

        return model

    def train(self,
              total_timesteps: int = None,
              callback: BaseCallback = None,
              eval_env: PortfolioEnv = None,
              eval_freq: int = 10000) -> Dict[str, Any]:
        """
        训练模型

        Args:
            total_timesteps: 总训练步数
            callback: 自定义回调
            eval_env: 评估环境
            eval_freq: 评估频率

        Returns:
            训练结果
        """
        total_timesteps = total_timesteps or self.total_timesteps

        logger.info("=" * 60)
        logger.info("开始 RL 训练")
        logger.info(f"算法: {self.algorithm.upper()}")
        logger.info(f"总步数: {total_timesteps}")
        logger.info("=" * 60)

        # 创建回调列表
        callbacks = []

        # 自定义回调
        if callback is not None:
            callbacks.append(callback)

        # 评估回调
        if eval_env is not None:
            eval_callback = EvalCallback(
                DummyVecEnv([lambda: Monitor(eval_env)]),
                best_model_save_path=self.model_dir,
                log_path=self.model_dir,
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)

        # 检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=eval_freq,
            save_path=self.model_dir,
            name_prefix=f'rl_{self.algorithm}',
        )
        callbacks.append(checkpoint_callback)

        # 投资组合回调
        portfolio_callback = PortfolioCallback(eval_freq=1000, verbose=1)
        callbacks.append(portfolio_callback)

        # 训练
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True,
            )
        except KeyboardInterrupt:
            logger.info("训练被中断")

        # 保存最终模型
        self._save_model('final')

        # 收集训练结果
        result = {
            'total_timesteps': total_timesteps,
            'best_mean_reward': portfolio_callback.best_mean_reward,
            'episode_rewards': portfolio_callback.episode_rewards,
            'algorithm': self.algorithm,
        }

        logger.info(f"训练完成: 最佳平均奖励 = {result['best_mean_reward']:.4f}")

        return result

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        预测动作

        Args:
            obs: 观察
            deterministic: 是否确定性策略

        Returns:
            动作
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def get_optimal_weights(self, obs: np.ndarray) -> np.ndarray:
        """
        获取最优权重

        Args:
            obs: 观察

        Returns:
            归一化权重
        """
        action = self.predict(obs, deterministic=True)

        # 归一化
        weights = np.clip(action, 0, 1)
        total = np.sum(weights)
        if total > 1e-8:
            weights = weights / total
        else:
            weights = np.ones_like(weights) / len(weights)

        return weights

    def evaluate(self,
                 eval_env: PortfolioEnv = None,
                 n_episodes: int = 10) -> Dict[str, float]:
        """
        评估模型

        Args:
            eval_env: 评估环境
            n_episodes: 评估 episode 数

        Returns:
            评估结果
        """
        eval_env = eval_env or self.env

        logger.info(f"评估 {n_episodes} 个 episodes...")

        episode_rewards = []
        episode_values = []
        episode_sharpes = []

        for ep in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            total_reward = 0
            returns = []

            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                total_reward += reward

                if 'portfolio_return' in info:
                    returns.append(info['portfolio_return'])

            episode_rewards.append(total_reward)
            episode_values.append(info.get('portfolio_value', 1.0))

            # 计算 Sharpe
            if len(returns) > 1:
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                episode_sharpes.append(sharpe)

        result = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_value': np.mean(episode_values),
            'mean_sharpe': np.mean(episode_sharpes) if episode_sharpes else 0.0,
        }

        logger.info(f"评估结果: 奖励={result['mean_reward']:.4f}, 价值={result['mean_value']:.4f}")

        return result

    def _save_model(self, name: str):
        """保存模型"""
        path = os.path.join(self.model_dir, f'{self.algorithm}_{name}.zip')
        self.model.save(path)
        logger.info(f"模型已保存: {path}")

    def load_model(self, path: str):
        """加载模型"""
        if self.algorithm == 'ppo':
            self.model = PPO.load(path, env=self.vec_env)
        elif self.algorithm == 'a2c':
            self.model = A2C.load(path, env=self.vec_env)
        elif self.algorithm == 'sac':
            self.model = SAC.load(path, env=self.vec_env)

        logger.info(f"模型已加载: {path}")


# ===================== RL 权重优化器 =====================

class RLWeightOptimizer:
    """
    RL 权重优化器

    使用训练好的 RL 模型优化投资组合权重
    """

    def __init__(self, model_path: str = None, config: dict = None):
        """
        Args:
            model_path: 预训练模型路径
            config: 配置字典
        """
        self.config = config or {}
        self.model = None
        self.algorithm = 'ppo'

        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str):
        """加载模型"""
        if not SB3_AVAILABLE:
            raise ImportError("需要安装 stable-baselines3")

        # 从文件名推断算法类型
        if 'ppo' in path.lower():
            self.algorithm = 'ppo'
            self.model = PPO.load(path)
        elif 'a2c' in path.lower():
            self.algorithm = 'a2c'
            self.model = A2C.load(path)
        elif 'sac' in path.lower():
            self.algorithm = 'sac'
            self.model = SAC.load(path)
        else:
            # 默认 PPO
            self.model = PPO.load(path)

        logger.info(f"加载 RL 模型: {path}")

    def optimize_weights(self,
                         factor_df: pd.DataFrame,
                         current_weights: np.ndarray = None,
                         n_stocks: int = 50) -> pd.DataFrame:
        """
        优化权重

        Args:
            factor_df: 因子数据
            current_weights: 当前权重
            n_stocks: 最大股票数

        Returns:
            包含权重的 DataFrame
        """
        if self.model is None:
            logger.warning("模型未加载，返回等权")
            return self._equal_weight(factor_df, n_stocks)

        # 构建观察
        obs = self._build_observation(factor_df, current_weights)

        # 预测动作
        action, _ = self.model.predict(obs, deterministic=True)

        # 转换为权重
        weights = self._action_to_weights(action, factor_df, n_stocks)

        return weights

    def _build_observation(self,
                           factor_df: pd.DataFrame,
                           current_weights: np.ndarray = None) -> np.ndarray:
        """构建观察"""
        # 因子特征
        exclude_cols = {'code', 'name', 'date', 'industry', 'label', 'forward_return', 'total_score'}
        feature_cols = [c for c in factor_df.columns
                       if c not in exclude_cols
                       and factor_df[c].dtype in [np.float64, np.float32, np.int64]]

        factor_features = factor_df[feature_cols].mean().values.astype(np.float32)
        factor_features = np.nan_to_num(factor_features, nan=0.0)

        # 当前权重
        if current_weights is None:
            current_weights = np.zeros(len(factor_df), dtype=np.float32)

        # 市场状态
        market_state = np.zeros(4, dtype=np.float32)

        # 组合观察
        obs = np.concatenate([factor_features, current_weights[:len(factor_df)], market_state])

        return obs.astype(np.float32)

    def _action_to_weights(self,
                           action: np.ndarray,
                           factor_df: pd.DataFrame,
                           n_stocks: int) -> pd.DataFrame:
        """动作转权重"""
        codes = factor_df['code'].values

        # 归一化权重
        weights = np.clip(action[:len(codes)], 0, 1)

        # Top-N 筛选
        if len(weights) > n_stocks:
            threshold = np.sort(weights)[-n_stocks]
            weights = np.where(weights >= threshold, weights, 0)

        # 归一化
        total = np.sum(weights)
        if total > 1e-8:
            weights = weights / total
        else:
            # 等权
            top_n = min(n_stocks, len(weights))
            weights = np.zeros_like(weights)
            weights[:top_n] = 1.0 / top_n

        # 创建结果
        result = factor_df[['code']].copy()
        result['weight'] = weights
        result = result[result['weight'] > 0].sort_values('weight', ascending=False)

        return result

    def _equal_weight(self, factor_df: pd.DataFrame, n_stocks: int) -> pd.DataFrame:
        """等权分配"""
        result = factor_df[['code']].head(n_stocks).copy()
        result['weight'] = 1.0 / len(result)
        return result


# ===================== 工厂函数 =====================

def create_rl_trainer(env: PortfolioEnv,
                      config: dict = None,
                      algorithm: str = 'ppo') -> RLPortfolioTrainer:
    """
    创建 RL 训练器

    Args:
        env: 训练环境
        config: 配置字典
        algorithm: RL 算法

    Returns:
        RLPortfolioTrainer 实例
    """
    if not SB3_AVAILABLE:
        raise ImportError("需要安装 stable-baselines3: pip install stable-baselines3")

    return RLPortfolioTrainer(env, config, algorithm)


def train_rl_agent(factor_data: pd.DataFrame,
                   config: dict = None,
                   algorithm: str = 'ppo',
                   total_timesteps: int = 100000,
                   save_path: str = None) -> Tuple[RLPortfolioTrainer, Dict[str, Any]]:
    """
    训练 RL 智能体

    Args:
        factor_data: 因子数据
        config: 配置字典
        algorithm: RL 算法
        total_timesteps: 训练步数
        save_path: 保存路径

    Returns:
        训练器和结果
    """
    if not check_gym_available():
        raise ImportError("需要安装 gymnasium: pip install gymnasium")

    if not SB3_AVAILABLE:
        raise ImportError("需要安装 stable-baselines3: pip install stable-baselines3")

    # 创建环境
    env = create_portfolio_env(factor_data, config)

    # 创建训练器
    trainer = create_rl_trainer(env, config, algorithm)

    # 训练
    result = trainer.train(total_timesteps=total_timesteps)

    # 保存
    if save_path:
        trainer._save_model(save_path)

    return trainer, result


def check_sb3_available() -> bool:
    """检查 Stable-Baselines3 是否可用"""
    return SB3_AVAILABLE
