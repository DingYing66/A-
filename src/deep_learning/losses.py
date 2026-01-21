"""
排序损失函数模块

针对股票选股的排序学习损失函数:
- ListwiseLoss (ListMLE): 列表级损失
- PairwiseLoss (RankNet): 对级损失
- NDCGLoss: 可微分NDCG近似
- ApproxNDCGLoss: 近似NDCG损失

参考:
- Xia et al., "Listwise Approach to Learning to Rank" (2008)
- Burges et al., "Learning to Rank with Nonsmooth Cost Functions" (2006)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ListwiseLoss(nn.Module):
    """
    ListMLE 损失函数

    将排序问题转化为列表概率最大化问题。
    给定预测分数，最大化正确排序的似然。

    公式: -sum(log(exp(s_i) / sum(exp(s_j) for j >= i)))

    优点:
    - 直接优化排序
    - 考虑全局顺序
    - 数值稳定
    """

    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        计算 ListMLE 损失

        Args:
            predictions: 预测分数 [batch_size] 或 [batch_size, 1]
            targets: 真实标签/收益率 [batch_size]
            mask: 有效样本掩码 [batch_size] (可选)

        Returns:
            标量损失值
        """
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]

        if len(predictions) < 2:
            return torch.tensor(0.0, device=predictions.device)

        # 按真实值排序
        sorted_indices = torch.argsort(targets, descending=True)
        sorted_preds = predictions[sorted_indices]

        # 计算 ListMLE
        n = len(sorted_preds)
        loss = 0.0

        for i in range(n - 1):
            # log-sum-exp trick for numerical stability
            remaining = sorted_preds[i:]
            max_val = remaining.max()
            log_sum_exp = max_val + torch.log(torch.sum(torch.exp(remaining - max_val)) + self.eps)
            loss += sorted_preds[i] - log_sum_exp

        return -loss / max(n - 1, 1)


class PairwiseLoss(nn.Module):
    """
    RankNet 对级损失

    对每对样本计算交叉熵损失，优化相对顺序。

    公式: sum(log(1 + exp(-sigma * (s_i - s_j)))) for i > j

    优点:
    - 计算效率高（可采样）
    - 对相对顺序敏感
    - 梯度稳定
    """

    def __init__(self, sigma: float = 1.0, sample_ratio: float = 0.1):
        """
        Args:
            sigma: 缩放因子
            sample_ratio: 采样比例（减少计算量）
        """
        super().__init__()
        self.sigma = sigma
        self.sample_ratio = sample_ratio

    def forward(self, predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        计算 RankNet 损失

        Args:
            predictions: 预测分数 [batch_size]
            targets: 真实标签 [batch_size]
            mask: 有效样本掩码 [batch_size] (可选)

        Returns:
            标量损失值
        """
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]

        n = len(predictions)
        if n < 2:
            return torch.tensor(0.0, device=predictions.device)

        # 生成配对
        # 采样以减少计算量
        n_pairs = int(n * (n - 1) / 2 * self.sample_ratio)
        n_pairs = max(n_pairs, min(100, n * (n - 1) // 2))

        # 随机采样配对
        indices_i = torch.randint(0, n, (n_pairs,), device=predictions.device)
        indices_j = torch.randint(0, n, (n_pairs,), device=predictions.device)

        # 确保 i != j
        same_mask = indices_i == indices_j
        indices_j[same_mask] = (indices_j[same_mask] + 1) % n

        # 计算配对损失
        pred_i = predictions[indices_i]
        pred_j = predictions[indices_j]
        target_i = targets[indices_i]
        target_j = targets[indices_j]

        # S_ij: 1 if target_i > target_j, -1 otherwise
        S_ij = torch.sign(target_i - target_j)

        # RankNet loss
        loss = torch.log(1 + torch.exp(-self.sigma * S_ij * (pred_i - pred_j)))

        return loss.mean()


class NDCGLoss(nn.Module):
    """
    可微分 NDCG 损失

    通过软排序近似实现可微分的 NDCG 优化。

    NDCG = DCG / IDCG
    DCG = sum((2^rel - 1) / log2(rank + 1))
    """

    def __init__(self, temperature: float = 1.0, k: int = None):
        """
        Args:
            temperature: softmax 温度参数
            k: 只考虑 top-k (可选)
        """
        super().__init__()
        self.temperature = temperature
        self.k = k

    def _soft_rank(self, scores: torch.Tensor) -> torch.Tensor:
        """计算软排名"""
        n = len(scores)
        # 使用 softmax 实现软排序
        diffs = scores.unsqueeze(0) - scores.unsqueeze(1)  # [n, n]
        soft_ranks = torch.sigmoid(diffs / self.temperature).sum(dim=1)
        return soft_ranks

    def forward(self, predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        计算近似 NDCG 损失

        Args:
            predictions: 预测分数 [batch_size]
            targets: 真实相关度/收益率 [batch_size]
            mask: 有效样本掩码

        Returns:
            1 - NDCG 作为损失
        """
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]

        n = len(predictions)
        if n < 2:
            return torch.tensor(0.0, device=predictions.device)

        # 计算软排名
        soft_ranks = self._soft_rank(predictions)

        # 计算 DCG
        # 将 targets 归一化到 [0, 1]
        targets_norm = (targets - targets.min()) / (targets.max() - targets.min() + 1e-10)
        gains = 2 ** targets_norm - 1
        discounts = torch.log2(soft_ranks + 2)  # +2 因为排名从1开始

        dcg = (gains / discounts).sum()

        # 计算 IDCG (理想 DCG)
        ideal_ranks = torch.argsort(torch.argsort(targets, descending=True)).float() + 1
        ideal_discounts = torch.log2(ideal_ranks + 1)
        idcg = (gains / ideal_discounts).sum()

        ndcg = dcg / (idcg + 1e-10)

        return 1 - ndcg


class ApproxNDCGLoss(nn.Module):
    """
    ApproxNDCG 损失 (LambdaRank 风格)

    使用 sigmoid 近似实现可微分的 NDCG 优化。
    来自 TensorFlow Ranking 的实现。
    """

    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        计算 ApproxNDCG 损失
        """
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]

        n = len(predictions)
        if n < 2:
            return torch.tensor(0.0, device=predictions.device)

        # 计算近似排名
        pred_diffs = predictions.unsqueeze(0) - predictions.unsqueeze(1)
        approx_ranks = torch.sum(torch.sigmoid(-self.alpha * pred_diffs), dim=1) + 0.5

        # 计算增益
        targets_norm = (targets - targets.min()) / (targets.max() - targets.min() + 1e-10)
        gains = 2 ** targets_norm - 1

        # 计算折扣
        discounts = torch.log2(approx_ranks + 1)

        # DCG
        dcg = (gains / discounts).sum()

        # IDCG
        sorted_targets, _ = torch.sort(targets, descending=True)
        sorted_norm = (sorted_targets - sorted_targets.min()) / (sorted_targets.max() - sorted_targets.min() + 1e-10)
        ideal_gains = 2 ** sorted_norm - 1
        ideal_ranks = torch.arange(1, n + 1, device=predictions.device, dtype=torch.float32)
        ideal_discounts = torch.log2(ideal_ranks + 1)
        idcg = (ideal_gains / ideal_discounts).sum()

        return 1 - dcg / (idcg + 1e-10)


class RankMSELoss(nn.Module):
    """
    排名加权 MSE 损失

    对预测误差按排名加权，高收益样本误差权重更大。
    """

    def __init__(self, top_weight: float = 2.0, bottom_weight: float = 0.5):
        super().__init__()
        self.top_weight = top_weight
        self.bottom_weight = bottom_weight

    def forward(self, predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """计算排名加权 MSE"""
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]

        n = len(predictions)
        if n < 1:
            return torch.tensor(0.0, device=predictions.device)

        # 计算排名权重
        ranks = torch.argsort(torch.argsort(targets, descending=True)).float()
        normalized_ranks = ranks / (n - 1) if n > 1 else ranks

        # 线性插值权重: top -> top_weight, bottom -> bottom_weight
        weights = self.top_weight + (self.bottom_weight - self.top_weight) * normalized_ranks

        # 加权 MSE
        mse = (predictions - targets) ** 2
        weighted_mse = (mse * weights).mean()

        return weighted_mse


class CombinedLoss(nn.Module):
    """
    组合损失函数

    结合多个损失函数，支持加权组合。
    """

    def __init__(self, losses: dict):
        """
        Args:
            losses: 损失函数字典 {name: (loss_fn, weight)}
        """
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}

        for name, (loss_fn, weight) in losses.items():
            self.losses[name] = loss_fn
            self.weights[name] = weight

    def forward(self, predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """计算组合损失"""
        total_loss = 0.0

        for name, loss_fn in self.losses.items():
            loss = loss_fn(predictions, targets, mask)
            total_loss += self.weights[name] * loss

        return total_loss


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    获取损失函数

    Args:
        loss_type: 损失类型 ('mse', 'listwise', 'pairwise', 'ndcg', 'approx_ndcg', 'rank_mse')
        **kwargs: 额外参数

    Returns:
        损失函数实例
    """
    loss_map = {
        'mse': nn.MSELoss,
        'listwise': ListwiseLoss,
        'pairwise': PairwiseLoss,
        'ndcg': NDCGLoss,
        'approx_ndcg': ApproxNDCGLoss,
        'rank_mse': RankMSELoss,
    }

    if loss_type not in loss_map:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_map.keys())}")

    return loss_map[loss_type](**kwargs)
