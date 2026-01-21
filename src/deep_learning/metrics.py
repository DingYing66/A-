"""
评估指标模块 (Phase 0)

统一的评估指标体系:
- Rank-IC / IC-IR: 日度排序相关性
- Top-N 超额收益: 选股组合表现
- 成本调整收益: 考虑交易成本后的收益
- 换手率: 组合调仓频率
- Block Bootstrap: 时间序列置信区间
- Transfer Coefficient: 组合实现效率
- 风格归因: 规模/价值/动量/波动归因

参考: López de Prado, Advances in Financial Machine Learning (2018)
"""

from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from scipy import stats


def compute_rank_ic(predictions: np.ndarray,
                    returns: np.ndarray,
                    dates: np.ndarray = None) -> Dict[str, float]:
    """
    计算 Rank-IC (Spearman 相关系数)

    Rank-IC 是预测排名与实际收益排名的相关性,
    是量化因子有效性的核心指标。

    Args:
        predictions: 预测分数数组
        returns: 实际收益数组
        dates: 日期数组（可选，用于计算日度IC）

    Returns:
        字典包含:
        - rank_ic: 总体Rank-IC
        - mean_daily_ic: 日均IC
        - ic_std: IC标准差
        - ic_ir: IC信息比率 (IC均值/标准差)
        - ic_positive_ratio: IC为正的比例
    """
    if len(predictions) != len(returns):
        raise ValueError("predictions 和 returns 长度必须相同")

    # 移除无效值
    valid_mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_valid = predictions[valid_mask]
    ret_valid = returns[valid_mask]

    if len(pred_valid) < 10:
        return {
            'rank_ic': np.nan,
            'mean_daily_ic': np.nan,
            'ic_std': np.nan,
            'ic_ir': np.nan,
            'ic_positive_ratio': np.nan,
        }

    # 总体 Rank-IC
    overall_ic, _ = stats.spearmanr(pred_valid, ret_valid)

    # 如果有日期，计算日度IC
    if dates is not None:
        dates_valid = dates[valid_mask]
        daily_ics = []

        for date in np.unique(dates_valid):
            date_mask = dates_valid == date
            if date_mask.sum() < 20:  # 每天至少20只股票
                continue

            pred_day = pred_valid[date_mask]
            ret_day = ret_valid[date_mask]

            ic, _ = stats.spearmanr(pred_day, ret_day)
            if not np.isnan(ic):
                daily_ics.append(ic)

        if len(daily_ics) > 0:
            mean_ic = np.mean(daily_ics)
            std_ic = np.std(daily_ics)
            ic_ir = mean_ic / (std_ic + 1e-10)
            ic_positive = np.mean([ic > 0 for ic in daily_ics])
        else:
            mean_ic = overall_ic
            std_ic = 0.0
            ic_ir = 0.0
            ic_positive = 0.5
    else:
        mean_ic = overall_ic
        std_ic = 0.0
        ic_ir = 0.0
        ic_positive = 0.5 if overall_ic > 0 else 0.0

    return {
        'rank_ic': overall_ic,
        'mean_daily_ic': mean_ic,
        'ic_std': std_ic,
        'ic_ir': ic_ir,
        'ic_positive_ratio': ic_positive,
    }


def compute_ic_ir(daily_ics: List[float]) -> float:
    """
    计算 IC-IR (信息比率)

    IC-IR = mean(IC) / std(IC)
    衡量因子预测的稳定性

    Args:
        daily_ics: 日度IC列表

    Returns:
        IC-IR 值
    """
    if len(daily_ics) == 0:
        return 0.0

    mean_ic = np.mean(daily_ics)
    std_ic = np.std(daily_ics)

    return mean_ic / (std_ic + 1e-10)


def compute_top_n_excess(predictions: np.ndarray,
                         returns: np.ndarray,
                         dates: np.ndarray,
                         n: int = 20,
                         benchmark_returns: np.ndarray = None) -> Dict[str, float]:
    """
    计算 Top-N 超额收益

    每期选择预测分数最高的N只股票，计算:
    - 组合收益 vs 全市场等权收益
    - 超额收益率
    - 胜率

    Args:
        predictions: 预测分数数组
        returns: 实际收益数组
        dates: 日期数组
        n: 选股数量
        benchmark_returns: 基准收益数组（可选）

    Returns:
        字典包含:
        - top_n_return: Top-N组合累计收益
        - market_return: 市场等权累计收益
        - excess_return: 超额收益
        - win_rate: 胜率（超越市场的比例）
        - avg_period_excess: 单期平均超额
    """
    # 移除无效值
    valid_mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_valid = predictions[valid_mask]
    ret_valid = returns[valid_mask]
    dates_valid = dates[valid_mask]

    unique_dates = np.unique(dates_valid)
    unique_dates = np.sort(unique_dates)

    top_n_returns = []
    market_returns = []
    excess_returns = []

    for date in unique_dates:
        date_mask = dates_valid == date
        pred_day = pred_valid[date_mask]
        ret_day = ret_valid[date_mask]

        if len(pred_day) < n:
            continue

        # Top-N 选股
        top_idx = np.argsort(pred_day)[-n:]
        top_n_ret = np.mean(ret_day[top_idx])

        # 市场等权
        market_ret = np.mean(ret_day)

        # 超额
        excess = top_n_ret - market_ret

        top_n_returns.append(top_n_ret)
        market_returns.append(market_ret)
        excess_returns.append(excess)

    if len(top_n_returns) == 0:
        return {
            'top_n_return': np.nan,
            'market_return': np.nan,
            'excess_return': np.nan,
            'win_rate': np.nan,
            'avg_period_excess': np.nan,
        }

    # 累计收益 (简单加和，非复利)
    cum_top_n = np.sum(top_n_returns)
    cum_market = np.sum(market_returns)
    cum_excess = cum_top_n - cum_market

    # 胜率
    win_rate = np.mean([e > 0 for e in excess_returns])

    # 平均单期超额
    avg_excess = np.mean(excess_returns)

    return {
        'top_n_return': cum_top_n,
        'market_return': cum_market,
        'excess_return': cum_excess,
        'win_rate': win_rate,
        'avg_period_excess': avg_excess,
        'n_periods': len(top_n_returns),
    }


def compute_cost_adjusted_return(predictions: np.ndarray,
                                  returns: np.ndarray,
                                  dates: np.ndarray,
                                  n: int = 20,
                                  commission: float = 0.0003,
                                  stamp_duty: float = 0.001,
                                  slippage: float = 0.001) -> Dict[str, float]:
    """
    计算成本调整后收益

    考虑:
    - 换手成本: 买入佣金 + 卖出佣金 + 印花税
    - 滑点成本
    - 市场冲击成本（暂不考虑）

    Args:
        predictions: 预测分数数组
        returns: 实际收益数组
        dates: 日期数组
        n: 选股数量
        commission: 佣金率 (双向)
        stamp_duty: 印花税 (仅卖出)
        slippage: 滑点

    Returns:
        字典包含:
        - gross_return: 总收益（未扣成本）
        - cost_adjusted_return: 成本调整后收益
        - total_cost: 总交易成本
        - avg_turnover: 平均换手率
    """
    valid_mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_valid = predictions[valid_mask]
    ret_valid = returns[valid_mask]
    dates_valid = dates[valid_mask]

    unique_dates = np.sort(np.unique(dates_valid))

    gross_returns = []
    turnovers = []
    prev_holdings = set()

    for date in unique_dates:
        date_mask = dates_valid == date
        indices = np.where(date_mask)[0]

        if len(indices) < n:
            continue

        pred_day = pred_valid[date_mask]
        ret_day = ret_valid[date_mask]

        # Top-N 选股
        top_local_idx = np.argsort(pred_day)[-n:]
        current_holdings = set(indices[top_local_idx])

        # 计算收益
        gross_ret = np.mean(ret_day[top_local_idx])
        gross_returns.append(gross_ret)

        # 计算换手率
        if len(prev_holdings) > 0:
            new_stocks = len(current_holdings - prev_holdings)
            turnover = new_stocks / n
            turnovers.append(turnover)

        prev_holdings = current_holdings

    if len(gross_returns) == 0:
        return {
            'gross_return': np.nan,
            'cost_adjusted_return': np.nan,
            'total_cost': np.nan,
            'avg_turnover': np.nan,
        }

    # 总收益
    total_gross = np.sum(gross_returns)

    # 平均换手率
    avg_turnover = np.mean(turnovers) if turnovers else 0.0

    # 单次换手成本: 买入(佣金+滑点) + 卖出(佣金+印花税+滑点)
    single_trade_cost = commission + slippage  # 买入
    single_trade_cost += commission + stamp_duty + slippage  # 卖出

    # 总成本 = 每期换手率 * 单次成本 * 期数
    n_periods = len(gross_returns)
    total_cost = avg_turnover * single_trade_cost * n_periods

    # 成本调整后收益
    cost_adjusted = total_gross - total_cost

    return {
        'gross_return': total_gross,
        'cost_adjusted_return': cost_adjusted,
        'total_cost': total_cost,
        'avg_turnover': avg_turnover,
        'n_periods': n_periods,
    }


def compute_turnover(predictions: np.ndarray,
                     dates: np.ndarray,
                     codes: np.ndarray,
                     n: int = 20) -> Dict[str, float]:
    """
    计算组合换手率

    Args:
        predictions: 预测分数数组
        dates: 日期数组
        codes: 股票代码数组
        n: 选股数量

    Returns:
        字典包含:
        - avg_turnover: 平均换手率
        - turnover_std: 换手率标准差
        - max_turnover: 最大换手率
    """
    unique_dates = np.sort(np.unique(dates))
    turnovers = []
    prev_holdings = set()

    for date in unique_dates:
        date_mask = dates == date
        pred_day = predictions[date_mask]
        codes_day = codes[date_mask]

        if len(pred_day) < n:
            continue

        # Top-N 股票代码
        top_idx = np.argsort(pred_day)[-n:]
        current_holdings = set(codes_day[top_idx])

        # 换手率
        if len(prev_holdings) > 0:
            new_stocks = len(current_holdings - prev_holdings)
            turnover = new_stocks / n
            turnovers.append(turnover)

        prev_holdings = current_holdings

    if len(turnovers) == 0:
        return {
            'avg_turnover': np.nan,
            'turnover_std': np.nan,
            'max_turnover': np.nan,
        }

    return {
        'avg_turnover': np.mean(turnovers),
        'turnover_std': np.std(turnovers),
        'max_turnover': np.max(turnovers),
    }


def evaluate_model_comprehensive(predictions: np.ndarray,
                                  returns: np.ndarray,
                                  dates: np.ndarray,
                                  codes: np.ndarray = None,
                                  n: int = 20,
                                  commission: float = 0.0003,
                                  stamp_duty: float = 0.001,
                                  slippage: float = 0.001) -> Dict[str, float]:
    """
    综合评估模型

    整合所有评估指标，输出统一报告

    Args:
        predictions: 预测分数
        returns: 实际收益
        dates: 日期
        codes: 股票代码（可选）
        n: 选股数量
        commission: 佣金率
        stamp_duty: 印花税
        slippage: 滑点

    Returns:
        综合评估指标字典
    """
    results = {}

    # 1. Rank-IC 指标
    ic_metrics = compute_rank_ic(predictions, returns, dates)
    results.update({f'ic_{k}': v for k, v in ic_metrics.items()})

    # 2. Top-N 超额
    excess_metrics = compute_top_n_excess(predictions, returns, dates, n)
    results.update({f'top{n}_{k}': v for k, v in excess_metrics.items()})

    # 3. 成本调整收益
    cost_metrics = compute_cost_adjusted_return(
        predictions, returns, dates, n, commission, stamp_duty, slippage
    )
    results.update({f'cost_{k}': v for k, v in cost_metrics.items()})

    # 4. 换手率（如果有代码）
    if codes is not None:
        turnover_metrics = compute_turnover(predictions, dates, codes, n)
        results.update({f'turnover_{k}': v for k, v in turnover_metrics.items()})

    return results


# ===================== 统计显著性检验 =====================

def compute_ic_significance(daily_ics: List[float],
                            null_hypothesis: float = 0.0) -> Dict[str, float]:
    """
    计算 IC 的统计显著性

    使用 t 检验验证 IC 均值是否显著不为零

    Args:
        daily_ics: 日度 IC 值列表
        null_hypothesis: 原假设的 IC 值 (默认 0)

    Returns:
        字典包含:
        - t_stat: t 统计量
        - p_value: p 值 (双尾)
        - is_significant_95: 95% 置信水平下是否显著
        - is_significant_99: 99% 置信水平下是否显著
        - mean_ic: IC 均值
        - std_error: 标准误
    """
    if len(daily_ics) < 2:
        return {
            't_stat': np.nan,
            'p_value': np.nan,
            'is_significant_95': False,
            'is_significant_99': False,
            'mean_ic': np.nan,
            'std_error': np.nan,
        }

    daily_ics = np.array(daily_ics)
    n = len(daily_ics)
    mean_ic = np.mean(daily_ics)
    std_ic = np.std(daily_ics, ddof=1)  # 样本标准差
    std_error = std_ic / np.sqrt(n)

    # t 统计量
    t_stat = (mean_ic - null_hypothesis) / (std_error + 1e-10)

    # p 值 (双尾 t 检验)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))

    return {
        't_stat': t_stat,
        'p_value': p_value,
        'is_significant_95': p_value < 0.05,
        'is_significant_99': p_value < 0.01,
        'mean_ic': mean_ic,
        'std_error': std_error,
        'n_samples': n,
    }


def bootstrap_ci(data: np.ndarray,
                 statistic_func=np.mean,
                 n_bootstrap: int = 1000,
                 confidence_level: float = 0.95,
                 random_state: int = 42) -> Tuple[float, float, float]:
    """
    Bootstrap 置信区间估计

    Args:
        data: 样本数据
        statistic_func: 统计量函数 (默认均值)
        n_bootstrap: Bootstrap 重抽样次数
        confidence_level: 置信水平
        random_state: 随机种子

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(random_state)
    n = len(data)

    if n < 2:
        point = statistic_func(data) if len(data) > 0 else np.nan
        return point, np.nan, np.nan

    # 计算点估计
    point_estimate = statistic_func(data)

    # Bootstrap 重抽样
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # 置信区间 (百分位法)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return point_estimate, ci_lower, ci_upper


def compute_ic_bootstrap_ci(daily_ics: List[float],
                            n_bootstrap: int = 1000,
                            confidence_level: float = 0.95) -> Dict[str, float]:
    """
    计算 IC 的 Bootstrap 置信区间

    Args:
        daily_ics: 日度 IC 值列表
        n_bootstrap: Bootstrap 次数
        confidence_level: 置信水平

    Returns:
        字典包含均值和置信区间
    """
    daily_ics = np.array(daily_ics)

    # IC 均值的 CI
    mean_ic, mean_lower, mean_upper = bootstrap_ci(
        daily_ics, np.mean, n_bootstrap, confidence_level
    )

    # IC-IR 的 CI
    def ic_ir_stat(x):
        return np.mean(x) / (np.std(x) + 1e-10)

    ic_ir, ir_lower, ir_upper = bootstrap_ci(
        daily_ics, ic_ir_stat, n_bootstrap, confidence_level
    )

    return {
        'mean_ic': mean_ic,
        'mean_ic_ci_lower': mean_lower,
        'mean_ic_ci_upper': mean_upper,
        'ic_ir': ic_ir,
        'ic_ir_ci_lower': ir_lower,
        'ic_ir_ci_upper': ir_upper,
        'confidence_level': confidence_level,
    }


def compute_excess_return_significance(excess_returns: List[float]) -> Dict[str, float]:
    """
    计算超额收益的显著性

    Args:
        excess_returns: 各期超额收益列表

    Returns:
        显著性检验结果
    """
    if len(excess_returns) < 2:
        return {
            't_stat': np.nan,
            'p_value': np.nan,
            'is_significant_95': False,
            'sharpe_ratio': np.nan,
        }

    excess = np.array(excess_returns)

    # t 检验 (均值是否显著为正)
    sig_result = compute_ic_significance(excess.tolist(), null_hypothesis=0.0)

    # 夏普比率 (假设无风险利率 = 0)
    sharpe = np.mean(excess) / (np.std(excess, ddof=1) + 1e-10) * np.sqrt(252)

    # Bootstrap CI for cumulative excess
    cum_excess = np.sum(excess)
    _, cum_lower, cum_upper = bootstrap_ci(excess, np.sum, n_bootstrap=1000)

    return {
        't_stat': sig_result['t_stat'],
        'p_value': sig_result['p_value'],
        'is_significant_95': sig_result['is_significant_95'],
        'is_significant_99': sig_result['is_significant_99'],
        'sharpe_ratio': sharpe,
        'cumulative_excess': cum_excess,
        'cum_excess_ci_lower': cum_lower,
        'cum_excess_ci_upper': cum_upper,
        'mean_excess': np.mean(excess),
        'n_periods': len(excess),
    }


def compare_models(model_a_ics: List[float],
                   model_b_ics: List[float]) -> Dict[str, float]:
    """
    比较两个模型的 IC (配对 t 检验)

    检验 Model A 的 IC 是否显著优于 Model B

    Args:
        model_a_ics: 模型 A 的日度 IC
        model_b_ics: 模型 B 的日度 IC (必须一一对应)

    Returns:
        比较结果
    """
    if len(model_a_ics) != len(model_b_ics):
        raise ValueError("两个模型的 IC 数量必须相同")

    if len(model_a_ics) < 2:
        return {
            't_stat': np.nan,
            'p_value': np.nan,
            'a_is_better': False,
        }

    a = np.array(model_a_ics)
    b = np.array(model_b_ics)
    diff = a - b

    # 配对 t 检验
    t_stat, p_value = stats.ttest_rel(a, b)

    # 单尾检验: A > B
    p_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2

    return {
        't_stat': t_stat,
        'p_value': p_value,
        'p_value_one_sided': p_one_sided,
        'a_is_better_95': p_one_sided < 0.05 and np.mean(diff) > 0,
        'mean_ic_a': np.mean(a),
        'mean_ic_b': np.mean(b),
        'mean_ic_diff': np.mean(diff),
    }


def evaluate_model_with_significance(predictions: np.ndarray,
                                      returns: np.ndarray,
                                      dates: np.ndarray,
                                      codes: np.ndarray = None,
                                      n: int = 20,
                                      n_bootstrap: int = 1000) -> Dict[str, float]:
    """
    综合评估模型 (包含显著性检验)

    Args:
        predictions: 预测分数
        returns: 实际收益
        dates: 日期
        codes: 股票代码
        n: 选股数量
        n_bootstrap: Bootstrap 次数

    Returns:
        包含显著性检验的完整评估结果
    """
    # 基础评估
    results = evaluate_model_comprehensive(
        predictions, returns, dates, codes, n
    )

    # 提取日度 IC
    valid_mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_valid = predictions[valid_mask]
    ret_valid = returns[valid_mask]
    dates_valid = dates[valid_mask]

    daily_ics = []
    for date in np.unique(dates_valid):
        date_mask = dates_valid == date
        if date_mask.sum() >= 20:
            ic, _ = stats.spearmanr(pred_valid[date_mask], ret_valid[date_mask])
            if not np.isnan(ic):
                daily_ics.append(ic)

    # IC 显著性
    if len(daily_ics) > 1:
        ic_sig = compute_ic_significance(daily_ics)
        results.update({f'ic_sig_{k}': v for k, v in ic_sig.items()})

        ic_ci = compute_ic_bootstrap_ci(daily_ics, n_bootstrap)
        results.update({f'ic_ci_{k}': v for k, v in ic_ci.items()})

    # 超额收益显著性
    excess_metrics = compute_top_n_excess(predictions, returns, dates, n)
    if 'n_periods' in excess_metrics and excess_metrics['n_periods'] > 1:
        # 重新计算日度超额用于显著性检验
        unique_dates = np.sort(np.unique(dates_valid))
        excess_returns = []

        for date in unique_dates:
            date_mask = dates_valid == date
            if date_mask.sum() >= n:
                pred_day = pred_valid[date_mask]
                ret_day = ret_valid[date_mask]
                top_idx = np.argsort(pred_day)[-n:]
                top_ret = np.mean(ret_day[top_idx])
                market_ret = np.mean(ret_day)
                excess_returns.append(top_ret - market_ret)

        if len(excess_returns) > 1:
            excess_sig = compute_excess_return_significance(excess_returns)
            results.update({f'excess_sig_{k}': v for k, v in excess_sig.items()})

    return results


def print_evaluation_report(metrics: Dict[str, float], model_name: str = 'Model', n: int = None):
    """打印评估报告

    Args:
        metrics: 评估指标字典
        model_name: 模型名称
        n: 选股数量，如果为 None 则自动从 metrics key 中推断
    """
    # 自动推断 n 值
    if n is None:
        for key in metrics.keys():
            if key.startswith('top') and '_' in key:
                try:
                    n = int(key.split('_')[0][3:])
                    break
                except ValueError:
                    continue
        if n is None:
            n = 20  # 默认值

    print("\n" + "=" * 60)
    print(f"              {model_name} 评估报告")
    print("=" * 60)

    print("\n【Rank-IC 指标】")
    print(f"  Rank-IC:        {metrics.get('ic_rank_ic', np.nan):.4f}")
    print(f"  日均IC:         {metrics.get('ic_mean_daily_ic', np.nan):.4f}")
    print(f"  IC标准差:       {metrics.get('ic_ic_std', np.nan):.4f}")
    print(f"  IC-IR:          {metrics.get('ic_ic_ir', np.nan):.4f}")
    print(f"  IC正比例:       {metrics.get('ic_ic_positive_ratio', np.nan):.2%}")

    # 显著性检验
    if 'ic_sig_t_stat' in metrics:
        print("\n【IC 显著性检验】")
        print(f"  t-stat:         {metrics.get('ic_sig_t_stat', np.nan):.4f}")
        print(f"  p-value:        {metrics.get('ic_sig_p_value', np.nan):.4f}")
        sig_95 = "✓" if metrics.get('ic_sig_is_significant_95', False) else "✗"
        sig_99 = "✓" if metrics.get('ic_sig_is_significant_99', False) else "✗"
        print(f"  95%显著:        {sig_95}")
        print(f"  99%显著:        {sig_99}")

    # Bootstrap CI
    if 'ic_ci_mean_ic_ci_lower' in metrics:
        print("\n【IC Bootstrap 置信区间】")
        print(f"  IC均值 95% CI:  [{metrics.get('ic_ci_mean_ic_ci_lower', np.nan):.4f}, "
              f"{metrics.get('ic_ci_mean_ic_ci_upper', np.nan):.4f}]")
        print(f"  IC-IR 95% CI:   [{metrics.get('ic_ci_ic_ir_ci_lower', np.nan):.4f}, "
              f"{metrics.get('ic_ci_ic_ir_ci_upper', np.nan):.4f}]")

    print(f"\n【Top-{n} 超额收益】")
    print(f"  Top-{n}累计收益: {metrics.get(f'top{n}_top_n_return', np.nan):.4f}")
    print(f"  市场累计收益:   {metrics.get(f'top{n}_market_return', np.nan):.4f}")
    print(f"  超额收益:       {metrics.get(f'top{n}_excess_return', np.nan):.4f}")
    print(f"  胜率:           {metrics.get(f'top{n}_win_rate', np.nan):.2%}")
    print(f"  单期平均超额:   {metrics.get(f'top{n}_avg_period_excess', np.nan):.4f}")

    # 超额收益显著性
    if 'excess_sig_t_stat' in metrics:
        print("\n【超额收益显著性】")
        print(f"  t-stat:         {metrics.get('excess_sig_t_stat', np.nan):.4f}")
        print(f"  p-value:        {metrics.get('excess_sig_p_value', np.nan):.4f}")
        sig_95 = "✓" if metrics.get('excess_sig_is_significant_95', False) else "✗"
        print(f"  95%显著:        {sig_95}")
        print(f"  年化夏普:       {metrics.get('excess_sig_sharpe_ratio', np.nan):.4f}")
        print(f"  累计超额CI:     [{metrics.get('excess_sig_cum_excess_ci_lower', np.nan):.4f}, "
              f"{metrics.get('excess_sig_cum_excess_ci_upper', np.nan):.4f}]")

    print("\n【成本调整收益】")
    print(f"  总收益(未扣):   {metrics.get('cost_gross_return', np.nan):.4f}")
    print(f"  成本调整后:     {metrics.get('cost_cost_adjusted_return', np.nan):.4f}")
    print(f"  总交易成本:     {metrics.get('cost_total_cost', np.nan):.4f}")
    print(f"  平均换手率:     {metrics.get('cost_avg_turnover', np.nan):.2%}")

    if 'turnover_avg_turnover' in metrics:
        print("\n【换手统计】")
        print(f"  平均换手率:     {metrics.get('turnover_avg_turnover', np.nan):.2%}")
        print(f"  换手率标准差:   {metrics.get('turnover_turnover_std', np.nan):.2%}")
        print(f"  最大换手率:     {metrics.get('turnover_max_turnover', np.nan):.2%}")

    # Block Bootstrap 结果
    if 'block_bootstrap_mean_ic' in metrics:
        print("\n【Block Bootstrap IC】")
        print(f"  IC均值:         {metrics.get('block_bootstrap_mean_ic', np.nan):.4f}")
        print(f"  95% CI:         [{metrics.get('block_bootstrap_mean_ic_ci_lower', np.nan):.4f}, "
              f"{metrics.get('block_bootstrap_mean_ic_ci_upper', np.nan):.4f}]")
        print(f"  块大小:         {metrics.get('block_bootstrap_block_size', 10)}")

    # Transfer Coefficient
    if 'tc_mean_tc' in metrics:
        print("\n【Transfer Coefficient】")
        print(f"  平均TC:         {metrics.get('tc_mean_tc', np.nan):.4f}")
        print(f"  TC标准差:       {metrics.get('tc_std', np.nan):.4f}")
        print(f"  TC正比例:       {metrics.get('tc_positive_ratio', np.nan):.2%}")

    # 风格归因
    if 'style_alpha' in metrics:
        print("\n【风格归因】")
        print(f"  Alpha:          {metrics.get('style_alpha', np.nan):.6f}")
        print(f"  Alpha t-stat:   {metrics.get('style_alpha_t', np.nan):.4f}")
        print(f"  R-squared:      {metrics.get('style_r_squared', np.nan):.4f}")
        for factor in ['size', 'value', 'momentum', 'volatility']:
            beta_key = f'style_beta_{factor}'
            if beta_key in metrics:
                print(f"  Beta ({factor}): {metrics.get(beta_key, np.nan):.4f}")

    print("=" * 60)


# ===================== Block Bootstrap (时间序列感知) =====================

def block_bootstrap_ci(data: np.ndarray,
                        statistic_func: Callable = np.mean,
                        block_size: int = 10,
                        n_bootstrap: int = 1000,
                        confidence_level: float = 0.95,
                        random_state: int = 42) -> Dict[str, float]:
    """
    Block Bootstrap 置信区间

    保持时间序列结构的 Bootstrap 方法，适用于存在自相关的数据。
    与标准 Bootstrap 不同，Block Bootstrap 以连续块为单位重抽样，
    保留了时间序列的局部依赖结构。

    建议 block_size 与调仓周期对齐 (如双周调仓用 10)

    Args:
        data: 时间序列数据
        statistic_func: 统计量函数
        block_size: 块大小 (默认 10，对应双周调仓)
        n_bootstrap: 重抽样次数
        confidence_level: 置信水平
        random_state: 随机种子

    Returns:
        字典包含:
        - point_estimate: 点估计
        - ci_lower: 置信区间下界
        - ci_upper: 置信区间上界
        - std: 标准误
        - block_size: 使用的块大小
    """
    np.random.seed(random_state)
    data = np.asarray(data)
    n = len(data)

    if n < block_size:
        raise RuntimeError("insufficient data for block bootstrap")

    # 计算点估计
    point_estimate = statistic_func(data)

    # 创建重叠块
    n_blocks = n - block_size + 1
    blocks = [data[i:i+block_size] for i in range(n_blocks)]

    # 需要采样的块数量 (重建与原序列等长的序列)
    blocks_needed = (n + block_size - 1) // block_size

    # Block Bootstrap
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # 随机采样块索引
        block_indices = np.random.choice(n_blocks, size=blocks_needed, replace=True)

        # 拼接成新序列
        resampled = np.concatenate([blocks[i] for i in block_indices])[:n]

        # 计算统计量
        bootstrap_stats.append(statistic_func(resampled))

    bootstrap_stats = np.array(bootstrap_stats)

    # 置信区间 (百分位法)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return {
        'point_estimate': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std': np.std(bootstrap_stats),
        'block_size': block_size,
        'method': 'block_bootstrap',
    }


def block_bootstrap_ic_ci(daily_ics: List[float],
                           block_size: int = 10,
                           n_bootstrap: int = 1000,
                           confidence_level: float = 0.95) -> Dict[str, float]:
    """
    计算 IC 的 Block Bootstrap 置信区间

    Args:
        daily_ics: 日度 IC 值列表
        block_size: 块大小
        n_bootstrap: Bootstrap 次数
        confidence_level: 置信水平

    Returns:
        IC 和 IC-IR 的置信区间
    """
    daily_ics = np.array(daily_ics)

    # IC 均值的 Block Bootstrap CI
    ic_result = block_bootstrap_ci(
        daily_ics, np.mean, block_size, n_bootstrap, confidence_level
    )

    # IC-IR 的 Block Bootstrap CI
    def ic_ir_stat(x):
        return np.mean(x) / (np.std(x) + 1e-10)

    ir_result = block_bootstrap_ci(
        daily_ics, ic_ir_stat, block_size, n_bootstrap, confidence_level
    )

    return {
        'mean_ic': ic_result['point_estimate'],
        'mean_ic_ci_lower': ic_result['ci_lower'],
        'mean_ic_ci_upper': ic_result['ci_upper'],
        'mean_ic_std': ic_result['std'],
        'ic_ir': ir_result['point_estimate'],
        'ic_ir_ci_lower': ir_result['ci_lower'],
        'ic_ir_ci_upper': ir_result['ci_upper'],
        'block_size': block_size,
        'method': ic_result['method'],
    }


# ===================== Transfer Coefficient =====================

def compute_transfer_coefficient(scores: np.ndarray,
                                  weights: np.ndarray,
                                  returns: np.ndarray) -> float:
    """
    计算 Transfer Coefficient (TC)

    TC = corr(预测分数, 实现贡献)
    实现贡献 = 权重 * 收益

    TC 衡量预测信号转化为实际收益贡献的效率。
    TC = 1 表示完美转化，TC < 1 表示存在约束损耗。

    Args:
        scores: 预测分数
        weights: 组合权重
        returns: 实际收益

    Returns:
        Transfer Coefficient
    """
    # 移除无效值
    valid_mask = ~(np.isnan(scores) | np.isnan(weights) | np.isnan(returns))
    if valid_mask.sum() < 10:
        return np.nan

    scores = scores[valid_mask]
    weights = weights[valid_mask]
    returns = returns[valid_mask]

    # 实现贡献
    contribution = weights * returns

    # 标准化
    scores_z = (scores - scores.mean()) / (scores.std() + 1e-10)
    contrib_z = (contribution - contribution.mean()) / (contribution.std() + 1e-10)

    # 相关系数
    tc = np.corrcoef(scores_z, contrib_z)[0, 1]

    return tc


def compute_tc_by_period(predictions: np.ndarray,
                          returns: np.ndarray,
                          dates: np.ndarray,
                          n: int = 20) -> Dict[str, float]:
    """
    按期计算 Transfer Coefficient

    Args:
        predictions: 预测分数
        returns: 实际收益
        dates: 日期
        n: 选股数量

    Returns:
        TC 统计结果
    """
    valid_mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_valid = predictions[valid_mask]
    ret_valid = returns[valid_mask]
    dates_valid = dates[valid_mask]

    unique_dates = np.sort(np.unique(dates_valid))
    tc_values = []

    for date in unique_dates:
        date_mask = dates_valid == date
        if date_mask.sum() < n:
            continue

        pred_day = pred_valid[date_mask]
        ret_day = ret_valid[date_mask]

        # 计算 Top-N 权重 (等权)
        top_idx = np.argsort(pred_day)[-n:]
        weights = np.zeros(len(pred_day))
        weights[top_idx] = 1.0 / n

        # 计算 TC
        tc = compute_transfer_coefficient(pred_day, weights, ret_day)
        if not np.isnan(tc):
            tc_values.append(tc)

    if len(tc_values) == 0:
        return {
            'mean_tc': np.nan,
            'tc_std': np.nan,
            'tc_positive_ratio': np.nan,
        }

    return {
        'mean_tc': np.mean(tc_values),
        'tc_std': np.std(tc_values),
        'tc_positive_ratio': np.mean([tc > 0 for tc in tc_values]),
        'min_tc': np.min(tc_values),
        'max_tc': np.max(tc_values),
        'n_periods': len(tc_values),
    }


# ===================== 风格归因分析 =====================

def compute_style_attribution(portfolio_returns: np.ndarray,
                               factor_df: pd.DataFrame,
                               style_factors: List[str] = None) -> Dict[str, float]:
    """
    风格归因分析

    使用多因子回归将组合收益归因到不同风格因子:
    r_p = α + Σ β_i * F_i + ε

    默认风格因子:
    - 规模 (size): log(market_cap)
    - 价值 (value): 1/pb 或 ep
    - 动量 (momentum): ret_60d
    - 波动 (volatility): vol_20d

    Args:
        portfolio_returns: 组合收益序列
        factor_df: 包含风格因子的 DataFrame (需含 date, size, value, momentum, volatility)
        style_factors: 自定义风格因子列表

    Returns:
        字典包含:
        - alpha: 回归截距 (选股能力)
        - betas: 各因子暴露
        - t_stats: t 统计量
        - r_squared: 解释度
    """
    # 默认风格因子
    default_factors = ['size', 'value', 'momentum', 'volatility']
    style_factors = style_factors or default_factors

    # 检查因子是否存在
    available_factors = [f for f in style_factors if f in factor_df.columns]
    if len(available_factors) == 0:
        return {
            'alpha': np.nan,
            'r_squared': np.nan,
            'betas': {},
            't_stats': {},
            'error': 'No style factors found in factor_df',
        }

    # 构建回归数据
    # 假设 factor_df 是日度数据，需要按日期聚合
    if 'date' in factor_df.columns:
        factor_returns = factor_df.groupby('date')[available_factors].mean()
    else:
        factor_returns = factor_df[available_factors]

    # 对齐长度
    n = min(len(portfolio_returns), len(factor_returns))
    if n < 10:
        return {
            'alpha': np.nan,
            'r_squared': np.nan,
            'betas': {},
            't_stats': {},
            'error': 'Insufficient data points',
        }

    y = np.array(portfolio_returns[:n])
    X = factor_returns.iloc[:n].values

    # 添加常数项
    X_with_const = np.column_stack([np.ones(n), X])

    try:
        # OLS 回归
        # β = (X'X)^(-1) X'y
        XtX = X_with_const.T @ X_with_const
        XtX_inv = np.linalg.inv(XtX)
        betas_all = XtX_inv @ X_with_const.T @ y

        # 预测和残差
        y_pred = X_with_const @ betas_all
        residuals = y - y_pred

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # 标准误
        mse = ss_res / (n - len(betas_all))
        var_betas = mse * np.diag(XtX_inv)
        se_betas = np.sqrt(var_betas)

        # t 统计量
        t_stats = betas_all / (se_betas + 1e-10)

        # 整理结果
        alpha = betas_all[0]
        betas = dict(zip(available_factors, betas_all[1:]))
        t_stat_dict = {'alpha': t_stats[0]}
        t_stat_dict.update(dict(zip(available_factors, t_stats[1:])))

        # p 值
        p_values = {k: 2 * (1 - stats.t.cdf(abs(v), df=n-len(betas_all)))
                    for k, v in t_stat_dict.items()}

        return {
            'alpha': alpha,
            'alpha_t_stat': t_stats[0],
            'alpha_p_value': p_values['alpha'],
            'alpha_significant': p_values['alpha'] < 0.05,
            'r_squared': r_squared,
            'betas': betas,
            't_stats': t_stat_dict,
            'p_values': p_values,
            'n_observations': n,
        }

    except np.linalg.LinAlgError:
        return {
            'alpha': np.nan,
            'r_squared': np.nan,
            'betas': {},
            't_stats': {},
            'error': 'Singular matrix in regression',
        }


def compute_style_exposure(predictions: np.ndarray,
                            factor_df: pd.DataFrame,
                            dates: np.ndarray,
                            n: int = 20,
                            style_factors: List[str] = None) -> Dict[str, float]:
    """
    计算组合的风格暴露

    根据 Top-N 选股，计算组合在各风格因子上的平均暴露

    Args:
        predictions: 预测分数
        factor_df: 因子 DataFrame
        dates: 日期
        n: 选股数量
        style_factors: 风格因子列表

    Returns:
        风格暴露统计
    """
    default_factors = ['size', 'value', 'momentum', 'volatility']
    style_factors = style_factors or default_factors
    available_factors = [f for f in style_factors if f in factor_df.columns]

    if len(available_factors) == 0 or 'date' not in factor_df.columns:
        return {'error': 'Missing required columns'}

    exposures = {f: [] for f in available_factors}

    unique_dates = np.sort(np.unique(dates))

    for date in unique_dates:
        # 获取当日数据
        date_mask = dates == date
        if date_mask.sum() < n:
            continue

        # 当日因子
        factor_day = factor_df[factor_df['date'] == date]
        if len(factor_day) < n:
            continue

        pred_day = predictions[date_mask]

        # Top-N 选股
        top_idx = np.argsort(pred_day)[-n:]

        # 计算风格暴露 (Top-N 均值 vs 全市场均值的偏离)
        for factor in available_factors:
            if factor in factor_day.columns:
                factor_values = factor_day[factor].values
                if len(factor_values) == len(pred_day):
                    top_mean = np.mean(factor_values[top_idx])
                    market_mean = np.mean(factor_values)
                    market_std = np.std(factor_values) + 1e-10
                    # 标准化暴露
                    exposure = (top_mean - market_mean) / market_std
                    exposures[factor].append(exposure)

    # 汇总
    results = {}
    for factor in available_factors:
        if len(exposures[factor]) > 0:
            results[f'{factor}_exposure_mean'] = np.mean(exposures[factor])
            results[f'{factor}_exposure_std'] = np.std(exposures[factor])
        else:
            results[f'{factor}_exposure_mean'] = np.nan
            results[f'{factor}_exposure_std'] = np.nan

    return results


# ===================== 综合评估 (增强版) =====================

def evaluate_model_enhanced(predictions: np.ndarray,
                             returns: np.ndarray,
                             dates: np.ndarray,
                             codes: np.ndarray = None,
                             factor_df: pd.DataFrame = None,
                             n: int = 20,
                             block_size: int = 10,
                             n_bootstrap: int = 1000) -> Dict[str, float]:
    """
    增强版综合评估

    包含所有评估指标:
    - 基础指标 (IC, 超额收益, 成本)
    - 显著性检验 (t 检验, Bootstrap CI)
    - Block Bootstrap (时序感知)
    - Transfer Coefficient
    - 风格归因 (如提供 factor_df)

    Args:
        predictions: 预测分数
        returns: 实际收益
        dates: 日期
        codes: 股票代码
        factor_df: 因子 DataFrame (可选，用于风格归因)
        n: 选股数量
        block_size: Block Bootstrap 块大小
        n_bootstrap: Bootstrap 次数

    Returns:
        完整评估结果
    """
    # 基础评估 + 显著性
    results = evaluate_model_with_significance(
        predictions, returns, dates, codes, n, n_bootstrap
    )

    # 提取日度 IC 用于 Block Bootstrap
    valid_mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_valid = predictions[valid_mask]
    ret_valid = returns[valid_mask]
    dates_valid = dates[valid_mask]

    daily_ics = []
    for date in np.unique(dates_valid):
        date_mask = dates_valid == date
        if date_mask.sum() >= 20:
            ic, _ = stats.spearmanr(pred_valid[date_mask], ret_valid[date_mask])
            if not np.isnan(ic):
                daily_ics.append(ic)

    # Block Bootstrap IC CI
    if len(daily_ics) > block_size:
        block_ci = block_bootstrap_ic_ci(daily_ics, block_size, n_bootstrap)
        results.update({f'block_bootstrap_{k}': v for k, v in block_ci.items()})

    # Transfer Coefficient
    tc_result = compute_tc_by_period(predictions, returns, dates, n)
    results.update({f'tc_{k}': v for k, v in tc_result.items()})

    # 风格归因 (如果提供因子数据)
    if factor_df is not None and len(factor_df) > 0:
        # 计算组合日收益用于风格归因
        unique_dates = np.sort(np.unique(dates_valid))
        portfolio_returns = []
        for date in unique_dates:
            date_mask = dates_valid == date
            if date_mask.sum() >= n:
                pred_day = pred_valid[date_mask]
                ret_day = ret_valid[date_mask]
                top_idx = np.argsort(pred_day)[-n:]
                portfolio_returns.append(np.mean(ret_day[top_idx]))

        if len(portfolio_returns) > 10:
            style_result = compute_style_attribution(
                np.array(portfolio_returns), factor_df
            )
            if 'error' not in style_result:
                results['style_alpha'] = style_result['alpha']
                results['style_alpha_t'] = style_result.get('alpha_t_stat', np.nan)
                results['style_r_squared'] = style_result['r_squared']
                for factor, beta in style_result.get('betas', {}).items():
                    results[f'style_beta_{factor}'] = beta

        # 风格暴露
        exposure_result = compute_style_exposure(predictions, factor_df, dates, n)
        results.update(exposure_result)

    return results


# ===================== RL 策略评估 =====================

def evaluate_rl_strategy(weights_history: List[np.ndarray],
                          returns_history: List[float],
                          baseline_returns: List[float] = None,
                          dates: List = None) -> Dict[str, float]:
    """
    评估 RL 策略表现

    专门用于评估 RL 仓位管理策略的效果，包括:
    - 基础收益指标 (总收益、年化收益、波动率、Sharpe)
    - 风险指标 (最大回撤、Calmar、Sortino)
    - 换手率指标 (平均/最大换手)
    - 与基线对比 (超额收益、信息比率、胜率)

    Args:
        weights_history: 每期权重列表 [np.array(n_stocks), ...]
        returns_history: 每期组合收益列表 [0.01, -0.02, ...]
        baseline_returns: 基线 (如 Top-N 等权) 收益列表 (可选)
        dates: 日期列表 (可选，用于报告)

    Returns:
        评估指标字典
    """
    returns = np.array(returns_history)
    n_periods = len(returns)

    if n_periods < 2:
        return {
            'total_return': np.nan,
            'annual_return': np.nan,
            'volatility': np.nan,
            'sharpe_ratio': np.nan,
            'max_drawdown': np.nan,
            'error': 'Insufficient data',
        }

    # ========== 基础收益指标 ==========
    # 总收益 (复利)
    cumulative = np.cumprod(1 + returns)
    total_return = cumulative[-1] - 1

    # 年化收益 (假设每期为日度)
    annualization_factor = 252 / n_periods if n_periods < 252 else 252
    annual_return = (1 + total_return) ** (annualization_factor / n_periods) - 1

    # 波动率 (年化)
    volatility = np.std(returns) * np.sqrt(252)

    # Sharpe (假设无风险利率 = 0)
    sharpe = annual_return / volatility if volatility > 1e-8 else 0

    # ========== 风险指标 ==========
    # 最大回撤
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    max_drawdown = np.max(drawdown)

    # Calmar Ratio (年化收益 / 最大回撤)
    calmar = annual_return / max_drawdown if max_drawdown > 1e-8 else 0

    # Sortino Ratio (下行波动率)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-10
    sortino = annual_return / downside_std

    # ========== 换手率指标 ==========
    turnovers = []
    if len(weights_history) > 1:
        for i in range(1, len(weights_history)):
            prev_w = weights_history[i-1]
            curr_w = weights_history[i]

            # 确保长度一致
            min_len = min(len(prev_w), len(curr_w))
            if min_len > 0:
                # 换手率 = 权重变化绝对值之和 / 2
                turnover = np.sum(np.abs(curr_w[:min_len] - prev_w[:min_len])) / 2
                turnovers.append(turnover)

    avg_turnover = np.mean(turnovers) if turnovers else 0
    max_turnover = np.max(turnovers) if turnovers else 0
    turnover_std = np.std(turnovers) if turnovers else 0

    # ========== 持仓集中度 ==========
    # 平均有效持仓数 (使用 HHI 指数的倒数)
    effective_n_stocks = []
    for w in weights_history:
        w_nonzero = w[w > 1e-8]
        if len(w_nonzero) > 0:
            hhi = np.sum(w_nonzero ** 2)
            effective_n = 1 / hhi if hhi > 1e-8 else len(w_nonzero)
            effective_n_stocks.append(effective_n)

    avg_effective_n = np.mean(effective_n_stocks) if effective_n_stocks else 0

    # ========== 结果汇总 ==========
    result = {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'sortino_ratio': sortino,
        'avg_turnover': avg_turnover,
        'max_turnover': max_turnover,
        'turnover_std': turnover_std,
        'avg_effective_n_stocks': avg_effective_n,
        'n_periods': n_periods,
    }

    # ========== 与基线对比 ==========
    if baseline_returns is not None:
        baseline = np.array(baseline_returns)

        if len(baseline) == len(returns):
            # 超额收益
            excess = returns - baseline
            excess_cumulative = np.prod(1 + excess) - 1
            result['excess_return'] = excess_cumulative

            # 信息比率
            tracking_error = np.std(excess) * np.sqrt(252)
            info_ratio = np.mean(excess) * 252 / tracking_error if tracking_error > 1e-8 else 0
            result['information_ratio'] = info_ratio

            # 胜率 (跑赢基线的比例)
            win_rate = np.mean(returns > baseline)
            result['win_rate'] = win_rate

            # 平均超额
            result['avg_excess'] = np.mean(excess)

            # 基线统计
            baseline_total = np.prod(1 + baseline) - 1
            result['baseline_total_return'] = baseline_total

    # ========== 收益分布统计 ==========
    result['return_mean'] = np.mean(returns)
    result['return_std'] = np.std(returns)
    result['return_skew'] = stats.skew(returns) if len(returns) > 2 else 0
    result['return_kurtosis'] = stats.kurtosis(returns) if len(returns) > 3 else 0

    # 正收益比例
    result['positive_return_ratio'] = np.mean(returns > 0)

    # 最佳/最差单期收益
    result['best_period_return'] = np.max(returns)
    result['worst_period_return'] = np.min(returns)

    return result


def print_rl_evaluation_report(metrics: Dict[str, float],
                                 strategy_name: str = 'RL Strategy'):
    """打印 RL 策略评估报告"""
    print("\n" + "=" * 70)
    print(f"              {strategy_name} 评估报告")
    print("=" * 70)

    print("\n【收益指标】")
    print(f"  总收益:         {metrics.get('total_return', np.nan):.4%}")
    print(f"  年化收益:       {metrics.get('annual_return', np.nan):.4%}")
    print(f"  年化波动率:     {metrics.get('volatility', np.nan):.4%}")
    print(f"  Sharpe Ratio:   {metrics.get('sharpe_ratio', np.nan):.4f}")

    print("\n【风险指标】")
    print(f"  最大回撤:       {metrics.get('max_drawdown', np.nan):.4%}")
    print(f"  Calmar Ratio:   {metrics.get('calmar_ratio', np.nan):.4f}")
    print(f"  Sortino Ratio:  {metrics.get('sortino_ratio', np.nan):.4f}")

    print("\n【换手指标】")
    print(f"  平均换手率:     {metrics.get('avg_turnover', np.nan):.4%}")
    print(f"  最大换手率:     {metrics.get('max_turnover', np.nan):.4%}")
    print(f"  换手率标准差:   {metrics.get('turnover_std', np.nan):.4%}")

    print("\n【持仓特征】")
    print(f"  平均有效持仓数: {metrics.get('avg_effective_n_stocks', np.nan):.2f}")
    print(f"  评估期数:       {metrics.get('n_periods', 0)}")

    # 基线对比
    if 'excess_return' in metrics:
        print("\n【与基线对比】")
        print(f"  基线总收益:     {metrics.get('baseline_total_return', np.nan):.4%}")
        print(f"  超额收益:       {metrics.get('excess_return', np.nan):.4%}")
        print(f"  信息比率:       {metrics.get('information_ratio', np.nan):.4f}")
        print(f"  胜率:           {metrics.get('win_rate', np.nan):.2%}")
        print(f"  平均超额:       {metrics.get('avg_excess', np.nan):.6f}")

    print("\n【收益分布】")
    print(f"  平均收益:       {metrics.get('return_mean', np.nan):.6f}")
    print(f"  收益标准差:     {metrics.get('return_std', np.nan):.6f}")
    print(f"  偏度:           {metrics.get('return_skew', np.nan):.4f}")
    print(f"  峰度:           {metrics.get('return_kurtosis', np.nan):.4f}")
    print(f"  正收益比例:     {metrics.get('positive_return_ratio', np.nan):.2%}")
    print(f"  最佳单期收益:   {metrics.get('best_period_return', np.nan):.4%}")
    print(f"  最差单期收益:   {metrics.get('worst_period_return', np.nan):.4%}")

    print("=" * 70)
