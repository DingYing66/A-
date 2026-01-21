"""
绩效分析模块
负责计算和展示回测绩效指标
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .utils import (
    load_config, setup_logger, get_project_root, ensure_dir,
    calc_max_drawdown, calc_max_drawdown_duration, format_pct, format_number
)

logger = setup_logger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class PerformanceAnalyzer:
    """绩效分析器"""

    def __init__(self, config: dict = None):
        """
        初始化绩效分析器

        Args:
            config: 配置字典
        """
        self.config = config or load_config()
        self.root = get_project_root()

    def calc_annual_return(self, nav: pd.Series) -> float:
        """
        计算年化收益率

        Args:
            nav: 净值序列

        Returns:
            年化收益率
        """
        if len(nav) < 2:
            return 0.0

        total_return = nav.iloc[-1] / nav.iloc[0] - 1
        days = (nav.index[-1] - nav.index[0]).days

        if days <= 0:
            return 0.0

        annual_return = (1 + total_return) ** (365 / days) - 1
        return annual_return

    def calc_annual_volatility(self, returns: pd.Series) -> float:
        """
        计算年化波动率

        Args:
            returns: 日收益率序列

        Returns:
            年化波动率
        """
        if len(returns) < 2:
            return 0.0

        return returns.std() * np.sqrt(252)

    def calc_sharpe_ratio(self, returns: pd.Series, rf: float = 0.02) -> float:
        """
        计算夏普比率

        Args:
            returns: 日收益率序列
            rf: 无风险利率（年化）

        Returns:
            夏普比率
        """
        if len(returns) < 2:
            return 0.0

        excess_return = returns.mean() * 252 - rf
        volatility = returns.std() * np.sqrt(252)

        if volatility == 0:
            return 0.0

        return excess_return / volatility

    def calc_sortino_ratio(self, returns: pd.Series, rf: float = 0.02) -> float:
        """
        计算索提诺比率（只考虑下行波动）

        Args:
            returns: 日收益率序列
            rf: 无风险利率（年化）

        Returns:
            索提诺比率
        """
        if len(returns) < 2:
            return 0.0

        excess_return = returns.mean() * 252 - rf
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return np.inf

        downside_std = downside_returns.std() * np.sqrt(252)

        if downside_std == 0:
            return 0.0

        return excess_return / downside_std

    def calc_calmar_ratio(self, nav: pd.Series) -> float:
        """
        计算卡尔玛比率（年化收益/最大回撤）

        Args:
            nav: 净值序列

        Returns:
            卡尔玛比率
        """
        annual_return = self.calc_annual_return(nav)
        max_dd = abs(calc_max_drawdown(nav))

        if max_dd == 0:
            return np.inf

        return annual_return / max_dd

    def calc_win_rate(self, returns: pd.Series) -> float:
        """
        计算胜率

        Args:
            returns: 收益率序列

        Returns:
            胜率
        """
        if len(returns) == 0:
            return 0.0

        wins = (returns > 0).sum()
        return wins / len(returns)

    def calc_profit_loss_ratio(self, returns: pd.Series) -> float:
        """
        计算盈亏比

        Args:
            returns: 收益率序列

        Returns:
            盈亏比
        """
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(losses) == 0 or losses.mean() == 0:
            return np.inf

        return abs(wins.mean() / losses.mean()) if len(wins) > 0 else 0

    def calc_information_ratio(self, returns: pd.Series,
                                benchmark_returns: pd.Series) -> float:
        """
        计算信息比率

        Args:
            returns: 策略收益率
            benchmark_returns: 基准收益率

        Returns:
            信息比率
        """
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)

        if tracking_error == 0:
            return 0.0

        return excess_returns.mean() * 252 / tracking_error

    def calc_beta(self, returns: pd.Series,
                  benchmark_returns: pd.Series) -> float:
        """
        计算Beta

        Args:
            returns: 策略收益率
            benchmark_returns: 基准收益率

        Returns:
            Beta值
        """
        if len(returns) < 2:
            return 0.0

        cov = returns.cov(benchmark_returns)
        var = benchmark_returns.var()

        if var == 0:
            return 0.0

        return cov / var

    def calc_alpha(self, returns: pd.Series,
                   benchmark_returns: pd.Series,
                   rf: float = 0.02) -> float:
        """
        计算Alpha（年化）

        Args:
            returns: 策略收益率
            benchmark_returns: 基准收益率
            rf: 无风险利率

        Returns:
            Alpha值
        """
        beta = self.calc_beta(returns, benchmark_returns)
        strategy_return = returns.mean() * 252
        benchmark_return = benchmark_returns.mean() * 252

        alpha = strategy_return - rf - beta * (benchmark_return - rf)
        return alpha

    def generate_report(self, backtest_result: pd.DataFrame) -> Dict:
        """
        生成完整绩效报告

        Args:
            backtest_result: 回测结果DataFrame

        Returns:
            绩效指标字典
        """
        report = {}

        nav = backtest_result['nav']
        returns = backtest_result['daily_return'].dropna()

        # 基础指标
        report['total_return'] = nav.iloc[-1] / nav.iloc[0] - 1
        report['annual_return'] = self.calc_annual_return(nav)
        report['annual_volatility'] = self.calc_annual_volatility(returns)
        report['sharpe_ratio'] = self.calc_sharpe_ratio(returns)
        report['sortino_ratio'] = self.calc_sortino_ratio(returns)
        report['max_drawdown'] = calc_max_drawdown(nav)
        report['max_drawdown_duration'] = calc_max_drawdown_duration(nav)
        report['calmar_ratio'] = self.calc_calmar_ratio(nav)

        # 交易统计
        report['win_rate'] = self.calc_win_rate(returns)
        report['profit_loss_ratio'] = self.calc_profit_loss_ratio(returns)
        report['trading_days'] = len(returns)

        # 相对基准指标
        if 'benchmark' in backtest_result.columns:
            benchmark_nav = backtest_result['benchmark'].dropna()
            benchmark_returns = backtest_result['benchmark_return'].dropna()

            # 对齐日期
            common_idx = returns.index.intersection(benchmark_returns.index)
            returns_aligned = returns.loc[common_idx]
            benchmark_aligned = benchmark_returns.loc[common_idx]

            report['benchmark_return'] = benchmark_nav.iloc[-1] / benchmark_nav.iloc[0] - 1
            report['benchmark_annual_return'] = self.calc_annual_return(benchmark_nav)
            report['excess_return'] = report['annual_return'] - report['benchmark_annual_return']
            report['information_ratio'] = self.calc_information_ratio(
                returns_aligned, benchmark_aligned
            )
            report['beta'] = self.calc_beta(returns_aligned, benchmark_aligned)
            report['alpha'] = self.calc_alpha(returns_aligned, benchmark_aligned)

        # 分年度统计
        nav_with_year = nav.copy()
        nav_with_year.index = pd.to_datetime(nav_with_year.index)
        yearly_returns = {}

        for year in nav_with_year.index.year.unique():
            year_data = nav_with_year[nav_with_year.index.year == year]
            if len(year_data) > 1:
                yearly_returns[year] = year_data.iloc[-1] / year_data.iloc[0] - 1

        report['yearly_returns'] = yearly_returns

        return report

    def print_report(self, report: Dict) -> None:
        """
        打印绩效报告

        Args:
            report: 绩效指标字典
        """
        print("\n" + "=" * 60)
        print("                    绩效报告")
        print("=" * 60)

        print("\n【收益指标】")
        print(f"  总收益率:          {format_pct(report['total_return'])}")
        print(f"  年化收益率:        {format_pct(report['annual_return'])}")
        print(f"  年化波动率:        {format_pct(report['annual_volatility'])}")

        if 'benchmark_return' in report:
            print(f"  基准总收益率:      {format_pct(report['benchmark_return'])}")
            print(f"  基准年化收益率:    {format_pct(report['benchmark_annual_return'])}")
            print(f"  超额收益(年化):    {format_pct(report['excess_return'])}")

        print("\n【风险指标】")
        print(f"  最大回撤:          {format_pct(report['max_drawdown'])}")
        print(f"  最大回撤持续(天):  {report['max_drawdown_duration']}")
        print(f"  夏普比率:          {format_number(report['sharpe_ratio'])}")
        print(f"  索提诺比率:        {format_number(report['sortino_ratio'])}")
        print(f"  卡尔玛比率:        {format_number(report['calmar_ratio'])}")

        if 'alpha' in report:
            print(f"  Alpha(年化):       {format_pct(report['alpha'])}")
            print(f"  Beta:              {format_number(report['beta'])}")
            print(f"  信息比率:          {format_number(report['information_ratio'])}")

        print("\n【交易统计】")
        print(f"  胜率:              {format_pct(report['win_rate'])}")
        print(f"  盈亏比:            {format_number(report['profit_loss_ratio'])}")
        print(f"  交易天数:          {report['trading_days']}")

        if 'yearly_returns' in report and report['yearly_returns']:
            print("\n【分年度收益】")
            for year, ret in sorted(report['yearly_returns'].items()):
                print(f"  {year}年:            {format_pct(ret)}")

        print("\n" + "=" * 60)

    def plot_nav_curve(self, backtest_result: pd.DataFrame,
                       title: str = "策略净值曲线",
                       save_path: str = None) -> None:
        """
        绘制净值曲线

        Args:
            backtest_result: 回测结果
            title: 图表标题
            save_path: 保存路径
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        nav = backtest_result['nav']
        dates = pd.to_datetime(nav.index)

        # 1. 净值曲线
        ax1 = axes[0]
        ax1.plot(dates, nav.values, label='策略', linewidth=1.5, color='#1f77b4')

        if 'benchmark' in backtest_result.columns:
            benchmark = backtest_result['benchmark']
            ax1.plot(dates, benchmark.values, label='基准', linewidth=1.5,
                     color='#ff7f0e', alpha=0.7)

        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('净值')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # 2. 回撤曲线
        ax2 = axes[1]
        peak = nav.expanding().max()
        drawdown = (nav - peak) / peak

        ax2.fill_between(dates, drawdown.values, 0, color='red', alpha=0.3)
        ax2.plot(dates, drawdown.values, color='red', linewidth=1)
        ax2.set_ylabel('回撤')
        ax2.set_title('回撤曲线')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # 3. 超额收益
        ax3 = axes[2]
        if 'excess_return' in backtest_result.columns:
            excess = backtest_result['excess_return'].dropna()
            cum_excess = (1 + excess).cumprod()
            ax3.plot(cum_excess.index, cum_excess.values,
                     color='green', linewidth=1.5)
            ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
            ax3.set_title('累计超额收益')
        else:
            # 日收益分布
            returns = backtest_result['daily_return'].dropna()
            ax3.hist(returns, bins=50, color='#1f77b4', alpha=0.7, edgecolor='white')
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax3.set_title('日收益分布')

        ax3.set_ylabel('累计超额' if 'excess_return' in backtest_result.columns else '频数')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"图表已保存: {save_path}")

        plt.show()

    def plot_monthly_returns(self, backtest_result: pd.DataFrame,
                              save_path: str = None) -> None:
        """
        绘制月度收益热力图

        Args:
            backtest_result: 回测结果
            save_path: 保存路径
        """
        nav = backtest_result['nav']
        nav.index = pd.to_datetime(nav.index)

        # 计算月度收益
        monthly_nav = nav.resample('M').last()
        monthly_returns = monthly_nav.pct_change().dropna()

        # 创建年-月矩阵
        years = monthly_returns.index.year.unique()
        months = range(1, 13)

        data = pd.DataFrame(index=years, columns=months)

        for date, ret in monthly_returns.items():
            data.loc[date.year, date.month] = ret

        data = data.astype(float)

        # 绘制热力图
        fig, ax = plt.subplots(figsize=(12, max(4, len(years) * 0.5)))

        import seaborn as sns
        sns.heatmap(data, annot=True, fmt='.1%', cmap='RdYlGn',
                    center=0, ax=ax, cbar_kws={'label': '月度收益率'})

        ax.set_title('月度收益热力图', fontsize=14, fontweight='bold')
        ax.set_xlabel('月份')
        ax.set_ylabel('年份')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def plot_holdings_analysis(self, trades_df: pd.DataFrame,
                                save_path: str = None) -> None:
        """
        绘制持仓分析图

        Args:
            trades_df: 交易记录
            save_path: 保存路径
        """
        if len(trades_df) == 0:
            logger.warning("无交易记录")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 交易金额分布
        ax1 = axes[0, 0]
        trades_df['amount'].hist(bins=30, ax=ax1, color='#1f77b4', alpha=0.7)
        ax1.set_title('交易金额分布')
        ax1.set_xlabel('金额')
        ax1.set_ylabel('频数')

        # 2. 买卖比例
        ax2 = axes[0, 1]
        direction_counts = trades_df['direction'].value_counts()
        ax2.pie(direction_counts, labels=direction_counts.index,
                autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
        ax2.set_title('买卖比例')

        # 3. 日交易金额
        ax3 = axes[1, 0]
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        daily_amount = trades_df.groupby('date')['amount'].sum()
        ax3.bar(daily_amount.index, daily_amount.values, color='#3498db', alpha=0.7)
        ax3.set_title('日交易金额')
        ax3.set_xlabel('日期')
        ax3.set_ylabel('金额')
        ax3.tick_params(axis='x', rotation=45)

        # 4. 交易成本累计
        ax4 = axes[1, 1]
        trades_df['cum_cost'] = trades_df['cost'].cumsum()
        ax4.plot(trades_df['date'], trades_df['cum_cost'],
                 color='#e74c3c', linewidth=1.5)
        ax4.set_title('累计交易成本')
        ax4.set_xlabel('日期')
        ax4.set_ylabel('成本')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def save_report(self, report: Dict, backtest_result: pd.DataFrame,
                    name: str = None) -> None:
        """
        保存完整报告（含图表）

        Args:
            report: 绩效指标
            backtest_result: 回测结果
            name: 报告名称
        """
        output_dir = ensure_dir(self.root / 'output' / 'reports')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = name or f"report_{timestamp}"

        # 保存绩效指标
        report_path = output_dir / f"{name}_metrics.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("绩效报告\n")
            f.write("=" * 60 + "\n\n")

            for key, value in report.items():
                if key == 'yearly_returns':
                    f.write("\n分年度收益:\n")
                    for year, ret in sorted(value.items()):
                        f.write(f"  {year}: {format_pct(ret)}\n")
                elif isinstance(value, float):
                    if 'ratio' in key.lower() or 'beta' in key.lower() or 'alpha' in key.lower():
                        f.write(f"{key}: {format_number(value)}\n")
                    else:
                        f.write(f"{key}: {format_pct(value)}\n")
                else:
                    f.write(f"{key}: {value}\n")

        # 保存图表
        nav_path = output_dir / f"{name}_nav.png"
        self.plot_nav_curve(backtest_result, save_path=str(nav_path))

        logger.info(f"报告已保存: {output_dir / name}")


if __name__ == "__main__":
    # 测试代码
    analyzer = PerformanceAnalyzer()

    # 创建模拟回测结果
    dates = pd.date_range('2020-01-01', '2024-12-01', freq='D')
    np.random.seed(42)

    # 模拟净值
    returns = np.random.randn(len(dates)) * 0.02 + 0.0003
    nav = (1 + pd.Series(returns, index=dates)).cumprod()

    # 模拟基准
    benchmark_returns = np.random.randn(len(dates)) * 0.015 + 0.0002
    benchmark = (1 + pd.Series(benchmark_returns, index=dates)).cumprod()

    result_df = pd.DataFrame({
        'nav': nav,
        'benchmark': benchmark,
        'daily_return': pd.Series(returns, index=dates),
        'benchmark_return': pd.Series(benchmark_returns, index=dates)
    })
    result_df['excess_return'] = result_df['daily_return'] - result_df['benchmark_return']

    # 生成报告
    report = analyzer.generate_report(result_df)
    analyzer.print_report(report)
