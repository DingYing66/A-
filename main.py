"""
A股多因子选股系统 - 主程序入口

使用方法:
    1. 数据下载: python main.py download
    2. 因子计算: python main.py compute-factors [--freq daily] [--workers 4]
    3. 运行回测: python main.py backtest
    4. 生成信号: python main.py signal
    5. 完整流程: python main.py run
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.data_fetcher import DataFetcher
from src.data_processor import DataProcessor
from src.factor_engine import FactorEngine
from src.scorer import FactorScorer
from src.ml_model import MLModel, EnsembleModel
from src.backtester import Backtester, WalkForwardBacktester
from src.performance import PerformanceAnalyzer
from src.adaptive_weights import AdaptiveWeightManager
from src.utils import load_config, setup_logger, ensure_dir, get_project_root

logger = setup_logger('main')


def download_data(args):
    """数据下载"""
    logger.info("=" * 60)
    logger.info("开始数据下载")
    logger.info("=" * 60)

    fetcher = DataFetcher()

    codes = getattr(args, 'codes', None)
    if codes:
        # 下载指定股票
        codes = codes.split(',')
        fetcher.run_full_download(codes)
    else:
        # 下载全市场
        fetcher.run_full_download()

    logger.info("数据下载完成!")


def run_backtest(args):
    """运行回测"""
    logger.info("=" * 60)
    logger.info("开始回测")
    logger.info("=" * 60)

    config = load_config()

    # 获取回测参数
    start_date = args.start or config['backtest']['backtest_start']
    end_date = args.end or config['backtest']['backtest_end']
    top_n = args.top_n or config['backtest']['top_n']

    logger.info(f"回测区间: {start_date} ~ {end_date}")
    logger.info(f"选股数量: {top_n}")

    if args.walk_forward:
        # 滚动回测
        bt = WalkForwardBacktester(config)
        result = bt.run_walk_forward(
            start_date=start_date,
            end_date=end_date,
            model_type=args.model or 'xgboost'
        )
    else:
        # 普通回测
        bt = Backtester(config)
        bt.top_n = top_n

        if args.model:
            # 使用ML模型
            logger.info(f"使用ML模型: {args.model}")
            ml = MLModel(config)

            # 训练模型
            train_end = start_date
            train_start = str(int(train_end[:4]) - 3) + train_end[4:]
            X, y = ml.build_training_samples(train_start, train_end)

            if len(X) > 100:
                ml.train(X, y, model_type=args.model)

                def ml_score_func(factor_df):
                    return ml.select_stocks_ml(factor_df, n=top_n)

                result = bt.run(start_date, end_date, score_func=ml_score_func)
            else:
                logger.warning("训练样本不足，使用多因子打分")
                result = bt.run(start_date, end_date)
        else:
            result = bt.run(start_date, end_date)

    if len(result) == 0:
        logger.error("回测失败，无结果")
        return

    # 绩效分析
    analyzer = PerformanceAnalyzer(config)
    report = analyzer.generate_report(result)
    analyzer.print_report(report)

    # 保存结果
    bt.save_results(result, args.name)

    # 绘图
    if not args.no_plot:
        analyzer.plot_nav_curve(result)

    logger.info("回测完成!")


def generate_signal(args):
    """生成选股信号（含仓位管理）"""
    logger.info("=" * 60)
    logger.info("生成选股信号")
    logger.info("=" * 60)

    config = load_config()
    root = get_project_root()

    # 计算当前因子
    factor_engine = FactorEngine(config)
    scorer = FactorScorer(config)

    # 自适应权重管理器
    adaptive_manager = AdaptiveWeightManager(config)
    use_adaptive = config.get('adaptive_weights', {}).get('enabled', False)

    # 使用最新数据
    processor = DataProcessor(config)
    calendar = processor.trade_calendar

    # 获取最近的交易日
    today = datetime.now()
    recent_dates = calendar[calendar <= today]
    if len(recent_dates) == 0:
        logger.error("无可用交易日")
        return

    latest_date = recent_dates[-1]
    logger.info(f"信号日期: {latest_date.date()}")

    # 计算因子（生成信号不需要前向收益率）
    factor_df = factor_engine.compute_factor_cross_section(latest_date, add_forward_return=False)
    if len(factor_df) == 0:
        logger.error("因子计算失败")
        return

    factor_df = factor_engine.standardize_factors(factor_df)

    # ========== 市场环境分析 ==========
    market_trend = factor_df['market_trend'].iloc[0] if 'market_trend' in factor_df.columns else 0
    market_breadth = factor_df['market_breadth'].iloc[0] if 'market_breadth' in factor_df.columns else 0.5
    market_volatility = factor_df['market_volatility'].iloc[0] if 'market_volatility' in factor_df.columns else 0.2

    # 获取自适应权重和市场状态
    if use_adaptive:
        current_weights, market_status = adaptive_manager.get_weights_from_factor_df(factor_df)
        regime_info = adaptive_manager.get_regime_info(market_trend, market_breadth, market_volatility)
    else:
        current_weights = config['weights']
        market_status = "均衡型(默认)"
        regime_info = None

    # 计算建议仓位比例
    position_ratio = 1.0
    risk_level = "正常"

    if market_trend < -0.05:  # 市场20日跌超5%
        position_ratio *= 0.6
        risk_level = "高风险"
    elif market_trend < 0:
        position_ratio *= 0.8
        risk_level = "中风险"

    if market_breadth < 0.3:  # 赚钱效应差
        position_ratio *= 0.7
        risk_level = "高风险" if risk_level != "高风险" else risk_level

    if market_volatility > 0.3:  # 高波动
        position_ratio *= 0.8

    position_ratio = max(0.3, min(1.0, position_ratio))

    # ========== 选股 ==========
    top_n = args.top_n or config['backtest']['top_n']

    if args.model and args.model_path:
        # 使用已训练的ML模型
        ml = MLModel(config)
        ml.load_model(args.model_path)
        selected = ml.select_stocks_ml(factor_df, n=top_n)
    else:
        # 使用自适应权重选股
        selected = scorer.select_top_stocks(factor_df, n=top_n, use_adaptive=use_adaptive)

    # ========== 计算权重 ==========
    # 检查空选股情况
    if len(selected) == 0:
        logger.warning(f"选股结果为空，无法生成信号")
        print("\n" + "=" * 70)
        print("警告: 选股结果为空，无符合条件的股票")
        print("=" * 70)
        return None

    # 基础等权
    base_weight = 1.0 / len(selected)
    max_position = config['backtest'].get('max_position', 0.08)

    # 应用仓位比例调整
    adjusted_weight = min(base_weight, max_position) * position_ratio
    selected = selected.copy()
    selected['weight'] = adjusted_weight
    selected['position_ratio'] = position_ratio

    # 计算实际总仓位
    total_position = adjusted_weight * len(selected)

    # ========== 行业分布统计 ==========
    industry_dist = {}
    if 'industry' in selected.columns:
        industry_dist = selected.groupby('industry').size().to_dict()

    # ========== 输出结果 ==========
    print("\n" + "=" * 70)
    print("                      选 股 信 号 报 告")
    print("=" * 70)

    # 市场环境
    print(f"\n【市场环境分析】")
    print(f"  市场趋势(20日): {market_trend:+.2%}")
    print(f"  市场广度(赚钱效应): {market_breadth:.1%}")
    print(f"  市场波动率: {market_volatility:.1%}")
    print(f"  风险等级: {risk_level}")
    print(f"  市场状态: {market_status}")

    # 自适应权重信息
    print(f"\n【因子权重配置】")
    print(f"  自适应权重: {'已启用' if use_adaptive else '未启用'}")
    for factor, weight in current_weights.items():
        print(f"    {factor}: {weight:.0%}")

    # 仓位建议
    print(f"\n【仓位建议】")
    print(f"  建议总仓位: {position_ratio:.0%}")
    print(f"  实际配置仓位: {total_position:.1%}")
    print(f"  现金保留: {(1 - total_position):.1%}")
    print(f"  单票权重: {adjusted_weight:.2%}")

    # 选股结果
    print(f"\n【选股结果】共 {len(selected)} 只")
    print("-" * 70)
    print(f"{'排名':<5}{'代码':<10}{'得分':<10}{'权重':<10}{'行业':<15}")
    print("-" * 70)

    for _, row in selected.iterrows():
        score = row.get('total_score', row.get('ml_score', 0))
        industry = row.get('industry', '-')[:12] if 'industry' in row else '-'
        print(f"{row['rank']:<5}{row['code']:<10}{score:<10.4f}{row['weight']:<10.2%}{industry:<15}")

    # 行业分布
    if industry_dist:
        print(f"\n【行业分布】")
        for ind, count in sorted(industry_dist.items(), key=lambda x: -x[1])[:10]:
            print(f"  {ind}: {count}只 ({count/len(selected)*100:.1f}%)")

    # 风险提示
    print(f"\n【风险提示】")
    if risk_level == "高风险":
        print("  ⚠️  当前市场环境较差，建议降低仓位或观望")
    elif risk_level == "中风险":
        print("  ⚠️  市场存在一定风险，建议控制仓位")
    else:
        print("  ✓  市场环境正常，可按计划操作")

    print("=" * 70)

    # 保存信号
    output_dir = ensure_dir(root / 'output' / 'signals')
    date_str = latest_date.strftime('%Y%m%d')
    signal_path = output_dir / f"signal_{date_str}.csv"

    selected.to_csv(signal_path, index=False, encoding='utf-8-sig')
    logger.info(f"\n信号已保存: {signal_path}")


def full_run(args):
    """完整流程"""
    logger.info("=" * 60)
    logger.info("运行完整流程")
    logger.info("=" * 60)

    # 1. 下载数据
    if not args.skip_download:
        logger.info("\n>>> 步骤1: 下载数据")
        download_data(args)

    # 2. 运行回测
    logger.info("\n>>> 步骤2: 运行回测")
    run_backtest(args)

    # 3. 生成信号
    logger.info("\n>>> 步骤3: 生成最新信号")
    generate_signal(args)

    logger.info("\n完整流程执行完毕!")


def compute_factors(args):
    """批量计算因子 (用于深度学习训练)"""
    logger.info("=" * 60)
    logger.info("批量计算因子")
    logger.info("=" * 60)

    config = load_config()

    # 获取参数
    start_date = args.start or config['data_fetch']['start_date']
    end_date = args.end or datetime.now().strftime('%Y%m%d')
    freq = args.freq or config['factors'].get('dl_compute_freq', 'daily')
    parallel_dates = not args.no_parallel
    date_workers = args.workers or config['factors'].get('date_workers', 4)

    logger.info(f"计算范围: {start_date} ~ {end_date}")
    logger.info(f"计算频率: {freq}")
    logger.info(f"日期并行: {parallel_dates}, workers: {date_workers}")

    # 初始化因子引擎
    factor_engine = FactorEngine(config)

    # 执行计算
    factor_engine.compute_and_save_factors(
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        parallel_dates=parallel_dates,
        date_workers=date_workers
    )

    logger.info("因子计算完成!")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='A股多因子选股系统',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 下载命令
    download_parser = subparsers.add_parser('download', help='下载数据')
    download_parser.add_argument('--codes', type=str, help='指定股票代码，逗号分隔')

    # 回测命令
    backtest_parser = subparsers.add_parser('backtest', help='运行回测')
    backtest_parser.add_argument('--start', type=str, help='开始日期 (YYYYMMDD)')
    backtest_parser.add_argument('--end', type=str, help='结束日期 (YYYYMMDD)')
    backtest_parser.add_argument('--top-n', type=int, help='选股数量')
    backtest_parser.add_argument('--model', type=str,
                                  choices=['xgboost', 'lightgbm', 'linear'],
                                  help='ML模型类型')
    backtest_parser.add_argument('--walk-forward', action='store_true',
                                  help='使用滚动回测')
    backtest_parser.add_argument('--name', type=str, help='结果保存名称')
    backtest_parser.add_argument('--no-plot', action='store_true', help='不显示图表')

    # 信号命令
    signal_parser = subparsers.add_parser('signal', help='生成选股信号')
    signal_parser.add_argument('--top-n', type=int, help='选股数量')
    signal_parser.add_argument('--model', type=str, help='模型类型')
    signal_parser.add_argument('--model-path', type=str, help='模型文件路径')

    # 完整流程命令
    run_parser = subparsers.add_parser('run', help='运行完整流程')
    run_parser.add_argument('--skip-download', action='store_true', help='跳过数据下载')
    run_parser.add_argument('--start', type=str, help='回测开始日期')
    run_parser.add_argument('--end', type=str, help='回测结束日期')
    run_parser.add_argument('--top-n', type=int, help='选股数量')
    run_parser.add_argument('--model', type=str, help='ML模型类型')
    run_parser.add_argument('--walk-forward', action='store_true', help='滚动回测')
    run_parser.add_argument('--name', type=str, help='结果名称')
    run_parser.add_argument('--no-plot', action='store_true', help='不显示图表')

    # 因子计算命令 (用于深度学习训练)
    factor_parser = subparsers.add_parser('compute-factors', help='批量计算因子 (深度学习训练)')
    factor_parser.add_argument('--start', type=str, help='开始日期 (YYYYMMDD)')
    factor_parser.add_argument('--end', type=str, help='结束日期 (YYYYMMDD)')
    factor_parser.add_argument('--freq', type=str, choices=['daily', 'weekly', 'biweekly', 'monthly'],
                               help='计算频率 (默认: daily)')
    factor_parser.add_argument('--no-parallel', action='store_true', help='禁用日期级并行')
    factor_parser.add_argument('--workers', type=int, help='并行线程数')

    args = parser.parse_args()

    if args.command == 'download':
        download_data(args)
    elif args.command == 'backtest':
        run_backtest(args)
    elif args.command == 'signal':
        generate_signal(args)
    elif args.command == 'run':
        full_run(args)
    elif args.command == 'compute-factors':
        compute_factors(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
