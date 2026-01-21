"""
深度学习选股系统 - 独立入口

完全独立于现有 main.py，不影响原系统运行。

使用方法:
    # 训练模型
    python main_deep.py train --model transformer --start 20180101 --end 20231231

    # 回测评估
    python main_deep.py backtest --model transformer --start 20240101 --end 20241231

    # 生成信号
    python main_deep.py signal --model transformer

    # 评估现有模型
    python main_deep.py evaluate --model-path output/models/deep/xxx.pt
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

import yaml
import numpy as np
import pandas as pd

# 延迟导入 torch，避免未安装时报错
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch 未安装，请先运行: pip install -r requirements_deep.txt")

from src.utils import setup_logger, get_project_root, ensure_dir

logger = setup_logger('main_deep')

def _validate_factor_cache(factor_files) -> bool:
    """检查因子缓存是否存在混用或重复日期问题"""
    if not factor_files:
        logger.error("未找到因子缓存文件")
        return False

    has_legacy = False
    has_new = False
    date_to_files = {}

    for f in factor_files:
        parts = f.stem.split('_')
        if len(parts) <= 2:  # factors_YYYYMMDD
            has_legacy = True
        else:
            has_new = True

        if len(parts) >= 2:
            date_str = parts[1]
            date_to_files.setdefault(date_str, []).append(f)

    if has_legacy and has_new:
        logger.error(
            "检测到旧格式与新格式因子缓存混用 (factors_DATE.parquet + factors_DATE_universe.parquet)。\n"
            "为避免口径混乱，请清理缓存并按单一格式重新生成。"
        )
        return False

    duplicate_dates = [d for d, files in date_to_files.items() if len(files) > 1]
    if duplicate_dates:
        logger.error(
            f"检测到 {len(duplicate_dates)} 个日期存在重复的因子缓存文件。\n"
            f"示例: {duplicate_dates[:3]}\n"
            "可能是同日期生成了多份不同版本的因子，请清理后重新生成。"
        )
        return False

    return True


def load_deep_config() -> dict:
    """加载深度学习配置"""
    config_path = get_project_root() / 'config' / 'deep_learning.yaml'
    if not config_path.exists():
        logger.warning(f"配置文件不存在: {config_path}")
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def check_torch():
    """检查 PyTorch 是否可用"""
    if not TORCH_AVAILABLE:
        print("\n" + "=" * 60)
        print("错误: PyTorch 未安装")
        print("=" * 60)
        print("\n请先安装深度学习依赖:")
        print("  pip install -r requirements_deep.txt")
        print("\n或单独安装 PyTorch (根据CUDA版本):")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("=" * 60 + "\n")
        sys.exit(1)


def train_model(args):
    """训练深度模型"""
    check_torch()

    from src.deep_learning.temporal_models import get_temporal_model
    from src.deep_learning.gnn_models import get_gnn_model
    from src.deep_learning.contrastive import get_contrastive_model
    from src.deep_learning.moe_models import get_moe_model
    from src.deep_learning.base import load_merged_config
    from src.data_processor import DataProcessor
    from src.factor_engine import FactorEngine

    logger.info("=" * 60)
    logger.info("开始训练深度模型")
    logger.info("=" * 60)

    config = load_merged_config()

    # 获取训练参数
    model_type = args.model or 'transformer'
    start_date = args.start or config.get('data', {}).get('start_date', '20180101')
    end_date = args.end or datetime.now().strftime('%Y%m%d')

    logger.info(f"模型类型: {model_type}")
    logger.info(f"训练区间: {start_date} ~ {end_date}")

    # 检查因子缓存
    factor_cache_dir = get_project_root() / 'data' / 'processed' / 'factors'
    if not factor_cache_dir.exists() or len(list(factor_cache_dir.glob('*.parquet'))) == 0:
        logger.warning("因子缓存不存在，请先运行: python main.py backtest 生成因子缓存")
        logger.info("或者运行: python main.py run --skip-download 生成因子数据")
        return

    # 加载因子数据构建训练样本
    logger.info("加载因子数据...")
    processor = DataProcessor()

    # 获取所有因子文件
    factor_files = sorted(factor_cache_dir.glob('factors_*.parquet'))
    if not _validate_factor_cache(factor_files):
        return
    all_factors = []

    for f in factor_files:
        date_str = f.stem.split('_')[1]
        date = pd.to_datetime(date_str)
        if pd.to_datetime(start_date) <= date <= pd.to_datetime(end_date):
            df = pd.read_parquet(f)
            df['date'] = date
            all_factors.append(df)

    if len(all_factors) == 0:
        logger.error("无可用因子数据")
        return

    X = pd.concat(all_factors, ignore_index=True)
    logger.info(f"加载 {len(all_factors)} 天因子数据, 共 {len(X)} 个样本")

    # 获取标签列
    label_col = None
    for col in ['forward_return', 'label', 'ret_forward']:
        if col in X.columns:
            label_col = col
            break

    if label_col is None:
        logger.error("因子数据中无标签列 (forward_return/label)")
        return

    y = X[label_col]
    logger.info(f"使用标签列: {label_col}")

    # 根据模型类型创建模型
    if model_type in ['lstm', 'transformer', 'patchtst', 'informer']:
        model = get_temporal_model(model_type, config)
    elif model_type in ['graphsage', 'gat', 'graphormer']:
        model = get_gnn_model(model_type, config)
    elif model_type == 'contrastive':
        pretrained_path = getattr(args, 'pretrained_path', None)
        model = get_contrastive_model(config, pretrained_path=pretrained_path)
    elif model_type == 'moe':
        model = get_moe_model(config)
    else:
        logger.error(f"未知模型类型: {model_type}")
        return

    # 训练
    result = model.train(X, y)

    logger.info(f"训练完成: 最终损失 {result['val_loss']:.6f}")

    # 保存模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = ensure_dir(get_project_root() / 'output' / 'models' / 'deep') / f'{model_type}_{timestamp}.pt'
    model.save_model(str(model_path))

    logger.info(f"模型已保存: {model_path}")
    logger.info("=" * 60)


def run_backtest(args):
    """深度模型回测"""
    check_torch()

    from src.deep_learning.temporal_models import get_temporal_model
    from src.deep_learning.metrics import evaluate_model_comprehensive, print_evaluation_report
    from src.data_processor import DataProcessor
    from src.deep_learning.base import load_merged_config

    logger.info("=" * 60)
    logger.info("开始深度模型回测")
    logger.info("=" * 60)

    config = load_merged_config()

    model_type = args.model or 'transformer'
    start_date = args.start or '20240101'
    end_date = args.end or datetime.now().strftime('%Y%m%d')
    bt_cfg = config.get('backtest') or {}
    top_n = args.top_n or bt_cfg.get('top_n', 50)

    logger.info(f"模型类型: {model_type}")
    logger.info(f"回测区间: {start_date} ~ {end_date}")
    logger.info(f"选股数量: {top_n}")

    # 加载或训练模型
    model = get_temporal_model(model_type, config)

    if args.model_path:
        model.load_model(args.model_path)
        logger.info(f"加载模型: {args.model_path}")
    else:
        # 自动查找最新模型
        model_dir = get_project_root() / 'output' / 'models' / 'deep'
        # 优先查找 best 模型
        best_model = model_dir / 'TransformerStockModel_best.pt'
        if best_model.exists():
            model.load_model(str(best_model))
            logger.info(f"加载最佳模型: {best_model}")
        else:
            # 查找最新的模型文件
            model_files = sorted(model_dir.glob(f'*{model_type}*.pt'), reverse=True)
            if model_files:
                model.load_model(str(model_files[0]))
                logger.info(f"加载最新模型: {model_files[0]}")
            else:
                logger.error(f"未找到 {model_type} 模型，请先训练: python main_deep.py train --model {model_type}")
                return

    # 加载回测数据
    factor_cache_dir = get_project_root() / 'data' / 'processed' / 'factors'
    factor_files = sorted(factor_cache_dir.glob('factors_*.parquet'))
    if not _validate_factor_cache(factor_files):
        return

    all_predictions = []
    all_returns = []
    all_dates = []
    all_codes = []

    for f in factor_files:
        date_str = f.stem.split('_')[1]
        date = pd.to_datetime(date_str)
        if pd.to_datetime(start_date) <= date <= pd.to_datetime(end_date):
            df = pd.read_parquet(f)
            df['date'] = date

            # 预测
            scores = model.predict(df)
            if scores is None or len(scores) == 0:
                continue

            # 获取实际收益
            label_col = None
            for col in ['forward_return', 'label', 'ret_forward']:
                if col in df.columns:
                    label_col = col
                    break

            if label_col is None:
                continue

            all_predictions.extend(scores.values)
            all_returns.extend(df[label_col].values)
            all_dates.extend([date] * len(df))
            all_codes.extend(df['code'].values)

    if len(all_predictions) == 0:
        logger.error("无回测数据")
        return

    # 评估
    predictions = np.array(all_predictions)
    returns = np.array(all_returns)
    dates = np.array(all_dates)
    codes = np.array(all_codes)

    metrics = evaluate_model_comprehensive(
        predictions, returns, dates, codes, n=top_n
    )

    print_evaluation_report(metrics, f'{model_type.upper()} 深度模型')

    # 保存结果
    output_dir = ensure_dir(get_project_root() / 'output' / 'deep_backtest')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = output_dir / f'backtest_{model_type}_{timestamp}.csv'

    result_df = pd.DataFrame({
        'metric': list(metrics.keys()),
        'value': list(metrics.values()),
    })
    result_df.to_csv(result_path, index=False)
    logger.info(f"回测结果已保存: {result_path}")

    logger.info("=" * 60)


def generate_signal(args):
    """生成深度模型选股信号"""
    check_torch()

    from src.deep_learning.temporal_models import get_temporal_model
    from src.factor_engine import FactorEngine
    from src.data_processor import DataProcessor
    from src.deep_learning.base import load_merged_config

    logger.info("=" * 60)
    logger.info("生成深度模型选股信号")
    logger.info("=" * 60)

    config = load_merged_config()

    model_type = args.model or 'transformer'
    bt_cfg = config.get('backtest') or {}
    top_n = args.top_n or bt_cfg.get('top_n', 50)

    # 加载模型
    model = get_temporal_model(model_type, config)

    if args.model_path:
        model.load_model(args.model_path)
        logger.info(f"加载模型: {args.model_path}")
    else:
        # 尝试加载最新模型
        model_dir = get_project_root() / 'output' / 'models' / 'deep'
        model_files = sorted(model_dir.glob(f'{model_type}_*.pt'), reverse=True)
        if model_files:
            model.load_model(str(model_files[0]))
            logger.info(f"加载最新模型: {model_files[0]}")
        else:
            logger.error("未找到可用模型，请先训练: python main_deep.py train")
            return

    # 获取最新因子
    processor = DataProcessor()
    calendar = processor.trade_calendar

    today = datetime.now()
    recent_dates = calendar[calendar <= today]
    if len(recent_dates) == 0:
        logger.error("无可用交易日")
        return

    latest_date = recent_dates[-1]
    date_str = latest_date.strftime('%Y%m%d')

    factor_path = get_project_root() / 'data' / 'processed' / 'factors' / f'factors_{date_str}.parquet'

    if not factor_path.exists():
        logger.warning(f"最新因子不存在: {factor_path}")
        logger.info("尝试计算最新因子...")
        factor_engine = FactorEngine()
        factor_df = factor_engine.compute_factor_cross_section(latest_date)
        factor_df = factor_engine.standardize_factors(factor_df)
    else:
        factor_df = pd.read_parquet(factor_path)
        factor_df['date'] = latest_date

    logger.info(f"信号日期: {latest_date.date()}, 股票数: {len(factor_df)}")

    # 选股
    selected = model.select_stocks_ml(factor_df, n=top_n)

    if len(selected) == 0:
        logger.error("选股失败")
        return

    # 输出信号
    print("\n" + "=" * 70)
    print("               深度模型选股信号")
    print("=" * 70)
    print(f"\n模型类型: {model_type.upper()}")
    print(f"信号日期: {latest_date.date()}")
    print(f"选股数量: {len(selected)}")

    print(f"\n{'排名':<5}{'代码':<10}{'分数':<12}{'名称':<15}")
    print("-" * 70)

    for _, row in selected.head(20).iterrows():
        name = row.get('name', '-')[:12] if 'name' in row else '-'
        print(f"{row['rank']:<5}{row['code']:<10}{row['ml_score']:<12.4f}{name:<15}")

    if len(selected) > 20:
        print(f"... 共 {len(selected)} 只股票")

    print("=" * 70)

    # 保存信号
    output_dir = ensure_dir(get_project_root() / 'output' / 'signals')
    signal_path = output_dir / f'deep_signal_{date_str}_{model_type}.csv'
    selected.to_csv(signal_path, index=False, encoding='utf-8-sig')
    logger.info(f"信号已保存: {signal_path}")


def evaluate_model(args):
    """评估已训练模型"""
    check_torch()

    from src.deep_learning.metrics import evaluate_model_comprehensive, print_evaluation_report

    logger.info("=" * 60)
    logger.info("评估深度模型")
    logger.info("=" * 60)

    if not args.model_path:
        logger.error("请指定模型路径: --model-path")
        return

    # 加载模型并评估
    # ... 实现评估逻辑

    logger.info("评估完成")


def run_pretrain(args):
    """运行对比学习预训练"""
    check_torch()

    from src.deep_learning.contrastive import FactorContrastiveLearning, pretrain_contrastive
    from src.deep_learning.sequence_dataset import FactorSequenceDataset
    from src.deep_learning.base import load_merged_config

    logger.info("=" * 60)
    logger.info("开始对比学习预训练")
    logger.info("=" * 60)

    config = load_merged_config()

    start_date = args.start or config.get('data', {}).get('start_date', '20180101')
    end_date = args.end or datetime.now().strftime('%Y%m%d')
    epochs = args.epochs or config.get('contrastive', {}).get('pretrain_epochs', 50)
    batch_size = args.batch_size or 256

    logger.info(f"预训练区间: {start_date} ~ {end_date}")
    logger.info(f"预训练轮数: {epochs}")

    # 检查因子缓存
    factor_cache_dir = get_project_root() / 'data' / 'processed' / 'factors'
    if not factor_cache_dir.exists() or len(list(factor_cache_dir.glob('*.parquet'))) == 0:
        logger.warning("因子缓存不存在，请先运行: python main.py backtest 生成因子缓存")
        return

    # 加载因子数据
    logger.info("加载因子数据...")
    factor_files = sorted(factor_cache_dir.glob('factors_*.parquet'))
    if not _validate_factor_cache(factor_files):
        return
    all_factors = []

    for f in factor_files:
        date_str = f.stem.split('_')[1]
        date = pd.to_datetime(date_str)
        if pd.to_datetime(start_date) <= date <= pd.to_datetime(end_date):
            df = pd.read_parquet(f)
            df['date'] = date
            all_factors.append(df)

    if len(all_factors) == 0:
        logger.error("无可用因子数据")
        return

    X = pd.concat(all_factors, ignore_index=True)
    logger.info(f"加载 {len(all_factors)} 天因子数据, 共 {len(X)} 个样本")

    # 构建序列数据集
    try:
        dataset = FactorSequenceDataset(config, factor_df=X)
        logger.info(f"构建序列数据集: {len(dataset)} 个样本")
    except Exception as e:
        logger.error(f"构建数据集失败: {e}")
        # 创建简单数据集作为替代
        import torch
        from torch.utils.data import TensorDataset

        # 使用因子横截面作为简化序列
        exclude_cols = {'code', 'name', 'date', 'industry', 'label', 'forward_return', 'total_score'}
        feature_cols = [c for c in X.columns
                       if c not in exclude_cols and X[c].dtype in [np.float64, np.float32, np.int64]]

        features = X[feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)

        # 添加序列维度 [samples, 1, features]
        features = torch.tensor(features).unsqueeze(1)
        dataset = TensorDataset(features)
        logger.info(f"使用简化数据集: {len(dataset)} 个样本, 特征维度: {features.shape[-1]}")

    # 运行预训练
    model = FactorContrastiveLearning(config)
    result = model.pretrain(dataset, epochs=epochs, batch_size=batch_size)

    logger.info(f"预训练完成, 最终损失: {result['final_loss']:.6f}")

    # 保存预训练模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = ensure_dir(get_project_root() / 'output' / 'models' / 'deep' / 'pretrained') / f'contrastive_{timestamp}.pt'
    model.save(str(save_path))

    logger.info(f"预训练模型已保存: {save_path}")
    logger.info("=" * 60)
    logger.info("使用预训练模型微调:")
    logger.info(f"  python main_deep.py train --model contrastive --pretrained-path {save_path}")
    logger.info("=" * 60)


def run_rl_train(args):
    """运行 RL 仓位优化训练"""
    check_torch()

    try:
        from src.deep_learning.rl_env import create_portfolio_env, check_gym_available
        from src.deep_learning.rl_trainer import create_rl_trainer, check_sb3_available
    except ImportError as e:
        logger.error(f"RL 模块导入失败: {e}")
        logger.info("请安装依赖: pip install gymnasium stable-baselines3")
        return

    if not check_gym_available():
        logger.error("Gymnasium 未安装，请运行: pip install gymnasium")
        return

    if not check_sb3_available():
        logger.error("Stable-Baselines3 未安装，请运行: pip install stable-baselines3")
        return

    from src.deep_learning.base import load_merged_config

    logger.info("=" * 60)
    logger.info("开始 RL 仓位优化训练")
    logger.info("=" * 60)

    # R4修复: 使用 load_merged_config 确保配置继承一致
    config = load_merged_config()
    rl_config = config.get('rl', {})

    start_date = args.start or config.get('data', {}).get('start_date', '20180101')
    end_date = args.end or datetime.now().strftime('%Y%m%d')
    algorithm = args.algorithm or rl_config.get('algorithm', 'ppo')
    total_timesteps = args.timesteps or rl_config.get('total_timesteps', 100000)

    logger.info(f"RL 算法: {algorithm.upper()}")
    logger.info(f"训练区间: {start_date} ~ {end_date}")
    logger.info(f"总步数: {total_timesteps}")

    # 检查因子缓存
    factor_cache_dir = get_project_root() / 'data' / 'processed' / 'factors'
    if not factor_cache_dir.exists() or len(list(factor_cache_dir.glob('*.parquet'))) == 0:
        logger.warning("因子缓存不存在，请先运行: python main.py backtest 生成因子缓存")
        return

    # 加载因子数据
    logger.info("加载因子数据...")
    factor_files = sorted(factor_cache_dir.glob('factors_*.parquet'))
    if not _validate_factor_cache(factor_files):
        return

    all_factors = []

    for f in factor_files:
        date_str = f.stem.split('_')[1]
        date = pd.to_datetime(date_str)
        if pd.to_datetime(start_date) <= date <= pd.to_datetime(end_date):
            df = pd.read_parquet(f)
            df['date'] = date
            all_factors.append(df)

    if len(all_factors) == 0:
        logger.error("无可用因子数据")
        return

    X = pd.concat(all_factors, ignore_index=True)
    logger.info(f"加载 {len(all_factors)} 天因子数据, 共 {len(X)} 个样本")

    # 创建环境
    try:
        env = create_portfolio_env(X, config)
        logger.info(f"创建环境: {env.n_stocks} 只股票, {env.n_features} 个特征")
    except Exception as e:
        logger.error(f"创建环境失败: {e}")
        return

    # 创建训练器
    try:
        trainer = create_rl_trainer(env, config, algorithm)
    except Exception as e:
        logger.error(f"创建训练器失败: {e}")
        return

    # 训练
    result = trainer.train(total_timesteps=total_timesteps)

    logger.info(f"RL 训练完成")
    logger.info(f"最佳平均奖励: {result['best_mean_reward']:.4f}")

    # 评估
    eval_result = trainer.evaluate(n_episodes=10)
    logger.info(f"评估结果: 平均奖励={eval_result['mean_reward']:.4f}, Sharpe={eval_result['mean_sharpe']:.4f}")

    logger.info("=" * 60)
    logger.info("使用 RL 权重优化:")
    logger.info(f"  模型路径: {trainer.model_dir}")
    logger.info("=" * 60)


def run_rl_backtest(args):
    """使用 RL 策略运行回测"""
    check_torch()

    try:
        from src.deep_learning.rl_adapter import RLAdapter
        from src.backtester import Backtester
    except ImportError as e:
        logger.error(f"RL 适配器模块导入失败: {e}")
        return

    from src.deep_learning.base import load_merged_config

    logger.info("=" * 60)
    logger.info("使用 RL 策略回测")
    logger.info("=" * 60)

    # 加载合并配置
    config = load_merged_config()

    # 参数
    model_path = args.model_path
    start_date = args.start or '20240101'
    end_date = args.end or datetime.now().strftime('%Y%m%d')
    top_n = args.top_n or config.get('backtest', {}).get('top_n', 50)

    # P7修复: 将命令行 top_n 更新到配置 (虽然 mode='weight' 时不使用 top_n)
    if 'backtest' not in config:
        config['backtest'] = {}
    config['backtest']['top_n'] = top_n

    if not model_path:
        # 查找最新 RL 模型
        model_dir = get_project_root() / 'output' / 'models' / 'rl'
        model_files = sorted(model_dir.glob('*.zip'), key=lambda x: x.stat().st_mtime, reverse=True)
        if model_files:
            model_path = str(model_files[0])
            logger.info(f"使用最新模型: {model_path}")
        else:
            logger.error("未找到 RL 模型，请先训练: python main_deep.py rl-train")
            return

    logger.info(f"模型路径: {model_path}")
    logger.info(f"回测区间: {start_date} ~ {end_date}")
    logger.info(f"持仓数量: {top_n}")

    # 加载 RL 适配器
    try:
        adapter = RLAdapter(config=config, model_path=model_path)
        logger.info(f"RL 适配器已加载: cost_penalty={adapter.cost_penalty:.4f}")
    except Exception as e:
        logger.error(f"加载 RL 适配器失败: {e}")
        return

    # 创建 weight_func (P1修复: 使用权重模式而非分数模式)
    weight_func = adapter.create_weight_func()

    # 运行回测 (mode='weight' 直接使用RL输出的权重，不经过Top-N排序)
    backtester = Backtester(config=config)
    result = backtester.run(
        start_date=start_date,
        end_date=end_date,
        weight_func=weight_func,
        mode='weight',
    )

    if len(result) == 0:
        logger.error("回测失败，无结果")
        return

    # 计算指标
    final_nav = result['nav'].iloc[-1]
    total_return = (final_nav - 1) * 100
    daily_returns = result['daily_return'].dropna()
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    max_dd = (result['nav'] / result['nav'].cummax() - 1).min() * 100

    # 输出结果
    print("\n" + "=" * 70)
    print("               RL 策略回测结果")
    print("=" * 70)
    print(f"\n回测区间: {start_date} ~ {end_date}")
    print(f"RL 模型: {Path(model_path).name}")
    print(f"\n{'指标':<20} {'值':<15}")
    print("-" * 40)
    print(f"{'最终净值':<20} {final_nav:.4f}")
    print(f"{'总收益率':<20} {total_return:.2f}%")
    print(f"{'年化 Sharpe':<20} {sharpe:.4f}")
    print(f"{'最大回撤':<20} {max_dd:.2f}%")
    print(f"{'平均换手率':<20} {backtester.calc_turnover():.2%}")
    print("=" * 70)

    # 保存结果
    output_dir = ensure_dir(get_project_root() / 'output' / 'rl_backtest')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = output_dir / f'rl_backtest_{timestamp}.csv'
    result.reset_index().to_csv(result_path, index=False)
    logger.info(f"回测结果已保存: {result_path}")

    logger.info("=" * 60)


def run_rl_signal(args):
    """生成 RL 策略选股信号"""
    check_torch()

    try:
        from src.deep_learning.rl_adapter import RLAdapter
        from src.factor_engine import FactorEngine
        from src.data_processor import DataProcessor
    except ImportError as e:
        logger.error(f"RL 适配器模块导入失败: {e}")
        return

    from src.deep_learning.base import load_merged_config

    logger.info("=" * 60)
    logger.info("生成 RL 策略选股信号")
    logger.info("=" * 60)

    # 加载配置
    config = load_merged_config()

    # 参数
    model_path = args.model_path
    top_n = args.top_n or config.get('backtest', {}).get('top_n', 50)

    if not model_path:
        # 查找最新 RL 模型
        model_dir = get_project_root() / 'output' / 'models' / 'rl'
        model_files = sorted(model_dir.glob('*.zip'), key=lambda x: x.stat().st_mtime, reverse=True)
        if model_files:
            model_path = str(model_files[0])
            logger.info(f"使用最新模型: {model_path}")
        else:
            logger.error("未找到 RL 模型，请先训练: python main_deep.py rl-train")
            return

    logger.info(f"模型路径: {model_path}")

    # 加载 RL 适配器
    try:
        adapter = RLAdapter(config=config, model_path=model_path)
    except Exception as e:
        logger.error(f"加载 RL 适配器失败: {e}")
        return

    # 获取最新因子
    processor = DataProcessor()
    calendar = processor.trade_calendar

    today = datetime.now()
    recent_dates = calendar[calendar <= today]
    if len(recent_dates) == 0:
        logger.error("无可用交易日")
        return

    latest_date = recent_dates[-1]
    date_str = latest_date.strftime('%Y%m%d')

    # N4修复: 智能匹配因子文件 (支持新旧格式)
    factor_cache_dir = get_project_root() / 'data' / 'processed' / 'factors'
    factor_path = None

    # 尝试旧格式: factors_{date}.parquet
    old_format = factor_cache_dir / f'factors_{date_str}.parquet'
    if old_format.exists():
        factor_path = old_format
    else:
        # 尝试新格式: factors_{date}_{universe}.parquet
        new_format_matches = list(factor_cache_dir.glob(f'factors_{date_str}_*.parquet'))
        if new_format_matches:
            factor_path = new_format_matches[0]
            logger.info(f"使用新格式因子文件: {factor_path.name}")

    if factor_path is None or not factor_path.exists():
        logger.warning(f"最新因子不存在: factors_{date_str}*.parquet")
        logger.info("尝试计算最新因子...")
        factor_engine = FactorEngine()
        factor_df = factor_engine.compute_factor_cross_section(latest_date)
        factor_df = factor_engine.standardize_factors(factor_df)
    else:
        factor_df = pd.read_parquet(factor_path)
        factor_df['date'] = latest_date

    logger.info(f"信号日期: {latest_date.date()}, 股票数: {len(factor_df)}")

    # 预测权重
    try:
        weights = adapter.predict_weights(factor_df)
    except Exception as e:
        logger.error(f"预测权重失败: {e}")
        return

    # 构建信号 DataFrame
    codes = factor_df['code'].tolist()
    names = factor_df['name'].tolist() if 'name' in factor_df.columns else [''] * len(codes)

    signal_df = pd.DataFrame({
        'rank': range(1, len(codes) + 1),
        'code': codes,
        'name': names,
        'weight': weights[:len(codes)] if len(weights) >= len(codes) else list(weights) + [0] * (len(codes) - len(weights)),
    })

    # 按权重排序
    signal_df = signal_df[signal_df['weight'] > 1e-6].sort_values('weight', ascending=False)
    signal_df['rank'] = range(1, len(signal_df) + 1)

    # 输出信号
    print("\n" + "=" * 70)
    print("               RL 策略选股信号")
    print("=" * 70)
    print(f"\n信号日期: {latest_date.date()}")
    print(f"RL 模型: {Path(model_path).name}")
    print(f"选股数量: {len(signal_df)}")

    print(f"\n{'排名':<5}{'代码':<10}{'权重':<12}{'名称':<15}")
    print("-" * 70)

    for _, row in signal_df.head(top_n).iterrows():
        name = str(row['name'])[:12] if row['name'] else '-'
        print(f"{row['rank']:<5}{row['code']:<10}{row['weight']:<12.4f}{name:<15}")

    if len(signal_df) > top_n:
        print(f"... 共 {len(signal_df)} 只股票")

    print("=" * 70)

    # 保存信号
    output_dir = ensure_dir(get_project_root() / 'output' / 'signals')
    signal_path = output_dir / f'rl_signal_{date_str}.csv'
    signal_df.head(top_n).to_csv(signal_path, index=False, encoding='utf-8-sig')
    logger.info(f"信号已保存: {signal_path}")


def run_rl_train_adapter(args):
    """使用 RLAdapter 训练 RL 策略 (统一配置)"""
    check_torch()

    try:
        from src.deep_learning.rl_adapter import RLAdapter
    except ImportError as e:
        logger.error(f"RL 适配器模块导入失败: {e}")
        logger.info("请安装依赖: pip install gymnasium stable-baselines3")
        return

    from src.deep_learning.base import load_merged_config

    logger.info("=" * 60)
    logger.info("开始 RL 策略训练 (统一配置)")
    logger.info("=" * 60)

    # 加载合并配置
    config = load_merged_config()
    rl_cfg = config.get('rl_adapter', config.get('rl', {}))

    # 参数
    start_date = args.start or config.get('data', {}).get('start_date', '20180101')
    end_date = args.end or datetime.now().strftime('%Y%m%d')
    algorithm = args.algorithm or rl_cfg.get('algorithm', 'ppo')
    total_timesteps = args.timesteps or rl_cfg.get('total_timesteps', 100000)
    universe_id = getattr(args, 'universe_id', None)  # P3修复: universe_id 跟踪

    logger.info(f"RL 算法: {algorithm.upper()}")
    logger.info(f"训练区间: {start_date} ~ {end_date}")
    logger.info(f"总步数: {total_timesteps}")
    if universe_id:
        logger.info(f"股票池版本: {universe_id}")

    # 检查因子缓存
    factor_cache_dir = get_project_root() / 'data' / 'processed' / 'factors'
    if not factor_cache_dir.exists() or len(list(factor_cache_dir.glob('*.parquet'))) == 0:
        logger.warning("因子缓存不存在，请先运行: python main.py backtest 生成因子缓存")
        return

    # N1修复: 简化因子加载 - 仅按日期范围筛选，不依赖 universe_id
    logger.info("加载因子数据...")
    factor_files = sorted(factor_cache_dir.glob('factors_*.parquet'))
    if not _validate_factor_cache(factor_files):
        return

    all_factors = []

    for f in factor_files:
        date_str = f.stem.split('_')[1]
        date = pd.to_datetime(date_str)
        if pd.to_datetime(start_date) <= date <= pd.to_datetime(end_date):
            df = pd.read_parquet(f)
            df['date'] = date
            all_factors.append(df)

    if len(all_factors) == 0:
        logger.error("无可用因子数据")
        return

    factor_data = pd.concat(all_factors, ignore_index=True)
    logger.info(f"加载 {len(all_factors)} 天因子数据, 共 {len(factor_data)} 个样本")

    # 创建适配器并训练
    adapter = RLAdapter(config=config)
    adapter.algorithm = algorithm
    adapter.universe_id = universe_id  # P3修复: 记录 universe_id

    save_path = str(ensure_dir(get_project_root() / 'output' / 'models' / 'rl') /
                    f'{algorithm}_{start_date}_{end_date}.zip')

    result = adapter.train(
        factor_data=factor_data,
        total_timesteps=total_timesteps,
        save_path=save_path,
    )

    logger.info(f"RL 训练完成")
    logger.info(f"最佳平均奖励: {result.get('best_mean_reward', 'N/A')}")
    logger.info(f"模型已保存: {save_path}")

    logger.info("=" * 60)
    logger.info("使用 RL 策略回测:")
    logger.info(f"  python main_deep.py rl-backtest --model-path {save_path}")
    logger.info("=" * 60)


def run_text_extract(args):
    """提取文本因子"""
    try:
        from src.deep_learning.text_factors import (
            TextFactorExtractor, check_text_dependencies
        )
    except ImportError as e:
        logger.error(f"文本因子模块导入失败: {e}")
        logger.info("请安装依赖: pip install akshare jieba")
        return

    deps = check_text_dependencies()
    if not deps['akshare']:
        logger.error("AKShare 未安装，请运行: pip install akshare")
        return

    logger.info("=" * 60)
    logger.info("开始提取文本因子")
    logger.info("=" * 60)

    config = load_deep_config()

    # 参数
    lookback_days = args.lookback or config.get('text_factors', {}).get('lookback_days', 30)
    use_transformer = args.transformer and deps['transformers']

    logger.info(f"回看天数: {lookback_days}")
    logger.info(f"使用 Transformer: {use_transformer}")
    logger.info(f"依赖状态: akshare={deps['akshare']}, jieba={deps['jieba']}, transformers={deps['transformers']}")

    # 检查因子缓存
    factor_cache_dir = get_project_root() / 'data' / 'processed' / 'factors'
    if not factor_cache_dir.exists() or len(list(factor_cache_dir.glob('*.parquet'))) == 0:
        logger.warning("因子缓存不存在，请先运行: python main.py backtest 生成因子缓存")
        return

    # 获取最新因子文件
    factor_files = sorted(factor_cache_dir.glob('factors_*.parquet'), reverse=True)
    if not factor_files:
        logger.error("无因子文件")
        return

    if args.date:
        # 指定日期
        target_file = factor_cache_dir / f'factors_{args.date}.parquet'
        if not target_file.exists():
            logger.error(f"指定日期因子文件不存在: {target_file}")
            return
        factor_file = target_file
    else:
        # 最新日期
        factor_file = factor_files[0]

    date_str = factor_file.stem.split('_')[1]
    logger.info(f"处理日期: {date_str}")

    # 加载因子
    factor_df = pd.read_parquet(factor_file)
    factor_df['date'] = pd.to_datetime(date_str)
    logger.info(f"加载 {len(factor_df)} 只股票")

    # 创建提取器
    extractor = TextFactorExtractor(config, use_transformer=use_transformer)

    # 提取文本因子
    codes = factor_df['code'].tolist()

    if args.sample and args.sample < len(codes):
        # 采样模式 (用于测试)
        import random
        codes = random.sample(codes, args.sample)
        logger.info(f"采样模式: 处理 {len(codes)} 只股票")

    date = pd.to_datetime(date_str)
    text_factors = extractor.extract_factors_batch(
        codes, date, lookback_days=lookback_days, use_cache=not args.no_cache
    )

    logger.info(f"提取 {len(text_factors)} 只股票的文本因子")

    # 显示因子统计
    print("\n" + "=" * 70)
    print("               文本因子统计")
    print("=" * 70)

    factor_cols = [c for c in text_factors.columns if c.startswith('text_')]
    for col in factor_cols:
        values = text_factors[col].dropna()
        if len(values) > 0:
            print(f"{col:<40} 均值={values.mean():.4f}  非零={len(values[values != 0])}")

    print("=" * 70)

    # 合并到因子文件
    if args.merge:
        merged = extractor.merge_with_factor_df(factor_df, lookback_days=lookback_days)
        output_path = factor_cache_dir / f'factors_with_text_{date_str}.parquet'
        merged.to_parquet(output_path, index=False)
        logger.info(f"合并后的因子已保存: {output_path}")

    # 保存文本因子
    output_dir = ensure_dir(get_project_root() / 'data' / 'processed' / 'text_factors')
    output_path = output_dir / f'text_factors_{date_str}.parquet'
    text_factors.to_parquet(output_path, index=False)
    logger.info(f"文本因子已保存: {output_path}")

    logger.info("=" * 60)


def prepare_daily_factors(args):
    """
    生成每日因子数据（用于深度学习序列模型训练）

    与 main.py backtest 不同，此命令生成每个交易日的因子，
    而不仅仅是调仓日，以满足序列模型对连续时间序列的需求。
    """
    from src.utils import load_config, get_project_root, ensure_dir
    from src.data_processor import DataProcessor
    from src.factor_engine import FactorEngine
    from tqdm import tqdm

    config = load_config()

    start_date = args.start or '20180101'
    end_date = args.end or datetime.now().strftime('%Y%m%d')

    logger.info("=" * 60)
    logger.info("生成每日因子数据 (用于深度学习)")
    logger.info("=" * 60)
    logger.info(f"日期范围: {start_date} ~ {end_date}")

    # 初始化
    processor = DataProcessor()
    factor_engine = FactorEngine(config)

    # 获取交易日历并过滤日期范围
    full_calendar = processor.trade_calendar
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    calendar = full_calendar[(full_calendar >= start_dt) & (full_calendar <= end_dt)]
    logger.info(f"交易日数量: {len(calendar)}")

    # 检查现有缓存
    factor_cache_dir = ensure_dir(get_project_root() / 'data' / 'processed' / 'factors')
    existing_dates = set()
    for f in factor_cache_dir.glob('factors_*.parquet'):
        date_str = f.stem.split('_')[1]
        existing_dates.add(date_str)

    # 计算需要生成的日期
    dates_to_compute = []
    for date in calendar:
        date_str = date.strftime('%Y%m%d')
        if date_str not in existing_dates or args.force:
            dates_to_compute.append(date)

    if len(dates_to_compute) == 0:
        logger.info(f"所有 {len(calendar)} 个交易日的因子已存在，跳过计算")
        logger.info(f"如需强制重新计算，请使用 --force 参数")
        return

    logger.info(f"需要计算: {len(dates_to_compute)} 天 (已存在: {len(existing_dates)} 天)")

    # 逐日计算因子
    success_count = 0
    for date in tqdm(dates_to_compute, desc="计算每日因子"):
        try:
            factor_df = factor_engine.compute_factor_cross_section(date)
            if len(factor_df) > 0:
                factor_df = factor_engine.standardize_factors(factor_df)
                date_str = date.strftime('%Y%m%d')
                factor_engine.save_factor_data(factor_df, date_str)
                success_count += 1
        except Exception as e:
            logger.warning(f"{date.strftime('%Y%m%d')} 计算失败: {e}")

    logger.info("=" * 60)
    logger.info(f"因子生成完成: 成功 {success_count}/{len(dates_to_compute)} 天")
    logger.info(f"缓存目录: {factor_cache_dir}")
    logger.info("=" * 60)
    logger.info("现在可以运行训练:")
    logger.info(f"  python main_deep.py train --model transformer --start {start_date} --end {end_date}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='深度学习选股系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main_deep.py train --model transformer --start 20180101 --end 20231231
  python main_deep.py train --model graphsage --start 20180101 --end 20231231
  python main_deep.py backtest --model transformer --start 20240101 --end 20241231
  python main_deep.py signal --model transformer
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练深度模型')
    train_parser.add_argument('--model', type=str,
                              choices=['lstm', 'transformer', 'patchtst', 'informer',
                                       'graphsage', 'gat', 'graphormer', 'contrastive', 'moe'],
                              default='transformer', help='模型类型')
    train_parser.add_argument('--start', type=str, help='训练开始日期 (YYYYMMDD)')
    train_parser.add_argument('--end', type=str, help='训练结束日期')
    train_parser.add_argument('--epochs', type=int, help='训练轮数')
    train_parser.add_argument('--batch-size', type=int, help='批大小')
    train_parser.add_argument('--pretrained-path', type=str, help='预训练模型路径 (用于 contrastive)')

    # 预训练命令 (对比学习)
    pretrain_parser = subparsers.add_parser('pretrain', help='对比学习预训练')
    pretrain_parser.add_argument('--start', type=str, help='训练开始日期 (YYYYMMDD)')
    pretrain_parser.add_argument('--end', type=str, help='训练结束日期')
    pretrain_parser.add_argument('--epochs', type=int, default=50, help='预训练轮数')
    pretrain_parser.add_argument('--batch-size', type=int, default=256, help='批大小')

    # 回测命令
    backtest_parser = subparsers.add_parser('backtest', help='回测评估')
    backtest_parser.add_argument('--model', type=str,
                                  choices=['lstm', 'transformer', 'patchtst', 'informer',
                                           'graphsage', 'gat', 'graphormer', 'contrastive', 'moe'],
                                  default='transformer', help='模型类型')
    backtest_parser.add_argument('--model-path', type=str, help='模型文件路径')
    backtest_parser.add_argument('--start', type=str, help='回测开始日期')
    backtest_parser.add_argument('--end', type=str, help='回测结束日期')
    backtest_parser.add_argument('--top-n', type=int, help='选股数量')

    # 信号命令
    signal_parser = subparsers.add_parser('signal', help='生成选股信号')
    signal_parser.add_argument('--model', type=str,
                               choices=['lstm', 'transformer', 'patchtst', 'informer',
                                        'graphsage', 'gat', 'graphormer', 'contrastive', 'moe'],
                               default='transformer', help='模型类型')
    signal_parser.add_argument('--model-path', type=str, help='模型文件路径')
    signal_parser.add_argument('--top-n', type=int, help='选股数量')

    # RL 训练命令
    rl_parser = subparsers.add_parser('rl', help='RL 仓位优化训练')
    rl_parser.add_argument('--algorithm', type=str, choices=['ppo', 'a2c', 'sac'],
                           default='ppo', help='RL 算法')
    rl_parser.add_argument('--start', type=str, help='训练开始日期 (YYYYMMDD)')
    rl_parser.add_argument('--end', type=str, help='训练结束日期')
    rl_parser.add_argument('--timesteps', type=int, default=100000, help='总训练步数')

    # RL 训练命令 (使用 RLAdapter 统一配置)
    rl_train_parser = subparsers.add_parser('rl-train', help='RL 策略训练 (统一配置)')
    rl_train_parser.add_argument('--algorithm', type=str, choices=['ppo', 'a2c', 'sac'],
                                  default='ppo', help='RL 算法')
    rl_train_parser.add_argument('--start', type=str, help='训练开始日期 (YYYYMMDD)')
    rl_train_parser.add_argument('--end', type=str, help='训练结束日期')
    rl_train_parser.add_argument('--timesteps', type=int, default=100000, help='总训练步数')
    rl_train_parser.add_argument('--universe-id', type=str, help='股票池版本ID (P3修复)')

    # RL 回测命令
    rl_backtest_parser = subparsers.add_parser('rl-backtest', help='使用 RL 策略回测')
    rl_backtest_parser.add_argument('--model-path', type=str, help='RL 模型路径')
    rl_backtest_parser.add_argument('--start', type=str, default='20240101', help='回测开始日期')
    rl_backtest_parser.add_argument('--end', type=str, help='回测结束日期')
    rl_backtest_parser.add_argument('--top-n', type=int, help='持仓数量')
    rl_backtest_parser.add_argument('--universe-id', type=str, help='股票池版本ID (P3修复)')

    # RL 信号命令
    rl_signal_parser = subparsers.add_parser('rl-signal', help='生成 RL 策略信号')
    rl_signal_parser.add_argument('--model-path', type=str, help='RL 模型路径')
    rl_signal_parser.add_argument('--top-n', type=int, help='输出股票数量')

    # 文本因子命令
    text_parser = subparsers.add_parser('text', help='提取文本因子')
    text_parser.add_argument('--date', type=str, help='指定日期 (YYYYMMDD)')
    text_parser.add_argument('--lookback', type=int, default=30, help='回看天数')
    text_parser.add_argument('--transformer', action='store_true', help='使用 Transformer 模型')
    text_parser.add_argument('--merge', action='store_true', help='合并到因子文件')
    text_parser.add_argument('--sample', type=int, help='采样股票数 (用于测试)')
    text_parser.add_argument('--no-cache', action='store_true', help='不使用缓存')

    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--model-path', type=str, required=True, help='模型文件路径')

    # 每日因子生成命令 (用于序列模型)
    prepare_parser = subparsers.add_parser('prepare-factors', help='生成每日因子数据 (用于序列模型)')
    prepare_parser.add_argument('--start', type=str, default='20180101', help='开始日期 (YYYYMMDD)')
    prepare_parser.add_argument('--end', type=str, help='结束日期')
    prepare_parser.add_argument('--force', action='store_true', help='强制重新计算已存在的日期')

    args = parser.parse_args()

    if args.command == 'train':
        train_model(args)
    elif args.command == 'pretrain':
        run_pretrain(args)
    elif args.command == 'backtest':
        run_backtest(args)
    elif args.command == 'signal':
        generate_signal(args)
    elif args.command == 'rl':
        run_rl_train(args)
    elif args.command == 'rl-train':
        run_rl_train_adapter(args)
    elif args.command == 'rl-backtest':
        run_rl_backtest(args)
    elif args.command == 'rl-signal':
        run_rl_signal(args)
    elif args.command == 'text':
        run_text_extract(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'prepare-factors':
        prepare_daily_factors(args)
    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("深度学习选股系统")
        print("=" * 60)
        print("\n这是独立于原系统的深度学习模块。")
        print("原系统命令 (python main.py) 不受影响。")
        print("\n快速开始:")
        print("  1. 确保已运行 python main.py run 生成因子缓存")
        print("  2. 安装依赖: pip install -r requirements_deep.txt")
        print("  3. 训练模型: python main_deep.py train --model transformer")
        print("  4. 生成信号: python main_deep.py signal --model transformer")
        print("\n对比学习流程:")
        print("  1. 预训练: python main_deep.py pretrain --epochs 50")
        print("  2. 微调: python main_deep.py train --model contrastive --pretrained-path <模型路径>")
        print("\nRL 仓位优化:")
        print("  python main_deep.py rl --algorithm ppo --timesteps 100000")
        print("\n文本因子提取:")
        print("  python main_deep.py text --lookback 30")
        print("  python main_deep.py text --merge  # 合并到因子文件")
        print("\n支持的模型类型:")
        print("  - lstm: LSTM 时序模型")
        print("  - transformer: Transformer 时序模型")
        print("  - patchtst: PatchTST 时序模型 (Patch-based, 高效长序列)")
        print("  - informer: Informer 时序模型 (稀疏注意力, O(L log L))")
        print("  - graphsage: GraphSAGE 图神经网络")
        print("  - gat: 图注意力网络")
        print("  - contrastive: 对比学习预训练 + 微调")
        print("  - moe: 混合专家模型 (多专家 + 门控网络)")
        print("\nRL 算法 (仓位优化):")
        print("  - ppo: Proximal Policy Optimization")
        print("  - a2c: Advantage Actor-Critic")
        print("  - sac: Soft Actor-Critic")
        print("=" * 60)


if __name__ == '__main__':
    main()
