"""
深度学习选股模块

该模块独立于现有系统，提供基于深度学习的选股能力：
- Phase 0: 评估指标 (metrics.py)
- Phase 1: 序列模型 LSTM/Transformer (temporal_models.py)
- Phase 2: 图模型 GraphSAGE/GAT (gnn_models.py)
- Phase 3: 对比学习预训练 (contrastive.py)
- Phase 4: MoE专家模型 (moe_models.py)
- Phase 5: RL仓位优化 (rl_env.py, rl_trainer.py)
- Phase 6: 文本因子 (text_factors.py)

使用方式:
    python main_deep.py train --model transformer
    python main_deep.py backtest --model transformer
    python main_deep.py signal --model transformer

注意: 需要先安装 PyTorch: pip install -r requirements_deep.txt
"""

# 延迟导入，避免在没有 PyTorch 时报错
_TORCH_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

# 评估指标模块不依赖 PyTorch，可以直接导入
from .metrics import (
    compute_rank_ic,
    compute_ic_ir,
    compute_top_n_excess,
    compute_cost_adjusted_return,
    compute_turnover,
    evaluate_model_comprehensive,
    print_evaluation_report,
    # 显著性检验
    compute_ic_significance,
    bootstrap_ci,
    compute_ic_bootstrap_ci,
    compute_excess_return_significance,
    compare_models,
    evaluate_model_with_significance,
    # Block Bootstrap (时序感知)
    block_bootstrap_ci,
    block_bootstrap_ic_ci,
    # Transfer Coefficient
    compute_transfer_coefficient,
    compute_tc_by_period,
    # 风格归因
    compute_style_attribution,
    compute_style_exposure,
    # 增强版评估
    evaluate_model_enhanced,
)

# 策略模块不依赖 PyTorch
from .policies import (
    PriceType,
    ExecutionPolicy,
    TradeabilityPolicy,
    CostModel,
    TradingPolicies,
    get_default_policies,
)

# 数据契约模块不依赖 PyTorch
from .contracts import (
    CacheMeta,
    FactorFrame,
    SequenceBatch,
    GraphBundle,
    generate_universe_id,
    validate_cache_compatibility,
    # Config Hash 机制
    CONFIG_HASH_KEYS,
    compute_config_hash,
    get_cache_config_hash,
    check_cache_validity,
    create_cache_meta,
)

# 组合构建器模块不依赖 PyTorch
from .portfolio_builder import (
    OptimizationLevel,
    PortfolioConfig,
    PortfolioResult,
    PortfolioBuilder,
    build_portfolio,
)

# 图构建器模块（基础功能不依赖 PyTorch）
from .graph_builder import (
    StockGraphBuilder,
    build_stock_graph,
)

# Regime 分析模块不依赖 PyTorch
from .regime_analysis import (
    RegimeAnalyzer,
    get_index_returns,
    evaluate_with_regime,
    print_regime_report,
)

# RL 模块（依赖 gymnasium，可选）
_RL_AVAILABLE = False
try:
    from .rl_env import (
        PortfolioEnv, SimplePortfolioEnv, RewardCalculator,
        create_portfolio_env, create_simple_env, check_gym_available
    )
    _RL_AVAILABLE = True
except ImportError:
    pass

# 文本因子模块（依赖 akshare，可选）
_TEXT_AVAILABLE = False
try:
    from .text_factors import (
        TextFactorExtractor, TextFactorBuilder,
        BaseSentimentAnalyzer, TransformerSentimentAnalyzer,
        ResearchReportFetcher, AnnouncementFetcher,
        create_text_extractor, extract_text_factors,
        check_text_dependencies,
        POSITIVE_WORDS, NEGATIVE_WORDS, RATING_MAP,
    )
    _TEXT_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    # 评估指标
    'compute_rank_ic',
    'compute_ic_ir',
    'compute_top_n_excess',
    'compute_cost_adjusted_return',
    'compute_turnover',
    'evaluate_model_comprehensive',
    'print_evaluation_report',
    # 显著性检验
    'compute_ic_significance',
    'bootstrap_ci',
    'compute_ic_bootstrap_ci',
    'compute_excess_return_significance',
    'compare_models',
    'evaluate_model_with_significance',
    # Block Bootstrap (时序感知)
    'block_bootstrap_ci',
    'block_bootstrap_ic_ci',
    # Transfer Coefficient
    'compute_transfer_coefficient',
    'compute_tc_by_period',
    # 风格归因
    'compute_style_attribution',
    'compute_style_exposure',
    # 增强版评估
    'evaluate_model_enhanced',
    # 策略
    'PriceType',
    'ExecutionPolicy',
    'TradeabilityPolicy',
    'CostModel',
    'TradingPolicies',
    'get_default_policies',
    # 数据契约
    'CacheMeta',
    'FactorFrame',
    'SequenceBatch',
    'GraphBundle',
    'generate_universe_id',
    'validate_cache_compatibility',
    # Config Hash 机制
    'CONFIG_HASH_KEYS',
    'compute_config_hash',
    'get_cache_config_hash',
    'check_cache_validity',
    'create_cache_meta',
    # 组合构建器
    'OptimizationLevel',
    'PortfolioConfig',
    'PortfolioResult',
    'PortfolioBuilder',
    'build_portfolio',
    # 图构建器
    'StockGraphBuilder',
    'build_stock_graph',
    # Regime 分析
    'RegimeAnalyzer',
    'get_index_returns',
    'evaluate_with_regime',
    'print_regime_report',
]

# RL 模块导出（如果可用）
if _RL_AVAILABLE:
    __all__.extend([
        'PortfolioEnv',
        'SimplePortfolioEnv',
        'RewardCalculator',
        'create_portfolio_env',
        'create_simple_env',
        'check_gym_available',
    ])

    # RL 训练器（依赖 stable-baselines3）
    try:
        from .rl_trainer import (
            RLPortfolioTrainer, RLWeightOptimizer,
            create_rl_trainer, train_rl_agent, check_sb3_available
        )
        __all__.extend([
            'RLPortfolioTrainer',
            'RLWeightOptimizer',
            'create_rl_trainer',
            'train_rl_agent',
            'check_sb3_available',
        ])
    except ImportError:
        pass

# 文本因子模块导出（如果可用）
if _TEXT_AVAILABLE:
    __all__.extend([
        'TextFactorExtractor',
        'TextFactorBuilder',
        'BaseSentimentAnalyzer',
        'TransformerSentimentAnalyzer',
        'ResearchReportFetcher',
        'AnnouncementFetcher',
        'create_text_extractor',
        'extract_text_factors',
        'check_text_dependencies',
        'POSITIVE_WORDS',
        'NEGATIVE_WORDS',
        'RATING_MAP',
    ])

# 如果 PyTorch 可用，导入深度学习模块
if _TORCH_AVAILABLE:
    from .base import DeepStockModel, EnhancedPurgedCV, load_deep_config
    from .losses import (
        ListwiseLoss, PairwiseLoss, NDCGLoss,
        ApproxNDCGLoss, RankMSELoss, get_loss_function
    )
    from .sequence_dataset import (
        FactorSequenceDataset, FactorSequenceBuilder, create_dataloaders
    )
    from .temporal_models import (
        LSTMEncoder, TransformerEncoder, PatchTSTEncoder, InformerEncoder,
        TemporalStockModel, LSTMStockModel, TransformerStockModel,
        PatchTSTStockModel, InformerStockModel, get_temporal_model
    )
    from .gnn_models import (
        GraphSAGEEncoder, GATEncoder, GraphormerEncoder, GNNStockModel,
        TemporalGNNModel, get_gnn_model, get_temporal_gnn_model
    )
    from .contrastive import (
        TimeSeriesAugmentation, IndustryAugmentation,
        InfoNCELoss, NTXentLoss, TripletLoss,
        FactorContrastiveLearning, ContrastiveStockModel,
        get_contrastive_model, pretrain_contrastive
    )
    from .moe_models import (
        MarketStateExtractor, LearnedGatingNetwork,
        Expert, MixtureOfExpertsLayer, MixtureOfExpertsModel,
        get_moe_model, create_expert_ensemble
    )

    __all__.extend([
        # 基础
        'DeepStockModel',
        'EnhancedPurgedCV',
        'load_deep_config',
        # 损失函数
        'ListwiseLoss',
        'PairwiseLoss',
        'NDCGLoss',
        'ApproxNDCGLoss',
        'RankMSELoss',
        'get_loss_function',
        # 序列数据集
        'FactorSequenceDataset',
        'FactorSequenceBuilder',
        'create_dataloaders',
        # 时序模型
        'LSTMEncoder',
        'TransformerEncoder',
        'PatchTSTEncoder',
        'InformerEncoder',
        'TemporalStockModel',
        'LSTMStockModel',
        'TransformerStockModel',
        'PatchTSTStockModel',
        'InformerStockModel',
        'get_temporal_model',
        # GNN 模型
        'GraphSAGEEncoder',
        'GATEncoder',
        'GraphormerEncoder',
        'GNNStockModel',
        'TemporalGNNModel',
        'get_gnn_model',
        'get_temporal_gnn_model',
        # 对比学习
        'TimeSeriesAugmentation',
        'IndustryAugmentation',
        'InfoNCELoss',
        'NTXentLoss',
        'TripletLoss',
        'FactorContrastiveLearning',
        'ContrastiveStockModel',
        'get_contrastive_model',
        'pretrain_contrastive',
        # MoE 专家模型
        'MarketStateExtractor',
        'LearnedGatingNetwork',
        'Expert',
        'MixtureOfExpertsLayer',
        'MixtureOfExpertsModel',
        'get_moe_model',
        'create_expert_ensemble',
    ])


def is_torch_available() -> bool:
    """检查 PyTorch 是否可用"""
    return _TORCH_AVAILABLE


def is_rl_available() -> bool:
    """检查 RL 模块是否可用"""
    return _RL_AVAILABLE


def is_text_available() -> bool:
    """检查文本因子模块是否可用"""
    return _TEXT_AVAILABLE
