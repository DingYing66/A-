# A股多因子选股系统

基于 Python + AKShare 的量化多因子选股系统，支持因子计算、机器学习模型训练、历史回测和实时信号生成。

## 功能特性

- **多因子选股**: 质量、估值、动量、资金流、情绪五大类因子
- **机器学习**: 支持 XGBoost/LightGBM 模型训练预测
- **历史回测**: 完整的回测引擎，支持交易成本、滑点、涨跌停处理
- **因子缓存**: 计算过的因子自动缓存，避免重复计算
- **仓位管理**: 根据市场环境动态调整仓位比例
- **信号生成**: 生成每日选股信号和权重建议

## 快速开始

### 环境要求

- Python 3.8+
- 依赖包见 `requirements.txt`

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd AKshare

# 安装依赖
pip install -r requirements.txt
```

### 基本用法

```bash
# 1. 下载数据
python main.py download

# 2. 运行回测
python main.py backtest --start 20180101 --end 20241218

# 3. 生成选股信号
python main.py signal

# 4. 完整流程 (下载 + 回测 + 信号)
python main.py run --start 20180101 --end 20241218
```

## 项目结构

```
AKshare/
├── main.py                 # 主程序入口
├── config/
│   └── config.yaml         # 配置文件
├── src/
│   ├── data_fetcher.py     # 数据采集 (AKShare接口)
│   ├── data_processor.py   # 数据处理
│   ├── factor_engine.py    # 因子计算
│   ├── scorer.py           # 因子打分
│   ├── ml_model.py         # 机器学习模型
│   ├── backtester.py       # 回测引擎
│   ├── performance.py      # 绩效分析
│   └── utils.py            # 工具函数
├── data/
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后数据 (含因子缓存)
├── output/                 # 回测结果和信号输出
└── docs/
    ├── ALGORITHM.md        # 算法详解
    └── DATA_FUNCTIONS.md   # 数据函数说明
```

## 命令行参数

### download - 数据下载

```bash
python main.py download [--codes 000001,000002]
```

| 参数 | 说明 |
|------|------|
| `--codes` | 指定股票代码，逗号分隔；不指定则下载全市场 |

### backtest - 运行回测

```bash
python main.py backtest [options]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--start` | 回测开始日期 (YYYYMMDD) | 配置文件 |
| `--end` | 回测结束日期 | 配置文件 |
| `--top-n` | 选股数量 | 50 |
| `--model` | ML模型类型 (xgboost/lightgbm/linear) | 无 |
| `--walk-forward` | 使用滚动回测 | False |
| `--name` | 结果保存名称 | 日期时间 |
| `--no-plot` | 不显示图表 | False |

### signal - 生成信号

```bash
python main.py signal [--top-n 50] [--model xgboost] [--model-path path]
```

### run - 完整流程

```bash
python main.py run [--skip-download] [其他回测参数]
```

## 因子体系

### 因子分类

| 类别 | 权重 | 主要因子 |
|------|------|----------|
| 质量 (Quality) | 30% | ROE, ROA, 毛利率, 净利率, 现金流比率 |
| 估值 (Valuation) | 20% | PE, PB, PS, 股息率 |
| 动量 (Momentum) | 15% | 20/60/120日收益率, 创新高, 连涨天数 |
| 资金 (Flow) | 20% | 主力净流入, 北向资金变化 |
| 情绪 (Sentiment) | 15% | 相对强度, 短期反转, 量价配合 |

### 因子缓存

系统自动缓存计算过的因子数据：

```
data/processed/factors/
├── factors_20180115.parquet
├── factors_20180130.parquet
└── ...
```

首次运行会计算并保存因子，后续运行直接加载缓存。

清除缓存：
```bash
# Windows
del /Q data\processed\factors\*.parquet

# Linux/Mac
rm -rf data/processed/factors/*.parquet
```

## 配置说明

主要配置项 (`config/config.yaml`)：

```yaml
# 因子权重
weights:
  quality: 0.30
  valuation: 0.20
  momentum: 0.15
  flow: 0.20
  sentiment: 0.15

# 回测参数
backtest:
  rebalance_freq: biweekly  # 调仓频率
  top_n: 50                 # 选股数量
  max_position: 0.5         # 单票最大仓位
  commission: 0.001         # 佣金率
  slippage: 0.001           # 滑点
  benchmark: "000300"       # 基准指数
```

## 输出示例

### 回测绩效报告

```
================== 回测绩效报告 ==================
回测区间: 2018-01-01 ~ 2024-12-18

【收益指标】
  总收益率:      156.32%
  年化收益率:    14.56%
  基准收益率:    23.45%
  超额收益:      132.87%

【风险指标】
  年化波动率:    22.34%
  最大回撤:      -28.67%
  夏普比率:      0.65
  信息比率:      0.89

【其他指标】
  胜率:          56.78%
  盈亏比:        1.45
  换手率:        125.6%
```

### 选股信号报告

```
======================================================================
                      选 股 信 号 报 告
======================================================================

【市场环境分析】
  市场趋势(20日): +2.35%
  市场广度(赚钱效应): 58.2%
  市场波动率: 18.5%
  风险等级: 正常

【仓位建议】
  建议总仓位: 100%
  单票权重: 2.00%

【选股结果】共 50 只
----------------------------------------------------------------------
排名   代码        得分        权重        行业
----------------------------------------------------------------------
1     000001     0.8523      2.00%       银行
2     000002     0.8456      2.00%       房地产
...
```

## 文档

- [算法原理详解](docs/ALGORITHM.md) - 因子定义、打分模型、回测逻辑
- [数据函数说明](docs/DATA_FUNCTIONS.md) - 模块依赖、数据接口、数据流向

## 注意事项

1. **数据源**: 使用 AKShare 免费数据接口，请求间隔建议 1.5 秒以上
2. **回测局限**: 回测结果仅供参考，实盘需考虑流动性、涨跌停等因素
3. **因子时效**: 因子可能随市场环境变化而失效，建议定期评估
4. **风险提示**: 量化投资有风险，请谨慎决策

## 技术栈

- **数据获取**: AKShare
- **数据处理**: Pandas, NumPy
- **机器学习**: Scikit-learn, XGBoost, LightGBM
- **可视化**: Matplotlib
- **存储格式**: Parquet

## License

MIT License

## 更新日志

### v2.0 (2024-12)
- 新增情绪因子类别 (sentiment)
- 新增因子缓存机制，加速回测
- 新增仓位管理功能，根据市场环境动态调整
- 统一执行配置 (收盘价换仓)
- 优化机器学习训练 (PurgedGroupTimeSeriesSplit)

### v1.0 (2024-11)
- 初始版本
- 支持质量、估值、动量、资金流四大类因子
- 支持 XGBoost/LightGBM 模型
- 完整回测引擎
