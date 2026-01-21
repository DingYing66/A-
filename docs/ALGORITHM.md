# A股多因子选股系统 - 算法原理与流程详解

## 目录
1. [系统概述](#1-系统概述)
2. [整体架构](#2-整体架构)
3. [因子计算引擎](#3-因子计算引擎)
4. [多因子打分模型](#4-多因子打分模型)
5. [机器学习模型](#5-机器学习模型)
6. [回测引擎](#6-回测引擎)
7. [仓位管理与风控](#7-仓位管理与风控)
8. [绩效分析](#8-绩效分析)
9. [因子缓存机制](#9-因子缓存机制)
10. [数学公式汇总](#10-数学公式汇总)
11. [参数配置说明](#11-参数配置说明)

---

## 1. 系统概述

### 1.1 系统目标
构建一个量化多因子选股系统，通过分析股票的**质量、估值、动量、资金流、情绪**等多维度因子，结合机器学习模型，筛选出具有投资价值的股票组合。

### 1.2 核心理念
- **多因子模型**: 单一因子容易失效，多因子组合更稳健
- **因子分类**: 将因子分为质量、估值、动量、资金流、情绪五大类
- **避免过拟合**: 使用时间序列交叉验证、滚动训练、Purge隔离
- **防止未来函数**: 财务数据按公告日对齐，而非报告期
- **因子缓存**: 计算过的因子数据缓存至本地，避免重复计算
- **动态仓位**: 根据市场环境动态调整仓位比例

### 1.3 适用场景
- 持仓周期: 1-8周（中短期）
- 调仓频率: 双周调仓
- 选股数量: 20-50只
- 适合资金量: 50万-5000万

---

## 2. 整体架构

### 2.1 系统流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据层 (Data Layer)                        │
├─────────────────────────────────────────────────────────────────┤
│  股票列表  │  日K线数据  │  财务数据  │  资金流向  │  北向资金    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      因子层 (Factor Layer)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ 质量因子 │  │ 估值因子 │  │ 动量因子 │  │ 资金因子 │        │
│  │  (30%)   │  │  (15%)   │  │  (15%)   │  │  (15%)   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ 情绪因子 │  │ 趋势质量 │  │ 机构因子 │  │ 市场环境 │        │
│  │  (25%)   │  │  (合并)  │  │  (合并)  │  │ (仓位控制)│        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│       │             │             │             │               │
│       └─────────────┴─────────────┴─────────────┘               │
│                           │                                      │
│              ┌────────────┴────────────┐                        │
│              │      因子标准化          │                        │
│              │ (Winsorize + Z-score)   │                        │
│              └─────────────────────────┘                        │
│                           │                                      │
│              ┌────────────┴────────────┐                        │
│              │      因子缓存            │                        │
│              │ (Parquet文件)           │                        │
│              └─────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      模型层 (Model Layer)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌────────────────────┐    ┌────────────────────┐            │
│    │   多因子打分模型    │    │   机器学习模型      │            │
│    │   (Factor Score)   │    │  (XGBoost/LGBM)    │            │
│    └────────────────────┘    └────────────────────┘            │
│              │                         │                        │
│              └───────────┬─────────────┘                        │
│                          │                                      │
│                   ┌──────▼──────┐                               │
│                   │  集成模型    │                               │
│                   │  (Ensemble) │                               │
│                   └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      回测层 (Backtest Layer)                     │
├─────────────────────────────────────────────────────────────────┤
│  调仓执行  │  交易成本  │  涨跌停处理  │  滑点模拟  │  净值计算  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐│
│  │                    仓位管理 (P5)                            ││
│  │  市场环境分析 → 动态仓位比例 → 风险控制                      ││
│  └────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      分析层 (Analysis Layer)                     │
├─────────────────────────────────────────────────────────────────┤
│  年化收益  │  夏普比率  │  最大回撤  │  Alpha/Beta  │  信息比率  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 模块关系

| 模块 | 文件 | 功能 |
|------|------|------|
| 主程序入口 | `main.py` | 命令行接口，调度各模块 |
| 数据采集 | `src/data_fetcher.py` | 从AKShare获取行情、财务、资金流数据 |
| 数据处理 | `src/data_processor.py` | 数据加载、对齐、预处理、交易日历 |
| 因子计算 | `src/factor_engine.py` | 计算各类因子并标准化，含缓存机制 |
| 因子打分 | `src/scorer.py` | 多因子加权打分、约束选股 |
| 机器学习 | `src/ml_model.py` | XGBoost/LightGBM 训练预测 |
| 回测引擎 | `src/backtester.py` | 模拟交易、计算净值、仓位管理 |
| 绩效分析 | `src/performance.py` | 计算绩效指标、生成报告 |
| 工具函数 | `src/utils.py` | 通用工具函数 |

---

## 3. 因子计算引擎

### 3.1 因子分类与定义

#### 3.1.1 质量因子 (Quality Factors) - 权重30%
评估公司的盈利质量和经营效率。

| 因子名 | 计算公式 | 方向 | 说明 |
|--------|----------|------|------|
| roe | 净利润 / 净资产 | + | 净资产收益率，越高越好 |
| roa | 净利润 / 总资产 | + | 总资产收益率 |
| gross_margin | (营收-成本) / 营收 | + | 毛利率 |
| net_margin | 净利润 / 营收 | + | 净利率 |
| ocf_ratio | 经营现金流 / 净利润 | + | 现金流质量 |

#### 3.1.2 估值因子 (Valuation Factors) - 权重15%
评估股票的相对估值水平。

| 因子名 | 计算公式 | 方向 | 说明 |
|--------|----------|------|------|
| pe_ttm | 股价 / EPS(TTM) | - | 市盈率，越低越好 |
| pb | 股价 / 每股净资产 | - | 市净率 |
| ps_ttm | 市值 / 营收(TTM) | - | 市销率 |
| div_yield | 每股股息 / 股价 | + | 股息率，越高越好 |
| pe_relative | PE / 历史均值 | - | 相对PE |
| pb_relative | PB / 历史均值 | - | 相对PB |

#### 3.1.3 动量因子 (Momentum Factors) - 权重15%
捕捉价格趋势和市场情绪。

| 因子名 | 计算公式 | 方向 | 说明 |
|--------|----------|------|------|
| ret_20d | (P_t - P_{t-20}) / P_{t-20} | + | 20日收益率 |
| ret_60d | (P_t - P_{t-60}) / P_{t-60} | + | 60日收益率 |
| ret_120d | (P_t - P_{t-120}) / P_{t-120} | + | 120日收益率 |
| new_high | 是否创60日新高 | + | 创新高信号 |
| consecutive_up | 连续上涨天数 | + | 趋势强度 |

**动量因子的"跳空"处理**:
```
实际计算: ret_N = (P_{t-skip} - P_{t-skip-N}) / P_{t-skip-N}

其中 skip = momentum_skip_days (默认5天)
```
跳过最近几天是为了**避免短期反转效应**，捕捉更稳定的中期动量。

#### 3.1.4 资金流因子 (Flow Factors) - 权重15%
跟踪机构资金动向。

| 因子名 | 计算公式 | 方向 | 说明 |
|--------|----------|------|------|
| net_inflow_5d | Σ(5日主力净流入) | + | 5日资金净流入 |
| net_inflow_20d | Σ(20日主力净流入) | + | 20日资金净流入 |
| north_change | 北向持股变化 | + | 外资动向 |
| north_hold_change_20d | 20日北向持股变化 | + | 北向中期趋势 |
| north_hold_change_5d | 5日北向持股变化 | + | 北向短期趋势 |

#### 3.1.5 情绪/赚钱效应因子 (Sentiment Factors) - 权重25%
**核心改进因子**，衡量市场情绪和赚钱效应。

| 因子名 | 计算公式 | 方向 | 说明 |
|--------|----------|------|------|
| relative_strength_20d | 个股涨幅 / 市场涨幅 | + | 20日相对强度 |
| relative_strength_60d | 个股涨幅 / 市场涨幅 | + | 60日相对强度 |
| reversal_5d | 5日收益率 | - | 短期反转（均值回归） |
| reversal_10d | 10日收益率 | - | 短期反转 |
| price_position | 价格在N日区间位置 | + | 价格位置 (0-1) |
| volume_ratio | 成交量 / MA成交量 | + | 量能放大 |
| up_down_volume_ratio | 上涨日成交量/下跌日成交量 | + | 量价配合 |
| atr_contraction | ATR收缩比例 | + | 波动收敛 |

#### 3.1.6 趋势质量因子 (Trend Quality Factors)
评估趋势的质量和可持续性，合并到动量类别中使用。

| 因子名 | 计算公式 | 方向 | 说明 |
|--------|----------|------|------|
| dist_to_high_20d | 当前价/20日高点 | + | 距高点比例 |
| dist_to_high_60d | 当前价/60日高点 | + | 距高点比例 |
| ma20_bias | (价格-MA20)/MA20 | + | 乖离率 |
| ma20_slope | MA20斜率 | + | 均线趋势 |
| trend_strength | 趋势强度综合 | + | 综合趋势 |
| current_drawdown | 当前回撤 | - | 越小越好 |
| consecutive_up_days | 连涨天数 | + | 趋势持续性 |

#### 3.1.7 机构因子 (Institution Factors)
追踪机构持股变化，合并到资金流类别中使用。

| 因子名 | 计算公式 | 方向 | 说明 |
|--------|----------|------|------|
| inst_count_change | 机构数量变化 | + | 机构关注度 |
| inst_ratio_change | 持股比例变化 | + | 机构增仓 |
| new_inst_count | 新进机构数 | + | 新增关注 |

#### 3.1.8 市场环境因子 (Market Regime Factors)
用于动态仓位调整，不参与选股打分。

| 因子名 | 计算公式 | 方向 | 说明 |
|--------|----------|------|------|
| market_trend | 市场指数20日涨幅 | + | 市场趋势 |
| market_breadth | 上涨股票占比 | + | 赚钱效应 |
| market_volatility | 市场波动率 | - | 低波动好 |

### 3.2 因子标准化

#### 3.2.1 Winsorize (缩尾处理)
去除极端值的影响：
```python
def winsorize(x, lower=0.01, upper=0.99):
    """将数据截断到指定分位数范围"""
    lower_bound = x.quantile(lower)
    upper_bound = x.quantile(upper)
    return x.clip(lower_bound, upper_bound)
```

#### 3.2.2 Z-Score 标准化
将因子值转换为标准正态分布：
```
z = (x - μ) / σ

其中:
- μ = 横截面均值
- σ = 横截面标准差
```

#### 3.2.3 完整标准化流程
```python
def standardize(x):
    """Winsorize + Z-score"""
    x = winsorize(x, 0.01, 0.99)  # 1. 缩尾
    x = (x - x.mean()) / x.std()  # 2. Z-score
    return x
```

### 3.3 财务数据对齐（防止未来函数）

**问题**: 财务报表有"报告期"和"公告日"之分
- 报告期: 如2024Q3报告期为2024-09-30
- 公告日: 实际发布日期，如2024-10-28

**错误做法**: 使用报告期对齐数据（会用到未来数据）

**正确做法**: 按公告日对齐
```python
def align_financial_by_announce_date(fin_df, dates):
    """
    将财务数据按公告日对齐到交易日
    确保在任意交易日T，只使用T之前已公告的财务数据
    """
    result = []
    for date in dates:
        # 找到该日期之前最近一次公告的财务数据
        available = fin_df[fin_df['公告日期'] <= date]
        if len(available) > 0:
            latest = available.iloc[-1]  # 最新一条
            result.append(latest)
    return pd.DataFrame(result)
```

---

## 4. 多因子打分模型

### 4.1 打分流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 因子横截面   │ ──▶ │ 单因子打分   │ ──▶ │ 类内加权    │ ──▶ │ 类间加权    │
│ Factor_df   │     │ Percentile  │     │ Category    │     │ Total Score │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### 4.2 单因子打分
将因子值转换为百分位得分（0-1之间）：

```python
def compute_factor_score(values, direction):
    """
    direction = 1:  值越大得分越高
    direction = -1: 值越小得分越高
    """
    ranks = values.rank(ascending=(direction == 1))
    scores = (ranks - 1) / (len(ranks) - 1)  # 归一化到 [0, 1]
    return scores
```

**示例**:
| 股票 | ROE原值 | 排名 | ROE得分 |
|------|---------|------|---------|
| A | 20% | 5 | 1.00 |
| B | 15% | 4 | 0.75 |
| C | 12% | 3 | 0.50 |
| D | 8% | 2 | 0.25 |
| E | 5% | 1 | 0.00 |

### 4.3 类内加权
同一类别内的因子等权平均：

```
Quality_Score = (ROE_score + ROA_score + gross_margin_score) / 3
```

### 4.4 类间加权
不同类别按配置权重加权（当前版本配置）：

```
Total_Score = w_quality × Quality_Score
            + w_valuation × Valuation_Score
            + w_momentum × Momentum_Score
            + w_flow × Flow_Score
            + w_sentiment × Sentiment_Score

当前权重配置:
- quality: 0.30 (质量)
- valuation: 0.15 (估值)
- momentum: 0.15 (动量，熊市容易失效故降权)
- flow: 0.15 (资金流)
- sentiment: 0.25 (情绪/赚钱效应，核心改进)
```

### 4.5 约束选股器 (ConstrainedSelector)

除了基本的得分排序，还支持约束选股：

```python
class ConstrainedSelector:
    """约束选股器 (P1-3)"""

    def select_with_constraints(self, factor_df, n=50):
        # 1. 基础得分筛选
        candidates = factor_df.nlargest(n * 2, 'total_score')

        # 2. 行业约束：单行业不超过20%
        candidates = self.apply_industry_constraint(candidates)

        # 3. 流动性约束：日均成交额>1000万
        candidates = self.apply_liquidity_constraint(candidates)

        # 4. 最终选取top N
        return candidates.head(n)
```

---

## 5. 机器学习模型

### 5.1 模型选择

| 模型 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **XGBoost** | 精度高、可解释 | 训练慢、易过拟合 | 特征较多时 |
| **LightGBM** | 速度快、内存小 | 对小数据集不友好 | 大数据集 |
| **Ridge** | 简单、稳定 | 非线性捕捉弱 | 基准模型 |

### 5.2 特征工程

**输入特征**: 标准化后的所有因子值
```
X = [roe, roa, gross_margin, pe_ttm, pb, ret_20d, ret_60d,
     relative_strength_20d, volume_ratio, ...]
```

**标签(Y)**: 未来N日超额收益率（动态标签）
```python
def compute_forward_returns(codes, start_date, forward_days=20):
    """
    计算未来收益率（作为训练标签）

    P4-2改进: 使用动态标签，forward_days根据实际调仓间隔确定

    Y = (P_{t+N} - P_t) / P_t - benchmark_return
    """
    stock_returns = {}
    for code in codes:
        # 计算个股收益（close→close统一口径）
        ret = (future_close - current_close) / current_close
        stock_returns[code] = ret

    # 减去基准收益（超额收益）
    benchmark_ret = get_benchmark_return(start_date, forward_days)
    return stock_returns - benchmark_ret
```

### 5.3 时间序列交叉验证 (PurgedGroupTimeSeriesSplit)

**为什么不能用普通K-Fold?**
- 股票数据有时间依赖性
- 普通K-Fold会导致"未来信息泄露"

**TimeSeriesSplit + Purge 方法**:
```
时间 ─────────────────────────────────────────▶

Fold 1: [Train     ] [Purge] [Val]
Fold 2: [Train          ] [Purge] [Val]
Fold 3: [Train               ] [Purge] [Val]
Fold 4: [Train                    ] [Purge] [Val]
Fold 5: [Train                         ] [Purge] [Val]

Purge区间 = 标签计算窗口，确保训练集标签不与验证集时间重叠
```

```python
class PurgedGroupTimeSeriesSplit:
    """
    时间序列交叉验证 + Purge (P0-3修复版)

    参考: López de Prado, Advances in Financial Machine Learning (2018)
    """

    def __init__(self, n_splits=5, purge_days=20):
        self.n_splits = n_splits
        self.purge_days = purge_days  # 标签窗口

    def split(self, X, y=None, groups=None):
        # Purge: 训练集不能包含 val_start - purge_days 之后的日期
        # 这确保训练集的标签不会与验证集日期重叠
        ...
```

### 5.4 IC (Information Coefficient)

**定义**: 预测值与实际收益的相关系数
```
IC = corr(y_pred, y_actual)
```

**解读**:
| IC值 | 评价 |
|------|------|
| > 0.05 | 优秀 |
| 0.03-0.05 | 良好 |
| 0.01-0.03 | 一般 |
| < 0.01 | 较弱 |

### 5.5 集成模型

结合多因子打分和ML预测：

```python
def predict_ensemble(factor_df, ml_weight=0.5):
    """
    Ensemble_Score = (1-w) × Factor_Score + w × ML_Score

    参数:
    - ml_weight: ML模型权重 (0-1)
    """
    # 多因子打分
    factor_score = scorer.compute_total_score(factor_df)

    # ML预测
    ml_score = ml_model.predict(factor_df)
    ml_score = normalize(ml_score)  # 归一化到 [0, 1]

    # 集成
    ensemble = (1 - ml_weight) * factor_score + ml_weight * ml_score
    return ensemble
```

### 5.6 Walk-Forward 滚动训练

**问题**: 一次性训练的模型可能在后期失效

**解决**: 定期重新训练模型

```
时间轴: ════════════════════════════════════════════▶

训练窗口1: [████████████]
                        ▼ 预测期1
                        [====]

训练窗口2:    [████████████]
                            ▼ 预测期2
                            [====]

训练窗口3:        [████████████]
                                ▼ 预测期3
                                [====]
```

```python
class WalkForwardBacktester:
    """滚动回测器"""

    def run_walk_forward(self, start_date, end_date,
                         train_years=3, retrain_freq='yearly'):
        for rebalance_date in rebalance_dates:
            if need_retrain(rebalance_date):
                # 使用过去3年数据重新训练
                train_end = rebalance_date - 1day
                train_start = train_end - 3years

                X, y = build_training_samples(train_start, train_end)
                model.fit(X, y)

            # 使用当前模型预测
            factor_df = compute_factors(rebalance_date)
            selected = model.select_stocks(factor_df)

            # 执行调仓
            rebalance(selected)
```

---

## 6. 回测引擎

### 6.1 回测流程

```
┌────────────────────────────────────────────────────────────────┐
│                      初始化                                      │
│  - 设置初始资金 (默认100万)                                      │
│  - 清空持仓                                                      │
│  - 获取交易日历                                                  │
└────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                    遍历每个交易日                                 │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 是否为调仓日?                                              │  │
│  │                                                            │  │
│  │  是 ──▶ 1. 加载/计算因子（优先使用缓存）                   │  │
│  │         2. 打分选股                                        │  │
│  │         3. 市场环境分析                                    │  │
│  │         4. 计算目标仓位                                    │  │
│  │         5. 执行调仓（收盘价）                              │  │
│  │                                                            │  │
│  │  否 ──▶ 保持持仓不变                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                               │                                  │
│                               ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 计算当日组合市值 = Σ(持仓股数 × 收盘价) + 现金            │  │
│  │ 记录净值                                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                      输出结果                                    │
│  - 净值曲线                                                      │
│  - 交易记录                                                      │
│  - 持仓历史                                                      │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 执行配置 (P5统一口径)

系统使用统一的收盘价执行口径：

```yaml
execution:
  execution_price: "close"      # 执行价格类型
  trade_at_close: true          # 同日收盘执行
  limit_check_price: "close"    # 涨跌停判断基准
  label_entry: "close"          # ML标签入场价格
  label_exit: "close"           # ML标签出场价格
```

### 6.3 调仓执行逻辑

#### 6.3.1 收盘换仓模式 (默认)
```
T日收盘后:
  - 计算T日因子（使用T日收盘价）
  - 打分选股
  - 确定目标持仓
  - 按T日收盘价执行调仓（模拟收盘集合竞价）
```

#### 6.3.2 先卖后买
```python
def rebalance(current_holdings, target_holdings, date, cash):
    # 1. 先卖出需要减仓的
    for code in current_holdings:
        if code not in target_holdings or target_holdings[code] < current_holdings[code]:
            sell_shares = current_holdings[code] - target_holdings.get(code, 0)
            cash += sell_shares * sell_price - transaction_cost

    # 2. 再买入需要加仓的
    for code in target_holdings:
        if code not in current_holdings or target_holdings[code] > current_holdings[code]:
            buy_shares = target_holdings[code] - current_holdings.get(code, 0)
            if cash >= buy_shares * buy_price + transaction_cost:
                cash -= buy_shares * buy_price + transaction_cost
                holdings[code] = target_holdings[code]
```

### 6.4 交易成本模型

```python
def calc_transaction_cost(amount, direction):
    """
    交易成本 = 佣金 + 印花税(仅卖出)

    当前配置:
    - 佣金: 0.1% (千一)
    - 印花税: 0.1% (千一，仅卖出)
    - 滑点: 0.1%
    """
    commission = amount * 0.001  # 千一佣金
    slippage = amount * 0.001    # 0.1%滑点

    if direction == 'sell':
        stamp_tax = amount * 0.001  # 千一印花税(单边)
        return commission + stamp_tax + slippage

    return commission + slippage
```

### 6.5 涨跌停处理

```python
def check_tradeable(code, date, direction):
    """
    涨停不买，跌停不卖
    """
    pct_change = get_pct_change(code, date)

    if direction == 'buy' and pct_change >= 9.9:
        return False  # 涨停无法买入

    if direction == 'sell' and pct_change <= -9.9:
        return False  # 跌停无法卖出

    return True
```

### 6.6 停牌股处理 (P0-1修复)

```python
def get_last_valid_price(code, date):
    """
    获取最近有效收盘价（用于停牌股估值）

    A股停牌时使用最近一次有效收盘价估值，
    而非将持仓市值视为0。
    """
    if code in self._last_valid_prices:
        return self._last_valid_prices[code]

    # 查找历史最近有效价格
    price_df = load_daily_price(code)
    valid_prices = price_df[price_df['date'] <= date]
    if len(valid_prices) > 0:
        return valid_prices.iloc[-1]['close']

    return None
```

---

## 7. 仓位管理与风控

### 7.1 市场环境分析

信号生成时自动分析市场环境：

```python
def analyze_market_environment(factor_df):
    """市场环境分析"""
    market_trend = factor_df['market_trend'].iloc[0]      # 20日涨跌
    market_breadth = factor_df['market_breadth'].iloc[0]  # 赚钱效应
    market_volatility = factor_df['market_volatility'].iloc[0]  # 波动率

    return {
        'trend': market_trend,
        'breadth': market_breadth,
        'volatility': market_volatility
    }
```

### 7.2 动态仓位计算

根据市场环境动态调整仓位：

```python
def calculate_position_ratio(market_env):
    """
    计算建议仓位比例

    规则:
    - 市场20日跌超5% → 仓位×0.6
    - 市场20日下跌 → 仓位×0.8
    - 赚钱效应<30% → 仓位×0.7
    - 高波动率>30% → 仓位×0.8
    - 最终仓位限制在 [30%, 100%]
    """
    position_ratio = 1.0

    if market_env['trend'] < -0.05:
        position_ratio *= 0.6
    elif market_env['trend'] < 0:
        position_ratio *= 0.8

    if market_env['breadth'] < 0.3:
        position_ratio *= 0.7

    if market_env['volatility'] > 0.3:
        position_ratio *= 0.8

    return max(0.3, min(1.0, position_ratio))
```

### 7.3 风险等级评估

```python
def assess_risk_level(market_env):
    """风险等级评估"""
    if market_env['trend'] < -0.05 or market_env['breadth'] < 0.3:
        return "高风险"
    elif market_env['trend'] < 0:
        return "中风险"
    else:
        return "正常"
```

### 7.4 单票仓位限制

```python
# 单票最大仓位 = min(等权权重, 配置上限) × 总仓位比例
base_weight = 1.0 / top_n  # 等权
max_position = config['backtest']['max_position']  # 默认5%
adjusted_weight = min(base_weight, max_position) * position_ratio
```

---

## 8. 绩效分析

### 8.1 收益指标

#### 8.1.1 总收益率
```
Total Return = (NAV_end / NAV_start) - 1
```

#### 8.1.2 年化收益率
```
Annual Return = (1 + Total Return)^(365/days) - 1
```

#### 8.1.3 超额收益
```
Excess Return = Strategy Return - Benchmark Return
```

### 8.2 风险指标

#### 8.2.1 年化波动率
```
Volatility = std(daily_returns) × √252
```

#### 8.2.2 最大回撤
```
Max Drawdown = max((Peak - Trough) / Peak)

其中:
- Peak = 历史最高净值
- Trough = Peak之后的最低净值
```

```python
def calc_max_drawdown(nav):
    peak = nav.expanding().max()
    drawdown = (nav - peak) / peak
    return drawdown.min()  # 最小值（最大回撤是负数）
```

### 8.3 风险调整收益

#### 8.3.1 夏普比率 (Sharpe Ratio)
```
Sharpe = (R_p - R_f) / σ_p

其中:
- R_p = 策略年化收益
- R_f = 无风险利率 (默认2%)
- σ_p = 策略年化波动率
```

**解读**:
| Sharpe | 评价 |
|--------|------|
| > 2.0 | 优秀 |
| 1.0-2.0 | 良好 |
| 0.5-1.0 | 一般 |
| < 0.5 | 较差 |

#### 8.3.2 索提诺比率 (Sortino Ratio)
只考虑下行风险：
```
Sortino = (R_p - R_f) / σ_downside

其中:
- σ_downside = std(负收益率) × √252
```

#### 8.3.3 卡尔玛比率 (Calmar Ratio)
```
Calmar = Annual Return / |Max Drawdown|
```

### 8.4 相对基准指标

#### 8.4.1 Alpha
策略相对基准的超额收益（经风险调整）：
```
Alpha = R_p - R_f - β × (R_m - R_f)

其中:
- R_m = 基准收益率
- β = 策略对基准的敏感度
```

#### 8.4.2 Beta
```
Beta = Cov(R_p, R_m) / Var(R_m)
```

#### 8.4.3 信息比率 (Information Ratio)
```
IR = (R_p - R_m) / Tracking_Error

其中:
- Tracking Error = std(R_p - R_m) × √252
```

---

## 9. 因子缓存机制

### 9.1 缓存原理

因子计算是回测中最耗时的步骤。系统实现了因子缓存机制，避免重复计算：

```
首次计算:
  计算日期因子 → 保存到 data/processed/factors/factors_YYYYMMDD.parquet

后续运行:
  检查缓存是否存在 → 存在则直接加载 → 不存在则计算并保存
```

### 9.2 缓存位置

```
data/
└── processed/
    └── factors/
        ├── factors_20150115.parquet
        ├── factors_20150130.parquet
        ├── factors_20150213.parquet
        └── ...
```

### 9.3 缓存实现

```python
# factor_engine.py

def load_factor_data(self, date: str) -> pd.DataFrame:
    """加载缓存的因子数据"""
    cache_path = self.root / 'data/processed/factors' / f"factors_{date}.parquet"
    if cache_path.exists():
        return load_parquet(cache_path)
    return pd.DataFrame()

def save_factor_data(self, factor_df: pd.DataFrame, date: str) -> None:
    """保存因子数据到缓存"""
    save_dir = ensure_dir(self.root / 'data/processed/factors')
    save_path = save_dir / f"factors_{date}.parquet"
    save_parquet(factor_df, save_path)
    logger.debug(f"因子数据已缓存: {save_path}")
```

### 9.4 回测中的缓存使用

```python
# backtester.py

def run(self, start_date, end_date):
    for rb_date in rebalance_dates:
        date_str = rb_date.strftime('%Y%m%d')

        # 1. 尝试加载缓存
        factor_df = self.factor_engine.load_factor_data(date_str)

        # 2. 缓存不存在则计算
        if len(factor_df) == 0:
            factor_df = self.factor_engine.compute_factor_cross_section(rb_date)

            if len(factor_df) > 0:
                # 标准化
                factor_df = self.factor_engine.standardize_factors(factor_df)
                # 保存缓存
                self.factor_engine.save_factor_data(factor_df, date_str)

        # 3. 使用因子数据进行选股
        selected = self.scorer.select_top_stocks(factor_df)
```

### 9.5 清除缓存

如果修改了因子计算逻辑，需要清除旧缓存：

```bash
# Windows
del /Q data\processed\factors\*.parquet

# Linux/Mac
rm -rf data/processed/factors/*.parquet
```

---

## 10. 数学公式汇总

### 10.1 因子计算

| 公式 | 说明 |
|------|------|
| `ret_N = (P_t - P_{t-N}) / P_{t-N}` | N日收益率 |
| `vol = std(ret) × √252` | 年化波动率 |
| `MDD = min((NAV - Peak) / Peak)` | 最大回撤 |
| `z = (x - μ) / σ` | Z-score标准化 |
| `RS = 个股涨幅 / 市场涨幅` | 相对强度 |

### 10.2 打分模型

| 公式 | 说明 |
|------|------|
| `Score = (Rank - 1) / (N - 1)` | 百分位得分 |
| `Category = mean(factor_scores)` | 类内得分 |
| `Total = Σ(w_i × Category_i)` | 加权总分 |

### 10.3 绩效指标

| 公式 | 说明 |
|------|------|
| `Sharpe = (R - Rf) / σ` | 夏普比率 |
| `Sortino = (R - Rf) / σ_down` | 索提诺比率 |
| `Calmar = R_annual / \|MDD\|` | 卡尔玛比率 |
| `IR = (R - Rm) / TE` | 信息比率 |
| `Alpha = R - Rf - β(Rm - Rf)` | Alpha |
| `IC = corr(pred, actual)` | 信息系数 |

---

## 11. 参数配置说明

### 11.1 config.yaml 关键参数

```yaml
# 路径配置
paths:
  data_root: "data"
  raw_data: "data/raw"
  processed_data: "data/processed"
  output: "output"

# 因子参数
factors:
  momentum_windows: [20, 60, 120]   # 动量计算窗口
  momentum_skip_days: 5              # 动量跳过最近N天
  volatility_window: 20              # 波动率计算窗口
  flow_windows: [5, 20]              # 资金流计算窗口

# 权重配置 (当前版本)
weights:
  quality: 0.30      # 质量因子权重
  valuation: 0.15    # 估值因子权重
  momentum: 0.15     # 动量因子权重 (降低，熊市易失效)
  flow: 0.15         # 资金因子权重
  sentiment: 0.25    # 情绪因子权重 (核心改进)

# 回测参数
backtest:
  rebalance_freq: biweekly  # 调仓频率
  top_n: 50                 # 选股数量
  max_position: 0.05        # 单票最大仓位
  commission: 0.001         # 佣金率 (千一)
  slippage: 0.001           # 滑点
  benchmark: "000300"       # 基准指数 (沪深300)

# 机器学习参数
ml:
  train_window_years: 3     # 训练窗口
  forward_days: 20          # 预测周期
  xgboost:
    n_estimators: 200
    max_depth: 4            # 防止过拟合
    learning_rate: 0.05
    reg_alpha: 0.1          # L1正则化
    reg_lambda: 1.0         # L2正则化

# 执行配置 (P5统一口径)
execution:
  execution_price: "close"  # 收盘价执行
  trade_at_close: true      # 同日收盘执行
  label_entry: "close"      # 标签入场价格
  label_exit: "close"       # 标签出场价格
```

### 11.2 参数调优建议

| 参数 | 保守配置 | 激进配置 | 说明 |
|------|----------|----------|------|
| top_n | 30-50 | 10-20 | 选股数量越少，集中度越高 |
| max_position | 0.03 | 0.10 | 单票仓位上限 |
| momentum权重 | 0.10 | 0.25 | 动量权重越高，换手越大 |
| sentiment权重 | 0.20 | 0.35 | 情绪因子在趋势市中更有效 |
| rebalance_freq | monthly | weekly | 调仓频率 |

---

## 附录：常见问题

### Q1: 为什么动量因子要跳过最近几天？
**A**: 短期（1-5天）存在"反转效应"，即涨多了容易跌。跳过最近几天可以捕捉更稳定的中期动量。

### Q2: 为什么情绪因子权重最高？
**A**: 情绪因子（赚钱效应、相对强度等）能较好地捕捉市场热度和资金偏好，在A股市场尤其有效。结合质量因子可以在保证基本面的同时把握市场情绪。

### Q3: 为什么要用时间序列交叉验证？
**A**: 股票数据有时间依赖性，普通K-Fold会导致用未来数据预测过去，造成过拟合假象。Purge机制进一步确保训练标签不与验证期重叠。

### Q4: 回测结果很好，实盘会一样吗？
**A**: 不一定。需要注意：
- 回测无法完全模拟真实成交
- 涨跌停、流动性问题
- 市场环境变化
- 建议用Walk-Forward验证
- 因子失效需要及时调整

### Q5: 因子计算很慢怎么办？
**A**: 系统已实现因子缓存机制。首次运行会计算并保存所有因子，后续运行直接加载缓存，速度会大幅提升。

### Q6: 如何添加新因子？
**A**:
1. 在 `factor_engine.py` 的 `factor_categories` 添加因子名
2. 在 `factor_direction` 定义因子方向
3. 实现因子计算逻辑
4. 清除因子缓存重新计算

---

*文档版本: v2.0*
*更新日期: 2024-12*
*主要更新: 添加情绪因子、因子缓存、仓位管理、执行配置统一*
