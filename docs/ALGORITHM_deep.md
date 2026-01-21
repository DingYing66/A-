# A股多因子选股系统 - 深度学习链路算法说明 (main_deep)

## 目录
1. [系统定位](#1-系统定位)
2. [入口与运行方式](#2-入口与运行方式)
3. [配置与依赖](#3-配置与依赖)
4. [数据流与缓存](#4-数据流与缓存)
5. [样本构造与标签口径](#5-样本构造与标签口径)
6. [模型类型与训练流程](#6-模型类型与训练流程)
7. [评估与回测](#7-评估与回测)
8. [信号生成](#8-信号生成)
9. [一致性与风险控制](#9-一致性与风险控制)
10. [关键文件索引](#10-关键文件索引)
11. [已知限制与改进建议](#11-已知限制与改进建议)

---

## 1. 系统定位

`main_deep.py` 是深度学习选股链路的独立入口，定位为：
- 与 `main.py`（传统多因子/树模型）并行，互不影响。
- 复用现有因子与数据层，但采用深度模型（时序/图/对比预训练/MoE/RL）。
- 输出以横截面评分/排名为主，便于与现有回测与选股逻辑对齐。

---

## 2. 入口与运行方式

```bash
# 训练
python main_deep.py train --model transformer --start 20180101 --end 20231231

# 回测评估
python main_deep.py backtest --model transformer --start 20240101 --end 20241231

# 生成信号
python main_deep.py signal --model transformer

# 对比学习预训练
python main_deep.py pretrain --start 20180101 --end 20231231 --epochs 50
```

支持的模型类型：
- 时序: `lstm`, `transformer`, `patchtst`, `informer`
- 图模型: `graphsage`, `gat`, `graphormer`
- 对比学习: `contrastive`
- MoE: `moe`

---

## 3. 配置与依赖

### 3.1 深度配置
默认使用 `config/deep_learning.yaml` 管理深度模型参数：
- `sequence`: 序列长度、batch_size、AMP 等
- `training`: 训练轮数、学习率、损失函数
- `graph`: 图构建方式（industry/correlation/hybrid）、相关窗口、Top-K
- `contrastive`, `moe`, `rl`, `text` 等模块

### 3.2 主配置联动
深度链路仍会间接依赖主配置：
- `DataProcessor`、`FactorEngine` 默认读取 `config/config.yaml`
- `ExecutionPolicy` 使用 `execution.*` 与 `data_fetch.adjust`
- 因子缓存路径、交易日历、复权口径从主配置读取

**重要约束**：深度链路的执行口径必须与主配置一致（`execution_price`, `label_entry`, `label_exit`, `adjust`）。

---

## 4. 数据流与缓存

### 4.1 输入因子缓存
深度链路以因子缓存为核心输入：
- 目录：`data/processed/factors/`
- 文件：
  - 新格式: `factors_YYYYMMDD_{universe_id}.parquet`
  - 旧格式: `factors_YYYYMMDD.parquet`
- 必含列：`code`, `date`
- 标签列：默认 `forward_return`（close-to-close）

### 4.2 序列缓存
`FactorSequenceDataset` 会构建序列样本并缓存：
- 目录：`data/processed/factor_seq/`
- 元信息：`CacheMeta`（config_hash, adjust, universe_id）

### 4.3 图结构缓存
`StockGraphBuilder` 构建图并缓存：
- 目录：`data/processed/graphs/`
- 边类型：industry/correlation/hybrid
- 图刷新频率：daily/weekly/monthly

### 4.4 行业映射缓存
`IndustryFetcher` 会同时写入：
- 原始：`data/raw/industry/industry_mapping_{source}.parquet`
- 规范化：`data/processed/industry_map.parquet`

### 4.5 文本因子缓存
`TextFactorExtractor` 写入：
- `data/processed/text_factors/text_factors_YYYYMMDD.parquet`

---

## 5. 样本构造与标签口径

### 5.1 序列样本
对每只股票在目标日构建长度 `seq_len` 的因子序列：
- 序列区间: `[t-seq_len, t-1]`
- 标签: 目标日因子表中的 `forward_return`

### 5.2 标签口径
当前默认标签为 **close-to-close**（`forward_return`）。
若执行口径为 T+1 open 或其他口径：
1. 需要在训练阶段将标签转为执行口径；
2. 或在因子缓存阶段提前生成对应标签列。

### 5.3 防泄露原则
- 序列仅使用目标日之前的数据；
- 标签使用目标日之后收益；
- 训练/验证切分基于时间顺序。

---

## 6. 模型类型与训练流程

### 6.1 时序模型
入口：`src/deep_learning/temporal_models.py`
流程：
1. 构建序列数据集（`FactorSequenceDataset`）
2. 训练 LSTM/Transformer/PatchTST/Informer
3. 输出单股票评分

### 6.2 图模型
入口：`src/deep_learning/gnn_models.py`
流程：
1. 基于行业/相关性构建图结构
2. GraphSAGE/GAT/Graphormer 编码节点
3. 输出评分

### 6.3 对比学习预训练
入口：`src/deep_learning/contrastive.py`
通过对比学习预训练因子表示，再用于下游监督任务。

### 6.4 MoE (混合专家)
入口：`src/deep_learning/moe_models.py`
按市场状态（趋势/波动/广度）选择专家模型。

### 6.5 RL 组合优化
入口：`src/deep_learning/rl_env.py`
以组合权重为动作，奖励包含收益与交易成本惩罚。

---

## 7. 评估与回测

入口：`src/deep_learning/metrics.py`
主要指标：
- Rank-IC / IC-IR
- Top-N 超额收益
- 成本调整收益
- Sharpe / Max Drawdown

`main_deep.py backtest` 采用因子缓存逐日评分评估，属于 **信号层回测**，不等同于完整交易撮合回测。

---

## 8. 信号生成

`main_deep.py signal`：
1. 加载最新模型（或指定模型）
2. 读取最新因子缓存（若缺失则现场计算）
3. 输出 Top-N 股票评分与排名
4. 保存至 `output/signals/deep_signal_YYYYMMDD_{model}.csv`

---

## 9. 一致性与风险控制

统一约束组件：
- `ExecutionPolicy`: 执行/标签价格口径
- `TradeabilityPolicy`: ST/停牌/涨跌停/流动性过滤
- `CostModel`: 佣金/印花税/滑点

关键一致性要求：
- 训练/回测/信号生成应使用相同的 `execution` 与 `adjust` 口径
- `universe_id` 必须与因子缓存一致，避免跨股票池复用
- 行业映射为空时需要跳过行业边，避免图边爆炸

---

## 10. 关键文件索引

| 模块 | 文件 | 说明 |
|------|------|------|
| 入口 | `main_deep.py` | 深度链路 CLI |
| 基类 | `src/deep_learning/base.py` | 深度模型基类 |
| 序列 | `src/deep_learning/sequence_dataset.py` | 序列样本构造 |
| 时序模型 | `src/deep_learning/temporal_models.py` | LSTM/Transformer |
| 图模型 | `src/deep_learning/gnn_models.py` | GraphSAGE/GAT/Graphormer |
| 图构建 | `src/deep_learning/graph_builder.py` | 行业/相关性图 |
| 对比学习 | `src/deep_learning/contrastive.py` | 预训练 |
| MoE | `src/deep_learning/moe_models.py` | 混合专家 |
| RL | `src/deep_learning/rl_env.py` | 组合环境 |
| 文本因子 | `src/deep_learning/text_factors.py` | 新闻/公告因子 |
| 行业映射 | `src/industry_fetcher.py` | 行业映射缓存 |

---

## 11. 已知限制与改进建议

- 深度配置与主配置仍分离，建议合并或显式同步关键字段。
- 标签口径默认 close-to-close，若执行口径为 open 需做转换。
- 对比学习与 RL 仍未完全接入统一撮合规则。
- 文本因子仍需完善发布时间对齐与可复现 meta。

---

**备注**：本文件用于说明 `main_deep.py` 的算法路径与口径约束，确保深度链路与现有系统保持可复现与可比性。  
