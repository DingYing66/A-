# Plan

本计划基于 `docs/ALGORITHM.md` 与 `docs/DATA_FUNCTIONS.md` 的现有架构与数据依赖，结合最新改进建议，从算法、架构、数据、评估与工具链五个维度给出可落地的系统升级路线，目标是提升因子表达力、跨市场/跨状态稳健性与真实可交易性。

## Goals and principles
- 训练/回测/部署口径一致：价格口径、标签入场/出场、执行价与 Purge 机制统一。
- 输出兼容现有流程：模型以横截面打分/排序为主，能直接接入 `Backtester` 与 `FactorScorer`。
- 评估以真实可交易为导向：成本调整收益、换手约束、流动性约束与风险归因不可缺失。
- 模块化与可扩展：支持模型、数据源、因子插件化插拔；保留树模型回退路径。
- GPU 训练启用：引入 PyTorch/PyG/Stable-Baselines3，并保持 CPU 退化路径可用。
- “单一事实来源”(Single Source of Truth)：执行/撮合/成本/可交易性规则只允许一份实现，供回测与 RL 环境共用，避免口径漂移。

## Confirmed defaults (initial)
- 序列窗口：主窗口 60 日；做 20/60/120 多尺度消融；若只能选一档，默认 60 日。
- 调仓频率：以 biweekly 为主基线，对比 weekly/monthly 后用成本调整收益与 IC-IR 选定最终值。
- 标签长度：优先使用动态标签（下一次调仓周期），只在缺失时回退固定 `forward_days`。
- 行业口径：优先 THS 行业板块（稳定），EM 作为兜底；行业映射需本地缓存。
- 文本数据：优先研报与公告，新闻源需异常回退与缓存。

## Hard constraints (must be identical across train/backtest/deploy)
### 1) 决策时点与执行时点（强约束）
- 默认模式（与现有 `config.execution` 保持一致）：T 日收盘后生成信号，按 T 日收盘价执行（`trade_at_close=true`, `execution_price=close`）。
- 若切换到 T+1 开盘执行（`trade_at_close=false`, `execution_price=open`），必须同步修改：标签入场价(`label_entry`)、涨跌停判断价(`limit_check_price`)、回测成交价与成本估计，且训练/回测/实盘三者一致。

### 2) 复权口径（adjust）一致性（强约束）
- `data_fetch.adjust`（如 `hfq/qfq/None`）必须作为**全局唯一**复权口径：日K、指数K、收益标签、动量/波动、相关性图构建全部使用同一口径。
- 任何缓存数据（factor/seq/graph/text-features）都必须记录 `adjust` 与关键配置 hash；不一致则强制重算，禁止混用。

### 3) 信息可用性边界（强约束）
- 财务数据：只允许使用 `公告日/披露日 <= decision_time` 的数据（按公告日对齐是唯一口径）。
- 文本数据：只允许使用 `publish_time <= decision_time` 的文本；若只有日期无具体时间，默认按**下一交易日可用**（保守对齐，避免未来信息）。
- 图边构建：相关性窗口只使用 t 日之前的数据，禁止跨期统计。

## Config schema v1 (single source)
目标：给 `config/config.yaml` 一个“唯一版本”的字段规范，避免实现时出现重复字段、hash 不稳定、入口读取不一致。

### Canonical sections (建议最终结构)
- `paths.*`：路径，不参与特征 hash（run-time only）。
- `data_fetch.*`：数据拉取与复权口径（`adjust` 必须进入 hash）。
- `stock_pool.*`：股票池过滤与行业上限（过滤规则必须进入 hash，因为会改变 universe_id 与样本集合）。
- `factors.*`：因子窗口与参数（进入 hash）。
- `weights.*`：因子类别权重（进入 hash）。
- `liquidity.*`：流动性阈值与窗口（进入 hash）。
- `execution.*`：执行/标签/涨跌停判断口径（进入 hash）。
- `backtest.*`：调仓频率、top_n、单票上限、成本参数（进入 hash；其中 `backtest_start/end` 可作为 run-time）。
- `optimization.*`：换手约束与成本惩罚（进入 hash）。
- `ml.*`：树模型相关配置（进入 hash）。
- `ml.sequence.*`：深度时序模型配置（进入 hash）。
- `graph.*`：图构建配置（进入 hash）。
- `contrastive.*` / `moe.*` / `rl.*` / `text.*`：可选模块配置（进入 hash；模块关闭时仅记录开关）。

### Key list (defaults + owners, v1)
说明：下表给出 v1 必需字段的“唯一命名”、推荐默认值（优先沿用当前 `config/config.yaml`），以及主要读取方（single owner）。

| Key | Default | Owner (read by) | Notes |
|---|---:|---|---|
| `data_fetch.adjust` | `hfq` | `DataFetcher` / `DataProcessor` | 全链路复权口径，必须写入 meta 与缓存 key |
| `execution.trade_at_close` | `true` | `Backtester` / `MLModel` | 决策/执行时点口径 |
| `execution.execution_price` | `close` | `Backtester` | 成交价口径 |
| `execution.label_entry` / `execution.label_exit` | `close` / `close` | `MLModel` | 标签口径必须与执行一致 |
| `backtest.rebalance_freq` | `biweekly` | `Backtester` | 也影响 bootstrap block size 默认值 |
| `backtest.top_n` | `50` | `Backtester` / `FactorScorer` | Top-N 选股数量 |
| `backtest.max_position` | `0.05` | `Backtester` / `PortfolioBuilder` | 单票上限（V0/V1 必用） |
| `optimization.max_turnover` | `0.50` | `FactorScorer` / `PortfolioBuilder` | 换手上限（V1/V2） |
| `stock_pool.max_industry_weight` | `0.30` | `FactorScorer` / `PortfolioBuilder` | 行业集中度上限（V1/V2） |
| `liquidity.min_amount_threshold` | `10000000` | `FactorEngine` / `FactorScorer` | 流动性硬阈值 |
| `liquidity.min_turnover_pct` | `0.005` | `FactorEngine` / `FactorScorer` | 流动性硬阈值（小数形式） |
| `graph.edge_type` | `hybrid` | `GraphBuilder` | `industry`/`corr`/`hybrid` |
| `graph.corr_window` | `60` | `GraphBuilder` | 相关性窗口 |
| `graph.top_k` | `20` | `GraphBuilder` | 稀疏图邻居数 |
| `ml.sequence.model_type` | `patchtst` | `DeepModelTrainer` | `lstm`/`transformer`/`patchtst`/`informer` |
| `ml.sequence.seq_len` | `60` | `SequenceDataset` | 序列长度（与窗口消融一致） |

### Hash rules (params_hash / config_hash)
- `config_hash`：仅包含**影响数据契约与缓存**的字段（上述标记“进入 hash”的字段），采用稳定排序序列化（JSON with sorted keys）后取 sha256。
- `run-time only`（不进入 hash）：日志级别、device、num_workers、batch_size（若不改变数据契约）、可视化开关等。
- `params_hash` = `config_hash` + `code_version`（git commit/hash）+ `akshare_version`（写入 manifest，用于可追溯，不建议进入缓存 key）。
- 所有缓存文件写 `meta.json`：`config_hash, universe_id, adjust, execution_mode, source_versions`；读取时强校验。

## Target architecture upgrades
### 1) 模块化流水线与接口解耦
- 引入统一的数据契约：
  - `FactorFrame`：`code/date` + 因子列 + 行业列 + 市场状态列 + 可交易/停牌标记。
  - `SequenceBatch`：序列张量 + 目标标签 + 元信息（代码、日期、可交易标记、universe_id）。
  - `GraphBundle`：节点特征 + 边索引/权重 + 图版本 + 节点池快照。
- 统一模型接口（监督类）：`fit(X, y)`, `predict(X)`, `predict_scores(X)`；RL 使用 `PolicyAdapter`：`train(env)` 与 `act(state)`。
- 提供 `ScoreAdapter`：将 RL 的权重/动作映射为横截面 score，或在 `Backtester` 中走“权重模式”。
- 训练/回测数据管道分层：`DataProvider` → `FeatureBuilder` → `ModelTrainer` → `PortfolioBuilder` → `Backtester`。
- 样本约束：序列/图样本必须由 `HistoricalUniverse` 或 `get_available_codes(date)` 生成；退市/停牌在样本构造阶段剔除或显式 mask。

### 2) 事件驱动回测与RL环境一致性
- 为 `Backtester` 增加事件队列与执行器接口：
  - 事件类型：调仓、成交、停牌、涨跌停、现金结算、企业行为（可选）。
  - 在不破坏现有日频回测前提下，新增“事件驱动模式”开关。
- 构建 RL 环境（Gym-like）对接回测执行器，实现训练与回测逻辑一致。
- 单一事实来源组件（Backtester 与 RL env 共用同一实现与配置）：
  - `ExecutionPolicy`：决策时点/执行时点/成交价口径（close/open）与交易日推进规则。
  - `TradeabilityPolicy`：停牌、涨跌停、ST/退市、最小流动性、是否可买/可卖判定。
  - `CostModel`：佣金/印花税/滑点/冲击成本（含参数与默认值），统一计算回测成本与 RL 奖励扣减。
- 事件驱动最小实现范围（避免工程过大，先对齐现有回测规则）：
  - v0 (MVP)：step=每个 `rebalance_date`（非逐日）；事件仅覆盖 `rebalance → tradeability_check → execute_trades → apply_costs → update_holdings/cash`；企业行为暂不做（由 `adjust` 间接处理）。
  - v1 (增强)：step=逐交易日事件；支持更精细的成交/停牌/涨跌停队列；企业行为/分红拆股若无可靠数据则继续依赖 `adjust`，不强行引入新源。

### 3) 因子解耦与结构化重构
- 因子正交化/解耦：PCA/ICA 或 Autoencoder 生成“因子子空间”。
- 行业/风格中性化默认可选开关，模型训练期与实盘期保持一致。
- 支持事件驱动因子更新（公告/研报触发因子刷新）。

## Data sources & robustness plan (AKShare)
### Dependency tiers (可选分级与退化)
- L0 必须：日K/财务/资金流/北向/指数（现有已落盘）。
- L1 增强：行业映射/行业成分（关系图与行业约束用）。缺失时退化为无行业约束与静态图空图。
- L2 可选：研报/公告/新闻、机构持股等。缺失时退化为“数值因子+图结构”的单模态模型。

### DataFetcher/DataProcessor 扩展与文档同步
- 在 `docs/DATA_FUNCTIONS.md` 增加行业映射与文本接口的函数清单、字段说明、落盘路径、刷新频率与版本规则。
- 在 `src/data_fetcher.py` 增加：
  - `fetch_industry_mapping()` / `fetch_industry_constituents()`（THS 优先，EM 兜底）。
  - `fetch_text_research()` / `fetch_text_disclosure()` / `fetch_text_notice()` / `fetch_text_news()`。
- 在 `src/data_processor.py` 增加字段映射与 schema 校验，确保 AKShare 字段波动时可自动对齐。
- 新增落盘目录：`data/raw/industry/`、`data/raw/text/` 与版本化快照；默认使用本地缓存避免接口不稳定。

### Adjust (复权) 落地改造清单（最小闭环）
- DataFetcher 签名调整（示例）：
  - `fetch_daily_price(code: str, start_date: str=None, end_date: str=None, adjust: str=None) -> DataFrame`
  - `fetch_index_data(index_code: str, start_date: str=None, end_date: str=None, adjust: str=None) -> DataFrame`
  - `adjust=None` 时默认读 `config.data_fetch.adjust`，并将实际使用值写入落盘 meta。
- 落盘路径分区（避免不同复权混读）：
  - `data/raw/daily_price/{adjust}/{code}.parquet`
  - `data/raw/index/{adjust}/{index_code}.parquet`
- DataProcessor 加载策略：
  - `load_daily_price(code, adjust=None)` / `load_index_data(code, adjust=None)`：优先读取分区路径；若仅存在旧路径（无 adjust 分区），将其视作 `legacy_adjust` 并触发迁移或明确禁止复用。
- 旧数据迁移策略（建议在 Phase 0 完成）：
  - 提供一次性迁移脚本：将旧 `data/raw/daily_price/*.parquet` 移入 `data/raw/daily_price/{config.data_fetch.adjust}/` 并生成 meta。
  - 迁移后禁止在同一目录混放不同 adjust 数据。

### Storage schemas & versioning (落盘结构与版本号规则)
- 设计原则：与现有 `data/raw/*` parquet 风格一致；新增数据必须“可复现、可追溯、可回放”。任何训练/回测都要绑定具体快照版本。
- 行业数据落盘建议（示例）：
  - `data/raw/industry/industry_list_{source}_{asof_date}.parquet`：行业清单（含行业代码/名称/层级/更新时间/来源）。
  - `data/raw/industry/industry_constituents_{source}_{asof_date}.parquet`：行业成分（至少包含 `industry_code, code, name, asof_date, source`；若可得则加入 `start_date/end_date`）。
  - `data/raw/industry/industry_mapping_{source}_{asof_date}.parquet`：code→industry 的主映射（用于回测/训练对齐）。
- 文本数据落盘建议（示例，统一最小 schema）：
  - `data/raw/text/{type}/date=YYYYMMDD/*.parquet`（推荐按 date 分区，避免单文件无限增大）。
    - 最小字段：`code, title, publish_time, source, url, category, crawl_time`
    - 可选字段：`content/summary, author, sentiment_raw`
    - `publish_time` 必须可解析为 datetime；若只能得到日期则以 `publish_date` 存储并触发“下一交易日可用”的对齐策略
- 版本化策略（强约束）：
  - snapshot key = `{asof_date, source, adjust, akshare_version, params_hash}`
  - 每次训练/回测保存 `run_manifest.json`：记录使用的数据 snapshot 列表、配置 hash、代码版本（git commit 或 hash）、以及输出模型版本
  - 若 snapshot key 不一致，禁止复用缓存

### Cache consistency rules (universe_id as cache key)
- `universe_id` 必须参与缓存 key：同一 `date` 下不同股票池（过滤规则/数据补全/退市处理变动）不得复用缓存。
- 缓存建议（示例命名）：
  - 因子缓存：`data/processed/factors/factors_{date}_{universe_id}.parquet`
  - 序列缓存：`data/processed/factor_seq/seq_{date}_L{seq_len}_{universe_id}.parquet|npz`
  - 图缓存：`data/processed/graphs/graph_{date}_{edge_type}_{universe_id}.pt|parquet`
  - 每个缓存文件附带 `meta.json`（含 `universe_id/config_hash/adjust/source_versions`）
- Backtester/训练流程读取缓存时必须校验 meta：不一致则强制重算并写新缓存。
- universe_id 生成与快照绑定（建议写死，保证可复现）：
  - 先生成 `universe_snapshot`：`data/processed/universe/universe_{date}_{universe_id}.parquet`，字段至少包含 `code, date, reason_flags, config_hash`。
  - `universe_id = sha256(sorted_codes + date + stock_pool_filters_hash)`（推荐），并将 `sorted_codes_hash` 写入 meta；后续任何缓存都引用该 `universe_id`。

### Feature availability contract (可选数据域缺失处理)
目标：可选域（industry/text/institution）缺失时，训练/推理维度保持固定，避免 shape 漂移与分支爆炸。
- 固定维度原则：
  - 所有可选域输出必须包含 `*_available` 或 `*_mask` 列（0/1）。
  - 缺失时：值填 0，mask=0；可用时：值填实际，mask=1。
- 建议最小 mask 集合：
  - `industry_available`（code→industry 是否存在）
  - `text_available`（当日可用文本特征是否存在）
  - `institution_available`（机构相关数据是否存在）
- 对深度模型输入：将 mask 作为额外通道/特征输入，避免模型将“填0”误解为真实信号。
- 空域定义（建议写死，减少歧义）：
  - 行业缺失：`industry='unknown'`，`industry_available=0`；行业约束在 PortfolioBuilder 自动跳过；行业图退化为“仅相关性边”或空图。
  - 文本缺失：文本 embedding 全 0，`text_available=0`；多模态融合层必须支持 mask。
  - 机构缺失：机构相关因子全 0，`institution_available=0`；不删列、不变维。

### Industry mapping (优先级与回退)
- 优先：`stock_board_industry_name_ths` + `stock_board_industry_cons_em` 生成行业-成分映射。
- 兜底：`stock_board_industry_name_em` + `stock_board_industry_cons_em`，若接口变动或缺失则使用本地缓存。
- 缓存机制：定期刷新（如月度）并保留历史快照，保证回测可复现。

### Text/announcement sources (优先级与回退)
- 优先：
  - 研报：`stock_research_report_em`
  - 公告：`stock_zh_a_disclosure_report_cninfo`
  - 公告汇总：`stock_notice_report`（需日期/类型稳定性校验）
- 不稳定源：`stock_news_em` 仅作为补充，需异常处理与退化策略。
- 数据质量策略：统一字段映射、缺失回退、schema 校验、失败重试与本地落盘。
- 机构数据说明：若“机构持股/调研”无法稳定获取，则相关因子设为缺失并触发模型回退（不阻塞主链路）。

## Modeling roadmap (algorithm upgrades)
### 1) 时序模型优化
- 候选模型：
  - PatchTST（长序列局部语义）
  - Informer（稀疏注意力降复杂度）
  - 基线：LSTM/基础Transformer
- 评估策略：用 20/60/120 窗口做多尺度对比；加入排序损失（pairwise/listwise）。
- 结果输出：统一为横截面打分/排序，回测与选股复用。

### 2) 图结构建模优化
- 候选模型：GraphSAGE/GAT（基线）+ Graphormer（全局结构编码）
- 动态图：滚动相关性更新边（top-k），结合行业静态图做 hybrid graph。
- 图输入：行业、资金流、相关性、文本共现（可选）作为边权来源。
- 相关性定义（强约束，写死便于复现与对比）：
  - 使用与 `execution.execution_price` 一致的价格序列（默认 close），并使用 `data_fetch.adjust` 复权后的收益序列。
  - 日收益建议用 log-return：`r_t = log(p_t) - log(p_{t-1})`。
  - 窗口 `W=corr_window`（默认 60），最小重叠样本 `min_overlap`（如 40）；不足则不连边。
  - 相关性指标默认 Pearson；可选 Spearman 作为消融。
- 性能策略（默认可跑、可缓存）：
  - `refresh_freq` 默认 monthly；weekly/dayly 仅用于消融实验。
  - 计算按行业分块（block-wise）或按股票池分桶，避免全市场 O(N^2)。
  - 输出只保留 top-k 邻居（k=10~30），并缓存到 `data/processed/graphs/`。

### 3) 表示学习与对比预训练
- FactorGCL / TS2Vec 作为主线；可选 InfoTS/Mask-based 预训练。
- 预训练→监督微调：用小量标签微调到排序任务。
- 消融评估：预训练 vs 无预训练；不同增强策略对比。

### 4) 门控机制与混合专家
- 替换规则门控为学习式 gating network；
- 引入负载均衡/重要性正则，防止专家塌缩；
- 支持 TopK gating 与市场状态输入；
- 专家模型可复用 LSTM/Transformer/GNN backbone。

### 5) 强化学习策略优化
- 基线：PPO/A2C；增强：分布式 RL / 风险敏感奖励。
- 奖励函数：风险调整收益 - 交易成本 - 风险暴露惩罚。
- 可选离线决策：Decision Transformer（离线轨迹学习）。
- 与因子评分融合：RL 用于权重微调，避免完全替代因子信号。

## Data-driven factor expansion
### 1) 自监督因子构造
- 引入自监督模型从长期行情数据学习隐藏因子；
- 将隐含向量作为新因子参与打分或模型输入；
- 严格消融以验证增益，避免噪声因子引入。

### 2) LLM生成因子与多模态融合（可选）
- 文本处理路径：公告/研报 → LLM/FinBERT → 结构化事件/情绪因子。
- 提示模板：事件类型、影响方向、持续性；统一量化为数值因子。
- 多模态融合：文本嵌入与数值因子在中间层注意力融合。
- 时间戳对齐（强约束）：只使用 `publish_time <= decision_time` 的文本；date-only 文本默认 T+1 可用。
- 可复现策略（强约束）：LLM 输出必须落盘并可回放：记录 `model_name/model_version, temperature/top_p, prompt_version_hash, input_hash, output_json, generated_at`；禁止“在线实时再生成”覆盖历史。

## Evaluation & backtest improvements
### 1) 稳健性与跨状态检验
- Regime 分层：牛/熊、高/低波动、趋势/震荡。
- 滚动 IC/IR、年度/季度 Sharpe 序列稳定性；
- 极端行情压力测试（如重大回撤期）。

### 2) 交易可行性与成本指标
- Turnover-adjusted IR、成本调整后净收益；
- 流动性分层 IC（大/中/小盘）；
- 滑点/冲击成本敏感性分析。

### 3) Alpha 转化与风险归因
- 风格暴露回归（市场/规模/价值/动量/波动），优先用现有数据自建风格因子：
  - 规模：`log(market_cap)` 或 `log(amount*shares)`（用现有行情/财务估算）。
  - 价值：`1/pb` 或 `1/pe_ttm`（估值因子反向）。
  - 动量：`ret_60d` / `ret_120d`。
  - 波动：`vol_20d`。
  - 回归频率按月或按调仓频率，保证与持仓周期一致。
- Transfer Coefficient (TC) 衡量信号到组合的转化效率；
- Up/Down Capture、Pain/Ulcer Index、下行风险指标。

### 4) 组合构建规则与 TC 定义
- V0：Top-N 等权（与当前回测一致）。
- V1：按 score softmax 权重 + 单票上限 + 行业上限 + 换手约束。
  - 默认参数建议：`max_position=backtest.max_position`，`max_industry_weight=stock_pool.max_industry_weight(默认0.30)`，`max_turnover=optimization.max_turnover(默认0.50)`，流动性阈值复用 `liquidity.*` 与 `stock_pool.*`
- V2：含交易成本与换手惩罚的优化器（线性/二次规划，可选）。
- V2 默认目标函数（建议写死为 V2-0，确保实验可比）：
  - `maximize  Σ w_i * score_i  - λ_turnover * turnover_cost(w) - λ_risk * risk_proxy(w)`
  - `risk_proxy` 可用 `vol_20d` 加权近似或用协方差矩阵（可选）；λ 参数写入配置并记录到实验快照
- TC 默认定义：`corr(score, realized_contribution)`，其中 `realized_contribution` 为约束后的组合贡献；若走 V0，则以 `corr(score, forward_return)` 作为近似。

### 5) 统计显著性与多重比较控制
- 显著性：对 Rank-IC/Top-N 超额收益报告 t-stat 或 block bootstrap 置信区间（考虑时间自相关，避免 i.i.d. 假设）。
- 多重比较/数据挖掘控制：
  - 固定“最终测试集”（仅在模型与超参收敛后评估一次）
  - 模型选择仅用训练+验证的 walk-forward
  - 记录实验次数与配置/数据快照，必要时用 Reality Check/Deflated Sharpe（可选）评估过拟合风险
- 统计检验落地标准（建议写死默认参数，保证报告可比）：
  - bootstrap 对象：`daily_rank_ic` 序列、`topN_excess` 序列、策略 `daily_returns` 序列（至少三类可选，默认对前两类输出）
  - block size：默认取 `rebalance` 周期对应交易日数（biweekly≈10），无则回退 20
  - n_bootstrap：默认 1000（可配置，但不进入缓存 hash）
  - 输出：`mean, std, CI(2.5/97.5), p_value` 写入报告与 `run_manifest.json`

## Tooling & operations
- 训练、回测、部署一致性：配置快照 + 模型版本 + 数据快照。
- 实验管理：MLflow/Dagshub（可选）记录参数/指标/模型。
- 性能优化：向量化回测/批量计算因子；必要时并行计算。
- 安全与回退：数据异常 → 回退上期信号；模型异常 → 回退树模型基线。
- 数据契约校验：新增 schema validator 与字段映射日志，确保 AKShare 字段变动可追溯。
- 训练数据面板化（性能与一致性）：
  - 将 `FactorFrame` 落为按 `date=YYYYMMDD` 分区的 panel parquet（训练按日期批量读取）
  - 或构建 `np.memmap/npz` 序列缓存，减少训练期对单票 parquet 的随机 I/O
  - 面板化数据同样受 `universe_id/config_hash/adjust` 约束并写入 meta
- 默认选型（v1）：panel parquet 作为主格式；`npz/memmap` 作为可选加速缓存（不改变数据契约）。
- 性能验收（以 3080 Ti 笔记本为目标量级，可按实际微调）：
  - 单次构建 3 年训练窗口的 `FactorFrame(panel)`：端到端 < 30 分钟
  - 深度模型单 epoch 训练 I/O 占比 < 30%，GPU 利用率可稳定 > 50%
  - Walk-forward（单窗口训练+验证+一次回测）在可接受时间内完成（优先保证可复现实验，再优化速度）

## Implementation phases (detailed)
[ ] Phase 0 - 基线审计与消融实验
- 对齐文档与代码口径（标签/执行/成本/风控）。
- 输出基线指标（Rank-IC、Top-N、成本调整收益）。
- 序列窗口 × 调仓频率 消融，锁定默认组合。
- 完成 DATA_FUNCTIONS 的行业/文本接口补充，并实现 DataFetcher/DataProcessor 的落盘与 schema 校验。
- 在样本构造与训练集中强制使用 HistoricalUniverse/可交易池约束。
- 固化“强约束口径”：decision_time/exec_time、adjust、文本时间戳对齐、图窗口边界；写入配置与 manifest，并为缓存添加 meta 校验。
- 抽出 `ExecutionPolicy/TradeabilityPolicy/CostModel` 作为单一事实来源，供 Backtester 与 RL env 共用。
- 固化 Config schema v1：字段名、默认值、hash 规则、run_manifest 结构；并在 `main.py`/脚本入口统一读取（避免多入口漂移）。

[ ] Phase 1 - 序列数据集与时序模型
- 实现 `sequence_dataset.py` 与序列缓存。
- 引入 PatchTST/Informer/LSTM 作为模型候选。
- 完成训练/回测流程的统一接口对接。

[ ] Phase 2 - 行业/相关性图与GNN/Graphormer
- 生成行业映射缓存 + 相关性图缓存。
- 实现 GraphSAGE/GAT 与 Graphormer 组合输入。
- 引入动态图更新频率与回测一致性约束。

[ ] Phase 3 - 表示学习与对比预训练
- FactorGCL/TS2Vec 预训练 → 监督微调。
- 预训练开关与消融实验自动化。

[ ] Phase 4 - 学习门控与 MoE
- 引入 gating network + 负载均衡损失。
- 多专家模型训练与动态状态输入验证。

[ ] Phase 5 - RL 仓位优化与事件驱动回测
- 实现 RL 环境，确保训练/回测一致。
- 与因子评分融合，输出组合权重。
- 引入分布式RL或风险敏感奖励。

[ ] Phase 6 - 文本因子与多模态融合（可选）
- AKShare 文本源接入与缓存；
- LLM/FinBERT 向量化与事件因子生成；
- 与数值因子融合，做增益消融。

## Risks and mitigations
- 泄露风险：严格 Purge + 训练集统计量隔离；加入泄露单测。
- 标准化边界：Scaler 仅在训练窗口 `fit`，验证/测试仅 `transform`；预训练若使用全历史需明确隔离测试期。
- 图构建边界：相关性与图边仅使用 t 日之前窗口，禁止跨期统计。
- 财务口径：严格按公告日对齐，禁止报告期穿越。
- 数据稳定性：多源回退、本地缓存、字段校验。
- 深度模型过拟合：早停/正则/消融对比；不达标则回退树模型。
- 训练成本过高：优先小模型验证，逐步扩展模型规模。

## Deliverables & exit criteria
- Phase 0: 基线指标与默认窗口/调仓频率确认。
- Phase 1: 时序模型与序列缓存上线，可生成回测与信号。
- Phase 2: 图嵌入上线，验证对 IC/IR 的增益。
- Phase 3: 预训练方案验证与消融报告。
- Phase 4: MoE 性能与稳定性验证。
- Phase 5: RL 权重优化对比等权/分数权重的净收益。
- Phase 6: 文本因子增益可证实后再纳入默认流程。
