# 数据函数依赖文档

## 目录
1. [概述](#1-概述)
2. [模块依赖关系](#2-模块依赖关系)
3. [AKShare数据接口](#3-akshare数据接口)
4. [数据采集模块 (data_fetcher.py)](#4-数据采集模块)
5. [数据处理模块 (data_processor.py)](#5-数据处理模块)
6. [因子计算模块 (factor_engine.py)](#6-因子计算模块)
7. [数据流向图](#7-数据流向图)
8. [数据存储结构](#8-数据存储结构)

---

## 1. 概述

本文档详细描述系统中各模块的数据函数依赖关系，包括：
- AKShare外部数据接口
- 模块间的调用关系
- 数据加载和存储函数
- 数据流向

---

## 2. 模块依赖关系

### 2.1 模块调用层级

```
main.py
├── data_fetcher.py      [数据采集]
│   └── akshare (外部库)
├── data_processor.py    [数据处理]
│   └── utils.py
├── factor_engine.py     [因子计算]
│   └── data_processor.py
├── scorer.py            [因子打分]
│   └── factor_engine.py
├── ml_model.py          [机器学习]
│   ├── factor_engine.py
│   └── data_processor.py
├── backtester.py        [回测引擎]
│   ├── factor_engine.py
│   ├── scorer.py
│   └── data_processor.py
└── performance.py       [绩效分析]
    └── utils.py
```

### 2.2 模块依赖矩阵

| 模块 | utils | data_processor | factor_engine | scorer | ml_model | backtester |
|------|-------|----------------|---------------|--------|----------|------------|
| data_fetcher | √ | - | - | - | - | - |
| data_processor | √ | - | - | - | - | - |
| factor_engine | √ | √ | - | - | - | - |
| scorer | √ | - | √ | - | - | - |
| ml_model | √ | √ | √ | - | - | - |
| backtester | √ | √ | √ | √ | - | - |
| performance | √ | - | - | - | - | - |

---

## 3. AKShare数据接口

### 3.1 股票列表相关

| 函数 | 说明 | 返回字段 | 调用位置 |
|------|------|----------|----------|
| `ak.stock_info_a_code_name()` | 获取A股代码和名称 | code, name | `get_stock_list()` |
| `ak.stock_zh_a_spot_em()` | 获取A股实时行情 | 代码,名称,最新价,涨跌幅等 | `get_stock_list()` (备选) |
| `ak.stock_sh_a_spot_em()` | 获取沪市A股行情 | 同上 | `_get_stock_list_by_market()` |
| `ak.stock_sz_a_spot_em()` | 获取深市A股行情 | 同上 | `_get_stock_list_by_market()` |

### 3.2 行情数据相关

| 函数 | 说明 | 返回字段 | 调用位置 |
|------|------|----------|----------|
| `ak.stock_zh_a_hist(symbol, period, start_date, end_date, adjust)` | 获取个股日K线 | 日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率 | `fetch_daily_price()` |
| `ak.index_zh_a_hist(symbol, period, start_date, end_date)` | 获取指数日K线 | 日期,开盘,收盘,最高,最低,成交量 | `fetch_index_data()` |

### 3.3 财务数据相关

| 函数 | 说明 | 返回字段 | 调用位置 |
|------|------|----------|----------|
| `ak.stock_financial_report_sina(stock, symbol)` | 获取财务报表 | 报告期,各财务指标 | `fetch_financial_data()` |
| `ak.stock_financial_abstract_ths(symbol)` | 获取财务摘要 | 同上 | `fetch_financial_data()` (备选) |
| `ak.stock_yjbb_em(date)` | 获取业绩报表 | 股票代码,EPS,净利润等 | `fetch_financial_data()` |

### 3.4 资金流向相关

| 函数 | 说明 | 返回字段 | 调用位置 |
|------|------|----------|----------|
| `ak.stock_individual_fund_flow(stock, market)` | 获取个股资金流向 | 日期,主力净流入,散户净流入等 | `fetch_fund_flow()` |
| `ak.stock_hsgt_north_net_flow_in_em()` | 获取北向资金净流入 | 日期,沪股通,深股通,北向合计 | `fetch_north_flow()` |
| `ak.stock_hsgt_hold_stock_em()` | 获取北向持股明细 | 股票代码,持股数量,持股市值 | `fetch_north_holdings()` |

### 3.5 其他数据

| 函数 | 说明 | 返回字段 | 调用位置 |
|------|------|----------|----------|
| `ak.stock_board_industry_name_em()` | 获取行业板块 | 板块名称,成分股 | `fetch_industry_info()` |
| `ak.stock_info_change_name(symbol)` | 获取股票更名记录 | 变更日期,股票名称 | `fetch_name_change()` |

---

## 4. 数据采集模块

### 4.1 DataFetcher 类

**文件**: `src/data_fetcher.py`

#### 核心方法

```python
class DataFetcher:
    def __init__(self, config: dict = None)
    def _retry_request(self, func, *args, **kwargs) -> Any
    def _retry_request_safe(self, func, *args, **kwargs) -> Any
    def get_stock_list(self) -> pd.DataFrame
    def fetch_daily_price(self, code: str, start_date: str = None) -> pd.DataFrame
    def fetch_financial_data(self, code: str) -> pd.DataFrame
    def fetch_fund_flow(self, code: str) -> pd.DataFrame
    def fetch_north_flow(self) -> pd.DataFrame
    def fetch_index_data(self, index_code: str) -> pd.DataFrame
    def run_full_download(self, codes: List[str] = None) -> None
```

#### 方法详情

| 方法 | 功能 | 输入 | 输出 | 依赖的AKShare接口 |
|------|------|------|------|-------------------|
| `get_stock_list()` | 获取A股股票列表 | - | DataFrame | `stock_info_a_code_name`, `stock_zh_a_spot_em` |
| `fetch_daily_price()` | 获取个股日K线 | code, start_date | DataFrame | `stock_zh_a_hist` |
| `fetch_financial_data()` | 获取财务数据 | code | DataFrame | `stock_financial_report_sina` |
| `fetch_fund_flow()` | 获取资金流向 | code | DataFrame | `stock_individual_fund_flow` |
| `fetch_north_flow()` | 获取北向资金 | - | DataFrame | `stock_hsgt_north_net_flow_in_em` |
| `fetch_index_data()` | 获取指数数据 | index_code | DataFrame | `index_zh_a_hist` |

#### 数据保存位置

| 数据类型 | 保存路径 | 文件格式 |
|----------|----------|----------|
| 股票列表 | `data/raw/stock_list.parquet` | Parquet |
| 日K线 | `data/raw/daily_price/{code}.parquet` | Parquet |
| 财务数据 | `data/raw/financial/{code}.parquet` | Parquet |
| 资金流向 | `data/raw/fund_flow/{code}.parquet` | Parquet |
| 北向资金 | `data/raw/north_flow.parquet` | Parquet |
| 指数数据 | `data/raw/index/{code}.parquet` | Parquet |

---

## 5. 数据处理模块

### 5.1 DataProcessor 类

**文件**: `src/data_processor.py`

#### 核心方法

```python
class DataProcessor:
    def __init__(self, config: dict = None)

    # 交易日历
    @property
    def trade_calendar(self) -> pd.DatetimeIndex
    def build_trade_calendar(self) -> pd.DatetimeIndex
    def load_trade_calendar(self) -> pd.DatetimeIndex

    # 数据加载
    def load_stock_list(self) -> pd.DataFrame
    def load_daily_price(self, code: str) -> pd.DataFrame
    def load_financial_data(self, code: str) -> pd.DataFrame
    def load_fund_flow(self, code: str) -> pd.DataFrame
    def load_north_flow(self) -> pd.DataFrame
    def load_index_data(self, index_code: str) -> pd.DataFrame

    # 数据对齐
    def align_financial_data(self, code: str, date: pd.Timestamp) -> Dict
    def get_available_codes(self, date: pd.Timestamp) -> List[str]
    def filter_tradeable_codes(self, codes: List[str], date: pd.Timestamp) -> List[str]
```

#### 方法依赖关系

```
load_stock_list()
    └── load_parquet(data/raw/stock_list.parquet)

load_daily_price(code)
    └── load_parquet(data/raw/daily_price/{code}.parquet)

load_financial_data(code)
    └── load_parquet(data/raw/financial/{code}.parquet)

build_trade_calendar()
    └── load_parquet(data/raw/daily_price/*.parquet)
        └── 提取所有日期 → 排序 → 保存

get_available_codes(date)
    ├── load_stock_list()
    └── 过滤条件:
        ├── 排除ST
        ├── 上市天数 > min_list_days
        └── 成交额 > min_avg_amount
```

### 5.2 HistoricalUniverse 类

用于消除生存者偏差的历史股票池。

```python
class HistoricalUniverse:
    def __init__(self, config: dict = None)
    def build_universe(self) -> None
    def get_universe(self, date: pd.Timestamp) -> List[str]
    def was_tradeable(self, code: str, date: pd.Timestamp) -> bool
```

---

## 6. 因子计算模块

### 6.1 FactorEngine 类

**文件**: `src/factor_engine.py`

#### 核心方法

```python
class FactorEngine:
    def __init__(self, config: dict = None, use_universe: bool = True)

    # 因子计算 - 单股票
    def compute_momentum_factors(self, code: str, date: pd.Timestamp) -> Dict[str, float]
    def compute_volatility_factors(self, code: str, date: pd.Timestamp) -> Dict[str, float]
    def compute_quality_factors(self, code: str, date: pd.Timestamp) -> Dict[str, float]
    def compute_valuation_factors(self, code: str, date: pd.Timestamp) -> Dict[str, float]
    def compute_flow_factors(self, code: str, date: pd.Timestamp) -> Dict[str, float]
    def compute_sentiment_factors(self, code: str, date: pd.Timestamp) -> Dict[str, float]
    def compute_trend_quality_factors(self, code: str, date: pd.Timestamp) -> Dict[str, float]
    def compute_institution_factors(self, code: str, date: pd.Timestamp) -> Dict[str, float]

    # 因子计算 - 横截面
    def compute_factor_cross_section(self, date: pd.Timestamp) -> pd.DataFrame
    def compute_market_regime(self, date: pd.Timestamp) -> Dict[str, float]

    # 因子标准化
    def standardize_factors(self, factor_df: pd.DataFrame) -> pd.DataFrame

    # 因子缓存
    def load_factor_data(self, date: str) -> pd.DataFrame
    def save_factor_data(self, factor_df: pd.DataFrame, date: str) -> None
```

#### 方法依赖关系

```
compute_factor_cross_section(date)
├── processor.get_available_codes(date)      # 获取可交易股票
├── compute_momentum_factors(code, date)     # 动量因子
│   └── processor.load_daily_price(code)
├── compute_volatility_factors(code, date)   # 波动因子
│   └── processor.load_daily_price(code)
├── compute_quality_factors(code, date)      # 质量因子
│   └── processor.load_financial_data(code)
├── compute_valuation_factors(code, date)    # 估值因子
│   ├── processor.load_daily_price(code)
│   └── processor.load_financial_data(code)
├── compute_flow_factors(code, date)         # 资金因子
│   ├── processor.load_fund_flow(code)
│   └── processor.load_north_flow()
├── compute_sentiment_factors(code, date)    # 情绪因子
│   └── processor.load_daily_price(code)
├── compute_trend_quality_factors(code, date) # 趋势质量
│   └── processor.load_daily_price(code)
├── compute_institution_factors(code, date)  # 机构因子
│   └── 机构持股数据
└── compute_market_regime(date)              # 市场环境
    └── processor.load_index_data(benchmark)
```

#### 因子分类与计算依赖

| 因子类别 | 依赖数据 | 计算方法 |
|----------|----------|----------|
| momentum | 日K线 | `compute_momentum_factors()` |
| volatility | 日K线 | `compute_volatility_factors()` |
| quality | 财务数据 | `compute_quality_factors()` |
| valuation | 日K线 + 财务 | `compute_valuation_factors()` |
| flow | 资金流向 + 北向 | `compute_flow_factors()` |
| sentiment | 日K线 | `compute_sentiment_factors()` |
| trend_quality | 日K线 | `compute_trend_quality_factors()` |
| institution | 机构持股 | `compute_institution_factors()` |
| market_regime | 指数数据 | `compute_market_regime()` |

---

## 7. 数据流向图

### 7.1 数据采集流程

```
                    ┌─────────────┐
                    │   AKShare   │
                    │   (外部API) │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ DataFetcher │
                    │ (数据采集)   │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  日K线数据     │  │  财务数据     │  │  资金流向     │
│ daily_price/  │  │  financial/   │  │  fund_flow/   │
└───────────────┘  └───────────────┘  └───────────────┘
        │                  │                  │
        └──────────────────┴──────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  data/raw/  │
                    │ (Parquet)   │
                    └─────────────┘
```

### 7.2 因子计算流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        data/raw/                                 │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│ daily_price/ │  financial/  │  fund_flow/  │   north_flow      │
└──────┬───────┴──────┬───────┴──────┬───────┴─────────┬─────────┘
       │              │              │                 │
       └──────────────┴──────────────┴─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  DataProcessor   │
                    │ load_daily_price │
                    │ load_financial   │
                    │ load_fund_flow   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  FactorEngine    │
                    │ compute_factors  │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌──────────────────┐          ┌──────────────────┐
    │   standardize    │          │   save_factor    │
    │    _factors()    │          │     _data()      │
    └────────┬─────────┘          └────────┬─────────┘
             │                             │
             ▼                             ▼
    ┌──────────────────┐          ┌──────────────────┐
    │   factor_df      │          │ data/processed/  │
    │ (标准化后)        │          │    factors/      │
    └──────────────────┘          └──────────────────┘
```

### 7.3 回测执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         Backtester                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │    for date in rebalance_dates:   │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  1. load_factor_data(date)   │ ─── 尝试加载缓存
              └──────────────┬───────────────┘
                             │
                      缓存存在?
                      /        \
                    是          否
                    │            │
                    │            ▼
                    │   ┌────────────────────┐
                    │   │ compute_factor_    │
                    │   │ cross_section()    │
                    │   └─────────┬──────────┘
                    │             │
                    │             ▼
                    │   ┌────────────────────┐
                    │   │ standardize_factors│
                    │   └─────────┬──────────┘
                    │             │
                    │             ▼
                    │   ┌────────────────────┐
                    │   │ save_factor_data() │
                    │   └─────────┬──────────┘
                    │             │
                    └─────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  2. scorer.select_top_stocks │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  3. execute_rebalance()      │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  4. calculate_nav()          │
              └──────────────────────────────┘
```

---

## 8. 数据存储结构

### 8.1 目录结构

```
data/
├── raw/                          # 原始数据
│   ├── stock_list.parquet        # 股票列表
│   ├── daily_price/              # 日K线数据
│   │   ├── 000001.parquet
│   │   ├── 000002.parquet
│   │   └── ...
│   ├── financial/                # 财务数据
│   │   ├── 000001.parquet
│   │   └── ...
│   ├── fund_flow/                # 资金流向
│   │   ├── 000001.parquet
│   │   └── ...
│   ├── north_flow.parquet        # 北向资金
│   └── index/                    # 指数数据
│       ├── 000300.parquet
│       └── ...
│
└── processed/                    # 处理后数据
    ├── trade_calendar.parquet    # 交易日历
    ├── universe/                 # 历史股票池
    │   └── universe.parquet
    └── factors/                  # 因子缓存
        ├── factors_20150115.parquet
        ├── factors_20150130.parquet
        └── ...
```

### 8.2 数据格式说明

#### 日K线数据 (daily_price)

| 字段 | 类型 | 说明 |
|------|------|------|
| date | datetime | 交易日期 |
| open | float | 开盘价 |
| close | float | 收盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| volume | int | 成交量(股) |
| amount | float | 成交额(元) |
| pct_change | float | 涨跌幅(%) |
| turnover | float | 换手率(%) |

#### 财务数据 (financial)

| 字段 | 类型 | 说明 |
|------|------|------|
| report_date | datetime | 报告期 |
| announce_date | datetime | 公告日 |
| revenue | float | 营业收入 |
| net_profit | float | 净利润 |
| total_assets | float | 总资产 |
| net_assets | float | 净资产 |
| eps | float | 每股收益 |
| roe | float | 净资产收益率 |
| gross_margin | float | 毛利率 |

#### 因子缓存 (factors)

| 字段 | 类型 | 说明 |
|------|------|------|
| code | string | 股票代码 |
| roe | float | ROE因子值 |
| roa | float | ROA因子值 |
| pe_ttm | float | PE因子值 |
| ret_20d | float | 20日动量 |
| relative_strength_20d | float | 20日相对强度 |
| market_trend | float | 市场趋势 |
| ... | ... | 其他因子 |

---

## 附录：常用工具函数

### utils.py 核心函数

```python
# 配置加载
def load_config() -> dict

# 文件操作
def load_parquet(path: Path, columns: List[str] = None) -> pd.DataFrame
def save_parquet(df: pd.DataFrame, path: Path) -> None
def ensure_dir(path: Path) -> Path

# 数据处理
def standardize(x: pd.Series) -> pd.Series
def winsorize(x: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series
def zscore(x: pd.Series) -> pd.Series

# 日期处理
def get_rebalance_dates(start: str, end: str, freq: str) -> List[pd.Timestamp]
def get_next_trading_day(date: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp

# 绩效计算
def calc_max_drawdown(nav: pd.Series) -> float
def calc_max_drawdown_duration(nav: pd.Series) -> int
```

---

*文档版本: v1.0*
*更新日期: 2024-12*
