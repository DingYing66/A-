"""
数据预处理模块
负责数据清洗、对齐、交易日历生成等
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from functools import lru_cache

import pandas as pd
import numpy as np
from tqdm import tqdm

from .utils import (
    load_config, setup_logger, load_parquet, save_parquet,
    get_project_root, ensure_dir
)

logger = setup_logger(__name__)


class DataProcessor:
    """数据预处理器"""

    def __init__(self, config: dict = None):
        """
        初始化数据预处理器

        Args:
            config: 配置字典
        """
        self.config = config or load_config()
        self.root = get_project_root()
        self.raw_path = self.root / self.config['paths']['raw_data']
        self.processed_path = self.root / self.config['paths']['processed_data']
        self._trade_calendar = None

        # P6-1: 可交易代码缓存（性能优化）
        self._available_codes_cache = {}  # {date_str: [codes]}
        self._global_trading_index = None  # 全局交易日期索引 {code: set(dates)}

        # Fix 8: 价格数据缓存（性能优化）
        self._daily_price_cache = {}  # {code: DataFrame}

    @property
    def trade_calendar(self) -> pd.DatetimeIndex:
        """交易日历（懒加载，优先使用缓存）"""
        if self._trade_calendar is None:
            # Fix 4: 优先加载已保存的缓存，而非每次重新构建
            self._trade_calendar = self.load_trade_calendar()
        return self._trade_calendar

    def build_trade_calendar(self) -> pd.DatetimeIndex:
        """
        从行情数据构建交易日历 (支持 adjust 分区)

        Returns:
            交易日历 DatetimeIndex
        """
        logger.info("构建交易日历...")

        # 获取配置的 adjust 类型
        adjust = self.config['data_fetch']['adjust']
        adjust_dir = adjust if adjust else 'none'

        # 严格模式: 只使用分区路径，不回退旧路径
        price_dir = self.raw_path / "daily_price" / adjust_dir

        all_dates = set()

        # 从所有股票的日K数据中提取日期
        parquet_files = list(price_dir.glob("*.parquet")) if price_dir.exists() else []

        if len(parquet_files) == 0:
            raise RuntimeError(f"未找到日K数据文件 (路径: {price_dir})，无法构建交易日历")

        # 扫描所有文件构建完整日历（移除100文件限制）
        failed_files = []
        for f in tqdm(parquet_files, desc="构建交易日历"):
            try:
                df = load_parquet(f, columns=['date'])
                all_dates.update(pd.to_datetime(df['date']).tolist())
            except Exception as e:
                failed_files.append((f.name, str(e)))
                continue

        if failed_files:
            logger.warning(f"交易日历构建: {len(failed_files)} 个文件读取失败")
            for fname, err in failed_files[:5]:  # 只显示前5个
                logger.debug(f"  - {fname}: {err}")
            if len(failed_files) > 5:
                logger.debug(f"  ... 还有 {len(failed_files) - 5} 个文件失败")

        # 检查失败比例是否过高
        fail_ratio = len(failed_files) / len(parquet_files) if parquet_files else 0
        if fail_ratio > 0.1:  # 超过10%文件失败则告警
            logger.warning(f"交易日历构建: 失败文件比例过高 ({fail_ratio:.1%})，日历可能不完整")

        calendar = pd.DatetimeIndex(sorted(all_dates))
        logger.info(f"交易日历构建完成: {len(calendar)} 个交易日, "
                    f"范围 {calendar.min().date()} ~ {calendar.max().date()}")

        # 保存交易日历
        save_path = self.processed_path / "trade_calendar.parquet"
        save_parquet(pd.DataFrame({'date': calendar}), save_path)

        return calendar

    def load_trade_calendar(self) -> pd.DatetimeIndex:
        """
        加载已保存的交易日历

        Returns:
            交易日历
        """
        path = self.processed_path / "trade_calendar.parquet"
        if path.exists():
            df = load_parquet(path)
            return pd.DatetimeIndex(df['date'])
        return self.build_trade_calendar()

    def load_daily_price(self, code: str, adjust: str = None) -> Optional[pd.DataFrame]:
        """Load daily price data for a single stock (with cache)."""
        adjust = adjust if adjust is not None else self.config['data_fetch']['adjust']
        adjust_dir = adjust if adjust else 'none'

        # Fix 8: 使用缓存避免重复IO
        cache_key = f"{code}_{adjust_dir}"
        if cache_key in self._daily_price_cache:
            return self._daily_price_cache[cache_key]

        path = self.raw_path / 'daily_price' / adjust_dir / f"{code}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"daily price not found: {path}")

        df = load_parquet(path)

        # 缓存数据（限制缓存大小为500只股票）
        if len(self._daily_price_cache) < 500:
            self._daily_price_cache[cache_key] = df

        return df

    def clear_price_cache(self):
        """清除价格数据缓存"""
        self._daily_price_cache.clear()
        logger.debug("价格数据缓存已清除")

    def load_all_daily_prices(self, adjust: str = None) -> pd.DataFrame:
        """Load all daily price data (strict)."""
        logger.info("loading all daily price data...")
        adjust = adjust if adjust is not None else self.config['data_fetch']['adjust']
        adjust_dir = adjust if adjust else 'none'
        price_dir = self.raw_path / 'daily_price' / adjust_dir
        if not price_dir.exists():
            raise FileNotFoundError(f"daily price dir not found: {price_dir}")
        dfs = []
        for f in tqdm(list(price_dir.glob('*.parquet')), desc="load daily price"):
            df = load_parquet(f)
            dfs.append(df)
        if len(dfs) == 0:
            raise RuntimeError(f"no daily price files in {price_dir}")
        all_data = pd.concat(dfs, ignore_index=True)
        all_data['date'] = pd.to_datetime(all_data['date'])
        logger.info(f"loaded {len(all_data)} rows for {all_data['code'].nunique()} stocks")
        return all_data

    def load_financial_data(self, code: str) -> Optional[pd.DataFrame]:
        """Load financial data for a single stock (strict)."""
        path = self.raw_path / 'financial' / f"{code}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"financial data not found: {path}")
        return load_parquet(path)

    def align_financial_by_announce_date(self, financial_df: pd.DataFrame,
                                          trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Align financial data by announce_date (strict)."""
        if financial_df is None or len(financial_df) == 0:
            raise RuntimeError('financial_df is empty')
        if 'announce_date' not in financial_df.columns:
            raise RuntimeError('financial data missing announce_date')
        df = financial_df.copy()
        df['announce_date'] = pd.to_datetime(df['announce_date'])
        df = df.sort_values('announce_date')
        aligned_data = []
        for trade_date in trade_dates:
            available = df[df['announce_date'] <= trade_date]
            if len(available) > 0:
                latest = available.iloc[-1].copy()
                latest['trade_date'] = trade_date
                aligned_data.append(latest)
        if len(aligned_data) == 0:
            raise RuntimeError('no financial data available before trade dates')
        return pd.DataFrame(aligned_data)

    def get_price_on_date(self, code: str, date: pd.Timestamp,
                          strict: bool = False) -> Optional[dict]:
        """
        Get price record for a specific date.

        Args:
            code: 股票代码
            date: 日期
            strict: 严格模式。如果为True，当日无数据时返回None而非前向填充；
                   如果为False（默认），使用前向填充获取最近有效价格

        Returns:
            价格数据字典，严格模式下无数据时返回None
        """
        df = self.load_daily_price(code)
        df['date'] = pd.to_datetime(df['date'])
        row = df[df['date'] == date]
        if len(row) == 0:
            if strict:
                # 严格模式：当日无数据直接返回None，不做前向填充
                return None
            # 非严格模式：尝试获取最近的有效价格（向前查找）
            df_sorted = df[df['date'] <= date].sort_values('date', ascending=False)
            if len(df_sorted) == 0:
                raise RuntimeError(f"price not found for {code} on {date}")
            return df_sorted.iloc[0].to_dict()
        return row.iloc[0].to_dict()

    def get_prev_close(self, code: str, date: pd.Timestamp) -> Optional[float]:
        """Get previous close, with fallback for missing dates."""
        df = self.load_daily_price(code)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        # 找到小于当前日期的最近交易日
        df_before = df[df['date'] < date]
        if len(df_before) == 0:
            raise RuntimeError(f"no previous close for {code} on {date}")
        return df_before.iloc[-1]['close']

    def is_st_stock(self, code: str, date: pd.Timestamp = None) -> Optional[bool]:
        """Strict ST check using historical data only."""
        if date is None:
            raise ValueError('date is required for strict ST check')
        if not hasattr(self, '_st_history_cache'):
            self._st_history_cache = self._load_st_history()
        if not self._st_history_cache:
            # 历史数据缺失时返回False并告警，避免前视偏差
            logger.warning(f"ST历史数据缺失，无法判断 {code} 在 {date} 的ST状态，默认返回False")
            return False
        if code not in self._st_history_cache:
            return False
        st_periods = self._st_history_cache[code]
        for period in st_periods:
            start_date_str = period.get('start_date') or period.get('date')
            end_date_str = period.get('end_date')
            if not start_date_str or not end_date_str:
                continue
            start_ts = pd.to_datetime(start_date_str, errors='coerce')
            end_ts = pd.to_datetime(end_date_str, errors='coerce')
            if pd.isna(start_ts) or pd.isna(end_ts):
                continue
            if start_ts <= date <= end_ts:
                return True
        return False

    def _is_currently_st(self, code: str) -> bool:
        """
        P5: 检查股票当前是否为ST状态

        仅用于无法使用历史数据时的回退判断

        Args:
            code: 股票代码

        Returns:
            当前是否为ST
        """
        if not hasattr(self, '_st_codes_cache'):
            self._st_codes_cache = self._load_st_codes()

        if code in self._st_codes_cache:
            return True

        # 备选: 从股票名称判断
        stock_list = self.load_stock_list()
        if len(stock_list) == 0:
            return False

        if code in stock_list['code'].values:
            row = stock_list[stock_list['code'] == code].iloc[0]
            name = row.get('name', '')
            if 'ST' in name or '*ST' in name or '退' in name:
                return True

        return False

    def _load_st_history(self) -> dict:
        """Load historical ST data (strict)."""
        import json
        st_history_path = self.raw_path / 'st_stocks' / 'st_history.json'
        open_end = self.config.get('backtest', {}).get('backtest_end')
        if open_end:
            open_end = pd.to_datetime(open_end, errors='coerce')
            if pd.isna(open_end):
                open_end = datetime.now().strftime('%Y-%m-%d')
            else:
                open_end = open_end.strftime('%Y-%m-%d')
        else:
            open_end = datetime.now().strftime('%Y-%m-%d')

        if st_history_path.exists():
            try:
                with open(st_history_path, 'r', encoding='utf-8') as f:
                    raw_history = json.load(f)
            except Exception as e:
                logger.warning(f"failed to load st_history.json: {e}")
                raw_history = {}
            if raw_history:
                normalized = {}
                for code, periods in raw_history.items():
                    if not periods:
                        continue
                    cleaned = []
                    for period in periods:
                        start_date = period.get('start_date') or period.get('date')
                        end_date = period.get('end_date') or open_end
                        if not start_date:
                            continue
                        cleaned.append({
                            'start_date': start_date,
                            'end_date': end_date,
                            'st_type': period.get('st_type'),
                            'name': period.get('name'),
                        })
                    if cleaned:
                        normalized[code] = cleaned
                if normalized:
                    logger.info(f"loaded ST history: {len(normalized)} stocks")
                    return normalized

        st_path = self.raw_path / "st_stocks" / "st_list_latest.parquet"
        if not st_path.exists():
            logger.warning("st_history.json missing and st_list_latest.parquet not found")
            return {}

        try:
            df = load_parquet(st_path)
        except Exception as e:
            logger.warning(f"failed to load st_list_latest.parquet: {e}")
            return {}

        if 'code' not in df.columns:
            logger.warning("st_list_latest.parquet missing code column")
            return {}

        start_date = None
        if 'download_date' in df.columns:
            try:
                dates = pd.to_datetime(df['download_date'].astype(str), errors='coerce')
                if dates.notna().any():
                    start_date = dates.max().strftime('%Y-%m-%d')
            except Exception:
                start_date = None
        if not start_date:
            start_date = datetime.now().strftime('%Y-%m-%d')

        st_history = {}
        for code in df['code'].dropna():
            raw = str(code).strip()
            if not raw:
                continue
            if '.' in raw:
                raw = raw.split('.')[0]
            raw_lower = raw.lower()
            if raw_lower.startswith(('sh', 'sz', 'bj')):
                raw = raw[2:]
            raw = raw.strip()
            if raw.isdigit():
                raw = raw.zfill(6)
            st_history[raw] = [{
                'start_date': start_date,
                'end_date': open_end,
                'st_type': 'ST',
            }]

        if st_history:
            logger.info(f"loaded ST history from st list: {len(st_history)} stocks")
        return st_history

    def _load_st_codes(self) -> set:
        """
        加载ST股票代码集合

        Returns:
            ST股票代码集合
        """
        st_codes = set()

        # 尝试加载ST列表文件
        st_path = self.raw_path / "st_stocks" / "st_list_latest.parquet"
        if st_path.exists():
            try:
                df = load_parquet(st_path)
                if 'code' in df.columns:
                    st_codes = set(df['code'].tolist())
                    logger.info(f"加载ST股票列表: {len(st_codes)} 只")
            except Exception as e:
                logger.warning(f"加载ST列表失败: {e}")

        return st_codes

    def get_price_matrix(self, codes: List[str], start_date: str,
                         end_date: str, field: str = 'close') -> pd.DataFrame:
        """
        获取价格矩阵（行：日期，列：股票）

        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            field: 价格字段名

        Returns:
            价格矩阵DataFrame
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        price_dict = {}

        for code in tqdm(codes, desc="构建价格矩阵"):
            df = self.load_daily_price(code)
            if df is None:
                continue

            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start) & (df['date'] <= end)]
            df = df.set_index('date')[field]
            price_dict[code] = df

        if len(price_dict) == 0:
            return pd.DataFrame()

        matrix = pd.DataFrame(price_dict)
        matrix = matrix.sort_index()

        return matrix

    def get_return_matrix(self, codes: List[str], start_date: str,
                          end_date: str, periods: int = 1) -> pd.DataFrame:
        """
        获取收益率矩阵

        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            periods: 计算周期

        Returns:
            收益率矩阵DataFrame
        """
        price_matrix = self.get_price_matrix(codes, start_date, end_date, 'close')
        if price_matrix.empty:
            return pd.DataFrame()

        return price_matrix.pct_change(periods)

    def load_stock_list(self) -> pd.DataFrame:
        """Load stock list (strict)."""
        path = self.raw_path / "stock_list.parquet"
        if not path.exists():
            raise FileNotFoundError(f"stock list file not found: {path}")
        return load_parquet(path)

    def get_available_codes(self, date: pd.Timestamp) -> List[str]:
        """
        获取指定日期可交易的股票列表

        P6-1优化: 使用多级缓存替代逐文件遍历
        - 第1级: 内存缓存 (_available_codes_cache)
        - 第2级: 磁盘缓存 (processed/universe/*.parquet)
        - 第3级: 全局交易索引 (_global_trading_index)
        - 回退: 原始逐文件扫描

        Args:
            date: 日期

        Returns:
            股票代码列表
        """
        date_str = date.strftime('%Y%m%d')

        # 1. 检查内存缓存
        if date_str in self._available_codes_cache:
            return self._available_codes_cache[date_str]

        # 2. 检查磁盘缓存 (HistoricalUniverse生成的缓存)
        cache_path = self.processed_path / 'universe' / f'{date_str}.parquet'
        if cache_path.exists():
            try:
                df = load_parquet(cache_path)
                codes = df['code'].tolist()
                self._available_codes_cache[date_str] = codes
                return codes
            except Exception as e:
                logger.debug(f"P6-1: 读取universe缓存失败: {e}")

        # 3. 使用全局交易索引（如果已构建）
        if self._global_trading_index is not None:
            available = []
            for code, trading_dates in self._global_trading_index.items():
                if date in trading_dates:
                    available.append(code)
            self._available_codes_cache[date_str] = available
            return available

        # 4. 回退到原始方法（首次调用时自动构建索引）
        return self._get_available_codes_with_index(date)

    def _get_available_codes_with_index(self, date: pd.Timestamp) -> List[str]:
        """Build global trading index (strict partition path only)."""
        if self._global_trading_index is None:
            logger.info('building global trading index...')
            self._global_trading_index = {}
            adjust = self.config['data_fetch']['adjust']
            adjust_dir = adjust if adjust else 'none'
            price_dir = self.raw_path / 'daily_price' / adjust_dir
            if not price_dir.exists():
                raise FileNotFoundError(f"daily price dir not found: {price_dir}")
            parquet_files = list(price_dir.glob('*.parquet'))
            if len(parquet_files) == 0:
                raise RuntimeError(f"no daily price files in {price_dir}")
            for f in tqdm(parquet_files, desc='build trading index'):
                code = f.stem
                df = load_parquet(f, columns=['date'])
                df['date'] = pd.to_datetime(df['date'])
                self._global_trading_index[code] = set(df['date'])
            logger.info(f"index built: {len(self._global_trading_index)} stocks")
        available = [code for code, dates in self._global_trading_index.items() if date in dates]
        date_str = date.strftime('%Y%m%d')
        self._available_codes_cache[date_str] = available
        return available

    def load_index_data(self, symbol: str = 'sh000300') -> pd.DataFrame:
        """Load index data (strict)."""
        path = self.raw_path / 'index' / f"{symbol}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"index data not found: {path}")
        df = load_parquet(path)
        df['date'] = pd.to_datetime(df['date'])
        if len(df) == 0:
            raise RuntimeError(f"index data empty: {symbol}")
        return df

    def load_northbound_flow(self) -> pd.DataFrame:
        """Load northbound flow (strict)."""
        path = self.raw_path / 'northbound' / 'north_flow.parquet'
        if not path.exists():
            raise FileNotFoundError(f"northbound flow not found: {path}")
        df = load_parquet(path)
        if len(df) == 0:
            raise RuntimeError('northbound flow is empty')
        return df

    def load_northbound_holding(self) -> pd.DataFrame:
        """Load northbound holding (strict)."""
        path = self.raw_path / 'northbound' / 'north_holding.parquet'
        if not path.exists():
            raise FileNotFoundError(f"northbound holding not found: {path}")
        df = load_parquet(path)
        if len(df) == 0:
            raise RuntimeError('northbound holding is empty')
        return df

    def align_slow_freq_data_by_announce_date(self, data_df: pd.DataFrame,
                                                trade_date: pd.Timestamp,
                                                announce_col: str = 'announce_date',
                                                report_col: str = 'report_date') -> pd.DataFrame:
        """Align slow-frequency data by announce_date (strict)."""
        if data_df is None or len(data_df) == 0:
            raise RuntimeError('slow freq data is empty')
        df = data_df.copy()
        if announce_col not in df.columns:
            raise RuntimeError(f"missing announce column: {announce_col}")
        if 'code' not in df.columns:
            raise RuntimeError('slow freq data missing code column')
        df[announce_col] = pd.to_datetime(df[announce_col])
        available = df[df[announce_col] <= trade_date]
        if len(available) == 0:
            raise RuntimeError('no slow freq data available before trade date')
        latest = available.sort_values(announce_col).groupby('code').tail(1)
        return latest

    def load_institution_holding(self, code: str) -> Optional[pd.DataFrame]:
        """
        加载单只股票的机构持股数据

        Args:
            code: 股票代码

        Returns:
            机构持股数据
        """
        path = self.raw_path / "institution" / f"{code}.parquet"
        if path.exists():
            return load_parquet(path)
        return None

    def get_north_holding_change(self, code: str, date: pd.Timestamp, days: int = 20) -> Optional[float]:
        """Get northbound holding change (strict)."""
        holding_df = self.load_northbound_holding()
        code_col = 'code'
        date_col = 'date'
        hold_col = 'holding'
        if code_col not in holding_df.columns or date_col not in holding_df.columns or hold_col not in holding_df.columns:
            raise RuntimeError('northbound holding missing required columns')
        stock_data = holding_df[holding_df[code_col] == code].copy()
        if len(stock_data) == 0:
            raise RuntimeError(f"northbound holding missing for {code}")
        stock_data[date_col] = pd.to_datetime(stock_data[date_col])
        stock_data = stock_data.sort_values(date_col)
        target = stock_data[stock_data[date_col] <= date]
        if len(target) == 0:
            raise RuntimeError(f"no northbound holding before {date} for {code}")
        target = target.tail(days + 1)
        if len(target) < days + 1:
            raise RuntimeError(f"northbound holding insufficient history for {code}")
        start = target.iloc[0][hold_col]
        end = target.iloc[-1][hold_col]
        if start == 0:
            raise RuntimeError(f"northbound holding start is zero for {code}")
        return (end - start) / start

    def load_moneyflow(self, code: str) -> Optional[pd.DataFrame]:
        """
        加载单只股票的资金流向

        Args:
            code: 股票代码

        Returns:
            资金流向数据
        """
        path = self.raw_path / "moneyflow" / f"{code}.parquet"
        if path.exists():
            return load_parquet(path)
        return None

    def load_tech_signals(self, signal_type: str, date: str = None) -> pd.DataFrame:
        """
        加载技术选股信号

        Args:
            signal_type: 信号类型 ('new_high', 'consecutive_up')
            date: 日期，默认最新

        Returns:
            技术信号数据
        """
        signal_dir = self.raw_path / "tech_signals"

        if date is None:
            # 获取最新的文件
            files = sorted(signal_dir.glob(f"{signal_type}_*.parquet"), reverse=True)
            if len(files) > 0:
                return load_parquet(files[0])
        else:
            path = signal_dir / f"{signal_type}_{date}.parquet"
            if path.exists():
                return load_parquet(path)

        return pd.DataFrame()

    def prepare_factor_data(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        准备指定日期的因子计算所需数据

        Args:
            date: 日期

        Returns:
            汇总数据DataFrame
        """
        # 获取可用股票
        codes = self.get_available_codes(date)

        if len(codes) == 0:
            return pd.DataFrame()

        data_list = []

        for code in codes:
            row = {'code': code, 'date': date}

            # 行情数据
            price_data = self.get_price_on_date(code, date)
            if price_data:
                row.update(price_data)

            # 财务数据（按披露日对齐）
            fin_df = self.load_financial_data(code)
            if fin_df is not None and len(fin_df) > 0:
                aligned = self.align_financial_by_announce_date(
                    fin_df,
                    pd.DatetimeIndex([date])
                )
                if len(aligned) > 0:
                    fin_row = aligned.iloc[0].to_dict()
                    # 避免覆盖已有字段
                    for k, v in fin_row.items():
                        if k not in row:
                            row[k] = v

            # 资金流向
            flow_df = self.load_moneyflow(code)
            if flow_df is not None and len(flow_df) > 0:
                # 取最近的资金流数据
                row['has_moneyflow'] = True
            else:
                row['has_moneyflow'] = False

            data_list.append(row)

        return pd.DataFrame(data_list)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗

        Args:
            df: 输入数据

        Returns:
            清洗后的数据
        """
        # 删除全为空的列
        df = df.dropna(axis=1, how='all')

        # 删除关键字段为空的行
        key_fields = ['code', 'close']
        for field in key_fields:
            if field in df.columns:
                df = df[df[field].notna()]

        return df


class HistoricalUniverse:
    """
    P0-4: 历史可交易股票池

    解决生存者偏差:
    - 按日期返回当时可交易的股票
    - 排除未上市、已退市、停牌的股票
    """

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.root = get_project_root()
        self.processor = DataProcessor(config)

        # 上市/退市日期缓存
        self.listing_dates = {}
        self.delisting_dates = {}
        self.all_codes = []

        # 缓存目录
        self.cache_dir = ensure_dir(self.root / 'data' / 'processed' / 'universe')

        # P4-4: 交易日期缓存（每只股票的所有交易日集合）
        self._trading_dates_cache = {}  # {code: set(dates)}

        self._load_listing_data()

    def _load_listing_data(self):
        """加载上市/退市信息"""
        # 方案1: 从股票列表获取上市日期
        stock_list = self.processor.load_stock_list()
        if len(stock_list) > 0:
            # 直接使用标准化后的列名 'list_date'
            list_col = 'list_date' if 'list_date' in stock_list.columns else None

            if list_col:
                for _, row in stock_list.iterrows():
                    code = row.get('code', '')
                    if code:
                        self.all_codes.append(code)
                        # 严格模式: 直接解析日期，如有异常则传播
                        if pd.notna(row[list_col]):
                            list_date = pd.to_datetime(row[list_col])
                            self.listing_dates[code] = list_date
            else:
                # 没有上市日期，用所有代码（直接使用标准化后的 'code' 列）
                if 'code' in stock_list.columns:
                    self.all_codes = stock_list['code'].tolist()

        # 方案2: 从K线文件目录获取 (严格模式: 只使用分区路径)
        if len(self.all_codes) == 0:
            adjust = self.config['data_fetch']['adjust']
            adjust_dir = adjust if adjust else 'none'
            price_dir = self.root / self.config['paths']['raw_data'] / 'daily_price' / adjust_dir

            if price_dir.exists():
                for f in price_dir.glob('*.parquet'):
                    code = f.stem
                    self.all_codes.append(code)

        logger.info(f"HistoricalUniverse: 加载了 {len(self.all_codes)} 只股票, "
                   f"{len(self.listing_dates)} 个有上市日期")

    def get_tradeable_codes(self, date: pd.Timestamp) -> List[str]:
        """
        获取指定日期可交易的股票列表

        条件:
        1. 已上市 (上市日期 <= date)
        2. 未退市 (退市日期 > date 或 无退市日期)
        3. 未停牌 (当日有交易数据)
        4. 满足流动性条件 (可选)

        Args:
            date: 查询日期

        Returns:
            可交易股票代码列表
        """
        # 先尝试从缓存加载
        date_str = date.strftime('%Y%m%d')
        cache_path = self.cache_dir / f'{date_str}.parquet'
        if cache_path.exists():
            df = load_parquet(cache_path)
            return df['code'].tolist()

        # 计算可交易股票
        tradeable = []
        min_list_days = self.config.get('stock_pool', {}).get('min_list_days', 120)

        for code in self.all_codes:
            # 检查上市日期
            list_date = self.listing_dates.get(code)
            if list_date is not None:
                if list_date > date:
                    continue  # 尚未上市
                # 检查上市天数
                if (date - list_date).days < min_list_days:
                    continue  # 上市不满min_list_days天

            # 检查退市日期
            delist_date = self.delisting_dates.get(code)
            if delist_date is not None and delist_date <= date:
                continue  # 已退市

            # 检查当日是否有交易数据 (简化版停牌判断)
            if not self._has_trading_data(code, date):
                continue

            tradeable.append(code)

        return tradeable

    def _has_trading_data(self, code: str, date: pd.Timestamp) -> bool:
        """
        P4-4: 检查指定日期是否有交易数据（使用缓存优化）

        Args:
            code: 股票代码
            date: 日期

        Returns:
            是否有交易数据
        """
        # 先检查缓存
        if code in self._trading_dates_cache:
            return date in self._trading_dates_cache[code]

        # 缓存未命中，加载该股票的所有交易日
        trading_dates = self._load_trading_dates(code)
        return date in trading_dates

    def _load_trading_dates(self, code: str) -> set:
        """
        P4-4: 一次性加载该股票的所有交易日（缓存）

        Args:
            code: 股票代码

        Returns:
            交易日期集合
        """
        if code in self._trading_dates_cache:
            return self._trading_dates_cache[code]

        df = self.processor.load_daily_price(code)
        if df is None:
            self._trading_dates_cache[code] = set()
        else:
            df['date'] = pd.to_datetime(df['date'])
            self._trading_dates_cache[code] = set(df['date'])

        return self._trading_dates_cache[code]

    def build_universe_cache(self, start_date: str, end_date: str):
        """
        预构建Universe缓存 (加速回测)

        保存为: data/processed/universe/{YYYYMMDD}.parquet

        Args:
            start_date: 开始日期
            end_date: 结束日期
        """
        from .utils import get_rebalance_dates

        calendar = self.processor.trade_calendar
        # 使用日频以覆盖所有交易日
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = [d for d in dates if d in calendar]

        logger.info(f"构建历史Universe: {start_date} ~ {end_date}, 共 {len(dates)} 个交易日")

        for date in tqdm(dates, desc="构建历史Universe"):
            cache_path = self.cache_dir / f'{date.strftime("%Y%m%d")}.parquet'
            if cache_path.exists():
                continue  # 已有缓存

            codes = self.get_tradeable_codes(date)
            if len(codes) > 0:
                df = pd.DataFrame({'code': codes, 'date': date})
                save_parquet(df, cache_path)

        logger.info("Universe缓存构建完成")

    def get_tradeable_codes_fast(self, date: pd.Timestamp) -> List[str]:
        """
        快速获取可交易股票 (使用缓存)

        如果没有缓存，回退到实时计算

        Args:
            date: 查询日期

        Returns:
            可交易股票代码列表
        """
        date_str = date.strftime('%Y%m%d')
        cache_path = self.cache_dir / f'{date_str}.parquet'

        if cache_path.exists():
            df = load_parquet(cache_path)
            return df['code'].tolist()

        # 回退到实时计算
        return self.get_tradeable_codes(date)


if __name__ == "__main__":
    # 测试代码
    processor = DataProcessor()

    # 构建交易日历
    calendar = processor.build_trade_calendar()
    print(f"交易日历: {len(calendar)} 天")

    # 测试加载股票列表
    stock_list = processor.load_stock_list()
    print(f"股票列表: {len(stock_list)} 只")
