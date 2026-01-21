"""
数据采集模块
负责从AKShare获取股票列表、行情、财务、资金流等数据
"""

import os
import re
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import akshare as ak
import pandas as pd
import numpy as np
from tqdm import tqdm

from .utils import (
    load_config, setup_logger, ensure_dir, save_parquet,
    load_parquet, get_project_root, standardize_datetime_column
)

logger = setup_logger(__name__)


# ===================== 列名标准化映射 =====================
# 将 AKShare 返回的中文列名统一为英文，避免后续代码的兼容性处理
COLUMN_MAPPING = {
    # 日期
    '日期': 'date',
    '交易日': 'date',
    '交易日期': 'date',

    # 价格数据
    '开盘': 'open',
    '收盘': 'close',
    '最高': 'high',
    '最低': 'low',
    '开盘价': 'open',
    '收盘价': 'close',
    '最高价': 'high',
    '最低价': 'low',

    # 成交数据
    '成交量': 'volume',
    '成交额': 'amount',
    '换手率': 'turnover',
    '振幅': 'amplitude',
    '涨跌幅': 'pct_change',
    '涨跌额': 'change',

    # 资金流数据
    '主力净流入-净额': 'net_inflow',
    '主力净流入': 'net_inflow',
    '净流入': 'net_inflow',
    '大单净流入': 'large_net_inflow',
    '中单净流入': 'medium_net_inflow',
    '小单净流入': 'small_net_inflow',
    '主力净流入-净占比': 'net_inflow_pct',

    # 财务数据
    '净资产收益率': 'roe',
    '加权净资产收益率': 'roe',
    '总资产收益率': 'roa',
    '总资产净利率': 'roa',
    '毛利率': 'gross_margin',
    '销售毛利率': 'gross_margin',
    '净利率': 'net_margin',
    '净利润率': 'net_margin',
    '销售净利率': 'net_margin',
    '市盈率': 'pe',
    '市净率': 'pb',
    '市销率': 'ps',
    # 财务明细字段 (宽表)
    '净利润': 'net_profit',
    '归属于母公司股东的净利润': 'net_profit',
    '归属于母公司所有者的净利润': 'net_profit',
    '营业收入': 'operating_income',
    '营业总收入': 'operating_income_total',
    '营业成本': 'operating_costs',
    '营业总成本': 'operating_costs_total',
    '营业支出': 'operating_costs_total',
    '一、营业总收入': 'operating_income_total',
    '二、营业支出': 'operating_costs_total',
    '资产合计': 'assets_total',
    '总资产': 'assets_total',
    '资产总计': 'assets_total',
    '资产总额': 'assets_total',
    '负债和股东权益总计': 'debt_and_equity_total',
    '负债及股东权益总计': 'debt_and_equity_total',
    '负债和所有者权益总计': 'debt_and_equity_total',
    '负债及所有者权益总计': 'debt_and_equity_total',
    '归属于母公司所有者权益合计': 'parent_holder_equity_total',
    '归属于母公司股东权益合计': 'parent_holder_equity_total',
    '归属母公司股东权益合计': 'parent_holder_equity_total',
    '归属于母公司股东的净利润': 'net_profit',
    '归属于母公司所有者的净利润': 'net_profit',
    '归属母公司股东净利润': 'net_profit',
    '所有者权益合计': 'holder_equity_total',
    '股东权益合计': 'equity',
    # 指标名称/数值 (长表)
    '指标名称': 'metric_name',
    '指标': 'metric_name',
    '指标值': 'value',
    '指标数值': 'value',

    # 股票代码
    '股票代码': 'code',
    '代码': 'code',
    '证券代码': 'code',
    '公司代码': 'code',
    '股票名称': 'name',
    '名称': 'name',
    '证券简称': 'name',

    # 机构持股数据
    '机构数': 'inst_count',
    '机构家数': 'inst_count',
    '持股机构数': 'inst_count',
    '持股比例': 'inst_ratio',
    '机构持股比例': 'inst_ratio',
    '持仓占比': 'inst_ratio',
    '新进机构': 'new_inst',
    '新进机构数': 'new_inst',
    '新增机构': 'new_inst',

    # 公告日期/报告期 (财务数据对齐用)
    '公告日期': 'announce_date',
    '披露日期': 'announce_date',
    '更新日期': 'announce_date',
    '报告期': 'report_date',
    '报告日期': 'report_date',
    '终止上市日期': 'delist_date',
    '暂停上市日期': 'delist_date',

    # 上市信息
    '上市时间': 'list_date',
    '上市日期': 'list_date',

    # 北向持股
    '持股数量': 'holding',
    '持股股数': 'holding',
    '持股市值': 'holding_value',
    '今日持股-股数': 'holding',
    '今日持股-市值': 'holding_value',
    '今日持股-占流通股比': 'holding_ratio_float',
    '今日持股-占总股本比': 'holding_ratio_total',
    '今日增持估计-股数': 'north_increase',
    '今日增持估计-市值': 'north_increase_value',
    '今日增持估计-市值增幅': 'north_increase_pct',
    '今日增持估计-占流通股比': 'north_increase_ratio_float',
    '今日增持估计-占总股本比': 'north_increase_ratio_total',
    '所属板块': 'sector',
    '今日收盘价': 'close',
    '今日涨跌幅': 'pct_change',

    # 北向资金流
    '当日成交净买额': 'net_buy_amount',
    '买入成交额': 'buy_amount',
    '卖出成交额': 'sell_amount',
    '历史累计净买额': 'total_net_buy',
    '当日资金流入': 'daily_inflow',
    '当日余额': 'daily_balance',
    '领涨股': 'leading_stock',
    '领涨股-涨跌幅': 'leading_stock_pct',
    '领涨股-代码': 'leading_stock_code',

    # 资金流细分
    '超大单净流入-净额': 'super_large_net_inflow',
    '超大单净流入-净占比': 'super_large_net_inflow_pct',
    '大单净流入-净额': 'large_net_inflow',
    '大单净流入-净占比': 'large_net_inflow_pct',
    '中单净流入-净额': 'medium_net_inflow',
    '中单净流入-净占比': 'medium_net_inflow_pct',
    '小单净流入-净额': 'small_net_inflow',
    '小单净流入-净占比': 'small_net_inflow_pct',

    # 机构持股补充
    '持有基金家数': 'inst_count',
    '持股变化': 'holding_change',
    '持股变动数值': 'holding_change_value',
    '持股变动比例': 'holding_change_ratio',
    '股票简称': 'name',

    # 技术信号
    '连涨天数': 'consecutive_up_days',
    '连续涨跌幅': 'consecutive_return',
    '累计换手率': 'cumulative_turnover',
    '前期高点': 'prev_high',
    '前期高点日期': 'prev_high_date',
    '所属行业': 'industry',

    # 通用字段
    '序号': 'seq',
    '最新价': 'price',
    '昨收': 'prev_close',
    '今开': 'open',
    '量比': 'volume_ratio',
    '市盈率-动态': 'pe_ttm',

    # 财务增长指标
    '净利润同比增长率': 'net_profit_yoy',
    '营业总收入同比增长率': 'revenue_yoy',
    '扣非净利润': 'net_profit_deducted',
    '扣非净利润同比增长率': 'net_profit_deducted_yoy',
    '基本每股收益': 'eps',
    '每股净资产': 'bps',
    '每股资本公积金': 'capital_reserve_ps',
    '每股未分配利润': 'retained_earnings_ps',
    '每股经营现金流': 'ocf_ps',

    # 财务比率指标
    '资产负债率': 'debt_ratio',
    '流动比率': 'current_ratio',
    '速动比率': 'quick_ratio',
    '存货周转率': 'inventory_turnover',
    '应收账款周转天数': 'receivable_turnover_days',
    '营业周期': 'operating_cycle',
}


def _normalize_column_name(name: str) -> str:
    if name is None:
        return name
    normalized = str(name).strip()
    while normalized.startswith(('*', '＊')):
        normalized = normalized[1:].strip()
    normalized = re.sub(r"[（(].*?[）)]", "", normalized).strip()
    return normalized


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化 DataFrame 的列名

    将中文列名统一为英文，便于后续处理

    Args:
        df: 原始 DataFrame

    Returns:
        列名标准化后的 DataFrame
    """
    if df is None or df.empty:
        return df

    # 应用列名映射
    new_columns = {}
    for col in df.columns:
        normalized = _normalize_column_name(col)
        if normalized in COLUMN_MAPPING:
            new_columns[col] = COLUMN_MAPPING[normalized]

    if new_columns:
        df = df.rename(columns=new_columns)

    return df


class DataFetcher:
    """数据采集器"""

    def __init__(self, config: dict = None):
        """
        初始化数据采集器

        Args:
            config: 配置字典，默认从config.yaml加载
        """
        self.config = config or load_config()
        self.root = get_project_root()
        self.raw_path = self.root / self.config['paths']['raw_data']
        self.interval = self.config['data_fetch']['request_interval']
        self.max_retries = self.config['data_fetch']['max_retries']
        self._delisted_codes_cache = None
        self._failed_codes_cache = None

    def _get_latest_quarter_date(self) -> str:
        """
        Fix 7: 动态获取最新季度日期

        季报披露时间:
        - Q1 (3月31日): 4月30日前披露
        - Q2 (6月30日): 8月31日前披露
        - Q3 (9月30日): 10月31日前披露
        - Q4 (12月31日): 次年4月30日前披露

        Returns:
            最新可用季度日期，格式 YYYYMMDD
        """
        today = datetime.now()
        year = today.year
        month = today.month

        # 根据当前月份确定最新可用的季度
        if month >= 11:  # 11-12月: Q3已披露
            return f"{year}0930"
        elif month >= 9:  # 9-10月: Q2已披露
            return f"{year}0630"
        elif month >= 5:  # 5-8月: Q1已披露
            return f"{year}0331"
        else:  # 1-4月: 上年Q3已披露
            return f"{year - 1}0930"

    def _get_recent_date(self, days_back: int = 30) -> str:
        """
        Fix 7: 获取近期日期

        Args:
            days_back: 回溯天数

        Returns:
            日期字符串，格式 YYYYMMDD
        """
        recent = datetime.now() - timedelta(days=days_back)
        return recent.strftime('%Y%m%d')

    def _normalize_code(self, code: str) -> str:
        raw = str(code).strip()
        if not raw:
            raise RuntimeError("empty stock code")
        if '.' in raw:
            raw = raw.split('.')[0]
        lowered = raw.lower()
        if lowered.startswith(('sh', 'sz', 'bj')):
            raw = raw[2:]
        raw = raw.strip()
        if not raw.isdigit():
            raise RuntimeError(f"invalid stock code: {code}")
        if len(raw) < 6:
            raw = raw.zfill(6)
        return raw

    def _normalize_codes(self, codes: List[str]) -> List[str]:
        return [self._normalize_code(code) for code in codes]

    def _load_failed_codes(self, data_type: str = None) -> set:
        """
        加载下载失败的股票代码列表

        Args:
            data_type: 数据类型，可选值:
                - None: 加载所有类型的失败列表（用于股票池过滤）
                - 'daily_price': 只加载日K线失败列表
                - 'financial': 只加载财务数据失败列表
                - 'moneyflow': 只加载资金流向失败列表

        Returns:
            失败股票代码集合
        """
        # 如果指定了数据类型，不使用缓存（因为缓存是全部类型的合集）
        if data_type is None and self._failed_codes_cache is not None:
            return self._failed_codes_cache

        failed = set()
        adjust = self.config['data_fetch']['adjust']
        adjust_dir = adjust if adjust else 'none'

        # 根据数据类型选择要加载的失败列表
        if data_type == 'daily_price':
            candidate_paths = [self.raw_path / "daily_price" / adjust_dir / "_failed_codes.txt"]
        elif data_type == 'financial':
            candidate_paths = [self.raw_path / "financial" / "_failed_codes.txt"]
        elif data_type == 'moneyflow':
            candidate_paths = [self.raw_path / "moneyflow" / "_failed_codes.txt"]
        else:
            # 加载所有类型
            candidate_paths = [
                self.raw_path / "daily_price" / adjust_dir / "_failed_codes.txt",
                self.raw_path / "financial" / "_failed_codes.txt",
                self.raw_path / "moneyflow" / "_failed_codes.txt",
            ]

        for path in candidate_paths:
            if not path.exists():
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        code = line.strip()
                        if not code:
                            continue
                        try:
                            failed.add(self._normalize_code(code))
                        except Exception:
                            continue
            except Exception as e:
                logger.warning(f"加载失败列表失败: {path} {e}")

        # 只有加载全部类型时才缓存
        if data_type is None:
            self._failed_codes_cache = failed
        return failed

    def _load_delisted_codes(self) -> set:
        if self._delisted_codes_cache is not None:
            return self._delisted_codes_cache
        delisted = set()
        try:
            df_sh = self._retry_request(ak.stock_info_sh_delist)
            df_sh = standardize_columns(df_sh)
            if df_sh is not None and 'code' in df_sh.columns:
                for val in df_sh['code'].dropna():
                    try:
                        delisted.add(self._normalize_code(val))
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"加载沪市退市列表失败: {e}")
        try:
            df_sz = self._retry_request(ak.stock_info_sz_delist)
            df_sz = standardize_columns(df_sz)
            if df_sz is not None and 'code' in df_sz.columns:
                for val in df_sz['code'].dropna():
                    try:
                        delisted.add(self._normalize_code(val))
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"加载深市退市列表失败: {e}")
        self._delisted_codes_cache = delisted
        return delisted

    def _retry_request(self, func, *args, **kwargs) -> Any:
        """
        带重试的请求封装

        Args:
            func: 要调用的函数
            *args, **kwargs: 函数参数

        Returns:
            函数返回值
        """
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                time.sleep(self.interval)
                return result
            except Exception as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.interval * 2)
                else:
                    raise

    def _retry_request_safe(self, func, *args, **kwargs) -> Any:
        """Strict alias for _retry_request (no silent fallback)."""
        return self._retry_request(func, *args, **kwargs)

    def _get_latest_quarter_date(self) -> str:
        """
        获取最新已披露的季度日期

        财报披露规则:
        - 一季报: 4月底前披露 (0331)
        - 半年报: 8月底前披露 (0630)
        - 三季报: 10月底前披露 (0930)
        - 年报: 次年4月底前披露 (1231)

        Returns:
            季度日期字符串, 格式 YYYYMMDD
        """
        now = datetime.now()
        year = now.year
        month = now.month

        # 根据当前月份确定最新已披露的季度
        if month >= 11:
            # 11-12月: 三季报已披露
            return f"{year}0930"
        elif month >= 9:
            # 9-10月: 半年报已披露，三季报可能还在披露中
            return f"{year}0630"
        elif month >= 5:
            # 5-8月: 一季报已披露
            return f"{year}0331"
        else:
            # 1-4月: 上年年报可能还在披露中，使用上年三季报
            return f"{year - 1}0930"

    def _get_latest_trade_date(self) -> Optional[pd.Timestamp]:
        candidates: List[pd.Timestamp] = []
        processed_path = self.root / self.config['paths']['processed_data']
        calendar_path = processed_path / "trade_calendar.parquet"
        index_paths = [
            self.raw_path / "index" / "sh000300.parquet",
            self.raw_path / "index" / "sh000905.parquet",
        ]

        for path in [calendar_path, *index_paths]:
            if not path.exists():
                continue
            try:
                df = load_parquet(path, columns=['date'])
            except Exception:
                try:
                    df = load_parquet(path)
                except Exception:
                    continue
            if df is None or len(df) == 0 or 'date' not in df.columns:
                continue
            dates = pd.to_datetime(standardize_datetime_column(df['date']), errors='coerce')
            if dates.notna().any():
                candidates.append(dates.max())

        return max(candidates) if candidates else None

    def _is_in_financial_disclosure_period(self) -> bool:
        financial_config = self.config.get('data_fetch', {}).get('financial_update', {})
        months = financial_config.get('disclosure_months', [4, 8, 10])
        day_start = financial_config.get('disclosure_day_start', 15)
        now = datetime.now()
        return now.month in months and now.day >= day_start

    def _load_financial_update_status(self, save_dir: Path) -> dict:
        status_path = save_dir / "_update_status.json"
        if not status_path.exists():
            return {}
        try:
            with open(status_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_financial_update_status(self, save_dir: Path, latest_quarter: str = None) -> None:
        status_path = save_dir / "_update_status.json"
        status = self._load_financial_update_status(save_dir)
        status['last_update_date'] = datetime.now().strftime('%Y-%m-%d')
        if latest_quarter:
            latest_ts = pd.to_datetime(latest_quarter, errors='coerce')
            status['last_quarter'] = (
                latest_ts.strftime('%Y-%m-%d') if pd.notna(latest_ts) else str(latest_quarter)
            )
        try:
            with open(status_path, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def get_stock_list(self) -> pd.DataFrame:
        """Fetch A-share stock list (strict)."""
        logger.info("fetching A-share stock list...")
        df = self._get_stock_list_full()
        if df is None or len(df) == 0:
            raise RuntimeError("stock list empty or unavailable")
        logger.info(f"stock list size: {len(df)}")
        return df

    def _get_stock_list_full(self) -> pd.DataFrame:
        """Fetch full stock list with code/name only (strict)."""
        logger.info("fetching stock list via stock_info_a_code_name...")
        df_info = self._retry_request(ak.stock_info_a_code_name)
        if df_info is None or len(df_info) == 0:
            raise RuntimeError("stock_info_a_code_name returned empty")
        df_info = df_info.rename(columns={'code': 'code', 'name': 'name'})
        required_cols = {'code', 'name'}
        missing = required_cols.difference(df_info.columns)
        if missing:
            raise RuntimeError(f"stock list missing columns: {sorted(missing)}")
        df_info = df_info[['code', 'name']].copy()
        return df_info

    def _get_stock_list_by_market(self) -> pd.DataFrame:
        """
        分市场获取股票列表（沪市+深市）
        绕过东财单接口100条限制
        """
        try:
            dfs = []

            # 沪市A股
            logger.info("获取沪市A股...")
            try:
                df_sh = self._retry_request(ak.stock_sh_a_spot_em)
                if df_sh is not None and len(df_sh) > 0:
                    dfs.append(df_sh)
                    logger.info(f"  沪市: {len(df_sh)} 只")
            except Exception as e:
                logger.warning(f"沪市获取失败: {e}")

            time.sleep(self.interval)

            # 深市A股
            logger.info("获取深市A股...")
            try:
                df_sz = self._retry_request(ak.stock_sz_a_spot_em)
                if df_sz is not None and len(df_sz) > 0:
                    dfs.append(df_sz)
                    logger.info(f"  深市: {len(df_sz)} 只")
            except Exception as e:
                logger.warning(f"深市获取失败: {e}")

            if len(dfs) == 0:
                return None

            # 合并
            df = pd.concat(dfs, ignore_index=True)

            # 去重
            df = df.drop_duplicates(subset=['代码'])

            # 重命名列
            df = df.rename(columns={
                '代码': 'code', '名称': 'name', '最新价': 'price',
                '涨跌幅': 'pct_change', '换手率': 'turnover',
                '总市值': 'total_mv', '流通市值': 'circ_mv', '成交额': 'amount'
            })

            cols = ['code', 'name', 'price', 'pct_change', 'turnover',
                    'total_mv', 'circ_mv', 'amount']
            df = df[[c for c in cols if c in df.columns]]

            return df

        except Exception as e:
            logger.warning(f"分市场获取失败: {e}")
            return None

    def filter_stock_pool(self, stock_list: pd.DataFrame) -> pd.DataFrame:
        """
        过滤股票池（完整过滤，包含流动性筛选）

        Args:
            stock_list: 原始股票列表

        Returns:
            过滤后的股票列表
        """
        config = self.config['stock_pool']
        df = stock_list.copy()
        original_count = len(df)

        # 排除ST股票
        if config.get('exclude_st', True):
            df = df[~df['name'].str.contains('ST|\\*ST', na=False)]
            logger.info(f"排除ST后剩余: {len(df)} 只")

        # 排除退市股票
        if config.get('exclude_delisted', True):
            delisted_codes = self._load_delisted_codes()
            if delisted_codes:
                df = df[~df['code'].isin(delisted_codes)]
                logger.info(f"排除退市股票后剩余: {len(df)} 只")
            else:
                df = df[~df['name'].str.contains('退', na=False)]
                logger.info(f"排除含退市标识后剩余: {len(df)} 只")

        # 根据配置决定是否排除北交所股票 (代码以8、4、92开头)
        if config.get('exclude_bj', True):
            df = df[~df['code'].str.startswith(('8', '4', '92'))]
            logger.info(f"排除北交所后剩余: {len(df)} 只")

        # 排除已知无法下载的股票
        if config.get('exclude_failed', True):
            failed_codes = self._load_failed_codes()
            if failed_codes:
                df = df[~df['code'].isin(failed_codes)]
                logger.info(f"排除无法下载后剩余: {len(df)} 只")

        # 按成交额过滤
        min_amount = config['min_avg_amount'] * 10000  # 转换为元
        if 'amount' in df.columns:
            df = df[df['amount'] >= min_amount]
            logger.info(f"过滤低成交额后剩余: {len(df)} 只")

        # 按换手率过滤
        min_turnover = config['min_avg_turnover']
        if 'turnover' in df.columns:
            df = df[df['turnover'] >= min_turnover]
            logger.info(f"过滤低换手率后剩余: {len(df)} 只")

        logger.info(f"股票池从 {original_count} 过滤到 {len(df)}")
        return df

    def filter_stock_pool_for_download(self, stock_list: pd.DataFrame) -> pd.DataFrame:
        """
        下载阶段的股票池过滤（只做基本过滤，不排除历史失败的股票）

        下载完整数据，流动性过滤在选股阶段进行，这样可以：
        1. 保留科创板、创业板等所有板块的数据
        2. 在选股时根据需要灵活调整流动性要求
        3. 不排除历史失败的股票，给它们重试的机会

        Args:
            stock_list: 原始股票列表

        Returns:
            过滤后的股票列表
        """
        config = self.config['stock_pool']
        df = stock_list.copy()
        original_count = len(df)

        # 排除ST股票
        if config.get('exclude_st', True):
            df = df[~df['name'].str.contains('ST|\\*ST', na=False)]
            logger.info(f"排除ST后剩余: {len(df)} 只")

        # 排除退市股票
        if config.get('exclude_delisted', True):
            delisted_codes = self._load_delisted_codes()
            if delisted_codes:
                df = df[~df['code'].isin(delisted_codes)]
                logger.info(f"排除退市股票后剩余: {len(df)} 只")
            else:
                df = df[~df['name'].str.contains('退', na=False)]
                logger.info(f"排除含退市标识后剩余: {len(df)} 只")

        # 根据配置决定是否排除北交所股票 (代码以8、4、92开头)
        if config.get('exclude_bj', True):
            df = df[~df['code'].str.startswith(('8', '4', '92'))]
            logger.info(f"排除北交所后剩余: {len(df)} 只")

        # 注意：不做流动性过滤（成交额、换手率），保留完整数据
        logger.info(f"下载股票池从 {original_count} 过滤到 {len(df)} (不含流动性过滤)")
        return df

    def fetch_daily_price(self, code: str, start_date: str = None,
                          end_date: str = None, adjust: str = None) -> Optional[pd.DataFrame]:
        """
        获取单只股票的日K线数据
        使用新浪接口 stock_zh_a_daily (东财接口在某些版本有bug)

        Args:
            code: 股票代码
            start_date: 开始日期，格式 YYYYMMDD
            end_date: 结束日期，格式 YYYYMMDD
            adjust: 复权类型 (qfq/hfq/空字符串)，为 None 时从配置读取

        Returns:
            日K线DataFrame，包含 _adjust 元数据列
        """
        start_date = start_date or self.config['data_fetch']['start_date']
        end_date = end_date or datetime.now().strftime('%Y%m%d')
        adjust = adjust if adjust is not None else self.config['data_fetch']['adjust']

        # 转换股票代码格式: 000001 -> sz000001, 600000 -> sh600000
        # 北交所股票: 8/4/92开头 -> bj前缀
        code = self._normalize_code(code)
        if code.startswith('92') or code.startswith('8') or code.startswith('4'):
            symbol = f"bj{code}"  # 北交所
        elif code.startswith('6'):
            symbol = f"sh{code}"  # 沪市（含科创板68）
        elif code.startswith('0') or code.startswith('3'):
            symbol = f"sz{code}"  # 深市（含创业板3）
        else:
            symbol = f"sz{code}"

        def _validate_daily_df(daily_df: pd.DataFrame) -> pd.DataFrame:
            required_cols = {"date", "open", "high", "low", "close", "volume", "amount"}
            missing = required_cols.difference(daily_df.columns)
            if missing:
                raise RuntimeError(f"daily price missing columns: {sorted(missing)}")
            return daily_df

        errors = []

        try:
            # 使用新浪接口
            df = self._retry_request(
                ak.stock_zh_a_daily,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )
            if df is None or len(df) == 0:
                raise RuntimeError("daily price fetch failed")

            df = df.rename(columns={'outstanding_share': 'shares'})
            if 'date' not in df.columns and df.index.name == 'date':
                df = df.reset_index()
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['code'] = code
            if 'pct_change' not in df.columns and 'close' in df.columns:
                df['pct_change'] = df['close'].pct_change() * 100
            df = standardize_columns(df)
            return _validate_daily_df(df)
        except Exception as e:
            errors.append(f"sina: {e}")

        try:
            # 使用东财接口作为回退
            em_adjust = adjust if adjust else ""
            df = self._retry_request(
                ak.stock_zh_a_hist,
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=em_adjust
            )
            if df is None or len(df) == 0:
                raise RuntimeError("daily price fetch failed")

            df = standardize_columns(df)
            if 'date' not in df.columns and df.index.name == 'date':
                df = df.reset_index()
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['code'] = code
            if 'pct_change' not in df.columns and 'close' in df.columns:
                df['pct_change'] = df['close'].pct_change() * 100
            return _validate_daily_df(df)
        except Exception as e:
            errors.append(f"eastmoney: {e}")

        raise RuntimeError(f"daily price fetch failed ({'; '.join(errors)})")

    def fetch_all_daily_price(self, codes: List[str], start_date: str = None,
                              end_date: str = None, skip_existing: bool = True,
                              adjust: str = None, skip_failed: bool = True,
                              update_existing: bool = True) -> None:
        """
        批量下载日K线数据 (按 adjust 分区存储)

        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            skip_existing: 是否跳过已存在的文件
            adjust: 复权类型 (qfq/hfq/空字符串)，为 None 时从配置读取
            update_existing: 是否增量更新已存在的文件

        存储路径: data/raw/daily_price/{adjust}/{code}.parquet
        """
        codes = self._normalize_codes(codes)
        start_date = start_date or self.config['data_fetch']['start_date']
        end_date = end_date or datetime.now().strftime('%Y%m%d')
        end_ts = pd.to_datetime(end_date, errors='coerce')
        adjust = adjust if adjust is not None else self.config['data_fetch']['adjust']
        # 空字符串表示不复权，使用 'none' 作为目录名
        adjust_dir = adjust if adjust else 'none'
        save_dir = ensure_dir(self.raw_path / "daily_price" / adjust_dir)
        success_count = 0
        fail_count = 0
        skip_count = 0
        updated_count = 0
        update_fail_count = 0
        up_to_date_count = 0
        fail_fast = bool(self.config.get('data_fetch', {}).get('fail_fast', False))

        # 加载已知无法下载的股票列表
        failed_list_path = save_dir / "_failed_codes.txt"
        known_failed = set()
        if failed_list_path.exists():
            with open(failed_list_path, 'r', encoding='utf-8') as f:
                known_failed = set(line.strip() for line in f if line.strip())

        skip_failed_count = 0
        # 预先统计已存在文件数和待下载列表
        to_update = []
        if skip_existing:
            existing_files = set(f.stem for f in save_dir.glob("*.parquet"))
            # 跳过已下载和已知失败的
            skip_downloaded = len([c for c in codes if c in existing_files])
            skip_failed_count = len([c for c in codes if c in known_failed and c not in existing_files]) if skip_failed else 0
            to_download = [
                c for c in codes
                if c not in existing_files and (c not in known_failed or not skip_failed)
            ]
            if update_existing:
                # 优化：先批量检查哪些文件真正需要更新
                logger.info("检查需要更新的文件...")
                candidates = [c for c in codes if c in existing_files]
                for code in tqdm(candidates, desc="检查更新状态", leave=False):
                    save_path = save_dir / f"{code}.parquet"
                    try:
                        date_df = load_parquet(save_path, columns=['date'])
                        if date_df is None or len(date_df) == 0 or 'date' not in date_df.columns:
                            to_update.append(code)
                            continue
                        date_df['date'] = pd.to_datetime(date_df['date'], errors='coerce')
                        max_date = date_df['date'].max()
                        if pd.isna(max_date) or (pd.notna(end_ts) and max_date < end_ts):
                            to_update.append(code)
                        else:
                            up_to_date_count += 1
                    except Exception:
                        to_update.append(code)
            logger.info(
                f"日K线: 已存在 {skip_downloaded} 个, 已知失败 {skip_failed_count} 个, "
                f"待下载 {len(to_download)} 个, 需更新 {len(to_update)} 个, 已是最新 {up_to_date_count} 个"
            )
        else:
            to_download = [c for c in codes if (c not in known_failed or not skip_failed)]

        if len(to_download) == 0 and len(to_update) == 0:
            logger.info("日K线: 全部文件已存在或已跳过, 无需下载")
            return

        new_failed = []
        recovered = set()
        for code in tqdm(to_download, desc=f"下载日K线({adjust_dir})"):
            save_path = save_dir / f"{code}.parquet"
            try:
                df = self.fetch_daily_price(code, start_date, end_date, adjust=adjust)
            except Exception as e:
                logger.warning(f"日K线下载失败 {code}: {e}")
                fail_count += 1
                new_failed.append(code)
                continue

            if df is not None and len(df) > 0:
                save_parquet(df, save_path)
                success_count += 1
                if code in known_failed:
                    recovered.add(code)
            else:
                fail_count += 1
                new_failed.append(code)

        if to_update:
            for code in tqdm(to_update, desc=f"更新日K线({adjust_dir})"):
                save_path = save_dir / f"{code}.parquet"
                try:
                    date_df = load_parquet(save_path, columns=['date'])
                    if date_df is None or len(date_df) == 0 or 'date' not in date_df.columns:
                        raise RuntimeError("existing daily price missing date column")
                    date_df['date'] = pd.to_datetime(date_df['date'], errors='coerce')
                    max_date = date_df['date'].max()

                    update_start = (max_date + pd.Timedelta(days=1)).strftime('%Y%m%d')
                    new_df = self.fetch_daily_price(code, start_date=update_start, end_date=end_date, adjust=adjust)
                    if new_df is None or len(new_df) == 0:
                        # 无新数据，可能是非交易日或已是最新
                        continue

                    old_df = load_parquet(save_path)
                    old_df['date'] = pd.to_datetime(old_df['date'], errors='coerce')
                    new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
                    merged = pd.concat([old_df, new_df], ignore_index=True)
                    merged = merged.drop_duplicates(subset=['date'], keep='last')
                    merged = merged.sort_values('date').reset_index(drop=True)
                    if 'close' in merged.columns:
                        merged['pct_change'] = merged['close'].pct_change() * 100
                    save_parquet(merged, save_path)
                    updated_count += 1
                except Exception as e:
                    logger.warning(f"日K线更新失败 {code}: {e}")
                    update_fail_count += 1

        # 保存失败列表
        if skip_failed:
            if new_failed:
                with open(failed_list_path, 'a', encoding='utf-8') as f:
                    for code in new_failed:
                        f.write(f"{code}\n")
                logger.info(f"新增 {len(new_failed)} 个无法下载的股票已记录")
        else:
            remaining_failed = (known_failed - recovered).union(new_failed)
            if remaining_failed:
                with open(failed_list_path, 'w', encoding='utf-8') as f:
                    for code in sorted(remaining_failed):
                        f.write(f"{code}\n")
            elif failed_list_path.exists():
                failed_list_path.unlink()
            if recovered:
                logger.info(f"移除 {len(recovered)} 个已恢复下载的股票")
            self._failed_codes_cache = None

        skip_count = skip_failed_count + up_to_date_count
        logger.info(
            f"日K线下载完成: 新增 {success_count}, 更新 {updated_count}, "
            f"更新失败 {update_fail_count}, 失败 {fail_count}, 已跳过 {skip_count}"
        )
        if fail_fast and (fail_count > 0 or update_fail_count > 0):
            raise RuntimeError(
                f"daily price download failed for {fail_count + update_fail_count} codes"
            )

    def _estimate_announce_date(self, report_date: pd.Timestamp) -> pd.Timestamp:
        year = report_date.year
        month = report_date.month
        day = report_date.day
        if month == 3 and day >= 31:
            return pd.Timestamp(year, 4, 30)
        if month == 6 and day >= 30:
            return pd.Timestamp(year, 8, 31)
        if month == 9 and day >= 30:
            return pd.Timestamp(year, 10, 31)
        if month == 12 and day >= 31:
            return pd.Timestamp(year + 1, 4, 30)
        raise RuntimeError(f"unexpected report_date: {report_date}")

    def _parse_cn_number(self, value: Any) -> float:
        if value is None:
            return np.nan
        if isinstance(value, (int, float, np.number)):
            if pd.isna(value):
                return np.nan
            return float(value)
        text = str(value).strip()
        lowered = text.lower()
        if lowered in ("", "nan", "none", "na", "<na>", "n/a") or text in ("--", "-", "—"):
            return np.nan
        text = text.replace(",", "").replace(" ", "").replace("－", "-")
        for suffix in ("元", "股", "手"):
            if text.endswith(suffix):
                text = text[:-1]
        if text.endswith(("%", "％")):
            percent = True
            text = text[:-1]
        else:
            percent = False
        for unit, multiplier in (("万亿", 1e12), ("亿", 1e8), ("万", 1e4), ("千", 1e3), ("百", 1e2)):
            if text.endswith(unit):
                text = text[:-len(unit)]
                try:
                    val = float(text) * multiplier
                except Exception:
                    return np.nan
                return val / 100.0 if percent else val
        if not text:
            return np.nan
        try:
            val = float(text)
        except Exception:
            return np.nan
        return val / 100.0 if percent else val

    def _to_numeric_series(self, series: pd.Series) -> pd.Series:
        return series.apply(self._parse_cn_number)

    def _normalize_report_date(self, value: Any) -> pd.Timestamp:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return pd.NaT
        if isinstance(value, (int, np.integer)):
            year = int(value)
            if 1900 <= year <= 2100:
                return pd.Timestamp(year, 12, 31)
        text = str(value).strip()
        if not text:
            return pd.NaT
        if re.fullmatch(r"\d{4}", text):
            return pd.Timestamp(int(text), 12, 31)
        match = re.search(r"(\d{4})年(一季报|三季报|半年报|中报|年报|四季报)", text)
        if match:
            year = int(match.group(1))
            label = match.group(2)
            if label in ("一季报",):
                return pd.Timestamp(year, 3, 31)
            if label in ("半年报", "中报"):
                return pd.Timestamp(year, 6, 30)
            if label in ("三季报",):
                return pd.Timestamp(year, 9, 30)
            return pd.Timestamp(year, 12, 31)
        try:
            return pd.to_datetime(text, errors='raise')
        except Exception:
            return pd.NaT

    def _normalize_ratio(self, series: pd.Series) -> pd.Series:
        numeric = series.apply(self._parse_cn_number)
        return numeric.apply(
            lambda v: v / 100 if pd.notna(v) and abs(v) > 1 else v
        )

    def _pivot_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = standardize_columns(df.copy())
        required = {'report_date', 'metric_name', 'value'}
        if required.issubset(df.columns):
            tmp = df[['report_date', 'metric_name', 'value']].copy()
            tmp['report_date'] = tmp['report_date'].apply(self._normalize_report_date)
            wide = tmp.pivot_table(
                index='report_date',
                columns='metric_name',
                values='value',
                aggfunc='last'
            ).reset_index()
        else:
            wide = df
        wide = standardize_columns(wide)
        if wide.columns.duplicated().any():
            merged = {}
            for col in pd.unique(wide.columns):
                cols = wide.loc[:, wide.columns == col]
                if cols.shape[1] == 1:
                    merged[col] = cols.iloc[:, 0]
                else:
                    merged[col] = cols.bfill(axis=1).iloc[:, 0]
            wide = pd.DataFrame(merged)
        if 'report_date' not in wide.columns:
            raise RuntimeError("financial data missing report_date")
        wide['report_date'] = wide['report_date'].apply(self._normalize_report_date)
        if wide['report_date'].isna().any():
            raise RuntimeError("financial data invalid report_date")
        return wide

    def fetch_financial_data(self, code: str) -> Optional[pd.DataFrame]:
        """
        获取单只股票的财务数据

        Args:
            code: 股票代码

        Returns:
            财务数据DataFrame
        """
        code = self._normalize_code(code)
        try:
            def _fetch_financial_ths() -> pd.DataFrame:
                abstract_df = self._retry_request(
                    ak.stock_financial_abstract_ths,
                    symbol=code,
                    indicator="按报告期"
                )
                benefit_df = self._retry_request(
                    ak.stock_financial_benefit_ths,
                    symbol=code
                )
                debt_df = self._retry_request(
                    ak.stock_financial_debt_ths,
                    symbol=code
                )

                if abstract_df is None or len(abstract_df) == 0:
                    raise RuntimeError("financial abstract data empty")
                if benefit_df is None or len(benefit_df) == 0:
                    raise RuntimeError("financial benefit data empty")
                if debt_df is None or len(debt_df) == 0:
                    raise RuntimeError("financial debt data empty")

                abstract_wide = self._pivot_metrics(abstract_df)
                benefit_wide = self._pivot_metrics(benefit_df)
                debt_wide = self._pivot_metrics(debt_df)

                df = abstract_wide.merge(benefit_wide, on='report_date', how='outer')
                df = df.merge(debt_wide, on='report_date', how='outer')
                df['code'] = code
                df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
                if df['report_date'].isna().any():
                    raise RuntimeError("financial data invalid report_date")
                df['announce_date'] = df['report_date'].apply(self._estimate_announce_date)

                def _expand_cols(base_names):
                    expanded = []
                    for base in base_names:
                        if base not in expanded:
                            expanded.append(base)
                        expanded.extend([c for c in df.columns if c.startswith(f"{base}_")])
                    return expanded

                def coalesce(cols):
                    series = None
                    for col in cols:
                        if col in df.columns:
                            vals = self._to_numeric_series(df[col])
                            series = vals if series is None else series.fillna(vals)
                    return series

                net_profit = coalesce(_expand_cols([
                    'net_profit', 'parent_holder_net_profit', 'parent_common_profit_total',
                    'parent_company_net_profit', 'parent_net_profit', 'index_deduct_holder_net_profit'
                ]))
                revenue = coalesce(_expand_cols(['operating_income_total', 'operating_income']))
                operating_cost = coalesce(_expand_cols(['operating_costs_total', 'operating_costs']))
                assets_total = coalesce(_expand_cols([
                    'assets_total', 'total_assets', 'asset_total', 'debt_and_equity_total',
                    'liabilities_and_equity_total'
                ]))
                equity_total = coalesce(_expand_cols([
                    'parent_holder_equity_total', 'holder_equity_total', 'equity', 'equity_total',
                    'owner_equity_total', 'total_equity'
                ]))

                roe = coalesce(_expand_cols(['roe', 'index_weighted_avg_roe', 'index_full_diluted_roe']))
                roa = coalesce(_expand_cols(['roa']))
                gross_margin = coalesce(_expand_cols(['gross_margin', 'sale_gross_margin']))
                net_margin = coalesce(_expand_cols(['net_margin']))

                if roe is None:
                    if net_profit is None or equity_total is None:
                        raise RuntimeError("financial data missing roe components")
                    roe = net_profit / equity_total
                if roa is None:
                    if net_profit is None or assets_total is None:
                        raise RuntimeError("financial data missing roa components")
                    roa = net_profit / assets_total
                if gross_margin is None:
                    if revenue is None or operating_cost is None:
                        raise RuntimeError("financial data missing gross_margin components")
                    gross_margin = (revenue - operating_cost) / revenue
                if net_margin is None:
                    if net_profit is None or revenue is None:
                        raise RuntimeError("financial data missing net_margin components")
                    net_margin = net_profit / revenue

                df['roe'] = self._normalize_ratio(roe).replace([np.inf, -np.inf], np.nan)
                df['roa'] = self._normalize_ratio(roa).replace([np.inf, -np.inf], np.nan)
                df['gross_margin'] = self._normalize_ratio(gross_margin).replace([np.inf, -np.inf], np.nan)
                df['net_margin'] = self._normalize_ratio(net_margin).replace([np.inf, -np.inf], np.nan)

                required = ['roe', 'roa', 'gross_margin', 'net_margin', 'announce_date']
                df = df.dropna(subset=required)
                if df.empty:
                    raise RuntimeError("financial ratios empty after dropna")

                return df

            def _fetch_financial_em() -> pd.DataFrame:
                if not hasattr(ak, 'stock_financial_analysis_indicator_em'):
                    raise RuntimeError("financial EM endpoint not available")
                symbol = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
                df = self._retry_request(
                    ak.stock_financial_analysis_indicator_em,
                    symbol=symbol,
                    indicator="按报告期"
                )
                if df is None or len(df) == 0:
                    raise RuntimeError("financial data fetch failed")

                df = df.copy()
                df['code'] = code
                if 'REPORT_DATE' not in df.columns:
                    raise RuntimeError("financial data missing REPORT_DATE")

                df = df.rename(columns={'REPORT_DATE': 'report_date'})
                df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
                if df['report_date'].isna().any():
                    raise RuntimeError("financial data invalid report_date")
                df['announce_date'] = df['report_date'].apply(self._estimate_announce_date)

                def pick_col(candidates):
                    for col in candidates:
                        if col in df.columns:
                            return col
                    return None

                roe_col = pick_col(['ROE_DILUTED', 'ROEJQ', 'ROEKCJQ'])
                roa_col = pick_col(['JROA', 'TOTAL_ROI', 'NET_ROI', 'ROIC'])
                gross_col = pick_col(['GROSS_PROFIT_RATIO', 'XSMLL'])
                gross_profit_col = pick_col(['GROSS_PROFIT'])
                net_margin_col = pick_col(['NET_PROFIT_RATIO'])
                revenue_col = pick_col(['TOTALOPERATEREVE'])
                profit_col = pick_col(['PARENTNETPROFIT'])

                if roe_col:
                    df['roe'] = self._normalize_ratio(df[roe_col])
                if roa_col:
                    df['roa'] = self._normalize_ratio(df[roa_col])
                if gross_col:
                    df['gross_margin'] = self._normalize_ratio(df[gross_col])
                elif gross_profit_col and revenue_col:
                    df['gross_margin'] = self._normalize_ratio(df[gross_profit_col] / df[revenue_col])
                if net_margin_col:
                    df['net_margin'] = self._normalize_ratio(df[net_margin_col])
                elif revenue_col and profit_col:
                    df['net_margin'] = self._normalize_ratio(df[profit_col] / df[revenue_col])

                required = ['roe', 'roa', 'gross_margin', 'net_margin', 'announce_date']
                missing = [col for col in required if col not in df.columns]
                if missing:
                    raise RuntimeError(f"financial data missing columns: {sorted(missing)}")
                df = df.dropna(subset=required)
                if df.empty:
                    raise RuntimeError("financial ratios empty after dropna")

                return df

            try:
                return _fetch_financial_ths()
            except Exception as ths_error:
                logger.warning(f"THS 财务数据失败, 尝试 EM: {ths_error}")

            return _fetch_financial_em()

        except Exception as e:
            logger.warning(f"获取 {code} 财务数据失败: {e}")
            raise RuntimeError("financial data fetch failed")

    def fetch_financial_data_em(self, code: str) -> Optional[pd.DataFrame]:
        """EM financial fallback disabled in strict mode."""
        raise RuntimeError("EM financial fallback disabled in strict mode")

    def fetch_all_financial_data(self, codes: List[str],
                                 skip_existing: bool = True,
                                 skip_failed: bool = True,
                                 update_existing: bool = True,
                                 force_update: bool = False) -> None:
        """
        批量下载财务数据

        Args:
            codes: 股票代码列表
            skip_existing: 是否跳过已存在的文件
            update_existing: 是否增量更新已存在的文件
            force_update: 是否强制更新（忽略披露期限制）
        """
        codes = self._normalize_codes(codes)
        save_dir = ensure_dir(self.raw_path / "financial")
        success_count = 0
        fail_count = 0
        skip_count = 0
        updated_count = 0
        update_fail_count = 0
        up_to_date_count = 0
        fail_fast = bool(self.config.get('data_fetch', {}).get('fail_fast', False))
        financial_config = self.config.get('data_fetch', {}).get('financial_update', {})
        smart_skip = bool(financial_config.get('smart_skip', True))
        disclosure_only = bool(financial_config.get('update_in_disclosure_period_only', False))
        force_interval_days = int(financial_config.get('force_update_interval_days', 0) or 0)

        if update_existing and disclosure_only and not force_update:
            allow_update = self._is_in_financial_disclosure_period()
            if not allow_update and force_interval_days > 0:
                status = self._load_financial_update_status(save_dir)
                last_update_str = status.get('last_update_date')
                last_update = pd.to_datetime(last_update_str, errors='coerce') if last_update_str else pd.NaT
                if pd.isna(last_update):
                    allow_update = True
                elif (pd.Timestamp.now() - last_update).days >= force_interval_days:
                    allow_update = True
            if not allow_update:
                logger.info("当前非财报披露期，跳过已存在财务数据更新")
                update_existing = False

        # 加载已知无法下载的股票列表
        failed_list_path = save_dir / "_failed_codes.txt"
        known_failed = set()
        if failed_list_path.exists():
            with open(failed_list_path, 'r', encoding='utf-8') as f:
                known_failed = set(line.strip() for line in f if line.strip())

        skip_failed_count = 0
        to_update = []
        # 预先统计已存在文件数
        if skip_existing:
            existing_files = set(f.stem for f in save_dir.glob("*.parquet"))
            skip_downloaded = len([c for c in codes if c in existing_files])
            skip_failed_count = len([c for c in codes if c in known_failed and c not in existing_files]) if skip_failed else 0
            to_download = [
                c for c in codes
                if c not in existing_files and (c not in known_failed or not skip_failed)
            ]
            if update_existing:
                to_update = [c for c in codes if c in existing_files]
            logger.info(
                f"财务数据: 已存在 {skip_downloaded} 个, 已知失败 {skip_failed_count} 个, "
                f"待下载 {len(to_download)} 个, 待更新 {len(to_update)} 个"
            )
        else:
            to_download = [c for c in codes if (c not in known_failed or not skip_failed)]

        if len(to_download) == 0 and len(to_update) == 0:
            logger.info("财务数据: 全部文件已存在或已跳过, 无需下载")
            return

        latest_quarter = None
        if to_update and update_existing and smart_skip:
            latest_quarter = self._get_latest_quarter_date()
            latest_quarter_ts = pd.to_datetime(latest_quarter, errors='coerce')
            filtered = []
            smart_skipped = 0
            for code in to_update:
                save_path = save_dir / f"{code}.parquet"
                try:
                    date_col = None
                    try:
                        date_df = load_parquet(save_path, columns=['report_date'])
                        if 'report_date' in date_df.columns:
                            date_col = 'report_date'
                    except Exception:
                        date_df = load_parquet(save_path, columns=['date'])
                        if 'date' in date_df.columns:
                            date_col = 'date'

                    if date_col is None:
                        filtered.append(code)
                        continue

                    date_df[date_col] = pd.to_datetime(date_df[date_col], errors='coerce')
                    max_date = date_df[date_col].max()
                    if pd.notna(max_date) and pd.notna(latest_quarter_ts) and max_date >= latest_quarter_ts:
                        up_to_date_count += 1
                        smart_skipped += 1
                        continue
                    filtered.append(code)
                except Exception:
                    filtered.append(code)
            if smart_skipped > 0:
                logger.info(f"智能跳过: {smart_skipped} 只股票已到最新季度 {latest_quarter}")
            to_update = filtered

        new_failed = []
        recovered = set()
        for code in tqdm(to_download, desc="下载财务数据"):
            save_path = save_dir / f"{code}.parquet"
            try:
                df = self.fetch_financial_data(code)
            except Exception as e:
                logger.warning(f"财务数据下载失败 {code}: {e}")
                fail_count += 1
                new_failed.append(code)
                continue

            if df is not None and len(df) > 0:
                save_parquet(df, save_path)
                success_count += 1
                if code in known_failed:
                    recovered.add(code)
            else:
                fail_count += 1
                new_failed.append(code)

        if to_update:
            for code in tqdm(to_update, desc="更新财务数据"):
                save_path = save_dir / f"{code}.parquet"
                try:
                    old_df = load_parquet(save_path)
                    new_df = self.fetch_financial_data(code)
                    if new_df is None or len(new_df) == 0:
                        up_to_date_count += 1
                        continue

                    date_col = None
                    if 'report_date' in old_df.columns:
                        date_col = 'report_date'
                    elif 'date' in old_df.columns:
                        date_col = 'date'
                    if date_col is None:
                        if 'report_date' in new_df.columns:
                            date_col = 'report_date'
                        elif 'date' in new_df.columns:
                            date_col = 'date'

                    if date_col is None:
                        save_parquet(new_df, save_path)
                        updated_count += 1
                        continue

                    old_df[date_col] = pd.to_datetime(old_df[date_col], errors='coerce')
                    new_df[date_col] = pd.to_datetime(new_df[date_col], errors='coerce')
                    max_date = old_df[date_col].max()
                    if pd.notna(max_date):
                        new_df = new_df[new_df[date_col] > max_date]
                        if new_df.empty:
                            up_to_date_count += 1
                            continue

                    merged = pd.concat([old_df, new_df], ignore_index=True)
                    merged = merged.drop_duplicates(subset=[date_col], keep='last')
                    merged = merged.sort_values(date_col).reset_index(drop=True)
                    save_parquet(merged, save_path)
                    updated_count += 1
                except Exception as e:
                    logger.warning(f"财务数据更新失败 {code}: {e}")
                    update_fail_count += 1

        # 保存失败列表
        if skip_failed:
            if new_failed:
                with open(failed_list_path, 'a', encoding='utf-8') as f:
                    for code in new_failed:
                        f.write(f"{code}\n")
                logger.info(f"新增 {len(new_failed)} 个无法下载的股票已记录")
        else:
            remaining_failed = (known_failed - recovered).union(new_failed)
            if remaining_failed:
                with open(failed_list_path, 'w', encoding='utf-8') as f:
                    for code in sorted(remaining_failed):
                        f.write(f"{code}\n")
            elif failed_list_path.exists():
                failed_list_path.unlink()
            if recovered:
                logger.info(f"移除 {len(recovered)} 个已恢复下载的股票")
            self._failed_codes_cache = None

        if (success_count > 0 or updated_count > 0 or fail_count > 0
                or update_fail_count > 0 or up_to_date_count > 0):
            if latest_quarter is None:
                latest_quarter = self._get_latest_quarter_date()
            self._save_financial_update_status(save_dir, latest_quarter)

        skip_count = skip_failed_count + up_to_date_count
        logger.info(
            f"财务数据下载完成: 新增 {success_count}, 更新 {updated_count}, "
            f"更新失败 {update_fail_count}, 失败 {fail_count}, 已跳过 {skip_count}"
        )
        if fail_fast and (fail_count > 0 or update_fail_count > 0):
            raise RuntimeError(
                f"financial data download failed for {fail_count + update_fail_count} codes"
            )

    def fetch_moneyflow(self, code: str) -> Optional[pd.DataFrame]:
        """
        获取单只股票的资金流向数据

        Args:
            code: 股票代码

        Returns:
            资金流向DataFrame
        """
        code = self._normalize_code(code)
        try:
            df = self._retry_request(
                ak.stock_individual_fund_flow,
                stock=code,
                market="sh" if code.startswith('6') else "sz"
            )

            if df is None or len(df) == 0:
                raise RuntimeError("moneyflow fetch failed")

            df['code'] = code
            # 标准化列名
            df = standardize_columns(df)
            return df

        except Exception as e:
            logger.warning(f"获取 {code} 资金流向失败: {e}")
            raise RuntimeError("moneyflow fetch failed")

    def fetch_all_moneyflow(self, codes: List[str],
                            skip_existing: bool = True,
                            skip_failed: bool = True,
                            update_existing: bool = True) -> None:
        """
        批量下载资金流向数据

        Args:
            codes: 股票代码列表
            skip_existing: 是否跳过已存在的文件
            update_existing: 是否增量更新已存在的文件
        """
        codes = self._normalize_codes(codes)
        save_dir = ensure_dir(self.raw_path / "moneyflow")
        success_count = 0
        fail_count = 0
        skip_count = 0
        updated_count = 0
        update_fail_count = 0
        up_to_date_count = 0
        fail_fast = bool(self.config.get('data_fetch', {}).get('fail_fast', False))

        # 加载已知无法下载的股票列表
        failed_list_path = save_dir / "_failed_codes.txt"
        known_failed = set()
        if failed_list_path.exists():
            with open(failed_list_path, 'r', encoding='utf-8') as f:
                known_failed = set(line.strip() for line in f if line.strip())

        skip_failed_count = 0
        to_update = []
        # 预先统计已存在文件数
        if skip_existing:
            existing_files = set(f.stem for f in save_dir.glob("*.parquet"))
            skip_downloaded = len([c for c in codes if c in existing_files])
            skip_failed_count = len([c for c in codes if c in known_failed and c not in existing_files]) if skip_failed else 0
            to_download = [
                c for c in codes
                if c not in existing_files and (c not in known_failed or not skip_failed)
            ]
            if update_existing:
                to_update = [c for c in codes if c in existing_files]
            logger.info(
                f"资金流向: 已存在 {skip_downloaded} 个, 已知失败 {skip_failed_count} 个, "
                f"待下载 {len(to_download)} 个, 待更新 {len(to_update)} 个"
            )
        else:
            to_download = [c for c in codes if (c not in known_failed or not skip_failed)]

        if len(to_download) == 0 and len(to_update) == 0:
            logger.info("资金流向: 全部文件已存在或已跳过, 无需下载")
            return

        latest_trade_date = None
        if to_update and update_existing:
            latest_trade_date = self._get_latest_trade_date()

        new_failed = []
        recovered = set()
        for code in tqdm(to_download, desc="下载资金流向"):
            save_path = save_dir / f"{code}.parquet"
            try:
                df = self.fetch_moneyflow(code)
            except Exception as e:
                logger.warning(f"资金流向下载失败 {code}: {e}")
                fail_count += 1
                new_failed.append(code)
                continue

            if df is not None and len(df) > 0:
                save_parquet(df, save_path)
                success_count += 1
                if code in known_failed:
                    recovered.add(code)
            else:
                fail_count += 1
                new_failed.append(code)

        if to_update:
            for code in tqdm(to_update, desc="更新资金流向"):
                save_path = save_dir / f"{code}.parquet"
                try:
                    if latest_trade_date is not None:
                        date_df = load_parquet(save_path, columns=['date'])
                        if date_df is not None and 'date' in date_df.columns:
                            local_dates = pd.to_datetime(
                                standardize_datetime_column(date_df['date']),
                                errors='coerce'
                            )
                            max_date = local_dates.max()
                            if pd.notna(max_date) and max_date >= latest_trade_date:
                                up_to_date_count += 1
                                continue

                    old_df = load_parquet(save_path)
                    new_df = self.fetch_moneyflow(code)
                    if new_df is None or len(new_df) == 0:
                        up_to_date_count += 1
                        continue

                    date_col = 'date' if 'date' in old_df.columns else None
                    if date_col is None and 'date' in new_df.columns:
                        date_col = 'date'

                    if date_col is None:
                        save_parquet(new_df, save_path)
                        updated_count += 1
                        continue

                    old_df[date_col] = pd.to_datetime(
                        standardize_datetime_column(old_df[date_col]),
                        errors='coerce'
                    )
                    new_df[date_col] = pd.to_datetime(
                        standardize_datetime_column(new_df[date_col]),
                        errors='coerce'
                    )
                    max_date = old_df[date_col].max()
                    if pd.notna(max_date):
                        new_df = new_df[new_df[date_col] > max_date]
                        if new_df.empty:
                            up_to_date_count += 1
                            continue

                    merged = pd.concat([old_df, new_df], ignore_index=True)
                    merged = merged.drop_duplicates(subset=[date_col], keep='last')
                    merged = merged.sort_values(date_col).reset_index(drop=True)
                    save_parquet(merged, save_path)
                    updated_count += 1
                except Exception as e:
                    logger.warning(f"资金流向更新失败 {code}: {e}")
                    update_fail_count += 1

        # 保存失败列表
        if skip_failed:
            if new_failed:
                with open(failed_list_path, 'a', encoding='utf-8') as f:
                    for code in new_failed:
                        f.write(f"{code}\n")
                logger.info(f"新增 {len(new_failed)} 个无法下载的股票已记录")
        else:
            remaining_failed = (known_failed - recovered).union(new_failed)
            if remaining_failed:
                with open(failed_list_path, 'w', encoding='utf-8') as f:
                    for code in sorted(remaining_failed):
                        f.write(f"{code}\n")
            elif failed_list_path.exists():
                failed_list_path.unlink()
            if recovered:
                logger.info(f"移除 {len(recovered)} 个已恢复下载的股票")
            self._failed_codes_cache = None

        skip_count = skip_failed_count + up_to_date_count
        logger.info(
            f"资金流向下载完成: 新增 {success_count}, 更新 {updated_count}, "
            f"更新失败 {update_fail_count}, 失败 {fail_count}, 已跳过 {skip_count}"
        )
        if fail_fast and (fail_count > 0 or update_fail_count > 0):
            raise RuntimeError(
                f"moneyflow download failed for {fail_count + update_fail_count} codes"
            )

    def fetch_northbound_flow(self) -> pd.DataFrame:
        """
        获取北向资金历史流向数据

        Returns:
            北向资金流向DataFrame
        """
        logger.info("获取北向资金历史数据...")

        try:
            df = self._retry_request(
                ak.stock_hsgt_hist_em,
                symbol="北向资金"
            )
            if df is None or len(df) == 0:
                raise RuntimeError("northbound flow fetch failed")

            # 标准化列名
            df = standardize_columns(df)
            save_path = self.raw_path / "northbound" / "north_flow.parquet"
            save_parquet(df, save_path)
            logger.info(f"北向资金历史数据已保存: {len(df)} 条")

            return df

        except Exception as e:
            logger.error(f"获取北向资金历史失败: {e}")
            raise RuntimeError("northbound flow fetch failed")

    def fetch_northbound_holding(self) -> pd.DataFrame:
        """
        获取北向资金持股数据

        Returns:
            北向持股DataFrame
        """
        logger.info("获取北向资金持股数据...")

        try:
            # 获取沪股通持股
            df_sh = self._retry_request(
                ak.stock_hsgt_hold_stock_em,
                market="沪股通",
                indicator="今日排行"
            )
            df_sh['market'] = '沪股通'

            time.sleep(self.interval)

            # 获取深股通持股
            df_sz = self._retry_request(
                ak.stock_hsgt_hold_stock_em,
                market="深股通",
                indicator="今日排行"
            )
            df_sz['market'] = '深股通'

            df = pd.concat([df_sh, df_sz], ignore_index=True)
            if df is None or len(df) == 0:
                raise RuntimeError("northbound holding fetch failed")

            # 标准化列名
            df = standardize_columns(df)
            save_path = self.raw_path / "northbound" / "north_holding.parquet"
            save_parquet(df, save_path)
            logger.info(f"北向持股数据已保存: {len(df)} 条")

            return df

        except Exception as e:
            logger.error(f"获取北向持股失败: {e}")
            raise RuntimeError("northbound holding fetch failed")

    def fetch_tech_signals(self) -> Dict[str, pd.DataFrame]:
        """
        获取技术选股信号（创新高、连续上涨等）

        Returns:
            技术信号字典
        """
        logger.info("获取技术选股信号...")
        signals = {}

        try:
            # 创新高
            df_cxg = self._retry_request(
                ak.stock_rank_cxg_ths,
                symbol="创月新高"
            )
            signals['new_high'] = df_cxg
            if df_cxg is None or len(df_cxg) == 0:
                raise RuntimeError("tech signals new_high empty")
            logger.info(f"创新高股票: {len(df_cxg)} 只")

            time.sleep(self.interval)

            # 连续上涨
            df_lxsz = self._retry_request(ak.stock_rank_lxsz_ths)
            signals['consecutive_up'] = df_lxsz
            if df_lxsz is None or len(df_lxsz) == 0:
                raise RuntimeError("tech signals consecutive_up empty")
            logger.info(f"连续上涨股票: {len(df_lxsz)} 只")

            # 保存
            save_dir = ensure_dir(self.raw_path / "tech_signals")
            today = datetime.now().strftime('%Y%m%d')

            for name, df in signals.items():
                save_path = save_dir / f"{name}_{today}.parquet"
                save_parquet(df, save_path)

            return signals

        except Exception as e:
            logger.error(f"获取技术选股信号失败: {e}")
            raise RuntimeError("tech signals fetch failed")

    def fetch_st_stocks(self) -> pd.DataFrame:
        """
        获取当前ST股票列表 (P1-1: ST数据下载)

        用于标识哪些股票当前是ST状态，回测时需要排除。

        Returns:
            ST股票列表DataFrame
        """
        logger.info("获取ST股票列表...")

        try:
            # 尝试使用东方财富ST板块接口
            df = self._retry_request(ak.stock_zh_a_st_em)
            if df is None or len(df) == 0:
                raise RuntimeError("st list fetch failed")

            if df is not None and len(df) > 0:
                # 标准化列名
                df = standardize_columns(df)

                # 添加下载日期
                df['download_date'] = datetime.now().strftime('%Y%m%d')

                # 保存
                save_dir = ensure_dir(self.raw_path / "st_stocks")
                today = datetime.now().strftime('%Y%m%d')
                save_path = save_dir / f"st_list_{today}.parquet"
                save_parquet(df, save_path)

                # 同时保存一个latest版本
                latest_path = save_dir / "st_list_latest.parquet"
                save_parquet(df, latest_path)

                logger.info(f"ST股票列表已保存: {len(df)} 只")
                return df

        except Exception as e:
            logger.warning(f"获取ST股票列表失败: {e}")
            raise RuntimeError("st list fetch failed")

        # 备选方案: 从股票列表中筛选名称包含ST的
        try:
            stock_list = self.get_stock_list()
            if len(stock_list) > 0 and 'name' in stock_list.columns:
                st_df = stock_list[stock_list['name'].str.contains('ST|\\*ST', na=False, regex=True)]
                if len(st_df) > 0:
                    st_df = st_df[['code', 'name']].copy()
                    st_df['download_date'] = datetime.now().strftime('%Y%m%d')

                    save_dir = ensure_dir(self.raw_path / "st_stocks")
                    save_path = save_dir / "st_list_latest.parquet"
                    save_parquet(st_df, save_path)

                    logger.info(f"从股票列表筛选ST股票: {len(st_df)} 只")
                    return st_df

        except Exception as e:
            logger.error(f"备选方案获取ST列表也失败: {e}")

        raise RuntimeError("st list fetch failed")

    def fetch_stock_name_history(self) -> pd.DataFrame:
        """
        P4-3: 获取A股股票曾用名列表

        用于追溯历史ST状态。从名称变更中可以识别ST/*ST状态变化。

        Returns:
            股票名称变更历史DataFrame
        """
        logger.info("获取A股股票曾用名历史...")

        try:
            # 使用 stock_info_change_name 接口获取曾用名
            df = self._retry_request(ak.stock_info_change_name)

            if df is not None and len(df) > 0:
                # 添加下载日期
                df['download_date'] = datetime.now().strftime('%Y%m%d')

                # 保存
                save_dir = ensure_dir(self.raw_path / "st_stocks")
                save_path = save_dir / "stock_name_history.parquet"
                save_parquet(df, save_path)

                logger.info(f"股票曾用名历史已保存: {len(df)} 条")
                return df

        except Exception as e:
            logger.warning(f"获取股票曾用名历史失败: {e}")

        raise RuntimeError("stock name history fetch failed")

    def build_st_history(self) -> Dict[str, List[tuple]]:
        """
        P4-3: 从股票名称变更历史构建ST状态表

        解析曾用名历史，提取每只股票的ST状态变更记录。

        Returns:
            ST历史字典: {code: [(start_date, end_date, st_type), ...]}
            st_type: 'ST', '*ST', 'S', 'S*ST' 等
        """
        logger.info("构建历史ST状态表...")

        # 加载名称变更历史
        save_dir = self.raw_path / "st_stocks"
        history_path = save_dir / "stock_name_history.parquet"

        if not history_path.exists():
            logger.info("名称变更历史不存在，先下载...")
            self.fetch_stock_name_history()

        if not history_path.exists():
            logger.error("无法获取名称变更历史")
            return {}

        df = load_parquet(history_path)
        if df is None or len(df) == 0:
            return {}

        # 解析ST状态
        st_history = {}
        st_pattern = r'^(S\*ST|S\*|ST|\*ST|S)'  # ST相关前缀

        # 假设数据有 code, 曾用名, 变更日期 等列
        # 实际列名可能不同，需要适配
        code_col = None
        name_col = None
        date_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'code' in col_lower or '代码' in col:
                code_col = col
            if 'name' in col_lower or '名称' in col or '曾用名' in col:
                name_col = col
            if 'date' in col_lower or '日期' in col or '变更' in col:
                date_col = col

        if code_col is None or name_col is None:
            logger.warning(f"无法识别列名: {df.columns.tolist()}")
            # 尝试直接使用位置
            if len(df.columns) >= 2:
                code_col = df.columns[0]
                name_col = df.columns[1]

        if code_col is None or name_col is None:
            logger.error("无法解析名称变更历史格式")
            return {}

        logger.info(f"解析列: code={code_col}, name={name_col}, date={date_col}")

        # 按股票代码分组，分析名称变化
        df = df.copy()
        if code_col:
            def _normalize_code_value(value):
                if pd.isna(value):
                    return None
                raw = str(value).strip()
                if not raw:
                    return None
                if '.' in raw:
                    raw = raw.split('.')[0]
                raw_lower = raw.lower()
                if raw_lower.startswith(('sh', 'sz', 'bj')):
                    raw = raw[2:]
                raw = raw.strip()
                if raw.isdigit():
                    return raw.zfill(6)
                return raw

            df['_norm_code'] = df[code_col].apply(_normalize_code_value)
            df = df[df['_norm_code'].notna()]
        else:
            df['_norm_code'] = None

        open_end = self.config.get('backtest', {}).get('backtest_end')
        if open_end:
            open_end = pd.to_datetime(open_end, errors='coerce')
            if pd.isna(open_end):
                open_end = datetime.now().strftime('%Y-%m-%d')
            else:
                open_end = open_end.strftime('%Y-%m-%d')
        else:
            open_end = datetime.now().strftime('%Y-%m-%d')

        import re
        for code, stock_df in df.groupby('_norm_code'):
            stock_df = stock_df.copy()

            if date_col and date_col in stock_df.columns:
                stock_df['_change_date'] = pd.to_datetime(stock_df[date_col], errors='coerce')
                if stock_df['_change_date'].notna().any():
                    stock_df = stock_df.sort_values('_change_date')
                stock_df['_next_date'] = stock_df['_change_date'].shift(-1)
            else:
                stock_df['_change_date'] = pd.NaT
                stock_df['_next_date'] = pd.NaT

            st_periods = []
            for _, row in stock_df.iterrows():
                name = str(row[name_col]) if pd.notna(row[name_col]) else ''
                change_date = row['_change_date']
                next_date = row['_next_date']
                if pd.isna(change_date):
                    continue
                if pd.notna(next_date):
                    end_date = (next_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    end_date = open_end

                # 检查是否包含ST相关前缀
                match = re.match(st_pattern, name)
                if match:
                    st_type = match.group(1)
                    st_periods.append({
                        'start_date': change_date.strftime('%Y-%m-%d'),
                        'end_date': end_date,
                        'name': name,
                        'st_type': st_type
                    })

            if st_periods:
                st_history[code] = st_periods

        # 保存ST历史
        if st_history:
            import json
            st_history_path = save_dir / "st_history.json"
            with open(st_history_path, 'w', encoding='utf-8') as f:
                # 将日期转换为字符串
                serializable = {}
                for code, periods in st_history.items():
                    serializable[code] = []
                    for p in periods:
                        serializable[code].append({
                            'start_date': str(p['start_date']) if p.get('start_date') else None,
                            'end_date': str(p['end_date']) if p.get('end_date') else None,
                            'name': p.get('name'),
                            'st_type': p.get('st_type')
                        })
                json.dump(serializable, f, ensure_ascii=False, indent=2)

            logger.info(f"ST历史已保存: {len(st_history)} 只股票有ST记录")

        return st_history

    def fetch_institution_holding(self, code: str) -> Optional[pd.DataFrame]:
        """
        P2-2: 获取单只股票的机构持股数据

        使用AKShare接口获取基金持仓明细

        Args:
            code: 股票代码

        Returns:
            机构持股DataFrame
        """
        code = self._normalize_code(code)
        try:
            # Fix 7: 使用动态日期而非硬编码
            quarter_date = self._get_latest_quarter_date()
            # 尝试使用 stock_report_fund_hold_detail (基金持仓明细)
            df = self._retry_request(
                ak.stock_report_fund_hold_detail,
                symbol=code,
                date=quarter_date
            )
            if df is None or len(df) == 0:
                raise RuntimeError("institution holding fetch failed")

            if df is not None and len(df) > 0:
                # 标准化列名
                df = standardize_columns(df)
                df['code'] = code
                df['download_date'] = datetime.now().strftime('%Y%m%d')
                return df

        except Exception as e:
            logger.debug(f"获取 {code} 机构持股(基金)失败: {e}")
            # 继续尝试备选方案，不要raise

        # 备选方案: 尝试获取十大股东
        try:
            df = self._retry_request(
                ak.stock_gdfx_free_holding_detail_em,
                symbol=code
            )

            if df is not None and len(df) > 0:
                # 标准化列名
                df = standardize_columns(df)
                df['code'] = code
                df['download_date'] = datetime.now().strftime('%Y%m%d')
                return df

        except Exception as e:
            logger.debug(f"获取 {code} 十大股东失败: {e}")

        raise RuntimeError("institution holding fetch failed")

    def fetch_all_institution_holding(self, codes: List[str],
                                      skip_existing: bool = True) -> None:
        """
        P2-2: 批量下载机构持股数据

        Args:
            codes: 股票代码列表
            skip_existing: 是否跳过已存在的文件
        """
        codes = self._normalize_codes(codes)
        save_dir = ensure_dir(self.raw_path / "institution")
        success_count = 0
        fail_count = 0

        for code in tqdm(codes, desc="下载机构持股"):
            save_path = save_dir / f"{code}.parquet"

            if skip_existing and save_path.exists():
                continue

            df = self.fetch_institution_holding(code)
            if df is not None and len(df) > 0:
                save_parquet(df, save_path)
                success_count += 1
            else:
                fail_count += 1

        logger.info(f"机构持股下载完成: 成功 {success_count}, 失败 {fail_count}")

    def fetch_fund_holding_stocks(self) -> pd.DataFrame:
        """
        P2-2: 获取公募基金重仓股汇总数据

        获取全市场公募基金重仓股TOP股票

        Returns:
            基金重仓股DataFrame
        """
        logger.info("获取公募基金重仓股数据...")

        # 动态获取最新季度日期
        quarter_date = self._get_latest_quarter_date()
        logger.info(f"使用季度日期: {quarter_date}")

        try:
            # 方案1: 使用 stock_report_fund_hold 接口
            # 注意: symbol 参数应为 "基金持仓" 而非 "基金重仓股"
            df = self._retry_request(
                ak.stock_report_fund_hold,
                symbol="基金持仓",
                date=quarter_date
            )
            if df is None or len(df) == 0:
                raise RuntimeError("fund holding fetch failed")

            if df is not None and len(df) > 0:
                # 标准化列名
                df = standardize_columns(df)
                df['download_date'] = datetime.now().strftime('%Y%m%d')

                save_dir = ensure_dir(self.raw_path / "institution")
                save_path = save_dir / "fund_holding_summary.parquet"
                save_parquet(df, save_path)

                logger.info(f"基金重仓股数据已保存: {len(df)} 条")
                return df

        except Exception as e:
            logger.warning(f"stock_report_fund_hold 失败: {e}, 尝试备选接口...")
            # 继续尝试备选方案，不要raise

        try:
            # 方案2: 备选接口 fund_report_stock_cninfo (巨潮资讯)
            df = self._retry_request(
                ak.fund_report_stock_cninfo,
                date=quarter_date
            )

            if df is not None and len(df) > 0:
                # 标准化列名
                df = standardize_columns(df)
                df['download_date'] = datetime.now().strftime('%Y%m%d')

                save_dir = ensure_dir(self.raw_path / "institution")
                save_path = save_dir / "fund_holding_summary.parquet"
                save_parquet(df, save_path)

                logger.info(f"基金重仓股数据已保存 (备选接口): {len(df)} 条")
                return df

        except Exception as e:
            logger.warning(f"获取基金重仓股失败: {e}")

        raise RuntimeError("fund holding fetch failed")


    def fetch_institution_holding_change(self) -> pd.DataFrame:
        """
        P2-2: 获取机构持股变动数据

        获取机构增减持变动

        Returns:
            机构持股变动DataFrame
        """
        logger.info("获取机构持股变动数据...")

        try:
            # Fix 7: 使用动态日期而非硬编码
            recent_date = self._get_recent_date(days_back=30)
            # 尝试获取机构调研数据作为替代
            df = self._retry_request(
                ak.stock_jgdy_tj_em,
                date=recent_date
            )
            if df is None or len(df) == 0:
                raise RuntimeError("institution holding change fetch failed")

            if df is not None and len(df) > 0:
                # 标准化列名
                df = standardize_columns(df)
                df['download_date'] = datetime.now().strftime('%Y%m%d')

                save_dir = ensure_dir(self.raw_path / "institution")
                save_path = save_dir / "institution_research.parquet"
                save_parquet(df, save_path)

                logger.info(f"机构调研数据已保存: {len(df)} 条")
                return df

        except Exception as e:
            logger.warning(f"获取机构调研数据失败: {e}")
            raise RuntimeError("institution holding change fetch failed")

        raise RuntimeError("institution holding change fetch failed")

    def fetch_index_daily(self, symbol: str = "sh000300",
                          start_date: str = None,
                          end_date: str = None) -> pd.DataFrame:
        """
        获取指数日K线数据

        Args:
            symbol: 指数代码，如 sh000300(沪深300), sh000905(中证500)
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            指数日K线DataFrame
        """
        start_date = start_date or self.config['data_fetch']['start_date']
        end_date = end_date or datetime.now().strftime('%Y%m%d')

        logger.info(f"获取指数 {symbol} 日K线...")

        try:
            df = self._retry_request(
                ak.stock_zh_index_daily,
                symbol=symbol
            )
            if df is None or len(df) == 0:
                raise RuntimeError("index daily fetch failed")

            if df is not None and len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                # 按日期过滤
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                df = df[(df['date'] >= start) & (df['date'] <= end)]

                # 标准化列名
                df = standardize_columns(df)
                save_path = self.raw_path / "index" / f"{symbol}.parquet"
                save_parquet(df, save_path)
                logger.info(f"指数数据已保存: {len(df)} 条")

            return df

        except Exception as e:
            logger.error(f"获取指数数据失败: {e}")
            raise RuntimeError("index daily fetch failed")

    def run_full_download(self, codes: List[str] = None) -> None:
        """
        运行完整的数据下载流程

        Args:
            codes: 股票代码列表，默认获取全A股
        """
        logger.info("=" * 50)
        logger.info("开始完整数据下载流程")
        logger.info("=" * 50)

        manual_codes = codes is not None

        # 1. 获取股票列表
        if codes is None:
            stock_list = self.get_stock_list()
            # 只排除ST、退市、北交所，不做流动性过滤（流动性过滤在选股阶段进行）
            stock_list = self.filter_stock_pool_for_download(stock_list)
            codes = stock_list['code'].tolist()

            # 保存股票列表
            save_path = self.raw_path / "stock_list.parquet"
            save_parquet(stock_list, save_path)

        logger.info(f"待下载股票数: {len(codes)}")

        # 2. 下载日K线
        logger.info("\n--- 下载日K线 ---")
        self.fetch_all_daily_price(codes, skip_failed=not manual_codes)

        # 3. 下载财务数据
        logger.info("\n--- 下载财务数据 ---")
        self.fetch_all_financial_data(codes, skip_failed=not manual_codes)

        # 4. 下载资金流向
        logger.info("\n--- 下载资金流向 ---")
        self.fetch_all_moneyflow(codes, skip_failed=not manual_codes)

        # 5. 下载北向资金
        logger.info("\n--- 下载北向资金 ---")
        self.fetch_northbound_flow()
        self.fetch_northbound_holding()

        # 6. 下载技术选股信号
        logger.info("\n--- 下载技术选股信号 ---")
        self.fetch_tech_signals()

        # 7. 下载基准指数
        logger.info("\n--- 下载基准指数 ---")
        self.fetch_index_daily("sh000300")  # 沪深300
        self.fetch_index_daily("sh000905")  # 中证500

        # 8. P1-1: 下载ST股票列表
        logger.info("\n--- 下载ST股票列表 ---")
        self.fetch_st_stocks()
        try:
            self.build_st_history()
        except Exception as e:
            logger.warning(f"st history build failed: {e}")

        # 9. P2-2: 下载机构持股数据
        logger.info("\n--- 下载机构持股数据 ---")
        self.fetch_fund_holding_stocks()  # 基金重仓股汇总
        self.fetch_institution_holding_change()  # 机构调研数据
        # 注意: 批量下载单只股票的机构持股非常耗时，默认不执行
        # 如需要可单独调用: self.fetch_all_institution_holding(codes)

        logger.info("=" * 50)
        logger.info("数据下载完成!")
        logger.info("=" * 50)


if __name__ == "__main__":
    # 测试代码
    fetcher = DataFetcher()

    # 获取股票列表
    stock_list = fetcher.get_stock_list()
    print(f"股票数量: {len(stock_list)}")

    # 过滤股票池
    filtered = fetcher.filter_stock_pool(stock_list)
    print(f"过滤后: {len(filtered)}")

    # 测试下载单只股票
    df = fetcher.fetch_daily_price("000001")
    if df is not None:
        print(f"000001 日K线: {len(df)} 条")
