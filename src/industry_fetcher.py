"""
行业映射模块

提供股票行业分类数据的获取与缓存:
- 同花顺行业 (ths): 默认数据源
- 东方财富行业 (em): 备选数据源
- 申万行业 (sw): 传统行业分类

用法:
    from src.industry_fetcher import IndustryFetcher

    fetcher = IndustryFetcher()
    mapping = fetcher.get_industry_mapping()  # 获取 code -> industry 映射
    industry = fetcher.get_code_industry('600000')  # 获取单只股票行业
"""

from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from .utils import setup_logger, get_project_root, ensure_dir, load_config

logger = setup_logger(__name__)


class IndustryFetcher:
    """
    行业映射数据获取器

    支持多数据源，优先同花顺，兜底东财
    """

    def __init__(self, config: dict = None):
        """
        Args:
            config: 配置字典，为 None 时从配置文件加载
        """
        self.config = config or load_config()
        self.root = get_project_root()
        self.raw_path = self.root / self.config['paths']['raw_data']
        self.industry_dir = ensure_dir(self.raw_path / 'industry')

        # 缓存
        self._mapping_cache: Dict[str, pd.DataFrame] = {}
        self._code_industry_cache: Dict[str, str] = {}

    def fetch_industry_mapping(self, source: str = 'sw', fallback: bool = False) -> pd.DataFrame:
        """Fetch industry mapping (strict; fallback disabled)."""
        cache_path = self.industry_dir / f'industry_mapping_{source}.parquet'
        cache_meta_path = self.industry_dir / f'industry_mapping_{source}_meta.json'

        if cache_path.exists() and cache_meta_path.exists():
            import json
            with open(cache_meta_path, 'r') as f:
                meta = json.load(f)
            fetch_time = pd.to_datetime(meta.get('fetch_time'))
            if (datetime.now() - fetch_time).days < 7:
                df = pd.read_parquet(cache_path)
                logger.info(f"industry map cache hit ({source}): {len(df)} rows")
                return df

        if source == 'ths':
            df = self.fetch_industry_mapping_ths()
        elif source == 'em':
            df = self.fetch_industry_mapping_em()
        elif source == 'sw':
            df = self.fetch_industry_mapping_sw()
        else:
            raise RuntimeError(f"unknown industry source: {source}")

        if df is None or len(df) == 0:
            raise RuntimeError(f"industry mapping fetch failed for source: {source}")

        df.to_parquet(cache_path, index=False)
        import json
        with open(cache_meta_path, 'w') as f:
            json.dump({
                'source': source,
                'fetch_time': datetime.now().isoformat(),
                'n_stocks': len(df),
                'n_industries': df['industry'].nunique()
            }, f, ensure_ascii=False, indent=2)

        processed_dir = self.root / self.config['paths']['processed_data']
        ensure_dir(processed_dir)
        processed_path = processed_dir / 'industry_map.parquet'
        normalized_df = df[['code', 'industry', 'industry_code']].copy()
        normalized_df.to_parquet(processed_path, index=False)
        logger.info(f"industry map saved: {processed_path}")

        return df

    def fetch_industry_mapping_em(self) -> pd.DataFrame:
        """
        获取东方财富行业分类

        Returns:
            DataFrame with columns: [code, industry, industry_code]
        """
        try:
            import akshare as ak

            # 获取东财行业板块列表
            industry_list = ak.stock_board_industry_name_em()

            if industry_list is None or len(industry_list) == 0:
                logger.warning("东财行业列表为空")
                raise RuntimeError("em industry mapping fetch failed")

            all_mappings = []

            for _, row in industry_list.iterrows():
                industry_name = row.get('板块名称', '')
                industry_code = row.get('板块代码', '')

                if not industry_name:
                    continue

                try:
                    # 获取行业成分股
                    constituents = ak.stock_board_industry_cons_em(symbol=industry_name)
                    if constituents is not None and len(constituents) > 0:
                        for _, stock in constituents.iterrows():
                            code = stock.get('代码', '')
                            if code:
                                all_mappings.append({
                                    'code': code,
                                    'industry': industry_name,
                                    'industry_code': industry_code,
                                    'source': 'em'
                                })
                except Exception as e:
                    logger.debug(f"获取行业 {industry_name} 成分股失败: {e}")
                    continue

            if len(all_mappings) == 0:
                raise RuntimeError("em industry mapping fetch failed")

            df = pd.DataFrame(all_mappings)
            df = df.drop_duplicates(subset=['code'], keep='first')

            logger.info(f"东财行业映射: {len(df)} 只股票, {df['industry'].nunique()} 个行业")
            return df

        except ImportError:
            logger.warning("akshare 未安装")
            raise RuntimeError("em industry mapping fetch failed")
        except Exception as e:
            logger.warning(f"获取东财行业映射失败: {e}")
            raise RuntimeError("em industry mapping fetch failed")

    def fetch_industry_mapping_sw(self) -> pd.DataFrame:
        """
        获取申万行业分类

        Returns:
            DataFrame with columns: [code, industry, industry_code, level]
        """
        try:
            import akshare as ak

            # 获取申万一级行业
            sw_list = ak.sw_index_first_info()

            if sw_list is None or len(sw_list) == 0:
                logger.warning("申万行业列表为空")
                raise RuntimeError("sw industry mapping fetch failed")

            all_mappings = []

            for _, row in sw_list.iterrows():
                industry_name = row.get('行业名称', '')
                industry_code = row.get('行业代码', '')

                if not industry_name or not industry_code:
                    continue

                try:
                    # 获取行业成分股
                    constituents = ak.index_component_sw(symbol=industry_code)
                    if constituents is not None and len(constituents) > 0:
                        for _, stock in constituents.iterrows():
                            code = stock.get('股票代码', stock.get('证券代码', ''))
                            if code:
                                # 清理代码格式
                                code = code.replace('.SH', '').replace('.SZ', '')
                                all_mappings.append({
                                    'code': code,
                                    'industry': industry_name,
                                    'industry_code': industry_code,
                                    'level': 1,
                                    'source': 'sw'
                                })
                except Exception as e:
                    logger.debug(f"获取申万行业 {industry_name} 成分股失败: {e}")
                    continue

            if len(all_mappings) == 0:
                raise RuntimeError("sw industry mapping fetch failed")

            df = pd.DataFrame(all_mappings)
            df = df.drop_duplicates(subset=['code'], keep='first')

            logger.info(f"申万行业映射: {len(df)} 只股票, {df['industry'].nunique()} 个行业")
            return df

        except ImportError:
            logger.warning("akshare 未安装")
            raise RuntimeError("sw industry mapping fetch failed")
        except Exception as e:
            logger.warning(f"获取申万行业映射失败: {e}")
            raise RuntimeError("sw industry mapping fetch failed")

    def get_industry_mapping(self, source: str = 'ths') -> Dict[str, str]:
        """
        获取 code -> industry 字典映射

        Args:
            source: 数据源

        Returns:
            {code: industry} 字典
        """
        if source in self._mapping_cache:
            df = self._mapping_cache[source]
        else:
            df = self.fetch_industry_mapping(source)
            self._mapping_cache[source] = df

        if len(df) == 0:
            return {}

        return dict(zip(df['code'], df['industry']))

    def get_code_industry(self, code: str, source: str = 'ths') -> Optional[str]:
        """
        获取单只股票的行业

        Args:
            code: 股票代码
            source: 数据源

        Returns:
            行业名称，如果不存在返回 None
        """
        if code in self._code_industry_cache:
            return self._code_industry_cache[code]

        mapping = self.get_industry_mapping(source)
        industry = mapping.get(code)

        if industry:
            self._code_industry_cache[code] = industry

        return industry

    def get_industry_codes(self, industry: str, source: str = 'ths') -> List[str]:
        """
        获取指定行业的所有股票

        Args:
            industry: 行业名称
            source: 数据源

        Returns:
            股票代码列表
        """
        if source in self._mapping_cache:
            df = self._mapping_cache[source]
        else:
            df = self.fetch_industry_mapping(source)
            self._mapping_cache[source] = df

        if len(df) == 0:
            return []

        return df[df['industry'] == industry]['code'].tolist()

    def get_all_industries(self, source: str = 'ths') -> List[str]:
        """
        获取所有行业名称

        Args:
            source: 数据源

        Returns:
            行业名称列表
        """
        if source in self._mapping_cache:
            df = self._mapping_cache[source]
        else:
            df = self.fetch_industry_mapping(source)
            self._mapping_cache[source] = df

        if len(df) == 0:
            return []

        return df['industry'].unique().tolist()

    def add_industry_to_df(self, df: pd.DataFrame,
                            code_col: str = 'code',
                            source: str = 'ths') -> pd.DataFrame:
        """
        给 DataFrame 添加行业列

        Args:
            df: 输入 DataFrame
            code_col: 股票代码列名
            source: 数据源

        Returns:
            添加了 industry 列的 DataFrame
        """
        mapping = self.get_industry_mapping(source)

        if len(mapping) == 0:
            df['industry'] = None
            return df

        df['industry'] = df[code_col].map(mapping)
        return df

    def compute_industry_stats(self, df: pd.DataFrame,
                                value_col: str,
                                source: str = 'ths') -> pd.DataFrame:
        """
        计算行业统计

        Args:
            df: 包含股票数据的 DataFrame
            value_col: 要统计的值列
            source: 数据源

        Returns:
            行业统计 DataFrame
        """
        if 'industry' not in df.columns:
            df = self.add_industry_to_df(df, source=source)

        stats = df.groupby('industry')[value_col].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).reset_index()

        stats.columns = ['industry', 'count', 'mean', 'std', 'min', 'max']
        return stats.sort_values('mean', ascending=False)


def get_industry_fetcher(config: dict = None) -> IndustryFetcher:
    """获取行业映射器实例"""
    return IndustryFetcher(config)


# 便捷函数
def fetch_industry_mapping(source: str = 'ths') -> pd.DataFrame:
    """获取行业映射 DataFrame"""
    return IndustryFetcher().fetch_industry_mapping(source)


def get_code_industry(code: str, source: str = 'ths') -> Optional[str]:
    """获取单只股票的行业"""
    return IndustryFetcher().get_code_industry(code, source)
