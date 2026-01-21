"""
文本因子提取模块 (Phase 6)

从研报、公告等文本数据中提取因子:
- 研报情感 (research_sentiment)
- 研报数量 (research_count)
- 研报评级变化 (rating_change)
- 公告数量 (announcement_count)
- 公告情感 (announcement_sentiment)
- 关键词特征 (keyword_features)

数据源:
- AKShare: stock_research_report_em (东方财富研报)
- AKShare: stock_zh_a_disclosure_report_cninfo (巨潮公告)

用法:
    from src.deep_learning.text_factors import TextFactorExtractor

    extractor = TextFactorExtractor(config)
    factors = extractor.extract_factors(codes, date)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from collections import Counter
import re
import json

import numpy as np
import pandas as pd

# 可选依赖
try:
    import akshare as ak
    AK_AVAILABLE = True
except ImportError:
    AK_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger, get_project_root, ensure_dir, save_parquet

logger = setup_logger(__name__)


# ===================== 情感词典 =====================

# 金融领域积极词汇
POSITIVE_WORDS = {
    '增长', '上涨', '突破', '创新高', '利好', '超预期', '强劲', '领先',
    '优异', '提升', '扩张', '盈利', '增持', '买入', '推荐', '看好',
    '改善', '回暖', '复苏', '景气', '龙头', '核心', '优质', '稀缺',
    '高增长', '高景气', '高确定性', '低估', '安全边际', '护城河',
    '业绩释放', '拐点', '加速', '爆发', '翻倍', '井喷', '大幅',
    '首次覆盖', '强烈推荐', '重点推荐', '首推', '金股',
}

# 金融领域消极词汇
NEGATIVE_WORDS = {
    '下跌', '下滑', '下降', '亏损', '减少', '萎缩', '疲软', '承压',
    '利空', '风险', '警惕', '谨慎', '回调', '调整', '震荡', '波动',
    '减持', '卖出', '回避', '观望', '低于预期', '不及预期',
    '恶化', '放缓', '收缩', '困境', '困难', '挑战', '压力',
    '计提', '减值', '商誉', '暴雷', '爆仓', '违约', '诉讼',
    '处罚', '调查', '质押', '冻结', '限售', '解禁',
}

# 研报评级映射
RATING_MAP = {
    # 积极评级
    '买入': 5, '强烈推荐': 5, '强推': 5,
    '推荐': 4, '增持': 4, '优于大市': 4, '跑赢行业': 4,
    # 中性评级
    '持有': 3, '中性': 3, '同步大市': 3, '行业持平': 3, '审慎增持': 3,
    # 消极评级
    '减持': 2, '弱于大市': 2, '跑输行业': 2,
    '卖出': 1, '回避': 1, '强烈回避': 1,
}


# ===================== 基础情感分析器 =====================

class BaseSentimentAnalyzer:
    """
    基于词典的情感分析器

    不依赖外部模型，使用金融领域词典
    """

    def __init__(self,
                 positive_words: set = None,
                 negative_words: set = None):
        self.positive_words = positive_words or POSITIVE_WORDS
        self.negative_words = negative_words or NEGATIVE_WORDS

        # 是否使用 jieba 分词
        self.use_jieba = JIEBA_AVAILABLE
        if self.use_jieba:
            # 添加金融词汇到词典
            for word in self.positive_words | self.negative_words:
                jieba.add_word(word)

    def analyze(self, text: str) -> Dict[str, float]:
        """
        分析文本情感

        Args:
            text: 输入文本

        Returns:
            情感分析结果
        """
        if not text or not isinstance(text, str):
            raise RuntimeError("invalid text input for sentiment analysis")

        # 分词
        if self.use_jieba:
            words = list(jieba.cut(text))
        else:
            # 简单按字符切分 + 关键词匹配
            words = list(text)

        # 统计情感词
        positive_count = sum(1 for w in self.positive_words if w in text)
        negative_count = sum(1 for w in self.negative_words if w in text)

        # 计算情感得分 [-1, 1]
        total = positive_count + negative_count
        if total > 0:
            sentiment = (positive_count - negative_count) / total
            confidence = min(total / 10, 1.0)  # 词越多越置信
        else:
            sentiment = 0.0
            confidence = 0.0

        return {
            'sentiment': sentiment,
            'positive': positive_count,
            'negative': negative_count,
            'confidence': confidence,
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """批量分析"""
        return [self.analyze(text) for text in texts]


class TransformerSentimentAnalyzer:
    """
    基于 Transformer 的情感分析器

    使用预训练的中文金融情感模型
    """

    def __init__(self, model_name: str = None, device: str = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("需要安装 transformers: pip install transformers")

        # 默认使用中文金融情感模型
        self.model_name = model_name or "uer/roberta-base-finetuned-chinanews-chinese"
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        logger.info(f"加载情感分析模型: {self.model_name}")
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            device=0 if self.device == 'cuda' else -1,
        )

    def analyze(self, text: str) -> Dict[str, float]:
        """分析文本情感"""
        if not text or not isinstance(text, str):
            raise RuntimeError("invalid text input for sentiment analysis")

        # 截断过长文本
        text = text[:512]

        result = self.pipeline(text)[0]
        label = result['label']
        score = result['score']

        # 转换为情感得分
        if 'POSITIVE' in label.upper() or 'POS' in label.upper():
            sentiment = score
        elif 'NEGATIVE' in label.upper() or 'NEG' in label.upper():
            sentiment = -score
        else:
            sentiment = 0.0

        return {
            'sentiment': sentiment,
            'label': label,
            'confidence': score,
        }

    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """批量分析"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                if not text or not isinstance(text, str):
                    raise RuntimeError("invalid text input for sentiment analysis")
            batch = [t[:512] for t in batch]

            batch_results = self.pipeline(batch)
            for r in batch_results:
                label = r['label']
                score = r['score']
                if 'POSITIVE' in label.upper() or 'POS' in label.upper():
                    sentiment = score
                elif 'NEGATIVE' in label.upper() or 'NEG' in label.upper():
                    sentiment = -score
                else:
                    sentiment = 0.0
                results.append({'sentiment': sentiment, 'label': label, 'confidence': score})

        return results


# ===================== 数据获取器 =====================

class ResearchReportFetcher:
    """
    研报数据获取器

    数据源: AKShare stock_research_report_em
    """

    def __init__(self, cache_dir: str = None):
        if not AK_AVAILABLE:
            raise ImportError("需要安装 akshare: pip install akshare")

        self.cache_dir = Path(cache_dir or get_project_root() / 'data' / 'processed' / 'text')
        ensure_dir(self.cache_dir)

    def fetch(self,
              symbol: str,
              start_date: str = None,
              end_date: str = None) -> pd.DataFrame:
        """
        获取个股研报

        Args:
            symbol: 股票代码 (如 '000001')
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            研报 DataFrame
        """
        # 使用 AKShare 获取研报
        df = ak.stock_research_report_em(symbol=symbol)

        if df is None or df.empty:
            raise RuntimeError(f"research report data missing for {symbol}")

        # 标准化列名
        df.columns = [c.strip() for c in df.columns]

        # 解析日期
        if '日期' in df.columns:
            df['date'] = pd.to_datetime(df['日期'])
        elif '发布日期' in df.columns:
            df['date'] = pd.to_datetime(df['发布日期'])
        else:
            # 尝试找日期列
            for col in df.columns:
                if '日期' in col or 'date' in col.lower():
                    df['date'] = pd.to_datetime(df[col])
                    break

        if 'date' not in df.columns:
            raise RuntimeError("missing date column in research report data")

        # 日期过滤
        # C5 修复: 确保只使用决策时间之前发布的数据 (防止研报数据泄露)
        if start_date:
            start = pd.to_datetime(start_date)
            df = df[df['date'] >= start]
        if end_date:
            # C5: end_date 即为决策时间，严格过滤 publish_time <= decision_time
            decision_time = pd.to_datetime(end_date)
            original_count = len(df)
            df = df[df['date'] <= decision_time]
            filtered_count = original_count - len(df)
            if filtered_count > 0:
                logger.debug(f"研报时间对齐: 移除 {filtered_count} 条未来数据")

        return df

    def fetch_batch(self,
                    symbols: List[str],
                    start_date: str = None,
                    end_date: str = None) -> Dict[str, pd.DataFrame]:
        """批量获取研报"""
        results = {}
        for symbol in symbols:
            df = self.fetch(symbol, start_date, end_date)
            if not df.empty:
                results[symbol] = df
        return results


class AnnouncementFetcher:
    """
    公告数据获取器

    数据源: AKShare stock_zh_a_disclosure_report_cninfo
    """

    def __init__(self, cache_dir: str = None):
        if not AK_AVAILABLE:
            raise ImportError("需要安装 akshare: pip install akshare")

        self.cache_dir = Path(cache_dir or get_project_root() / 'data' / 'processed' / 'text')
        ensure_dir(self.cache_dir)

    def fetch(self,
              symbol: str,
              start_date: str = None,
              end_date: str = None) -> pd.DataFrame:
        """
        获取个股公告

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            公告 DataFrame
        """
        # 尝试获取公告
        # 注意: AKShare API 可能有变化，需要适配
        df = ak.stock_zh_a_disclosure_report_cninfo(symbol=symbol)

        if df is None or df.empty:
            raise RuntimeError(f"announcement data missing for {symbol}")

        # 标准化列名
        df.columns = [c.strip() for c in df.columns]

        # 解析日期
        date_cols = ['公告日期', '日期', '发布日期', 'date']
        for col in date_cols:
            if col in df.columns:
                df['date'] = pd.to_datetime(df[col])
                break

        if 'date' not in df.columns:
            raise RuntimeError("missing date column in announcement data")

        # 日期过滤
        # C5 修复: 确保只使用决策时间之前发布的数据 (防止公告数据泄露)
        if start_date:
            start = pd.to_datetime(start_date)
            df = df[df['date'] >= start]
        if end_date:
            # C5: end_date 即为决策时间，严格过滤 publish_time <= decision_time
            decision_time = pd.to_datetime(end_date)
            original_count = len(df)
            df = df[df['date'] <= decision_time]
            filtered_count = original_count - len(df)
            if filtered_count > 0:
                logger.debug(f"公告时间对齐: 移除 {filtered_count} 条未来数据")

        return df


# ===================== 文本因子提取器 =====================

class TextFactorExtractor:
    """
    文本因子提取器

    从研报和公告中提取文本因子
    """

    def __init__(self,
                 config: dict = None,
                 use_transformer: bool = False,
                 cache_dir: str = None):
        """
        Args:
            config: 配置字典
            use_transformer: 是否使用 Transformer 模型
            cache_dir: 缓存目录
        """
        self.config = config or {}
        if use_transformer and not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not available but use_transformer=True")
        self.use_transformer = use_transformer

        # 缓存目录
        self.cache_dir = Path(cache_dir or get_project_root() / 'data' / 'processed' / 'text_factors')
        ensure_dir(self.cache_dir)

        # 初始化情感分析器
        if self.use_transformer:
            self.sentiment_analyzer = TransformerSentimentAnalyzer()
        else:
            self.sentiment_analyzer = BaseSentimentAnalyzer()

        # 初始化数据获取器
        if AK_AVAILABLE:
            self.research_fetcher = ResearchReportFetcher(cache_dir)
            self.announcement_fetcher = AnnouncementFetcher(cache_dir)
        else:
            self.research_fetcher = None
            self.announcement_fetcher = None

        # 配置
        text_config = self.config.get('text', {})
        self.lookback_days = text_config.get('lookback_days', 30)  # 回看天数
        self.min_reports = text_config.get('min_reports', 1)  # 最少研报数

        logger.info(f"文本因子提取器初始化: transformer={self.use_transformer}")

    def extract_research_factors(self,
                                  code: str,
                                  date: pd.Timestamp,
                                  lookback_days: int = None) -> Dict[str, float]:
        """
        提取研报因子

        Args:
            code: 股票代码
            date: 截止日期
            lookback_days: 回看天数

        Returns:
            研报因子字典
        """
        lookback_days = lookback_days or self.lookback_days

        factors = {
            'research_count': 0,
            'research_sentiment': 0.0,
            'research_sentiment_std': 0.0,
            'research_positive_ratio': 0.0,
            'research_rating_avg': 0.0,
            'research_rating_change': 0.0,
            'research_target_price_ratio': 0.0,
        }

        if self.research_fetcher is None:
            raise RuntimeError("research fetcher not available")

        # 计算日期范围
        end_date = date.strftime('%Y%m%d')
        start_date = (date - timedelta(days=lookback_days)).strftime('%Y%m%d')

        # 获取研报
        df = self.research_fetcher.fetch(code, start_date, end_date)

        if df.empty:
            raise RuntimeError(f"no research reports for {code}")

        # 研报数量
        factors['research_count'] = len(df)

        # 情感分析
        text_cols = ['标题', '研报标题', 'title', '摘要', '内容']
        texts = []
        for col in text_cols:
            if col in df.columns:
                texts.extend(df[col].dropna().tolist())

        if not texts:
            raise RuntimeError(f"no research text available for {code}")
        sentiments = [self.sentiment_analyzer.analyze(t)['sentiment'] for t in texts]
        factors['research_sentiment'] = np.mean(sentiments)
        factors['research_sentiment_std'] = np.std(sentiments) if len(sentiments) > 1 else 0
        factors['research_positive_ratio'] = np.mean([1 if s > 0 else 0 for s in sentiments])

        # 评级分析
        rating_cols = ['评级', '投资评级', 'rating', '最新评级']
        for col in rating_cols:
            if col in df.columns:
                ratings = []
                for r in df[col].dropna():
                    r_str = str(r).strip()
                    for key, val in RATING_MAP.items():
                        if key in r_str:
                            ratings.append(val)
                            break

                if ratings:
                    factors['research_rating_avg'] = np.mean(ratings)
                    # 评级变化 (最新 vs 平均)
                    if len(ratings) > 1:
                        factors['research_rating_change'] = ratings[0] - np.mean(ratings[1:])
                break

        # 目标价分析
        price_cols = ['目标价', '目标价格', 'target_price']
        for col in price_cols:
            if col in df.columns:
                prices = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(prices) > 0:
                    # 目标价相对变化
                    factors['research_target_price_ratio'] = prices.iloc[0] / prices.mean() - 1
                break

        return factors

    def extract_announcement_factors(self,
                                      code: str,
                                      date: pd.Timestamp,
                                      lookback_days: int = None) -> Dict[str, float]:
        """
        提取公告因子

        Args:
            code: 股票代码
            date: 截止日期
            lookback_days: 回看天数

        Returns:
            公告因子字典
        """
        lookback_days = lookback_days or self.lookback_days

        factors = {
            'announcement_count': 0,
            'announcement_sentiment': 0.0,
            'announcement_important_count': 0,
            'announcement_risk_count': 0,
            'announcement_positive_count': 0,
            'announcement_negative_count': 0,
        }

        if self.announcement_fetcher is None:
            raise RuntimeError("announcement fetcher not available")

        # 计算日期范围
        end_date = date.strftime('%Y%m%d')
        start_date = (date - timedelta(days=lookback_days)).strftime('%Y%m%d')

        # 获取公告
        df = self.announcement_fetcher.fetch(code, start_date, end_date)

        if df.empty:
            raise RuntimeError(f"no announcements for {code}")

        # 公告数量
        factors['announcement_count'] = len(df)

        # 情感分析
        text_cols = ['公告标题', '标题', 'title', '公告内容', '内容']
        texts = []
        for col in text_cols:
            if col in df.columns:
                texts.extend(df[col].dropna().tolist())

        if not texts:
            raise RuntimeError(f"no announcement text available for {code}")
        sentiments = [self.sentiment_analyzer.analyze(t) for t in texts]
        factors['announcement_sentiment'] = np.mean([s['sentiment'] for s in sentiments])
        factors['announcement_positive_count'] = sum(1 for s in sentiments if s['sentiment'] > 0.1)
        factors['announcement_negative_count'] = sum(1 for s in sentiments if s['sentiment'] < -0.1)

        # 重要公告检测
        important_keywords = ['重大', '重要', '业绩预告', '业绩快报', '分红', '送转', '增发', '回购', '收购', '重组']
        risk_keywords = ['风险', '警示', '诉讼', '处罚', '调查', '违规', '质押', '冻结', '减持']

        for text in texts:
            if any(kw in text for kw in important_keywords):
                factors['announcement_important_count'] += 1
            if any(kw in text for kw in risk_keywords):
                factors['announcement_risk_count'] += 1

        return factors

    def extract_keyword_factors(self,
                                 code: str,
                                 date: pd.Timestamp,
                                 lookback_days: int = None) -> Dict[str, float]:
        """
        提取关键词因子

        Args:
            code: 股票代码
            date: 截止日期
            lookback_days: 回看天数

        Returns:
            关键词因子字典
        """
        lookback_days = lookback_days or self.lookback_days

        factors = {
            'keyword_growth': 0,      # 成长类词汇
            'keyword_value': 0,       # 价值类词汇
            'keyword_risk': 0,        # 风险类词汇
            'keyword_momentum': 0,    # 动量类词汇
            'keyword_tech': 0,        # 技术/创新类词汇
        }

        # 关键词组
        growth_keywords = {'增长', '扩张', '高增长', '快速增长', '景气', '爆发', '井喷', '翻倍'}
        value_keywords = {'低估', '安全边际', '分红', '股息', '价值', '便宜', '护城河'}
        risk_keywords = {'风险', '警惕', '谨慎', '减持', '质押', '诉讼', '暴雷', '违约'}
        momentum_keywords = {'突破', '新高', '创新高', '强势', '涨停', '大涨', '加速'}
        tech_keywords = {'创新', '技术', '研发', '专利', '首创', '领先', '突破性', 'AI', '人工智能'}

        # 收集文本
        texts = []

        if self.research_fetcher is None:
            raise RuntimeError("research fetcher not available")
        if self.research_fetcher:
            end_date = date.strftime('%Y%m%d')
            start_date = (date - timedelta(days=lookback_days)).strftime('%Y%m%d')
            df = self.research_fetcher.fetch(code, start_date, end_date)
            for col in ['标题', '研报标题', 'title', '摘要']:
                if col in df.columns:
                    texts.extend(df[col].dropna().tolist())

        if not texts:
            raise RuntimeError(f"no text available for keywords: {code}")

        # 统计关键词
        all_text = ' '.join(texts)

        factors['keyword_growth'] = sum(1 for kw in growth_keywords if kw in all_text)
        factors['keyword_value'] = sum(1 for kw in value_keywords if kw in all_text)
        factors['keyword_risk'] = sum(1 for kw in risk_keywords if kw in all_text)
        factors['keyword_momentum'] = sum(1 for kw in momentum_keywords if kw in all_text)
        factors['keyword_tech'] = sum(1 for kw in tech_keywords if kw in all_text)

        return factors

    def extract_factors(self,
                        code: str,
                        date: pd.Timestamp,
                        lookback_days: int = None) -> Dict[str, float]:
        """
        提取所有文本因子

        Args:
            code: 股票代码
            date: 截止日期
            lookback_days: 回看天数

        Returns:
            文本因子字典
        """
        factors = {}

        # 研报因子
        research_factors = self.extract_research_factors(code, date, lookback_days)
        factors.update({f'text_{k}': v for k, v in research_factors.items()})

        # 公告因子
        announcement_factors = self.extract_announcement_factors(code, date, lookback_days)
        factors.update({f'text_{k}': v for k, v in announcement_factors.items()})

        # 关键词因子
        keyword_factors = self.extract_keyword_factors(code, date, lookback_days)
        factors.update({f'text_{k}': v for k, v in keyword_factors.items()})

        return factors

    def extract_factors_batch(self,
                               codes: List[str],
                               date: pd.Timestamp,
                               lookback_days: int = None,
                               use_cache: bool = True) -> pd.DataFrame:
        """
        批量提取文本因子

        Args:
            codes: 股票代码列表
            date: 截止日期
            lookback_days: 回看天数
            use_cache: 是否使用缓存

        Returns:
            文本因子 DataFrame
        """
        date_str = date.strftime('%Y%m%d')
        cache_path = self.cache_dir / f'text_factors_{date_str}.parquet'

        # 检查缓存
        if use_cache and cache_path.exists():
            logger.info(f"加载文本因子缓存: {cache_path}")
            cached_df = pd.read_parquet(cache_path)
            # 检查是否包含所有请求的股票
            cached_codes = set(cached_df['code'].tolist())
            if set(codes).issubset(cached_codes):
                return cached_df[cached_df['code'].isin(codes)]

        # 提取因子
        logger.info(f"提取 {len(codes)} 只股票的文本因子 ({date_str})")

        results = []
        for i, code in enumerate(codes):
            if (i + 1) % 50 == 0:
                logger.info(f"进度: {i + 1}/{len(codes)}")

            factors = self.extract_factors(code, date, lookback_days)
            factors['code'] = code
            factors['date'] = date
            results.append(factors)

        df = pd.DataFrame(results)

        # 保存缓存
        if use_cache:
            save_parquet(df, cache_path)
            logger.info(f"保存文本因子缓存: {cache_path}")

            # C5 修复: 保存缓存元数据 (确保可复现性)
            import json
            import hashlib
            cache_meta = {
                'date': date.isoformat(),
                'lookback_days': lookback_days or self.lookback_days,
                'use_transformer': self.use_transformer,
                'created_at': datetime.now().isoformat(),
                'n_stocks': len(df),
                'n_factors': len([c for c in df.columns if c not in ['code', 'date']]),
                'config_hash': hashlib.sha256(
                    json.dumps({
                        'lookback': lookback_days or self.lookback_days,
                        'transformer': self.use_transformer
                    }, sort_keys=True).encode()
                ).hexdigest()[:16]
            }
            meta_path = cache_path.with_suffix('.meta.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(cache_meta, f, ensure_ascii=False, indent=2)
            logger.debug(f"保存文本因子元数据: {meta_path}")

        return df

    def merge_with_factor_df(self,
                              factor_df: pd.DataFrame,
                              lookback_days: int = None,
                              use_cache: bool = True) -> pd.DataFrame:
        """
        将文本因子合并到现有因子 DataFrame

        Args:
            factor_df: 现有因子 DataFrame
            lookback_days: 回看天数
            use_cache: 是否使用缓存

        Returns:
            合并后的 DataFrame
        """
        if 'code' not in factor_df.columns:
            raise RuntimeError("factor_df missing code column")

        codes = factor_df['code'].unique().tolist()

        # 获取日期
        if 'date' in factor_df.columns:
            date = pd.to_datetime(factor_df['date'].iloc[0])
        else:
            raise RuntimeError("factor_df missing date column")

        # 提取文本因子
        text_factors = self.extract_factors_batch(codes, date, lookback_days, use_cache)

        # 合并
        merged = factor_df.merge(
            text_factors.drop(columns=['date'], errors='ignore'),
            on='code',
            how='left'
        )

        text_cols = [c for c in merged.columns if c.startswith('text_')]
        if merged[text_cols].isna().any().any():
            raise RuntimeError("text factors contain NaN after merge")

        return merged


# ===================== 综合因子构建器 =====================

class TextFactorBuilder:
    """
    文本因子构建器

    用于构建和缓存文本因子序列
    """

    def __init__(self, config: dict = None, cache_dir: str = None):
        self.config = config or {}
        self.cache_dir = Path(cache_dir or get_project_root() / 'data' / 'processed' / 'text_factors')
        ensure_dir(self.cache_dir)

        text_cfg = self.config.get('text', {})
        use_transformer = bool(text_cfg.get('use_transformer', False))
        self.extractor = TextFactorExtractor(config, use_transformer=use_transformer)

    def build_for_date_range(self,
                              codes: List[str],
                              start_date: str,
                              end_date: str,
                              freq: str = 'W') -> pd.DataFrame:
        """
        构建日期范围内的文本因子

        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            freq: 频率 (D/W/M)

        Returns:
            文本因子 DataFrame
        """
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        all_factors = []
        for date in dates:
            logger.info(f"构建文本因子: {date.strftime('%Y-%m-%d')}")
            factors = self.extractor.extract_factors_batch(codes, date)
            all_factors.append(factors)

        return pd.concat(all_factors, ignore_index=True)

    def get_factor_names(self) -> List[str]:
        """获取文本因子名称列表"""
        return [
            # 研报因子
            'text_research_count',
            'text_research_sentiment',
            'text_research_sentiment_std',
            'text_research_positive_ratio',
            'text_research_rating_avg',
            'text_research_rating_change',
            'text_research_target_price_ratio',
            # 公告因子
            'text_announcement_count',
            'text_announcement_sentiment',
            'text_announcement_important_count',
            'text_announcement_risk_count',
            'text_announcement_positive_count',
            'text_announcement_negative_count',
            # 关键词因子
            'text_keyword_growth',
            'text_keyword_value',
            'text_keyword_risk',
            'text_keyword_momentum',
            'text_keyword_tech',
        ]


# ===================== 工厂函数 =====================

def create_text_extractor(config: dict = None,
                          use_transformer: bool = False) -> TextFactorExtractor:
    """
    创建文本因子提取器

    Args:
        config: 配置字典
        use_transformer: 是否使用 Transformer

    Returns:
        TextFactorExtractor 实例
    """
    return TextFactorExtractor(config, use_transformer)


def extract_text_factors(factor_df: pd.DataFrame,
                         config: dict = None,
                         use_transformer: bool = False,
                         lookback_days: int = 30) -> pd.DataFrame:
    """
    提取文本因子并合并到因子 DataFrame

    Args:
        factor_df: 因子 DataFrame
        config: 配置字典
        use_transformer: 是否使用 Transformer
        lookback_days: 回看天数

    Returns:
        合并后的 DataFrame
    """
    extractor = TextFactorExtractor(config, use_transformer)
    return extractor.merge_with_factor_df(factor_df, lookback_days)


def check_text_dependencies() -> Dict[str, bool]:
    """
    检查文本因子依赖

    Returns:
        依赖可用性字典
    """
    return {
        'akshare': AK_AVAILABLE,
        'transformers': TRANSFORMERS_AVAILABLE,
        'jieba': JIEBA_AVAILABLE,
    }
