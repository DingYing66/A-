"""
为现有因子数据补充行业信息

用法:
    python scripts/add_industry_to_factors.py

功能:
    1. 从申万行业分类获取 code -> industry 映射
    2. 遍历所有因子文件，添加 industry 列
    3. 保存更新后的因子文件
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from tqdm import tqdm
from src.industry_fetcher import IndustryFetcher
from src.utils import setup_logger, load_config

logger = setup_logger(__name__)


def main():
    config = load_config()

    # 1. 获取行业映射
    logger.info("正在获取行业映射数据...")
    fetcher = IndustryFetcher(config)

    try:
        # 优先使用申万行业分类
        industry_mapping = fetcher.get_industry_mapping(source='sw')
    except Exception as e:
        logger.warning(f"申万行业获取失败: {e}, 尝试东方财富...")
        try:
            industry_mapping = fetcher.get_industry_mapping(source='em')
        except Exception as e2:
            logger.error(f"行业数据获取失败: {e2}")
            return

    logger.info(f"获取到 {len(industry_mapping)} 只股票的行业映射")

    # 2. 遍历因子文件
    factors_dir = project_root / config['paths']['processed_data'] / 'factors'

    if not factors_dir.exists():
        logger.error(f"因子目录不存在: {factors_dir}")
        return

    factor_files = list(factors_dir.glob("factors_*.parquet"))
    logger.info(f"找到 {len(factor_files)} 个因子文件")

    if len(factor_files) == 0:
        logger.warning("没有找到因子文件")
        return

    # 3. 更新每个因子文件
    updated_count = 0
    skipped_count = 0

    for factor_file in tqdm(factor_files, desc="更新因子文件"):
        try:
            df = pd.read_parquet(factor_file)

            # 检查是否已有 industry 列
            if 'industry' in df.columns:
                # 检查是否有有效数据
                valid_industry = df['industry'].notna().sum()
                if valid_industry > len(df) * 0.5:
                    skipped_count += 1
                    continue

            # 添加行业信息
            df['industry'] = df['code'].map(industry_mapping)

            # 统计缺失
            missing = df['industry'].isna().sum()
            if missing > 0:
                # 对于缺失的行业，标记为"未知"
                df['industry'] = df['industry'].fillna('未知')

            # 保存更新后的文件
            df.to_parquet(factor_file, index=False)
            updated_count += 1

        except Exception as e:
            logger.error(f"处理文件 {factor_file.name} 失败: {e}")
            continue

    logger.info(f"完成! 更新了 {updated_count} 个文件, 跳过了 {skipped_count} 个已有行业数据的文件")

    # 4. 验证结果
    if len(factor_files) > 0:
        sample_file = factor_files[0]
        df = pd.read_parquet(sample_file)
        logger.info(f"验证样本文件 {sample_file.name}:")
        logger.info(f"  - 列: {list(df.columns)}")
        logger.info(f"  - 行业分布: {df['industry'].value_counts().head(10).to_dict()}")


if __name__ == "__main__":
    main()
