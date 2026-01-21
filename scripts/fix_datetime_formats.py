"""
修复数据文件中的时间格式

将所有 timestamp 格式的日期列转换为标准的 datetime64[ns] 格式
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm
from src.utils import load_parquet, save_parquet, standardize_datetime_column, setup_logger

logger = setup_logger('fix_datetime')


def fix_datetime_in_file(file_path: Path, dry_run: bool = False) -> dict:
    """
    修复单个文件中的时间格式

    Args:
        file_path: 文件路径
        dry_run: 是否只检查不修改

    Returns:
        修复信息字典
    """
    result = {
        'file': str(file_path),
        'fixed': False,
        'columns': [],
        'error': None
    }

    try:
        df = pd.read_parquet(file_path)

        # 查找需要修复的日期列
        date_cols = []
        date_keywords = ['date', 'time', 'datetime']

        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in date_keywords):
                # 检查是否需要转换
                if pd.api.types.is_numeric_dtype(df[col]):
                    sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else 0
                    # 如果是 timestamp 格式（>1e9）
                    if sample > 1e9:
                        date_cols.append(col)
                elif df[col].dtype == 'object' and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    # 字符串格式也标准化为 datetime64
                    date_cols.append(col)

        if date_cols:
            if not dry_run:
                # 修复日期列
                for col in date_cols:
                    df[col] = standardize_datetime_column(df[col])

                # 保存
                save_parquet(df, file_path, standardize_dates=False)  # 已经标准化了，不需要再处理

            result['fixed'] = True
            result['columns'] = date_cols
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}修复 {file_path.name}: {date_cols}")

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"处理 {file_path} 失败: {e}")

    return result


def fix_datetime_in_directory(directory: Path, dry_run: bool = False,
                               pattern: str = "*.parquet") -> dict:
    """
    批量修复目录中的时间格式

    Args:
        directory: 目录路径
        dry_run: 是否只检查不修改
        pattern: 文件匹配模式

    Returns:
        统计信息字典
    """
    if not directory.exists():
        logger.warning(f"目录不存在: {directory}")
        return {'total': 0, 'fixed': 0, 'failed': 0}

    files = list(directory.glob(pattern))

    if not files:
        logger.info(f"目录 {directory} 中没有找到 {pattern} 文件")
        return {'total': 0, 'fixed': 0, 'failed': 0}

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}处理目录: {directory}")
    logger.info(f"找到 {len(files)} 个文件")

    stats = {'total': len(files), 'fixed': 0, 'failed': 0}

    for file_path in tqdm(files, desc=f"修复 {directory.name}"):
        result = fix_datetime_in_file(file_path, dry_run=dry_run)
        if result['fixed']:
            stats['fixed'] += 1
        if result['error']:
            stats['failed'] += 1

    return stats


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='修复数据文件中的时间格式')
    parser.add_argument('--dry-run', action='store_true',
                        help='只检查不修改')
    parser.add_argument('--directories', nargs='+',
                        help='指定要处理的目录')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("开始修复时间格式")
    if args.dry_run:
        logger.info("【DRY RUN 模式】只检查，不修改文件")
    logger.info("=" * 60)

    # 默认处理的目录
    default_dirs = [
        'data/raw/daily_price/hfq',
        'data/raw/financial',
        'data/raw/moneyflow',
        'data/raw/northbound',
        'data/raw/tech_signals',
        'data/raw/st_stocks',
        'data/raw/institution',
        'data/raw/index',
        'data/processed/factors'
    ]

    dirs_to_process = args.directories if args.directories else default_dirs

    total_stats = {'total': 0, 'fixed': 0, 'failed': 0}

    for dir_path in dirs_to_process:
        dir_obj = Path(dir_path)
        stats = fix_datetime_in_directory(dir_obj, dry_run=args.dry_run)

        total_stats['total'] += stats['total']
        total_stats['fixed'] += stats['fixed']
        total_stats['failed'] += stats['failed']

        logger.info(f"{dir_path}: 总数={stats['total']}, 修复={stats['fixed']}, 失败={stats['failed']}")

    logger.info("=" * 60)
    logger.info("修复完成")
    logger.info(f"总计: {total_stats['total']} 个文件")
    logger.info(f"修复: {total_stats['fixed']} 个文件")
    logger.info(f"失败: {total_stats['failed']} 个文件")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("这是 DRY RUN 模式，文件未被修改")
        logger.info("如需实际修复，请运行: python scripts/fix_datetime_formats.py")


if __name__ == '__main__':
    main()
