#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据迁移脚本: 将旧版平铺日K数据迁移到 adjust 分区结构

旧结构: data/raw/daily_price/{code}.parquet
新结构: data/raw/daily_price/{adjust}/{code}.parquet

使用方法:
    python scripts/migrate_adjust_partition.py
    python scripts/migrate_adjust_partition.py --adjust qfq
    python scripts/migrate_adjust_partition.py --dry-run  # 仅预览，不执行
    python scripts/migrate_adjust_partition.py --copy     # 复制而非移动
"""

import argparse
import shutil
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logger, get_project_root, load_config, ensure_dir

logger = setup_logger('migrate_adjust')


def migrate_daily_price(adjust: str = None, dry_run: bool = False, copy_mode: bool = False):
    """
    迁移日K数据到分区结构

    Args:
        adjust: 目标 adjust 类型 (默认从配置读取)
        dry_run: 仅预览，不执行
        copy_mode: 复制文件而非移动
    """
    config = load_config()
    root = get_project_root()

    # 确定 adjust 类型
    adjust = adjust or config.get('data_fetch', {}).get('adjust', 'qfq')
    adjust_dir = adjust if adjust else 'none'

    # 路径
    base_dir = root / 'data' / 'raw' / 'daily_price'
    target_dir = base_dir / adjust_dir

    if not base_dir.exists():
        logger.error(f"源目录不存在: {base_dir}")
        return

    # 找到需要迁移的文件 (仅限根目录下的 parquet 文件)
    files_to_migrate = [f for f in base_dir.glob('*.parquet') if f.is_file()]

    if len(files_to_migrate) == 0:
        logger.info("未找到需要迁移的文件")
        # 检查是否已经是分区结构
        if target_dir.exists():
            existing = len(list(target_dir.glob('*.parquet')))
            logger.info(f"目标目录 {adjust_dir}/ 已存在 {existing} 个文件")
        return

    logger.info(f"发现 {len(files_to_migrate)} 个文件需要迁移")
    logger.info(f"目标目录: {target_dir}")
    logger.info(f"操作模式: {'复制' if copy_mode else '移动'}")

    if dry_run:
        logger.info("=== 预览模式 (不执行) ===")
        for f in files_to_migrate[:10]:
            logger.info(f"  {f.name} -> {adjust_dir}/{f.name}")
        if len(files_to_migrate) > 10:
            logger.info(f"  ... 共 {len(files_to_migrate)} 个文件")
        return

    # 创建目标目录
    ensure_dir(target_dir)

    # 迁移文件
    success_count = 0
    fail_count = 0

    for f in files_to_migrate:
        target_path = target_dir / f.name
        try:
            if copy_mode:
                shutil.copy2(f, target_path)
            else:
                shutil.move(str(f), str(target_path))
            success_count += 1
        except Exception as e:
            logger.warning(f"迁移失败 {f.name}: {e}")
            fail_count += 1

    logger.info(f"迁移完成: 成功 {success_count}, 失败 {fail_count}")

    # 迁移失败文件列表
    failed_list = base_dir / '_failed_codes.txt'
    if failed_list.exists():
        target_failed = target_dir / '_failed_codes.txt'
        if copy_mode:
            shutil.copy2(failed_list, target_failed)
        else:
            shutil.move(str(failed_list), str(target_failed))
        logger.info(f"迁移失败列表: {target_failed}")

    # 验证
    if not copy_mode:
        remaining = len(list(base_dir.glob('*.parquet')))
        if remaining == 0:
            logger.info("所有文件已迁移，根目录已清空")
        else:
            logger.warning(f"根目录仍有 {remaining} 个文件未迁移")


def check_current_structure():
    """检查当前数据目录结构"""
    root = get_project_root()
    base_dir = root / 'data' / 'raw' / 'daily_price'

    if not base_dir.exists():
        print(f"日K目录不存在: {base_dir}")
        return

    print(f"\n=== 日K数据目录结构 ===")
    print(f"路径: {base_dir}")

    # 根目录文件
    root_files = len([f for f in base_dir.glob('*.parquet') if f.is_file()])
    print(f"\n根目录 parquet 文件: {root_files}")

    # 分区目录
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            files = len(list(subdir.glob('*.parquet')))
            print(f"  {subdir.name}/: {files} 个文件")

    print("")


def main():
    parser = argparse.ArgumentParser(
        description='迁移日K数据到 adjust 分区结构',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/migrate_adjust_partition.py              # 迁移到配置的 adjust 类型
    python scripts/migrate_adjust_partition.py --adjust qfq  # 迁移到 qfq 目录
    python scripts/migrate_adjust_partition.py --dry-run     # 仅预览
    python scripts/migrate_adjust_partition.py --copy        # 复制而非移动
    python scripts/migrate_adjust_partition.py --check       # 检查当前结构
        """
    )

    parser.add_argument('--adjust', type=str, default=None,
                        help='目标 adjust 类型 (qfq/hfq/空字符串)')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅预览，不执行迁移')
    parser.add_argument('--copy', action='store_true',
                        help='复制文件而非移动')
    parser.add_argument('--check', action='store_true',
                        help='检查当前目录结构')

    args = parser.parse_args()

    if args.check:
        check_current_structure()
    else:
        migrate_daily_price(
            adjust=args.adjust,
            dry_run=args.dry_run,
            copy_mode=args.copy
        )


if __name__ == '__main__':
    main()
