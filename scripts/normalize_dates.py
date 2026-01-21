"""
Normalize date/time columns in data files to YYYY-MM-DD strings.

Targets:
  - Parquet files under data/ (date-like columns + date-like index)
  - Known JSON files (st_history.json, _update_status.json)
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from src.utils import load_parquet, save_parquet, standardize_datetime_column, setup_logger


DATE_KEYWORDS = ("date", "time", "datetime")
logger = setup_logger(__name__)


def _series_equivalent(left: pd.Series, right: pd.Series) -> bool:
    if len(left) != len(right):
        return False
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    left_na = pd.isna(left)
    right_na = pd.isna(right)
    if not (left_na == right_na).all():
        return False
    left_comp = left[~left_na].astype(str)
    right_comp = right[~right_na].astype(str)
    return left_comp.equals(right_comp)


def _normalize_date_value(value):
    if value is None:
        return value
    series = pd.Series([value])
    normalized = standardize_datetime_column(series)
    if _series_equivalent(series, normalized):
        return value
    return normalized.iloc[0]


def _normalize_df_dates(df: pd.DataFrame) -> bool:
    changed = False
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in DATE_KEYWORDS):
            before = df[col]
            after = standardize_datetime_column(before)
            if not _series_equivalent(before, after):
                df[col] = after
                changed = True
    return changed


def _normalize_index_dates(df: pd.DataFrame) -> bool:
    idx = df.index
    if isinstance(idx, pd.MultiIndex):
        return False
    idx_name = (idx.name or "").lower()
    if isinstance(idx, pd.DatetimeIndex):
        df.index = pd.Index(idx.strftime("%Y-%m-%d"), name=idx.name)
        return True
    if idx_name and any(keyword in idx_name for keyword in DATE_KEYWORDS):
        idx_series = pd.Series(idx.values)
        normalized = standardize_datetime_column(idx_series)
        if not _series_equivalent(idx_series, normalized):
            df.index = pd.Index(normalized, name=idx.name)
            return True
    return False


def _should_skip(path: Path, excludes: List[str], root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    for ex in excludes:
        ex_norm = ex.strip().rstrip("/")
        if not ex_norm:
            continue
        if rel.startswith(ex_norm):
            return True
    return False


def _iter_parquet_files(root: Path, excludes: List[str]) -> Iterable[Path]:
    for path in root.rglob("*.parquet"):
        if _should_skip(path, excludes, root):
            continue
        yield path


def normalize_parquet_file(path: Path, dry_run: bool) -> bool:
    try:
        df = load_parquet(path)
    except Exception as exc:
        logger.warning("skip parquet (read failed): %s (%s)", path, exc)
        return False

    changed = _normalize_df_dates(df)
    changed = _normalize_index_dates(df) or changed

    if not changed:
        return False

    if dry_run:
        logger.info("would update: %s", path)
        return True

    try:
        save_parquet(df, path)
        logger.info("updated: %s", path)
        return True
    except Exception as exc:
        logger.warning("save failed: %s (%s)", path, exc)
        return False


def normalize_st_history(path: Path, dry_run: bool) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("skip json (read failed): %s (%s)", path, exc)
        return False

    if not isinstance(data, dict):
        return False

    changed = False
    for _, periods in data.items():
        if not isinstance(periods, list):
            continue
        for period in periods:
            if not isinstance(period, dict):
                continue
            for key in ("start_date", "end_date", "date"):
                if key not in period:
                    continue
                new_val = _normalize_date_value(period.get(key))
                if new_val != period.get(key):
                    period[key] = new_val
                    changed = True

    if not changed:
        return False

    if dry_run:
        logger.info("would update: %s", path)
        return True

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("updated: %s", path)
        return True
    except Exception as exc:
        logger.warning("save failed: %s (%s)", path, exc)
        return False


def normalize_update_status(path: Path, dry_run: bool) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("skip json (read failed): %s (%s)", path, exc)
        return False

    if not isinstance(data, dict):
        return False

    changed = False
    for key in ("last_update_date", "last_quarter"):
        if key in data:
            new_val = _normalize_date_value(data.get(key))
            if new_val != data.get(key):
                data[key] = new_val
                changed = True

    if not changed:
        return False

    if dry_run:
        logger.info("would update: %s", path)
        return True

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("updated: %s", path)
        return True
    except Exception as exc:
        logger.warning("save failed: %s (%s)", path, exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize stored dates to YYYY-MM-DD.")
    parser.add_argument("--root", default="data", help="Root data directory (default: data)")
    parser.add_argument("--dry-run", action="store_true", help="Only report changes")
    parser.add_argument("--no-parquet", action="store_true", help="Skip parquet files")
    parser.add_argument("--no-json", action="store_true", help="Skip JSON files")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude subpaths (relative to root), e.g. processed/factors",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    updated_parquet = 0
    updated_json = 0

    if not args.no_parquet:
        for path in _iter_parquet_files(root, args.exclude):
            if normalize_parquet_file(path, args.dry_run):
                updated_parquet += 1

    if not args.no_json:
        st_history_path = root / "raw" / "st_stocks" / "st_history.json"
        if st_history_path.exists():
            if normalize_st_history(st_history_path, args.dry_run):
                updated_json += 1
        update_status_path = root / "raw" / "financial" / "_update_status.json"
        if update_status_path.exists():
            if normalize_update_status(update_status_path, args.dry_run):
                updated_json += 1

    logger.info(
        "done. updated parquet: %s, updated json: %s", updated_parquet, updated_json
    )


if __name__ == "__main__":
    main()
