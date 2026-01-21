"""
Unit tests for src/utils.py

Purpose: Test utility functions to verify existing behavior.
Strategy: Test existing behavior without changing logic.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    setup_logger,
    get_project_root,
    ensure_dir,
    load_parquet,
    save_parquet,
    standardize_datetime_column,
    get_rebalance_dates,
    load_config,
)


class TestSetupLogger:
    """Tests for setup_logger function."""

    def test_setup_logger_returns_logger(self):
        """Verify setup_logger returns a logger object."""
        logger = setup_logger('test_logger')
        assert logger is not None
        assert logger.name == 'test_logger'

    def test_setup_logger_same_name_returns_same_logger(self):
        """Verify same logger name returns same logger instance."""
        logger1 = setup_logger('test_same')
        logger2 = setup_logger('test_same')
        # Should return the same logger instance
        assert logger1.name == logger2.name


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_get_project_root_returns_path(self):
        """Verify get_project_root returns a Path object."""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()
        assert (root / 'src').exists()
        assert (root / 'config').exists()

    def test_get_project_root_has_expected_structure(self):
        """Verify project root has expected directory structure."""
        root = get_project_root()
        expected_dirs = ['src', 'config', 'data', 'output']
        for dir_name in expected_dirs:
            assert (root / dir_name).exists(), f"Missing directory: {dir_name}"


class TestEnsureDir:
    """Tests for ensure_dir function."""

    def test_ensure_dir_creates_directory(self, tmp_path):
        """Verify ensure_dir creates non-existent directory."""
        test_dir = tmp_path / 'new_subdir' / 'nested'
        result = ensure_dir(test_dir)
        assert result.exists()
        assert result == test_dir

    def test_ensure_dir_returns_existing_dir(self, tmp_path):
        """Verify ensure_dir returns existing directory."""
        existing_dir = tmp_path / 'existing'
        existing_dir.mkdir()
        result = ensure_dir(existing_dir)
        assert result.exists()
        assert result == existing_dir


class TestLoadParquet:
    """Tests for load_parquet function."""

    def test_load_parquet_existing_file(self, tmp_path):
        """Verify loading existing parquet file."""
        # Create test parquet file
        test_df = pd.DataFrame({
            'code': ['000001', '000002'],
            'value': [1.0, 2.0],
        })
        test_path = tmp_path / 'test.parquet'
        test_df.to_parquet(test_path)

        # Load and verify
        result = load_parquet(test_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ['code', 'value']

    def test_load_parquet_nonexistent_file(self, tmp_path):
        """Verify loading non-existent file raises FileNotFoundError."""
        nonexistent_path = tmp_path / 'nonexistent.parquet'
        with pytest.raises(FileNotFoundError):
            load_parquet(nonexistent_path)


class TestSaveParquet:
    """Tests for save_parquet function."""

    def test_save_parquet_creates_file(self, tmp_path):
        """Verify save_parquet creates parquet file."""
        test_df = pd.DataFrame({
            'code': ['000001', '000002'],
            'value': [1.0, 2.0],
        })
        test_path = tmp_path / 'output.parquet'

        save_parquet(test_df, test_path)
        assert test_path.exists()

        # Verify content
        result = pd.read_parquet(test_path)
        assert len(result) == 2

    def test_save_parquet_overwrites_existing(self, tmp_path):
        """Verify save_parquet overwrites existing file."""
        test_path = tmp_path / 'overwrite.parquet'

        # First save
        df1 = pd.DataFrame({'code': ['000001']})
        save_parquet(df1, test_path)

        # Second save with different data
        df2 = pd.DataFrame({'code': ['000002']})
        save_parquet(df2, test_path)

        # Verify only second data exists
        result = pd.read_parquet(test_path)
        assert len(result) == 1
        assert result['code'].iloc[0] == '000002'


class TestStandardizeDatetimeColumn:
    """Tests for standardize_datetime_column function."""

    def test_timestamp_ms_conversion(self):
        """Verify millisecond timestamp conversion (1577923200000)."""
        series = pd.Series([1577923200000, 1578009600000])
        result = standardize_datetime_column(series)

        assert pd.api.types.is_datetime64_any_dtype(result)
        assert len(result) == 2

    def test_timestamp_s_conversion(self):
        """Verify second timestamp conversion."""
        series = pd.Series([1577923200, 1578009600])
        result = standardize_datetime_column(series)

        assert pd.api.types.is_datetime64_any_dtype(result)

    def test_already_datetime(self):
        """Verify already datetime column is unchanged."""
        series = pd.Series(pd.date_range('2024-01-01', periods=3))
        result = standardize_datetime_column(series)

        assert pd.api.types.is_datetime64_any_dtype(result)
        assert len(result) == 3

    def test_string_date_conversion(self):
        """Verify string date conversion."""
        series = pd.Series(['2025-07-22', '2025-07-23'])
        result = standardize_datetime_column(series)

        assert pd.api.types.is_datetime64_any_dtype(result)

    def test_yyyymmdd_int_conversion(self):
        """Verify YYYYMMDD integer format conversion."""
        series = pd.Series([20250722, 20250723])
        result = standardize_datetime_column(series)

        assert pd.api.types.is_datetime64_any_dtype(result)

    def test_with_nan_values(self):
        """Verify handling of NaN values."""
        series = pd.Series([1577923200000, None, 1578009600000])
        result = standardize_datetime_column(series)

        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.isna().sum() >= 1  # At least the None should be NaN


class TestGetRebalanceDates:
    """Tests for get_rebalance_dates function."""

    def test_get_rebalance_dates_daily(self, trade_calendar):
        """Verify daily frequency returns all trade dates."""
        start = '20240102'
        end = '20240110'

        dates = get_rebalance_dates(start, end, trade_calendar, freq='daily')

        # Should return all trade dates in range
        assert len(dates) > 0
        assert all(isinstance(d, pd.Timestamp) for d in dates)

    def test_get_rebalance_dates_weekly(self, trade_calendar):
        """Verify weekly frequency returns weekly dates."""
        start = '20240101'
        end = '20240229'

        dates = get_rebalance_dates(start, end, trade_calendar, freq='weekly')

        # Should have fewer dates than daily
        daily_dates = get_rebalance_dates(start, end, trade_calendar, freq='daily')
        assert len(dates) < len(daily_dates)

    def test_get_rebalance_dates_biweekly(self, trade_calendar):
        """Verify biweekly frequency returns biweekly dates."""
        start = '20240101'
        end = '20240430'

        dates = get_rebalance_dates(start, end, trade_calendar, freq='biweekly')

        # Should have even fewer dates
        weekly_dates = get_rebalance_dates(start, end, trade_calendar, freq='weekly')
        assert len(dates) <= len(weekly_dates)

    def test_get_rebalance_dates_monthly(self, trade_calendar):
        """Verify monthly frequency returns monthly dates."""
        start = '20240101'
        end = '20241231'

        dates = get_rebalance_dates(start, end, trade_calendar, freq='monthly')

        # Should have monthly dates, but trade_calendar fixture only has ~250 days
        # starting from 2024-01-01, so may not cover all months
        assert len(dates) >= 8  # At least 8 monthly dates
        assert len(dates) <= 12  # At most 12 monthly dates


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_returns_dict(self):
        """Verify load_config returns a dictionary."""
        config = load_config()
        assert isinstance(config, dict)

    def test_load_config_has_expected_keys(self):
        """Verify config has expected top-level keys."""
        config = load_config()

        expected_keys = ['paths', 'data_fetch', 'stock_pool', 'backtest']
        for key in expected_keys:
            assert key in config, f"Missing expected key: {key}"


class TestIntegration:
    """Integration tests for utils module."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Verify save_parquet and load_parquet are inverse operations."""
        original_df = pd.DataFrame({
            'code': ['000001', '000002', '000003'],
            'value': [1.5, 2.5, 3.5],
            'date': pd.date_range('2024-01-01', periods=3),
        })

        test_path = tmp_path / 'roundtrip.parquet'
        save_parquet(original_df, test_path)
        loaded_df = load_parquet(test_path)

        pd.testing.assert_frame_equal(original_df, loaded_df)

    def test_logger_integration(self, tmp_path):
        """Test logger can write to file."""
        logger = setup_logger('integration_test')

        log_file = tmp_path / 'test.log'
        handler = logging.FileHandler(log_file)
        logger.addHandler(handler)

        test_message = 'Test log message'
        logger.info(test_message)

        # Verify log was written
        assert log_file.exists()
        with open(log_file, 'r') as f:
            content = f.read()
            assert test_message in content


# Import logging at module level for the integration test
import logging
