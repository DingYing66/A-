"""
Unit tests for src/data_processor.py

Purpose: Test data processing functions to verify existing behavior.
Strategy: Test existing behavior without changing logic.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processor import DataProcessor, HistoricalUniverse


class TestDataProcessorInit:
    """Tests for DataProcessor initialization."""

    def test_init_with_default_config(self):
        """Verify DataProcessor initializes with default config."""
        processor = DataProcessor()
        assert processor.config is not None
        assert isinstance(processor.config, dict)

    def test_init_with_custom_config(self, mock_config):
        """Verify DataProcessor initializes with custom config."""
        processor = DataProcessor(config=mock_config)
        assert processor.config == mock_config

    def test_init_loads_trade_calendar(self):
        """Verify DataProcessor loads trade calendar on init."""
        processor = DataProcessor()
        assert processor._trade_calendar is not None or processor.trade_calendar is not None


class TestTradeCalendar:
    """Tests for trade calendar functionality."""

    def test_trade_calendar_is_datetimeindex(self):
        """Verify trade_calendar returns DatetimeIndex."""
        processor = DataProcessor()
        calendar = processor.trade_calendar

        assert isinstance(calendar, pd.DatetimeIndex)
        assert len(calendar) > 0

    def test_trade_calendar_has_recent_dates(self):
        """Verify trade calendar includes recent dates."""
        processor = DataProcessor()
        calendar = processor.trade_calendar

        # Should have dates up to recent time
        recent_date = pd.Timestamp.now() - timedelta(days=30)
        has_recent = (calendar >= recent_date).any()
        assert has_recent, "Trade calendar should include recent dates"

    def test_trade_calendar_no_weekends(self):
        """Verify trade calendar excludes weekends."""
        processor = DataProcessor()
        calendar = processor.trade_calendar

        # Check no Saturdays or Sundays
        weekdays = calendar.dayofweek
        assert (weekdays < 5).all(), "Trade calendar should only contain weekdays"


class TestLoadDailyPrice:
    """Tests for load_daily_price function."""

    def test_load_daily_price_returns_dataframe(self, sample_price_data):
        """Verify load_daily_price returns DataFrame."""
        processor = DataProcessor()

        # Mock the data loading
        with patch.object(processor, 'load_all_daily_prices', return_value=sample_price_data):
            result = processor.load_daily_price('000001')
            assert isinstance(result, pd.DataFrame)

    def test_load_daily_price_with_adjust(self, sample_price_data):
        """Verify load_daily_price with adjust parameter.

        Note: This test requires actual parquet files or more complex mocking.
        Skipping to avoid filesystem dependency.
        """
        pytest.skip("Requires actual parquet files - use integration tests")


class TestGetPriceOnDate:
    """Tests for get_price_on_date function."""

    def test_get_price_on_date_returns_dict(self, sample_price_data):
        """Verify get_price_on_date returns dictionary."""
        processor = DataProcessor()
        test_date = pd.Timestamp(datetime(2024, 1, 2))

        with patch.object(processor, 'load_daily_price', return_value=sample_price_data):
            result = processor.get_price_on_date('000001', test_date)
            # Should return dict or None
            assert result is None or isinstance(result, dict)

    def test_get_price_on_date_missing_data(self, sample_price_data):
        """Verify get_price_on_date raises RuntimeError for missing data."""
        processor = DataProcessor()
        far_future_date = pd.Timestamp.now() + timedelta(days=365)

        with patch.object(processor, 'load_daily_price', return_value=sample_price_data):
            with pytest.raises(RuntimeError):
                processor.get_price_on_date('000001', far_future_date)


class TestGetPrevClose:
    """Tests for get_prev_close function."""

    def test_get_prev_close_returns_float(self, sample_price_data):
        """Verify get_prev_close returns float."""
        processor = DataProcessor()
        test_date = pd.Timestamp(datetime(2024, 2, 1))

        with patch.object(processor, 'load_daily_price', return_value=sample_price_data):
            result = processor.get_prev_close('000001', test_date)
            # Should return float or None
            assert result is None or isinstance(result, (int, float))


class TestIsStStock:
    """Tests for is_st_stock function."""

    def test_is_st_stock_returns_bool_or_none(self, sample_price_data):
        """Verify is_st_stock returns bool or None."""
        processor = DataProcessor()
        test_date = pd.Timestamp(datetime(2024, 1, 2))

        # Mock the ST data
        with patch.object(processor, '_load_st_codes', return_value=set()):
            result = processor.is_st_stock('000001', test_date)
            assert result is True or result is False or result is None

    def test_is_st_stock_with_none_date(self):
        """Verify is_st_stock raises ValueError with None date."""
        processor = DataProcessor()

        with pytest.raises(ValueError):
            processor.is_st_stock('000001', None)


class TestLoadStockList:
    """Tests for load_stock_list function."""

    def test_load_stock_list_returns_dataframe(self):
        """Verify load_stock_list returns DataFrame."""
        processor = DataProcessor()
        result = processor.load_stock_list()

        assert isinstance(result, pd.DataFrame)

    def test_load_stock_list_has_expected_columns(self):
        """Verify load_stock_list has expected columns."""
        processor = DataProcessor()
        result = processor.load_stock_list()

        # Should have code column at minimum
        assert 'code' in result.columns


class TestGetAvailableCodes:
    """Tests for get_available_codes function."""

    def test_get_available_codes_returns_list(self):
        """Verify get_available_codes returns list."""
        processor = DataProcessor()
        test_date = pd.Timestamp(datetime(2024, 3, 1))

        result = processor.get_available_codes(test_date)
        assert isinstance(result, list)

    def test_get_available_codes_not_empty(self):
        """Verify get_available_codes returns non-empty list."""
        processor = DataProcessor()
        test_date = pd.Timestamp(datetime(2024, 3, 1))

        result = processor.get_available_codes(test_date)
        assert len(result) > 0

    def test_get_available_codes_format(self):
        """Verify get_available_codes returns valid stock codes."""
        processor = DataProcessor()
        test_date = pd.Timestamp(datetime(2024, 3, 1))

        result = processor.get_available_codes(test_date)
        for code in result:
            # Stock codes should be 6 digits
            assert len(code) == 6
            assert code.isdigit()


class TestGetPriceMatrix:
    """Tests for get_price_matrix function."""

    def test_get_price_matrix_returns_dataframe(self, sample_price_data):
        """Verify get_price_matrix returns DataFrame."""
        processor = DataProcessor()
        codes = ['000001', '000002']
        start_date = '20240101'
        end_date = '20240131'

        with patch.object(processor, 'load_all_daily_prices', return_value=sample_price_data):
            result = processor.get_price_matrix(codes, start_date, end_date)
            assert isinstance(result, pd.DataFrame)

    def test_get_price_matrix_index(self, sample_price_data):
        """Verify get_price_matrix index is DatetimeIndex."""
        processor = DataProcessor()
        codes = ['000001', '000002']
        start_date = '20240101'
        end_date = '20240131'

        with patch.object(processor, 'load_all_daily_prices', return_value=sample_price_data):
            result = processor.get_price_matrix(codes, start_date, end_date)
            assert isinstance(result.index, pd.DatetimeIndex)


class TestGetReturnMatrix:
    """Tests for get_return_matrix function."""

    def test_get_return_matrix_returns_dataframe(self, sample_price_data):
        """Verify get_return_matrix returns DataFrame."""
        processor = DataProcessor()
        codes = ['000001', '000002']
        start_date = '20240101'
        end_date = '20240131'

        with patch.object(processor, 'load_all_daily_prices', return_value=sample_price_data):
            result = processor.get_return_matrix(codes, start_date, end_date)
            assert isinstance(result, pd.DataFrame)

    def test_get_return_matrix_values_numeric(self, sample_price_data):
        """Verify get_return_matrix values are numeric."""
        processor = DataProcessor()
        codes = ['000001', '000002']
        start_date = '20240101'
        end_date = '20240131'

        with patch.object(processor, 'load_all_daily_prices', return_value=sample_price_data):
            result = processor.get_return_matrix(codes, start_date, end_date)
            assert result.select_dtypes(include=[np.number]).shape[1] > 0


class TestAlignFinancialByAnnounceDate:
    """Tests for align_financial_by_announce_date function."""

    def test_align_financial_data(self):
        """Verify align_financial_by_announce_date works."""
        processor = DataProcessor()

        # Create test financial data with announce dates that are before trade dates
        financial_df = pd.DataFrame({
            'code': ['000001', '000002'],
            'report_date': pd.to_datetime(['20231231', '20231231']),
            'announce_date': pd.to_datetime(['20240115', '20240115']),
            'revenue': [1e9, 2e9],
            'net_profit': [1e8, 2e8],
        })

        # Create trade dates after announce dates
        trade_dates = pd.date_range('2024-02-01', periods=10, freq='D')

        # Should not raise exception
        result = processor.align_financial_by_announce_date(financial_df, trade_dates)
        assert isinstance(result, pd.DataFrame)


class TestCleanData:
    """Tests for clean_data function."""

    def test_clean_data_returns_dataframe(self, sample_factor_data):
        """Verify clean_data returns DataFrame."""
        processor = DataProcessor()
        result = processor.clean_data(sample_factor_data)
        assert isinstance(result, pd.DataFrame)

    def test_clean_data_handles_inf(self, sample_factor_data):
        """Verify clean_data does not remove inf values (current behavior)."""
        processor = DataProcessor()

        # Add inf values
        dirty_data = sample_factor_data.copy()
        dirty_data.loc[0, 'momentum_20d'] = np.inf

        result = processor.clean_data(dirty_data)
        # Current behavior: clean_data doesn't handle inf values
        # It just removes empty rows/columns and returns the data
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(dirty_data)

    def test_clean_data_handles_nan(self, sample_factor_data):
        """Verify clean_data handles NaN values."""
        processor = DataProcessor()

        # Add NaN values
        dirty_data = sample_factor_data.copy()
        dirty_data.loc[0, 'momentum_20d'] = np.nan

        result = processor.clean_data(dirty_data)
        # Should handle NaN without error
        assert isinstance(result, pd.DataFrame)


class TestHistoricalUniverse:
    """Tests for HistoricalUniverse functionality."""

    def test_get_tradeable_codes(self):
        """Verify get_tradeable_codes returns list."""
        universe = HistoricalUniverse()
        test_date = pd.Timestamp(datetime(2024, 3, 1))

        result = universe.get_tradeable_codes(test_date)
        assert isinstance(result, list)

    def test_tradeable_codes_not_empty(self):
        """Verify tradeable codes is not empty."""
        universe = HistoricalUniverse()
        test_date = pd.Timestamp(datetime(2024, 3, 1))

        result = universe.get_tradeable_codes(test_date)
        assert len(result) > 0

    def test_tradeable_codes_valid_format(self):
        """Verify tradeable codes have valid format."""
        universe = HistoricalUniverse()
        test_date = pd.Timestamp(datetime(2024, 3, 1))

        result = universe.get_tradeable_codes(test_date)
        for code in result:
            assert len(code) == 6
            assert code.isdigit()


class TestIntegration:
    """Integration tests for data_processor module."""

    def test_processor_initialization_complete(self):
        """Test that DataProcessor initializes completely."""
        processor = DataProcessor()

        # Verify all key attributes are initialized
        assert processor.config is not None
        assert processor._trade_calendar is not None or processor.trade_calendar is not None

    @pytest.mark.skip(reason="Requires real data files - use integration tests instead")
    def test_data_pipeline_integration(self, sample_price_data):
        """Test full data pipeline."""
        processor = DataProcessor()
        test_date = pd.Timestamp(datetime(2024, 3, 1))

        # Test get_available_codes
        codes = processor.get_available_codes(test_date)
        assert len(codes) > 0

        # Test get_tradeable_codes
        tradeable = processor.get_tradeable_codes(test_date)
        assert len(tradeable) > 0
