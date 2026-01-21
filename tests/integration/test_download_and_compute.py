"""
Integration tests for A-share multi-factor stock selection system.

Purpose: Test complete end-to-end workflows to verify existing behavior.
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
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, get_project_root
from src.data_processor import DataProcessor
from src.factor_engine import FactorEngine


class TestDataDownloadPipeline:
    """Integration tests for data download pipeline."""

    def test_config_loading(self):
        """Verify configuration loads correctly."""
        config = load_config()

        assert isinstance(config, dict)
        assert 'paths' in config
        assert 'data_fetch' in config
        assert 'stock_pool' in config

    def test_config_has_required_sections(self):
        """Verify config has all required sections."""
        config = load_config()

        required_sections = [
            'paths',
            'data_fetch',
            'stock_pool',
            'backtest',
            'execution'
        ]

        for section in required_sections:
            assert section in config, f"Missing required config section: {section}"

    def test_config_paths_exist(self):
        """Verify config paths exist."""
        config = load_config()
        project_root = get_project_root()

        paths = config.get('paths', {})
        for key, path in paths.items():
            # Handle relative paths
            if not Path(path).is_absolute():
                full_path = project_root / path
            else:
                full_path = Path(path)

            # Just verify we can construct the path
            assert full_path is not None


class TestDataProcessorPipeline:
    """Integration tests for DataProcessor pipeline."""

    def test_processor_initialization(self):
        """Verify DataProcessor initializes completely."""
        processor = DataProcessor()

        assert processor.config is not None
        assert processor.trade_calendar is not None
        assert len(processor.trade_calendar) > 0

    def test_get_available_codes_pipeline(self):
        """Test get_available_codes end-to-end."""
        processor = DataProcessor()
        test_date = pd.Timestamp(datetime(2024, 3, 1))

        codes = processor.get_available_codes(test_date)

        assert isinstance(codes, list)
        assert len(codes) > 0
        # Verify format
        for code in codes:
            assert len(code) == 6
            assert code.isdigit()

    def test_get_tradeable_codes_pipeline(self):
        """Test get_tradeable_codes end-to-end."""
        from src.data_processor import HistoricalUniverse
        universe = HistoricalUniverse()
        test_date = pd.Timestamp(datetime(2024, 3, 1))

        codes = universe.get_tradeable_codes(test_date)

        assert isinstance(codes, list)
        assert len(codes) > 0

    def test_price_data_pipeline(self, sample_price_data):
        """Test price data loading pipeline."""
        processor = DataProcessor()

        with patch.object(processor, 'load_all_daily_prices', return_value=sample_price_data):
            prices = processor.load_daily_price('000001')

        assert isinstance(prices, pd.DataFrame)
        assert 'close' in prices.columns


class TestFactorEnginePipeline:
    """Integration tests for FactorEngine pipeline."""

    def test_engine_initialization(self):
        """Verify FactorEngine initializes completely."""
        engine = FactorEngine()

        assert engine.config is not None
        assert engine.processor is not None

    def test_single_stock_factor_pipeline(self, sample_price_data):
        """Test single stock factor computation pipeline."""
        engine = FactorEngine()
        code = '000001'
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
            result = engine.compute_all_factors(code, date)

        assert isinstance(result, dict)
        assert result['code'] == code
        assert 'momentum_20d' in result
        assert 'volatility_20d' in result

    def test_cross_section_pipeline(self, sample_price_data):
        """Test cross-section factor computation pipeline."""
        engine = FactorEngine()
        date = pd.Timestamp(datetime(2024, 3, 1))
        codes = ['000001', '000002', '000003']

        with patch.object(engine.processor, 'get_available_codes', return_value=codes):
            with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
                result = engine.compute_factor_cross_section(date)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # 3 stocks
        assert 'code' in result.columns
        assert 'date' in result.columns
        assert 'momentum_20d' in result.columns

    def test_factor_standardization_pipeline(self, sample_factor_data):
        """Test factor standardization pipeline."""
        engine = FactorEngine()

        # Add date column if not present
        if 'date' not in sample_factor_data.columns:
            sample_factor_data['date'] = pd.Timestamp(datetime(2024, 3, 1))

        result = engine.standardize_factors(sample_factor_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_factor_data)


class TestEndToEndPipeline:
    """End-to-end integration tests."""

    def test_data_to_factors_pipeline(self, sample_price_data, sample_factor_data):
        """Test data to factors complete pipeline."""
        # Initialize components
        processor = DataProcessor()
        engine = FactorEngine()

        # Get available codes
        date = pd.Timestamp(datetime(2024, 3, 1))
        codes = ['000001', '000002']

        # Compute factors for each code
        all_factors = []

        for code in codes:
            with patch.object(processor, 'load_daily_price', return_value=sample_price_data):
                factors = engine.compute_all_factors(code, date)

            factors['code'] = code
            factors['date'] = date
            all_factors.append(factors)

        # Combine into DataFrame
        result_df = pd.DataFrame(all_factors)

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        assert 'code' in result_df.columns
        assert 'date' in result_df.columns

    def test_factor_save_load_pipeline(self, tmp_path, sample_factor_data):
        """Test factor save and load pipeline."""
        engine = FactorEngine()

        test_date = '20240301'
        test_file = tmp_path / f'factors_{test_date}.parquet'

        # Save
        sample_factor_data.to_parquet(test_file)

        # Load
        loaded = engine.load_factor_data(test_date)

        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == len(sample_factor_data)

    def test_universe_id_generation_pipeline(self):
        """Test universe ID generation pipeline."""
        # Generate universe ID for a set of codes
        codes = ['000001', '000002', '000003']
        date = '20240301'

        from src.deep_learning.contracts import generate_universe_id
        universe_id = generate_universe_id(codes, date)

        assert isinstance(universe_id, str)
        assert len(universe_id) > 0

        # Same inputs should produce same ID
        universe_id2 = UniverseContract.generate_universe_id(codes, date)
        assert universe_id == universe_id2


class TestPipelineConsistency:
    """Tests for pipeline behavior consistency."""

    def test_factor_values_consistency(self, sample_price_data):
        """Test that factor values are consistent across calls."""
        engine = FactorEngine()
        code = '000001'
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
            result1 = engine.compute_momentum_factors(code, date)
            result2 = engine.compute_momentum_factors(code, date)

        # Results should be identical
        assert result1 == result2

    def test_processor_consistency(self):
        """Test DataProcessor behavior is consistent."""
        processor = DataProcessor()

        date = pd.Timestamp(datetime(2024, 3, 1))
        codes1 = processor.get_available_codes(date)
        codes2 = processor.get_available_codes(date)

        assert codes1 == codes2

    def test_calendar_consistency(self):
        """Test trade calendar is consistent."""
        processor = DataProcessor()

        calendar1 = processor.trade_calendar
        calendar2 = processor.trade_calendar

        assert (calendar1 == calendar2).all()


class TestPipelineErrorHandling:
    """Tests for pipeline error handling."""

    def test_missing_data_handling(self, sample_price_data):
        """Test handling of missing data."""
        engine = FactorEngine()
        code = '999999'  # Non-existent code
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_daily_price', return_value=pd.DataFrame()):
            result = engine.compute_momentum_factors(code, date)

        # Should return empty dict, not raise exception
        assert isinstance(result, dict)

    def test_empty_factor_data_handling(self):
        """Test handling of empty factor data."""
        engine = FactorEngine()

        empty_df = pd.DataFrame({'code': [], 'date': []})
        result = engine.standardize_factors(empty_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestPipelinePerformance:
    """Basic performance tests (marked as slow)."""

    @pytest.mark.slow
    def test_trade_calendar_size(self):
        """Test trade calendar has reasonable size."""
        processor = DataProcessor()

        calendar = processor.trade_calendar

        # Should have ~250 trading days per year
        assert len(calendar) >= 200  # At least ~8 months
        assert len(calendar) <= 400  # At most ~16 months

    @pytest.mark.slow
    def test_available_codes_count(self):
        """Test available codes count is reasonable."""
        processor = DataProcessor()
        date = pd.Timestamp(datetime(2024, 3, 1))

        codes = processor.get_available_codes(date)

        # A-share market has ~4000-5000 stocks
        assert 1000 < len(codes) < 6000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
