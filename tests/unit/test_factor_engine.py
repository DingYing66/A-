"""
Unit tests for src/factor_engine.py

Purpose: Test factor calculation functions to verify existing behavior.
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

from src.factor_engine import FactorEngine
from src.deep_learning.contracts import generate_universe_id


class TestUniverseContract:
    """Tests for generate_universe_id function."""

    def test_generate_universe_id_consistency(self):
        """Verify generate_universe_id returns consistent IDs for same inputs."""
        codes = ['000001', '000002', '000003']
        date = '20240301'

        id1 = generate_universe_id(codes, date)
        id2 = generate_universe_id(codes, date)
        assert id1 == id2

    def test_generate_universe_id_different_dates(self):
        """Verify generate_universe_id returns different IDs for different dates."""
        codes = ['000001', '000002', '000003']

        id1 = generate_universe_id(codes, '20240301')
        id2 = generate_universe_id(codes, '20240302')
        assert id1 != id2

    def test_generate_universe_id_different_codes(self):
        """Verify generate_universe_id returns different IDs for different codes."""
        date = '20240301'

        id1 = generate_universe_id(['000001', '000002'], date)
        id2 = generate_universe_id(['000001', '000003'], date)
        assert id1 != id2

    def test_generate_universe_id_format(self):
        """Verify generate_universe_id returns expected format."""
        codes = ['000001', '000002']
        date = '20240301'

        universe_id = generate_universe_id(codes, date)
        assert isinstance(universe_id, str)
        assert len(universe_id) > 0


class TestFactorEngineInit:
    """Tests for FactorEngine initialization."""

    def test_init_with_default_config(self):
        """Verify FactorEngine initializes with default config."""
        engine = FactorEngine()
        assert engine.config is not None
        assert isinstance(engine.config, dict)

    def test_init_with_custom_config(self, mock_config):
        """Verify FactorEngine initializes with custom config."""
        engine = FactorEngine(config=mock_config)
        assert engine.config == mock_config

    def test_init_has_processor(self):
        """Verify FactorEngine initializes with DataProcessor."""
        engine = FactorEngine()
        assert engine.processor is not None


class TestComputeMomentumFactors:
    """Tests for compute_momentum_factors function."""

    def test_compute_momentum_factors_returns_dict(self, sample_price_data):
        """Verify compute_momentum_factors returns dictionary."""
        engine = FactorEngine()
        code = '000001'
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
            result = engine.compute_momentum_factors(code, date)
            assert isinstance(result, dict)

    def test_compute_momentum_factors_has_expected_keys(self, sample_price_data):
        """Verify compute_momentum_factors returns expected factor keys."""
        engine = FactorEngine()
        code = '000001'
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
            result = engine.compute_momentum_factors(code, date)

        expected_keys = ['ret_20d', 'ret_60d', 'ret_120d']
        for key in expected_keys:
            assert key in result, f"Missing expected key: {key}"

    def test_compute_momentum_factors_empty_data(self, sample_price_data):
        """Verify compute_momentum_factors handles empty data."""
        engine = FactorEngine()
        code = '000001'
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_daily_price', return_value=pd.DataFrame()):
            result = engine.compute_momentum_factors(code, date)
            # Should return dict with NaN values for no data
            assert isinstance(result, dict)
            # Should have keys for each momentum window
            expected_keys = ['ret_20d', 'ret_60d', 'ret_120d']
            for key in expected_keys:
                assert key in result
                assert pd.isna(result[key])


class TestComputeVolatilityFactors:
    """Tests for compute_volatility_factors function."""

    def test_compute_volatility_factors_returns_dict(self, sample_price_data):
        """Verify compute_volatility_factors returns dictionary."""
        engine = FactorEngine()
        code = '000001'
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
            result = engine.compute_volatility_factors(code, date)
            assert isinstance(result, dict)

    def test_compute_volatility_factors_has_expected_keys(self, sample_price_data):
        """Verify compute_volatility_factors returns expected factor keys."""
        engine = FactorEngine()
        code = '000001'
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
            result = engine.compute_volatility_factors(code, date)

        expected_keys = ['vol_20d', 'max_drawdown_20d']
        for key in expected_keys:
            assert key in result, f"Missing expected key: {key}"


class TestComputeQualityFactors:
    """Tests for compute_quality_factors function."""

    def test_compute_quality_factors_returns_dict(self, sample_price_data):
        """Verify compute_quality_factors returns dictionary."""
        engine = FactorEngine()
        code = '000001'
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
            result = engine.compute_quality_factors(code, date)
            assert isinstance(result, dict)

    def test_compute_quality_factors_values_numeric(self, sample_price_data):
        """Verify compute_quality_factors values are numeric."""
        engine = FactorEngine()
        code = '000001'
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
            result = engine.compute_quality_factors(code, date)

        for key, value in result.items():
            assert isinstance(value, (int, float, np.number)), f"Non-numeric value for {key}"


class TestComputeValuationFactors:
    """Tests for compute_valuation_factors function."""

    def test_compute_valuation_factors_returns_dict(self, sample_price_data):
        """Verify compute_valuation_factors returns dictionary."""
        engine = FactorEngine()
        code = '000001'
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
            result = engine.compute_valuation_factors(code, date)
            assert isinstance(result, dict)


class TestComputeFlowFactors:
    """Tests for compute_flow_factors function."""

    def test_compute_flow_factors_returns_dict(self, sample_price_data):
        """Verify compute_flow_factors returns dictionary."""
        engine = FactorEngine()
        code = '000001'
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
            result = engine.compute_flow_factors(code, date)
            assert isinstance(result, dict)


class TestComputeLiquidityFactors:
    """Tests for compute_liquidity_factors function."""

    def test_compute_liquidity_factors_returns_dict(self, sample_price_data):
        """Verify compute_liquidity_factors returns dictionary."""
        engine = FactorEngine()
        code = '000001'
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
            result = engine.compute_liquidity_factors(code, date)
            assert isinstance(result, dict)


class TestComputeMarketRegimeFactors:
    """Tests for compute_market_regime_factors function."""

    def test_compute_market_regime_factors_returns_dict(self, sample_price_data):
        """Verify compute_market_regime_factors returns dictionary."""
        engine = FactorEngine()
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_all_daily_prices', return_value=sample_price_data):
            result = engine.compute_market_regime_factors(date)
            assert isinstance(result, dict)

    def test_compute_market_regime_factors_has_breadth(self, sample_price_data):
        """Verify compute_market_regime_factors includes market breadth."""
        engine = FactorEngine()
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_all_daily_prices', return_value=sample_price_data):
            result = engine.compute_market_regime_factors(date)

        assert 'market_breadth' in result


class TestComputeAllFactors:
    """Tests for compute_all_factors function."""

    def test_compute_all_factors_returns_dict(self, sample_price_data):
        """Verify compute_all_factors returns dictionary."""
        engine = FactorEngine()
        code = '000001'
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
            result = engine.compute_all_factors(code, date)
            assert isinstance(result, dict)

    def test_compute_all_factors_has_code(self, sample_price_data):
        """Verify compute_all_factors returns result with code."""
        engine = FactorEngine()
        code = '000001'
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
            result = engine.compute_all_factors(code, date)

        assert 'code' in result
        assert result['code'] == code


class TestComputeFactorCrossSection:
    """Tests for compute_factor_cross_section function."""

    def test_compute_factor_cross_section_returns_dataframe(self, sample_price_data, sample_factor_data):
        """Verify compute_factor_cross_section returns DataFrame."""
        engine = FactorEngine()
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'get_available_codes', return_value=['000001', '000002']):
            with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
                result = engine.compute_factor_cross_section(date)

        assert isinstance(result, pd.DataFrame)

    def test_compute_factor_cross_section_has_code_column(self, sample_price_data, sample_factor_data):
        """Verify compute_factor_cross_section has code column."""
        engine = FactorEngine()
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'get_available_codes', return_value=['000001', '000002']):
            with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
                result = engine.compute_factor_cross_section(date)

        assert 'code' in result.columns

    def test_compute_factor_cross_section_date_in_data(self, sample_price_data, sample_factor_data):
        """Verify compute_factor_cross_section includes date column."""
        engine = FactorEngine()
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'get_available_codes', return_value=['000001', '000002']):
            with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
                result = engine.compute_factor_cross_section(date)

        assert 'date' in result.columns


class TestStandardizeFactors:
    """Tests for standardize_factors function."""

    def test_standardize_factors_returns_dataframe(self, sample_factor_data):
        """Verify standardize_factors returns DataFrame."""
        engine = FactorEngine()
        result = engine.standardize_factors(sample_factor_data)
        assert isinstance(result, pd.DataFrame)

    def test_standardize_factors_has_expected_columns(self, sample_factor_data):
        """Verify standardize_factors preserves expected columns."""
        engine = FactorEngine()

        result = engine.standardize_factors(sample_factor_data)

        assert 'code' in result.columns
        assert 'date' in result.columns

    def test_standardize_factors_handles_inf(self, sample_factor_data):
        """Verify standardize_factors handles inf values."""
        engine = FactorEngine()

        # Add inf values
        dirty_data = sample_factor_data.copy()
        dirty_data.loc[0, 'momentum_20d'] = np.inf

        result = engine.standardize_factors(dirty_data)
        assert isinstance(result, pd.DataFrame)


class TestNeutralizeIndustry:
    """Tests for neutralize_industry function."""

    def test_neutralize_industry_returns_dataframe(self, sample_factor_data):
        """Verify neutralize_industry returns DataFrame."""
        engine = FactorEngine()
        result = engine.neutralize_industry(sample_factor_data)
        assert isinstance(result, pd.DataFrame)

    def test_neutralize_industry_same_shape(self, sample_factor_data):
        """Verify neutralize_industry returns same shape."""
        engine = FactorEngine()

        result = engine.neutralize_industry(sample_factor_data)

        assert result.shape == sample_factor_data.shape


class TestLoadFactorData:
    """Tests for load_factor_data function."""

    def test_load_factor_data_returns_dataframe(self, tmp_path, sample_factor_data):
        """Verify load_factor_data returns DataFrame."""
        engine = FactorEngine()

        # Create test factor file
        test_date = '20240301'
        test_file = tmp_path / f'factors_{test_date}.parquet'
        sample_factor_data.to_parquet(test_file)

        with patch('src.factor_engine.get_project_root', return_value=tmp_path):
            result = engine.load_factor_data(test_date)

        assert isinstance(result, pd.DataFrame)

    def test_load_factor_data_nonexistent(self, tmp_path):
        """Verify load_factor_data handles nonexistent file."""
        engine = FactorEngine()

        with patch('src.factor_engine.get_project_root', return_value=tmp_path):
            result = engine.load_factor_data('19990101')

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestSaveFactorData:
    """Tests for save_factor_data function."""

    def test_save_factor_data_creates_file(self, tmp_path, sample_factor_data):
        """Verify save_factor_data creates parquet file."""
        engine = FactorEngine()

        test_date = '20240301'
        test_file = tmp_path / f'factors_{test_date}.parquet'

        with patch('src.factor_engine.get_project_root', return_value=tmp_path):
            engine.save_factor_data(sample_factor_data, test_date)

        assert test_file.exists()


class TestGetFactorNames:
    """Tests for get_factor_names function."""

    def test_get_factor_names_returns_list(self):
        """Verify get_factor_names returns list."""
        engine = FactorEngine()
        result = engine.get_factor_names()
        assert isinstance(result, list)

    def test_get_factor_names_not_empty(self):
        """Verify get_factor_names returns non-empty list."""
        engine = FactorEngine()
        result = engine.get_factor_names()
        assert len(result) > 0

    def test_get_factor_names_by_category(self):
        """Verify get_factor_names by category."""
        engine = FactorEngine()

        momentum_factors = engine.get_factor_names('momentum')
        assert isinstance(momentum_factors, list)
        assert len(momentum_factors) > 0

        # All should contain 'momentum' in name
        for factor in momentum_factors:
            assert 'momentum' in factor


class TestIntegration:
    """Integration tests for factor_engine module."""

    def test_single_stock_factor_computation(self, sample_price_data):
        """Test complete single stock factor computation."""
        engine = FactorEngine()
        code = '000001'
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
            result = engine.compute_all_factors(code, date)

        assert isinstance(result, dict)
        assert result['code'] == code
        # Should have multiple factors
        assert len(result) > 5

    def test_cross_section_factor_computation(self, sample_price_data):
        """Test cross-section factor computation."""
        engine = FactorEngine()
        date = pd.Timestamp(datetime(2024, 3, 1))

        with patch.object(engine.processor, 'get_available_codes', return_value=['000001', '000002']):
            with patch.object(engine.processor, 'load_daily_price', return_value=sample_price_data):
                result = engine.compute_factor_cross_section(date)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 2 stocks
        assert 'code' in result.columns

    def test_factor_standardization_pipeline(self, sample_factor_data):
        """Test factor standardization pipeline."""
        engine = FactorEngine()

        # Standardize
        standardized = engine.standardize_factors(sample_factor_data)

        # Verify standardization worked
        assert standardized is not None
        assert isinstance(standardized, pd.DataFrame)
