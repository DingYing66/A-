"""
Pytest configuration and fixtures for A-share multi-factor stock selection system tests.

Purpose: Provide reusable test fixtures and configuration for all tests.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ===================== Fixtures =====================

@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def trade_calendar():
    """Create a mock trade calendar for testing."""
    # Generate mock trade dates ( weekdays, exclude weekends )
    base_date = datetime(2024, 1, 1)
    dates = []

    for i in range(250):  # ~1 year of trading days
        date = base_date + timedelta(days=i)
        # Exclude weekends (5=Saturday, 6=Sunday)
        if date.weekday() < 5:
            dates.append(pd.Timestamp(date))

    return pd.DatetimeIndex(dates)


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    codes = ['000001', '000002', '000003', '000004', '000005']
    base_date = datetime(2024, 1, 1)

    records = []
    for code in codes:
        base_price = 10.0 + hash(code) % 20  # Random starting price
        for i in range(60):  # 60 days of data
            date = base_date + timedelta(days=i)
            if date.weekday() < 5:
                change = (np.random.random() - 0.5) * 0.5
                close_price = base_price + change * i + np.random.random() * 0.1

                records.append({
                    'code': code,
                    'date': pd.Timestamp(date),
                    'open': close_price * (1 + (np.random.random() - 0.5) * 0.02),
                    'high': close_price * (1 + np.random.random() * 0.03),
                    'low': close_price * (1 - np.random.random() * 0.03),
                    'close': close_price,
                    'volume': np.random.randint(1000000, 10000000),
                    'amount': close_price * np.random.randint(1000000, 10000000),
                })

    return pd.DataFrame(records)


@pytest.fixture
def sample_factor_data():
    """Create sample factor data for testing."""
    codes = ['000001', '000002', '000003', '000004', '000005']
    date = pd.Timestamp(datetime(2024, 3, 1))

    data = {
        'code': codes,
        'date': [date] * len(codes),
        'name': [f'股票{i}' for i in range(1, len(codes) + 1)],
        'industry': ['银行', '医药', '科技', '消费', '制造'],
    }

    # Generate random factor values
    np.random.seed(42)
    factor_names = [
        'momentum_20d', 'momentum_60d', 'momentum_120d',
        'pe_ratio', 'pb_ratio', 'ps_ratio',
        'roe', 'gross_margin', 'net_profit_margin',
        'volatility_20d', 'volatility_60d',
        'amount_20d_avg', 'volume_ratio',
        'institution_holding_ratio', 'fund_holding_ratio',
    ]

    for factor in factor_names:
        data[factor] = np.random.randn(len(codes))

    # Add forward return (target)
    data['forward_return'] = np.random.randn(len(codes)) * 0.05

    return pd.DataFrame(data)


@pytest.fixture
def sample_daily_factor_data(sample_factor_data):
    """Create sample daily factor data for multiple days."""
    base_date = pd.Timestamp(datetime(2024, 3, 1))
    all_data = []

    for i in range(20):  # 20 days
        day_data = sample_factor_data.copy()
        day_data['date'] = base_date + timedelta(days=i)
        day_data['forward_return'] = np.random.randn(len(day_data)) * 0.05
        all_data.append(day_data)

    return pd.concat(all_data, ignore_index=True)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory for tests."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return {
        'paths': {
            'data_dir': 'data',
            'raw_data': 'data/raw',
            'raw_data_dir': 'data/raw',
            'processed_data': 'data/processed',
            'processed_data_dir': 'data/processed',
            'output_dir': 'output',
        },
        'data_fetch': {
            'adjust': 'qfq',  # 前复权
            'start_date': '20240101',
            'end_date': '20241231',
        },
        'stock_pool': {
            'min_list_days': 60,
            'min_avg_amount': 10000000,
            'exclude_st': True,
            'exchange': ['sh', 'sz'],
        },
        'backtest': {
            'initial_capital': 1000000,
            'commission': 0.0003,
            'slippage': 0.001,
        },
        'factors': {
            'momentum_windows': [20, 60, 120],
            'volatility_window': 20,
            'momentum_skip_days': 1,
            'flow_windows': [5, 20, 60],
        },
    }


# ===================== Pytest Configuration =====================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_network: marks tests that require network"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Sort tests by path for better organization
    items.sort(key=lambda item: (item.fspath, item.name))


# ===================== Helper Functions =====================

def create_mock_price_series(start_date='20240101', periods=60, seed=42):
    """Create a mock price series for testing."""
    np.random.seed(seed)
    base_date = pd.to_datetime(start_date)

    dates = []
    closes = []

    for i in range(periods):
        date = base_date + timedelta(days=i)
        if date.weekday() < 5:  # Skip weekends
            dates.append(date)
            closes.append(10.0 + np.random.randn() * 0.5 + i * 0.02)

    return pd.DataFrame({
        'date': dates,
        'close': closes,
    })


def create_mock_financial_data(codes, report_dates):
    """Create mock financial data for testing."""
    data = []

    for code in codes:
        for report_date in report_dates:
            data.append({
                'code': code,
                'report_date': pd.to_datetime(report_date),
                'announce_date': pd.to_datetime(report_date) + timedelta(days=30),
                'total_assets': np.random.randn() * 1e10,
                'total_liabilities': np.random.randn() * 5e9,
                'total_equity': np.random.randn() * 5e9,
                'revenue': np.random.randn() * 1e9,
                'net_profit': np.random.randn() * 1e8,
                'eps': np.random.randn() * 0.5,
                'roe': np.random.randn() * 0.05,
            })

    return pd.DataFrame(data)
