"""
tests/test_fx.py
----------------
Tests for src/finance/fx.py
"""
import pytest
from datetime import date
import pandas as pd
from src.finance.fx import FXStore, init_fx, convert


def make_rates_df(rows):
    return pd.DataFrame(rows, columns=["rate_date", "from_currency", "to_currency", "rate"])


@pytest.fixture(autouse=True)
def init_store():
    """Initialise FX store before every test."""
    df = make_rates_df([
        ("2024-01-15", "USD", "EUR", "0.92"),
        ("2024-01-15", "USD", "IDR", "15833.33"),
        ("2024-01-15", "USD", "INR", "83.33"),
        ("2024-01-15", "EUR", "ZAR", "20"),
        ("2024-02-15", "USD", "EUR", "0.92"),
        ("2024-02-15", "USD", "IDR", "15833.33"),
        ("2024-02-15", "USD", "INR", "83.33"),
        ("2024-02-15", "EUR", "ZAR", "20"),
    ])
    init_fx(df)


def test_same_currency_is_noop():
    assert convert(1000.0, "ZAR", "ZAR", date(2024, 2, 1)) == 1000.0


def test_direct_rate():
    # 1 USD = 0.92 EUR
    result = convert(100.0, "USD", "EUR", date(2024, 2, 1))
    assert abs(result - 92.0) < 0.01


def test_inverse_rate():
    # EUR → USD should be 1/0.92
    result = convert(92.0, "EUR", "USD", date(2024, 2, 1))
    assert abs(result - 100.0) < 0.01


def test_bridge_via_usd():
    # IDR → EUR: IDR → USD (1/15833.33) → EUR (*0.92)
    # 15833.33 IDR = 1 USD = 0.92 EUR
    result = convert(15833.33, "IDR", "EUR", date(2024, 2, 1))
    assert abs(result - 0.92) < 0.01


def test_uses_rate_on_or_before_date():
    # Use jan rate for a feb date
    result = convert(100.0, "USD", "EUR", date(2024, 1, 20))
    assert abs(result - 92.0) < 0.01


def test_no_rate_raises():
    with pytest.raises(ValueError):
        # No rate between ZAR and INR in our fixture
        convert(100.0, "ZAR", "INR", date(2024, 2, 1))


def test_zero_amount():
    assert convert(0.0, "USD", "EUR", date(2024, 2, 1)) == 0.0


def test_eur_to_zar():
    # EUR → ZAR direct = 20
    result = convert(50.0, "EUR", "ZAR", date(2024, 2, 1))
    assert abs(result - 1000.0) < 0.01
