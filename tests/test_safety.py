"""
tests/test_safety.py
--------------------
Tests for src/finance/safety.py

Scenarios tested:
- Safe plan (payment fits in headroom)
- Unsafe plan (payment exceeds headroom)
- Payment causing minimum-balance violation
- Future expense causes later violation
- Payment becomes safe on a later date
- Exact minimum balance (boundary condition)
- Insufficient balance (full window)
- earliest_safe_date: found vs not found
- Installment plan safety
"""
import pytest
from datetime import date, timedelta
from typing import List

from src.finance.events import FinancialEvent
from src.finance.forecast import simulate, Forecast
from src.finance.safety import (
    max_safe_amount,
    earliest_safe_date,
    is_plan_safe,
    SafetyResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQ_DATE = date(2024, 3, 1)
BALANCE = 50_000.0
MIN_BAL = 10_000.0


def make_event(event_id, event_date, amount, direction="debit", category="rent"):
    return FinancialEvent(
        event_id=event_id,
        user_id="user_01",
        event_type="expense" if direction == "debit" else "income",
        description="test",
        category=category,
        direction=direction,
        amount=amount,
        home_currency="ZAR",
        event_date=event_date,
        settlement_date=event_date,
        status="settled",
        linked_event_id=None,
        flexibility="fixed",
        minimum_allowed_amount=None,
    )


def base_forecast(events=None) -> Forecast:
    return simulate(
        user_id="user_01",
        request_date=REQ_DATE,
        start_balance=BALANCE,
        minimum_balance=MIN_BAL,
        events=events or [],
    )


# ---------------------------------------------------------------------------
# max_safe_amount
# ---------------------------------------------------------------------------

def test_no_future_expenses_max_safe():
    """No future expenses: headroom = balance - min_bal."""
    f = base_forecast()
    safe = max_safe_amount(f, requested_amount=BALANCE)
    # min_future_balance = BALANCE (no expenses)
    # headroom = BALANCE - MIN_BAL = 40000
    assert safe == pytest.approx(40_000.0, abs=0.01)


def test_requested_amount_below_headroom():
    """Requested amount is less than headroom — full amount is safe."""
    f = base_forecast()
    safe = max_safe_amount(f, requested_amount=5_000.0)
    assert safe == pytest.approx(5_000.0, abs=0.01)


def test_future_expense_reduces_headroom():
    """A future expense reduces the minimum balance and thus the headroom."""
    future_expense = make_event("e", REQ_DATE + timedelta(days=30), 20_000.0)
    f = base_forecast([future_expense])
    # Without payment: min future balance = BALANCE - 20000 = 30000
    # headroom = 30000 - 10000 = 20000
    safe = max_safe_amount(f, requested_amount=25_000.0)
    assert safe == pytest.approx(20_000.0, abs=0.01)


def test_safe_amount_clamped_to_zero():
    """When headroom is negative, amount_safe_to_pay = 0."""
    # Large expense that already breaches minimum
    future_expense = make_event("e", REQ_DATE + timedelta(days=10), 45_000.0)
    f = base_forecast([future_expense])
    safe = max_safe_amount(f, requested_amount=10_000.0)
    assert safe == 0.0


def test_safe_amount_never_exceeds_requested():
    f = base_forecast()
    safe = max_safe_amount(f, requested_amount=100.0)
    assert safe <= 100.0


def test_exact_minimum_balance_boundary():
    """Payment that leaves exactly minimum_balance should be safe."""
    # Balance = 50000, min = 10000, headroom = 40000
    f = base_forecast()
    safe = max_safe_amount(f, requested_amount=40_000.0)
    assert safe == pytest.approx(40_000.0, abs=0.01)


# ---------------------------------------------------------------------------
# earliest_safe_date
# ---------------------------------------------------------------------------

def test_earliest_safe_date_is_today_when_affordable_now():
    f = base_forecast()
    d = earliest_safe_date(f, 10_000.0)
    assert d == REQ_DATE


def test_earliest_safe_date_after_salary():
    """Large payment only becomes safe after a salary arrives mid-window."""
    # Balance = 50000, min = 10000, requested = 45000
    # headroom today = 40000 → not enough
    # salary on day 15 → balance = 50000 + 30000 = 80000, headroom = 70000 → enough
    salary = make_event("sal", REQ_DATE + timedelta(days=15), 30_000.0, direction="credit", category="salary")
    f = base_forecast([salary])
    d = earliest_safe_date(f, requested_amount=45_000.0)
    # Before salary: headroom = 40000 < 45000
    # On day 15 (salary day): min_from_day15 = min(80000, 80000, ...) = 80000
    # 80000 - 45000 = 35000 >= 10000 → day 15 is the answer
    assert d == REQ_DATE + timedelta(days=15)


def test_earliest_safe_date_none_when_not_achievable():
    """Large expense means payment is never safe in the window."""
    large_expense = make_event("e", REQ_DATE + timedelta(days=5), 48_000.0)
    f = base_forecast([large_expense])
    # min_future from any day = 50000 - 48000 = 2000
    # headroom = 2000 - 10000 = -8000 < requested_amount for any positive request
    d = earliest_safe_date(f, requested_amount=5_000.0)
    assert d is None


def test_earliest_safe_date_zero_amount():
    """Zero-cost request is always immediately safe."""
    f = base_forecast()
    d = earliest_safe_date(f, requested_amount=0.0)
    assert d == REQ_DATE


# ---------------------------------------------------------------------------
# is_plan_safe
# ---------------------------------------------------------------------------

def test_safe_single_payment():
    f = base_forecast()
    result = is_plan_safe(f, [(REQ_DATE, 10_000.0)])
    assert result.safe is True
    assert result.first_failure_date is None


def test_unsafe_payment_today():
    """Payment that brings balance below minimum."""
    f = base_forecast()
    # BALANCE - payment = 50000 - 42000 = 8000 < 10000
    result = is_plan_safe(f, [(REQ_DATE, 42_000.0)])
    assert result.safe is False
    assert result.first_failure_date == REQ_DATE


def test_future_expense_causes_later_violation():
    """Payment today is fine, but a future expense then breaches the minimum."""
    future_expense = make_event("e", REQ_DATE + timedelta(days=20), 30_000.0)
    f = base_forecast([future_expense])
    # After payment today: balance = 30000
    # After future expense: balance = 30000 - 30000 = 0 < 10000
    result = is_plan_safe(f, [(REQ_DATE, 20_000.0)])
    assert result.safe is False
    assert result.first_failure_date == REQ_DATE + timedelta(days=20)


def test_installment_plan_safe():
    """Three installments that all fit within headroom."""
    payments = [
        (REQ_DATE, 5_000.0),
        (REQ_DATE + timedelta(days=30), 5_000.0),
        (REQ_DATE + timedelta(days=60), 5_000.0),
    ]
    f = base_forecast()
    result = is_plan_safe(f, payments)
    # After all payments: balance = 50000 - 15000 = 35000 > 10000
    assert result.safe is True


def test_installment_plan_unsafe_second_payment():
    """Second installment causes a breach."""
    big_expense = make_event("e", REQ_DATE + timedelta(days=25), 20_000.0)
    f = base_forecast([big_expense])
    # After payment 1 (day 0): balance = 40000
    # After big expense (day 25): balance = 20000
    # After payment 2 (day 30): balance = 10000 (exactly at limit — safe)
    payments = [
        (REQ_DATE, 10_000.0),
        (REQ_DATE + timedelta(days=30), 11_000.0),  # 20000 - 11000 = 9000 < 10000
    ]
    result = is_plan_safe(f, payments)
    assert result.safe is False


def test_safe_result_reports_minimum_balance():
    f = base_forecast()
    result = is_plan_safe(f, [(REQ_DATE, 5_000.0)])
    assert result.minimum_balance_observed == pytest.approx(BALANCE - 5_000.0, abs=0.01)


def test_plan_with_salary_restores_safety():
    """After a large payment, an incoming salary keeps things safe."""
    salary = make_event("sal", REQ_DATE + timedelta(days=10), 20_000.0, direction="credit", category="salary")
    f = base_forecast([salary])
    # Pay 35000 on day 0: balance = 15000 (>= 10000 → safe for that day)
    # Day 10: balance = 15000 + 20000 = 35000
    result = is_plan_safe(f, [(REQ_DATE, 35_000.0)])
    assert result.safe is True
    assert result.minimum_balance_observed >= MIN_BAL
