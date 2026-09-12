"""
tests/test_forecast.py
----------------------
Tests for src/finance/forecast.py

Scenarios tested:
- Simple income only
- Simple expense only
- Recurring income + recurring expense
- Payment overlay (override_payment)
- Spending change: stop
- Spending change: reduce_to
- 90-day boundary (exactly 91 rows)
- Month boundary (expense at month end)
- Minimum balance tracking
- Future event falling mid-window
"""
import copy
import pytest
from datetime import date, timedelta

from src.finance.events import FinancialEvent
from src.finance.forecast import simulate, SpendingChange, Forecast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(
    event_id: str,
    event_date: date,
    amount: float,
    direction: str = "debit",
    status: str = "settled",
    flexibility: str = "fixed",
    category: str = "rent",
    is_projected: bool = False,
) -> FinancialEvent:
    return FinancialEvent(
        event_id=event_id,
        user_id="user_01",
        event_type="expense" if direction == "debit" else "income",
        description="Test event",
        category=category,
        direction=direction,
        amount=amount,
        home_currency="ZAR",
        event_date=event_date,
        settlement_date=event_date,
        status=status,
        linked_event_id=None,
        flexibility=flexibility,
        minimum_allowed_amount=None,
        is_projected=is_projected,
    )


REQ_DATE = date(2024, 3, 1)
BALANCE = 50_000.0
MIN_BAL = 10_000.0


def run(events, override=None, spending_changes=None):
    return simulate(
        user_id="user_01",
        request_date=REQ_DATE,
        start_balance=BALANCE,
        minimum_balance=MIN_BAL,
        events=events,
        override_payment=override,
        spending_changes=spending_changes or [],
    )


# ---------------------------------------------------------------------------
# Core invariants
# ---------------------------------------------------------------------------

def test_forecast_has_91_days():
    f = run([])
    assert len(f.days) == 91  # day 0 through day 90 inclusive


def test_no_events_flat_balance():
    f = run([])
    for day in f.days:
        assert day.closing_balance == BALANCE


def test_single_income_event():
    income = make_event("e_01", REQ_DATE + timedelta(days=10), 5000.0, direction="credit")
    f = run([income])
    assert f.days[10].total_income == 5000.0
    assert f.days[10].closing_balance == BALANCE + 5000.0
    assert f.days[11].closing_balance == BALANCE + 5000.0  # stays up


def test_single_expense_event():
    expense = make_event("e_01", REQ_DATE + timedelta(days=5), 3000.0, direction="debit")
    f = run([expense])
    assert f.days[5].total_expenses == 3000.0
    assert f.days[5].closing_balance == BALANCE - 3000.0


def test_events_before_request_date_ignored():
    """Events before request_date are already in the starting balance."""
    past_expense = make_event("e_01", REQ_DATE - timedelta(days=1), 5000.0)
    f = run([past_expense])
    # Should have no effect
    assert f.days[0].closing_balance == BALANCE


def test_event_on_request_date_applied():
    on_date = make_event("e_01", REQ_DATE, 1000.0)
    f = run([on_date])
    assert f.days[0].total_expenses == 1000.0
    assert f.days[0].closing_balance == BALANCE - 1000.0


def test_event_on_last_day_applied():
    last_day = REQ_DATE + timedelta(days=90)
    expense = make_event("e_01", last_day, 2000.0)
    f = run([expense])
    assert f.days[-1].closing_balance == BALANCE - 2000.0


def test_event_after_window_ignored():
    beyond = make_event("e_01", REQ_DATE + timedelta(days=91), 2000.0)
    f = run([beyond])
    assert f.days[-1].closing_balance == BALANCE  # not applied


# ---------------------------------------------------------------------------
# Override payment
# ---------------------------------------------------------------------------

def test_override_payment_applied():
    f = run([], override=(REQ_DATE, 20_000.0))
    assert f.days[0].closing_balance == BALANCE - 20_000.0
    assert f.days[1].closing_balance == BALANCE - 20_000.0  # persists


def test_override_payment_propagates():
    """Remaining balance after payment persists for all future days."""
    pay_date = REQ_DATE + timedelta(days=5)
    f = run([], override=(pay_date, 15_000.0))
    # Before payment
    assert f.days[4].closing_balance == BALANCE
    # On payment day
    assert f.days[5].closing_balance == BALANCE - 15_000.0
    # After
    assert f.days[10].closing_balance == BALANCE - 15_000.0


# ---------------------------------------------------------------------------
# Month boundary
# ---------------------------------------------------------------------------

def test_month_boundary_february():
    """Test that an expense on Feb 28 works even in a non-leap year."""
    req = date(2025, 2, 1)
    feb_28 = date(2025, 2, 28)
    expense = make_event("e_01", feb_28, 1000.0)
    f = simulate("user_01", req, BALANCE, MIN_BAL, [expense])
    day_offset = (feb_28 - req).days
    assert f.days[day_offset].closing_balance == BALANCE - 1000.0


# ---------------------------------------------------------------------------
# Spending changes
# ---------------------------------------------------------------------------

def test_stop_spending_change():
    stoppable = make_event(
        "e_stop", REQ_DATE + timedelta(days=10), 500.0,
        flexibility="stoppable", category="streaming", is_projected=True
    )
    f_base = run([stoppable])
    assert f_base.days[10].closing_balance == BALANCE - 500.0

    f_stopped = run(
        [stoppable],
        spending_changes=[SpendingChange(event_id="e_stop", action="stop")]
    )
    assert f_stopped.days[10].closing_balance == BALANCE  # event removed


def test_reduce_spending_change():
    reducible = make_event(
        "e_reduce", REQ_DATE + timedelta(days=15), 1000.0,
        flexibility="reducible", category="dining", is_projected=True
    )
    import copy
    reducible.minimum_allowed_amount = 200.0

    f = run(
        [reducible],
        spending_changes=[SpendingChange(event_id="e_reduce", action="reduce_to", new_amount=300.0)]
    )
    assert f.days[15].total_expenses == 300.0


def test_fixed_event_not_stoppable():
    """A spending change on a fixed event should be silently ignored."""
    fixed_evt = make_event("e_fixed", REQ_DATE + timedelta(days=5), 2000.0, flexibility="fixed")
    f = run(
        [fixed_evt],
        spending_changes=[SpendingChange(event_id="e_fixed", action="stop")]
    )
    # Fixed event should still be applied
    assert f.days[5].total_expenses == 2000.0


# ---------------------------------------------------------------------------
# Minimum balance tracking
# ---------------------------------------------------------------------------

def test_min_balance_over_window():
    """min_balance_over_window should be the lowest closing balance."""
    expenses = [
        make_event(f"e_{i}", REQ_DATE + timedelta(days=i * 10), 5000.0)
        for i in range(4)
    ]
    f = run(expenses)
    expected_min = BALANCE - 4 * 5000.0  # day 30 = 50000 - 20000 = 30000
    assert f.min_balance_over_window == pytest.approx(expected_min, abs=0.01)


def test_salary_mid_window():
    """Salary arriving mid-window lifts the balance."""
    large_expense = make_event("e_expense", REQ_DATE + timedelta(days=5), 30_000.0)
    salary = make_event("e_salary", REQ_DATE + timedelta(days=20), 40_000.0, direction="credit", category="salary")
    f = run([large_expense, salary])

    # Day 5: balance drops by 30k
    assert f.days[5].closing_balance == pytest.approx(BALANCE - 30_000.0, abs=0.01)
    # Day 20: salary arrives, balance rises
    assert f.days[20].closing_balance == pytest.approx(BALANCE - 30_000.0 + 40_000.0, abs=0.01)
