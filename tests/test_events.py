"""
tests/test_events.py
--------------------
Tests for src/finance/events.py

All test fixtures use hand-built DataFrames derived from real dataset patterns.
We do NOT call the LLM — all amounts are provided directly.
"""
import math
import pytest
from datetime import date
import pandas as pd

from src.finance import fx as fx_module
from src.finance.events import (
    FinancialEvent,
    resolve_events,
    detect_and_project_recurring,
    _EXCLUDED_STATUSES,
    _INCLUDED_STATUSES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rates_df():
    return pd.DataFrame([
        {"rate_date": "2024-01-15", "from_currency": "USD", "to_currency": "EUR", "rate": "0.92"},
        {"rate_date": "2024-01-15", "from_currency": "USD", "to_currency": "ZAR", "rate": "18.0"},
        {"rate_date": "2024-01-15", "from_currency": "EUR", "to_currency": "ZAR", "rate": "20.0"},
    ])


def make_profiles_df(users=None):
    if users is None:
        users = [("user_01", "ZAR", "50000", "10000")]
    rows = []
    for uid, ccy, bal, min_bal in users:
        rows.append({
            "user_id": uid, "home_currency": ccy,
            "current_available_balance": bal,
            "minimum_balance_to_keep": min_bal,
            "financial_priorities": "",
            "expense_categories_to_protect": "",
            "expense_categories_user_is_willing_to_reduce": "",
            "expense_categories_user_is_willing_to_stop": "",
            "payment_methods_user_will_consider": "full_payment",
            "max_installment_months": "",
        })
    return pd.DataFrame(rows)


def make_events_row(**kwargs):
    defaults = {
        "event_id": "event_01",
        "user_id": "user_01",
        "event_type": "expense",
        "description": "Test expense",
        "category": "rent",
        "direction": "debit",
        "amount": "1000",
        "currency": "ZAR",
        "event_date": "2024-01-01",
        "settlement_date": "2024-01-01",
        "status": "settled",
        "linked_event_id": "",
        "flexibility": "fixed",
        "minimum_allowed_amount": "",
    }
    defaults.update(kwargs)
    return defaults


@pytest.fixture(autouse=True)
def init_fx():
    fx_module.init_fx(make_rates_df())


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------

def test_single_settled_event_included():
    events_df = pd.DataFrame([make_events_row()])
    profiles_df = make_profiles_df()
    result = resolve_events(events_df, profiles_df, pd.DataFrame())
    assert "user_01" in result
    assert len(result["user_01"]) == 1
    evt = result["user_01"][0]
    assert evt.event_id == "event_01"
    assert evt.amount == 1000.0
    assert evt.direction == "debit"
    assert evt.signed_amount == -1000.0


def test_income_event_credit():
    events_df = pd.DataFrame([make_events_row(
        event_id="event_02", event_type="income", direction="credit", amount="5000", category="salary"
    )])
    profiles_df = make_profiles_df()
    result = resolve_events(events_df, profiles_df, pd.DataFrame())
    evt = result["user_01"][0]
    assert evt.signed_amount == 5000.0


# ---------------------------------------------------------------------------
# Status filtering
# ---------------------------------------------------------------------------

def test_pending_event_excluded():
    events_df = pd.DataFrame([make_events_row(status="pending")])
    profiles_df = make_profiles_df()
    result = resolve_events(events_df, profiles_df, pd.DataFrame())
    assert len(result.get("user_01", [])) == 0


def test_failed_event_excluded():
    events_df = pd.DataFrame([make_events_row(status="failed")])
    profiles_df = make_profiles_df()
    result = resolve_events(events_df, profiles_df, pd.DataFrame())
    assert len(result.get("user_01", [])) == 0


def test_cancelled_event_excluded():
    events_df = pd.DataFrame([make_events_row(status="cancelled")])
    profiles_df = make_profiles_df()
    result = resolve_events(events_df, profiles_df, pd.DataFrame())
    assert len(result.get("user_01", [])) == 0


def test_scheduled_event_included():
    events_df = pd.DataFrame([make_events_row(status="scheduled")])
    profiles_df = make_profiles_df()
    result = resolve_events(events_df, profiles_df, pd.DataFrame())
    assert len(result["user_01"]) == 1


# ---------------------------------------------------------------------------
# Linked event chains
# ---------------------------------------------------------------------------

def test_refund_chain_both_included():
    """Original debit + refund credit: both should be in the forecast."""
    rows = [
        make_events_row(
            event_id="event_98", event_type="expense", direction="debit",
            amount="583", status="settled", linked_event_id=""
        ),
        make_events_row(
            event_id="event_99", event_type="refund", direction="credit",
            amount="583", status="settled", linked_event_id="event_98"
        ),
    ]
    events_df = pd.DataFrame(rows)
    profiles_df = make_profiles_df()
    result = resolve_events(events_df, profiles_df, pd.DataFrame())
    ids = {e.event_id for e in result["user_01"]}
    assert "event_98" in ids
    assert "event_99" in ids


def test_cancelled_replaced_by_settled():
    """Cancelled parent + settled replacement: parent excluded, child included."""
    rows = [
        make_events_row(
            event_id="event_100", event_type="expense", direction="debit",
            amount="816.2", status="cancelled", linked_event_id=""
        ),
        make_events_row(
            event_id="event_101", event_type="expense", direction="debit",
            amount="816.2", status="settled", linked_event_id="event_100"
        ),
    ]
    events_df = pd.DataFrame(rows)
    profiles_df = make_profiles_df()
    result = resolve_events(events_df, profiles_df, pd.DataFrame())
    ids = {e.event_id for e in result["user_01"]}
    # event_100 is cancelled → excluded regardless
    assert "event_100" not in ids
    assert "event_101" in ids


def test_settled_supersedes_pending_parent():
    """A settled child replaces a pending/scheduled parent."""
    rows = [
        make_events_row(
            event_id="event_A", status="scheduled", linked_event_id="",
            event_date="2024-02-01", amount="1000"
        ),
        make_events_row(
            event_id="event_B", status="settled", linked_event_id="event_A",
            event_date="2024-02-01", amount="950"
        ),
    ]
    events_df = pd.DataFrame(rows)
    profiles_df = make_profiles_df()
    result = resolve_events(events_df, profiles_df, pd.DataFrame())
    ids = {e.event_id for e in result["user_01"]}
    assert "event_A" not in ids
    assert "event_B" in ids


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def test_exact_duplicate_removed():
    """Two rows with identical (user, direction, amount, date, category, description)."""
    row = make_events_row()
    events_df = pd.DataFrame([row, dict(row, event_id="event_dupe")])
    profiles_df = make_profiles_df()
    result = resolve_events(events_df, profiles_df, pd.DataFrame())
    assert len(result["user_01"]) == 1


def test_same_description_different_date_not_duplicate():
    """Recurring transactions with same description on different dates are not duplicates."""
    rows = [
        make_events_row(event_id="event_01", event_date="2024-01-02"),
        make_events_row(event_id="event_02", event_date="2024-02-02"),
    ]
    events_df = pd.DataFrame(rows)
    profiles_df = make_profiles_df()
    result = resolve_events(events_df, profiles_df, pd.DataFrame())
    assert len(result["user_01"]) == 2


# ---------------------------------------------------------------------------
# Missing amount (blank)
# ---------------------------------------------------------------------------

def test_blank_amount_is_nan():
    events_df = pd.DataFrame([make_events_row(amount="")])
    profiles_df = make_profiles_df()
    result = resolve_events(events_df, profiles_df, pd.DataFrame())
    assert len(result["user_01"]) == 1
    evt = result["user_01"][0]
    assert math.isnan(evt.amount)


def test_extracted_fact_fills_blank_amount():
    """extracted_facts should resolve a blank amount before chain resolution."""
    events_df = pd.DataFrame([make_events_row(amount="")])
    profiles_df = make_profiles_df()
    extracted = {"event_01": {"amount": 2500.0}}
    result = resolve_events(events_df, profiles_df, pd.DataFrame(), extracted_facts=extracted)
    assert result["user_01"][0].amount == 2500.0


# ---------------------------------------------------------------------------
# Flexibility
# ---------------------------------------------------------------------------

def test_stoppable_is_flexible():
    events_df = pd.DataFrame([make_events_row(flexibility="stoppable")])
    profiles_df = make_profiles_df()
    result = resolve_events(events_df, profiles_df, pd.DataFrame())
    assert result["user_01"][0].is_stoppable
    assert result["user_01"][0].is_flexible


def test_fixed_is_not_flexible():
    events_df = pd.DataFrame([make_events_row(flexibility="fixed")])
    profiles_df = make_profiles_df()
    result = resolve_events(events_df, profiles_df, pd.DataFrame())
    assert not result["user_01"][0].is_flexible


def test_reducible_or_stoppable():
    events_df = pd.DataFrame([make_events_row(
        flexibility="reducible_or_stoppable", minimum_allowed_amount="50"
    )])
    profiles_df = make_profiles_df()
    result = resolve_events(events_df, profiles_df, pd.DataFrame())
    evt = result["user_01"][0]
    assert evt.is_reducible
    assert evt.is_stoppable
    assert evt.minimum_allowed_amount == 50.0


# ---------------------------------------------------------------------------
# Recurrence projection
# ---------------------------------------------------------------------------

def make_recurring_events(n=3):
    """Create n monthly events (rent) for user_01."""
    rows = []
    for i in range(n):
        month = i + 1
        rows.append(FinancialEvent(
            event_id=f"event_{i:03d}",
            user_id="user_01",
            event_type="expense",
            description="Apartment rent transfer",
            category="rent",
            direction="debit",
            amount=5000.0,
            home_currency="ZAR",
            event_date=date(2024, month, 2),
            settlement_date=date(2024, month, 2),
            status="settled",
            linked_event_id=None,
            flexibility="fixed",
            minimum_allowed_amount=None,
        ))
    return rows


def test_recurring_events_projected():
    events = make_recurring_events(n=3)
    forecast_start = date(2024, 4, 1)
    forecast_end = date(2024, 6, 30)
    projected = detect_and_project_recurring(events, forecast_start, forecast_end)
    # Should project April and May rent (next after March + ~30 days)
    assert len(projected) >= 1
    for p in projected:
        assert p.is_projected
        assert p.event_date >= forecast_start
        assert p.event_date <= forecast_end
        assert p.category == "rent"
        assert p.amount == pytest.approx(5000.0, abs=1.0)


def test_single_event_not_projected():
    """Only one historical occurrence → below _MIN_RECURRENCE_COUNT → no projection."""
    events = make_recurring_events(n=1)
    projected = detect_and_project_recurring(
        events, date(2024, 4, 1), date(2024, 6, 30)
    )
    assert len(projected) == 0


def test_projection_does_not_go_past_forecast_end():
    events = make_recurring_events(n=6)
    forecast_end = date(2024, 7, 31)
    projected = detect_and_project_recurring(events, date(2024, 7, 1), forecast_end)
    for p in projected:
        assert p.event_date <= forecast_end
