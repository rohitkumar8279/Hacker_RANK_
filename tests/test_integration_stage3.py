"""
tests/test_integration_stage3.py
---------------------------------
Integration test: builds a RequestContext using synthetic data modelled
after real sample_request_01 from the dataset, then feeds it through
the Stage 2 forecast + safety engine.

Verifies:
  - Stage 3 context is correct
  - Stage 2 still produces the correct result after Stage 3 wiring
  - No regression: 60 existing tests + this test all pass together
"""
from __future__ import annotations

import pytest
from datetime import date, timedelta

import pandas as pd

from src.finance import fx as fx_module
from src.finance.events import detect_and_project_recurring
from src.finance.forecast import simulate
from src.finance.safety import max_safe_amount, earliest_safe_date, is_plan_safe
from src.data.join import DataBundle, RequestContextBuilder


# ---------------------------------------------------------------------------
# Setup: FX rates (modelled after real exchange_rates.csv)
# ---------------------------------------------------------------------------

RATES_DF = pd.DataFrame([
    {"rate_date": "2024-01-15", "from_currency": "USD", "to_currency": "ZAR", "rate": "18.0"},
    {"rate_date": "2024-01-15", "from_currency": "EUR", "to_currency": "ZAR", "rate": "20.0"},
    {"rate_date": "2024-02-15", "from_currency": "USD", "to_currency": "ZAR", "rate": "18.0"},
    {"rate_date": "2024-02-15", "from_currency": "EUR", "to_currency": "ZAR", "rate": "20.0"},
    {"rate_date": "2024-03-15", "from_currency": "USD", "to_currency": "ZAR", "rate": "18.0"},
    {"rate_date": "2024-03-15", "from_currency": "EUR", "to_currency": "ZAR", "rate": "20.0"},
])

# Initialise FX
fx_module.init_fx(RATES_DF)


# ---------------------------------------------------------------------------
# Synthetic data closely matching sample_request_01
# ---------------------------------------------------------------------------

REQUESTS_DF = pd.DataFrame([{
    "request_id": "request_01",
    "user_id": "user_01",
    "request_date": "2024-03-03",
    "request_type": "purchase",
    "requested_amount": "25256",
    "desired_completion_date": "2024-03-20",
    "allows_partial_payment": "true",
    "request_text": "Would paying for the laptop today leave enough for my regular expenses? The laptop I'm looking at is ZAR 25,256.",
}], dtype=str)

PROFILES_DF = pd.DataFrame([{
    "user_id": "user_01",
    "home_currency": "ZAR",
    "current_available_balance": "58481.1",
    "minimum_balance_to_keep": "18000",
    "financial_priorities": "education|debt_repayment",
    "expense_categories_to_protect": "rent|education|groceries|debt_repayment",
    "expense_categories_user_is_willing_to_reduce": "dining",
    "expense_categories_user_is_willing_to_stop": "delivery_membership",
    "payment_methods_user_will_consider": "full_payment",
    "max_installment_months": "",
}], dtype=str)

# Minimal but representative events for user_01:
# - Jan salary (settled, credit) — in the past, already in balance
# - Feb rent (settled, debit) — in the past
# - March salary (scheduled, credit) — within the forecast window
# - No pending events
EVENTS_DF = pd.DataFrame([
    {
        "event_id": "event_jan_salary",
        "user_id": "user_01",
        "event_type": "income",
        "description": "Monthly salary",
        "category": "salary",
        "direction": "credit",
        "amount": "23320",
        "currency": "ZAR",
        "event_date": "2024-01-15",
        "settlement_date": "2024-01-15",
        "status": "settled",
        "linked_event_id": "",
        "flexibility": "fixed",
        "minimum_allowed_amount": "",
    },
    {
        "event_id": "event_feb_salary",
        "user_id": "user_01",
        "event_type": "income",
        "description": "Monthly salary",
        "category": "salary",
        "direction": "credit",
        "amount": "23320",
        "currency": "ZAR",
        "event_date": "2024-02-15",
        "settlement_date": "2024-02-15",
        "status": "settled",
        "linked_event_id": "",
        "flexibility": "fixed",
        "minimum_allowed_amount": "",
    },
    {
        "event_id": "event_mar_salary",
        "user_id": "user_01",
        "event_type": "income",
        "description": "Monthly salary",
        "category": "salary",
        "direction": "credit",
        "amount": "23320",
        "currency": "ZAR",
        "event_date": "2024-03-15",
        "settlement_date": "2024-03-15",
        "status": "scheduled",
        "linked_event_id": "",
        "flexibility": "fixed",
        "minimum_allowed_amount": "",
    },
    # Rent — settled, past (already in balance)
    {
        "event_id": "event_jan_rent",
        "user_id": "user_01",
        "event_type": "expense",
        "description": "Apartment rent transfer",
        "category": "rent",
        "direction": "debit",
        "amount": "5148",
        "currency": "ZAR",
        "event_date": "2024-01-02",
        "settlement_date": "2024-01-02",
        "status": "settled",
        "linked_event_id": "",
        "flexibility": "fixed",
        "minimum_allowed_amount": "",
    },
    {
        "event_id": "event_feb_rent",
        "user_id": "user_01",
        "event_type": "expense",
        "description": "Apartment rent transfer",
        "category": "rent",
        "direction": "debit",
        "amount": "5148",
        "currency": "ZAR",
        "event_date": "2024-02-02",
        "settlement_date": "2024-02-02",
        "status": "settled",
        "linked_event_id": "",
        "flexibility": "fixed",
        "minimum_allowed_amount": "",
    },
], dtype=str)

PAYMENT_OPTIONS_DF = pd.DataFrame([
    {
        "payment_option_id": "payment_option_01",
        "request_id": "request_01",
        "payment_method": "full_payment",
        "payment_amount": "25256",
        "number_of_payments": "1",
        "first_payment_date": "2024-03-03",
        "payment_frequency_days": "",
        "financing_fee": "0",
        "total_payable_amount": "25256",
    },
    {
        "payment_option_id": "payment_option_02",
        "request_id": "request_01",
        "payment_method": "installments",
        "payment_amount": "1852.11",
        "number_of_payments": "15",
        "first_payment_date": "2024-03-06",
        "payment_frequency_days": "30",
        "financing_fee": "2525.65",
        "total_payable_amount": "27781.65",
    },
], dtype=str)

MESSAGES_DF = pd.DataFrame([], columns=[
    "message_id", "user_id", "request_id", "related_event_id",
    "sent_at", "source_type", "message_text",
], dtype=str)

IMAGES_DF = pd.DataFrame([], columns=[
    "image_id", "user_id", "request_id", "related_event_id",
], dtype=str)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ctx():
    """Build the RequestContext once for all integration tests."""
    bundle = DataBundle(
        requests=REQUESTS_DF,
        financial_profiles=PROFILES_DF,
        financial_events=EVENTS_DF,
        exchange_rates=RATES_DF,
        request_payment_options=PAYMENT_OPTIONS_DF,
        messages=MESSAGES_DF,
        images=IMAGES_DF,
        dataset_dir="dataset",
    )
    builder = RequestContextBuilder(bundle)
    return builder.build("request_01")


def test_integration_request_id(ctx):
    assert ctx.request_id == "request_01"


def test_integration_user_id(ctx):
    assert ctx.user_id == "user_01"


def test_integration_currency(ctx):
    assert ctx.home_currency == "ZAR"


def test_integration_balance(ctx):
    assert ctx.available_balance == pytest.approx(58481.1)


def test_integration_minimum_balance(ctx):
    assert ctx.minimum_balance == pytest.approx(18000.0)


def test_integration_correct_events(ctx):
    """Resolved events should only include forecast-eligible (settled/scheduled) events."""
    # pending/failed/cancelled events should not appear
    for evt in ctx.resolved_events:
        assert evt.status in ("settled", "scheduled"), f"Unexpected status: {evt.status}"
        assert evt.user_id == "user_01"


def test_integration_scheduled_salary_included(ctx):
    """The March scheduled salary must be in the resolved events."""
    ids = [e.event_id for e in ctx.resolved_events]
    assert "event_mar_salary" in ids


def test_integration_past_events_available(ctx):
    """Past settled events (Jan/Feb salary, rent) should be in resolved events."""
    ids = [e.event_id for e in ctx.resolved_events]
    assert "event_jan_salary" in ids
    assert "event_feb_salary" in ids


def test_integration_payment_options_count(ctx):
    assert len(ctx.payment_options) == 2


def test_integration_full_payment_option(ctx):
    full = next(o for o in ctx.payment_options if o.payment_method == "full_payment")
    assert full.payment_amount == pytest.approx(25256.0)
    assert full.financing_fee == pytest.approx(0.0)
    assert full.number_of_payments == 1
    assert full.first_payment_date == date(2024, 3, 3)


def test_integration_installment_option(ctx):
    inst = next(o for o in ctx.payment_options if o.payment_method == "installments")
    assert inst.payment_amount == pytest.approx(1852.11)
    assert inst.number_of_payments == 15
    assert inst.payment_frequency_days == 30
    assert inst.financing_fee == pytest.approx(2525.65)
    assert inst.total_payable_amount == pytest.approx(27781.65)


def test_integration_no_unresolved_events(ctx):
    assert not ctx.has_unresolved_events


def test_integration_stage2_forecast_correct(ctx):
    """
    Run the full Stage 2 forecast + safety check through the Stage 3 context.
    Expected (from sample_requests.csv):
      amount_safe_to_pay: 25256
      earliest_date_for_full_payment: 2024-03-03
    """
    request_date = ctx.request_date
    forecast_end = request_date + timedelta(days=90)

    # Project recurring events (rent is fixed, salary recurs)
    projected = detect_and_project_recurring(ctx.resolved_events, request_date, forecast_end)
    all_events = ctx.resolved_events + projected

    forecast = simulate(
        user_id=ctx.user_id,
        request_date=request_date,
        start_balance=ctx.available_balance,
        minimum_balance=ctx.minimum_balance,
        events=all_events,
    )

    safe_amt = max_safe_amount(forecast, ctx.requested_amount)
    earliest = earliest_safe_date(forecast, ctx.requested_amount)

    # Core expected results from sample_requests.csv
    assert safe_amt == pytest.approx(ctx.requested_amount, abs=0.01), (
        f"amount_safe_to_pay: expected {ctx.requested_amount}, got {safe_amt}"
    )
    assert earliest == ctx.request_date, (
        f"earliest_safe_date: expected {ctx.request_date}, got {earliest}"
    )

    # Plan safety check
    result = is_plan_safe(forecast, [(ctx.request_date, ctx.requested_amount)])
    assert result.safe is True
    assert result.minimum_balance_observed >= ctx.minimum_balance


def test_integration_recurrence_projects_rent(ctx):
    """The recurring rent pattern should produce projected March rent."""
    request_date = ctx.request_date
    forecast_end = request_date + timedelta(days=90)
    projected = detect_and_project_recurring(ctx.resolved_events, request_date, forecast_end)

    rent_projected = [p for p in projected if p.category == "rent" and p.is_projected]
    assert len(rent_projected) >= 1, "Expected at least one projected rent event"
    for p in rent_projected:
        assert p.event_date >= request_date
        assert p.event_date <= forecast_end
        assert p.amount == pytest.approx(5148.0, abs=1.0)


def test_integration_allows_partial_payment(ctx):
    assert ctx.allows_partial_payment is True


def test_integration_desired_completion_date(ctx):
    assert ctx.desired_completion_date == date(2024, 3, 20)


def test_integration_request_type(ctx):
    assert ctx.request_type == "purchase"


def test_integration_payment_methods(ctx):
    assert "full_payment" in ctx.profile.payment_methods_user_will_consider


def test_integration_profile_priorities(ctx):
    assert "education" in ctx.profile.financial_priorities
    assert "debt_repayment" in ctx.profile.financial_priorities
