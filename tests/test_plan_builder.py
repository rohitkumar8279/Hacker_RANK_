"""
tests/test_plan_builder.py
--------------------------
Tests for Stage 4 payment planner.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from src.finance.events import FinancialEvent
from src.data.join import RequestContext, UserProfile, PaymentOption
from src.planning.models import PlanResult
from src.planning.plan_builder import build_plan


def make_ctx(
    amt: float = 1000.0,
    balance: float = 5000.0,
    methods: str = "full_payment",
    partial: bool = False,
    events=None,
    options=None,
    max_months=None,
    desired_date=None,
) -> RequestContext:
    if events is None:
        events = []
    if options is None:
        options = []
        
    profile = UserProfile(
        user_id="u1",
        home_currency="ZAR",
        current_available_balance=balance,
        minimum_balance_to_keep=1000.0,
        financial_priorities=[],
        expense_categories_to_protect=[],
        expense_categories_user_is_willing_to_reduce=[],
        expense_categories_user_is_willing_to_stop=[],
        payment_methods_user_will_consider=methods.split("|"),
        max_installment_months=max_months,
    )
    
    return RequestContext(
        request_id="r1",
        user_id="u1",
        request_date=date(2024, 1, 1),
        request_type="purchase",
        requested_amount=amt,
        desired_completion_date=desired_date,
        allows_partial_payment=partial,
        request_text="",
        profile=profile,
        resolved_events=events,
        payment_options=options,
        messages_for_user=[],
        messages_for_request=[],
        messages_for_events={},
        images_for_user=[],
        images_for_request=[],
        images_for_events={},
    )


def test_affordable_now_full_payment():
    ctx = make_ctx(amt=1000, balance=5000, methods="full_payment")
    result = build_plan(ctx)
    
    assert result.affordability_status == "affordable_now"
    assert result.recommended_payment_method == "full_payment"
    assert result.amount_safe_to_pay == 1000.0
    assert result.earliest_date_for_full_payment == date(2024, 1, 1)
    assert len(result.payments) == 1
    assert result.payments[0] == (date(2024, 1, 1), 1000.0)


def test_affordable_now_but_rejected_by_user():
    # User rejects full_payment, so no plans are available (since we gave no installment options)
    ctx = make_ctx(amt=1000, balance=5000, methods="installments")
    result = build_plan(ctx)
    
    assert result.affordability_status == "not_affordable"
    assert result.recommended_payment_method == "not_recommended"
    assert result.amount_safe_to_pay == 1000.0
    # earliest safe date is independent of user pref
    assert result.earliest_date_for_full_payment == date(2024, 1, 1)


def test_wait_plan_affordable_later():
    # Starting balance is 1000 (min balance is 1000), so 0 safe today.
    # An income of 2000 arrives on Jan 15.
    events = [
        FinancialEvent(
            event_id="e1", user_id="u1", event_type="income",
            description="", category="salary", direction="credit",
            amount=2000.0, home_currency="ZAR", event_date=date(2024, 1, 15),
            settlement_date=date(2024, 1, 15), status="scheduled",
            linked_event_id="", flexibility="fixed", minimum_allowed_amount=None
        )
    ]
    ctx = make_ctx(amt=1000, balance=1000, methods="full_payment|wait", events=events)
    result = build_plan(ctx)
    
    assert result.affordability_status == "affordable_later"
    assert result.recommended_payment_method == "wait"
    assert result.amount_safe_to_pay == 0.0
    assert result.earliest_date_for_full_payment == date(2024, 1, 15)
    assert result.payments == [(date(2024, 1, 15), 1000.0)]


def test_partial_payment_affordable_with_plan():
    # Start balance 1500, min 1000 -> 500 safe today.
    # Income on Jan 15 makes the rest safe.
    events = [
        FinancialEvent(
            event_id="e1", user_id="u1", event_type="income",
            description="", category="salary", direction="credit",
            amount=2000.0, home_currency="ZAR", event_date=date(2024, 1, 15),
            settlement_date=date(2024, 1, 15), status="scheduled",
            linked_event_id="", flexibility="fixed", minimum_allowed_amount=None
        )
    ]
    ctx = make_ctx(
        amt=1000, balance=1500, methods="partial_payment", 
        partial=True, events=events
    )
    result = build_plan(ctx)
    
    assert result.affordability_status == "affordable_with_plan"
    assert result.recommended_payment_method == "partial_payment"
    assert result.amount_safe_to_pay == 500.0
    assert result.earliest_date_for_full_payment == date(2024, 1, 15)
    assert result.payments == [
        (date(2024, 1, 1), 500.0),
        (date(2024, 1, 15), 500.0)
    ]


def test_installments_affordable_with_plan():
    # 0 safe today, but an installment option is available and safe
    options = [
        PaymentOption(
            payment_option_id="opt1", request_id="r1", payment_method="installments",
            payment_amount=50.0, number_of_payments=20, first_payment_date=date(2024, 1, 1),
            payment_frequency_days=30, financing_fee=0.0, total_payable_amount=1000.0
        )
    ]
    # To make it safe, we need enough balance or income to cover 50/mo. We start with balance 2000 (min 1000), 
    # so we can easily cover the 3 payments that hit in the 90 day window (150 total).
    ctx = make_ctx(amt=1000, balance=2000, methods="installments", options=options)
    result = build_plan(ctx)
    
    assert result.affordability_status == "affordable_with_plan"
    assert result.recommended_payment_method == "installments"
    assert result.selected_payment_option_id == "opt1"


def test_installments_max_months_violation():
    # Option spans ~19 months (20 payments * 30 days)
    # Profile limits to 12 months.
    options = [
        PaymentOption(
            payment_option_id="opt1", request_id="r1", payment_method="installments",
            payment_amount=50.0, number_of_payments=20, first_payment_date=date(2024, 1, 1),
            payment_frequency_days=30, financing_fee=0.0, total_payable_amount=1000.0
        )
    ]
    ctx = make_ctx(
        amt=1000, balance=2000, methods="installments", 
        options=options, max_months=12
    )
    result = build_plan(ctx)
    
    # Should be rejected
    assert result.affordability_status == "not_affordable"
    assert result.recommended_payment_method == "not_recommended"


def test_completion_date_violation():
    # Wait payment on Jan 15, but desired completion is Jan 10
    events = [
        FinancialEvent(
            event_id="e1", user_id="u1", event_type="income",
            description="", category="salary", direction="credit",
            amount=2000.0, home_currency="ZAR", event_date=date(2024, 1, 15),
            settlement_date=date(2024, 1, 15), status="scheduled",
            linked_event_id="", flexibility="fixed", minimum_allowed_amount=None
        )
    ]
    ctx = make_ctx(
        amt=1000, balance=1000, methods="wait", 
        events=events, desired_date=date(2024, 1, 10)
    )
    result = build_plan(ctx)
    
    assert result.affordability_status == "not_affordable"
    assert result.recommended_payment_method == "not_recommended"

def test_installment_ranking_cost():
    # opt1 has higher cost (financing fee), opt2 has lower cost
    options = [
        PaymentOption(
            payment_option_id="opt1", request_id="r1", payment_method="installments",
            payment_amount=55.0, number_of_payments=20, first_payment_date=date(2024, 1, 1),
            payment_frequency_days=30, financing_fee=100.0, total_payable_amount=1100.0
        ),
        PaymentOption(
            payment_option_id="opt2", request_id="r1", payment_method="installments",
            payment_amount=50.0, number_of_payments=20, first_payment_date=date(2024, 1, 1),
            payment_frequency_days=30, financing_fee=0.0, total_payable_amount=1000.0
        )
    ]
    ctx = make_ctx(amt=1000, balance=5000, methods="installments", options=options)
    result = build_plan(ctx)
    
    assert result.selected_payment_option_id == "opt2"
    assert result.total_amount_paid == 1000.0

def test_boundary_affordability_epsilon():
    # Test P0-B: amount_safe_to_pay exactly equals requested_amount but float precision issues
    # would previously fail the simulation.
    
    events = [
        FinancialEvent(
            event_id="e1", user_id="u1", event_type="expense",
            description="", category="", direction="debit",
            amount=4000.0, home_currency="ZAR", event_date=date(2024, 1, 1),
            settlement_date=date(2024, 1, 1), status="scheduled",
            linked_event_id="", flexibility="fixed", minimum_allowed_amount=None
        )
    ]
    # Balance 5000, min 1000, scheduled expense 4000 on day 1 -> 0 safe.
    # But wait, we want safe amount == requested amount.
    # Balance 6000, min 1000, scheduled expense 4000 -> 1000 safe.
    events[0].amount = 4000.0
    ctx = make_ctx(amt=1000.0, balance=6000.0, methods="full_payment", events=events)
    result = build_plan(ctx)
    
    assert result.affordability_status == "affordable_now"
    assert result.recommended_payment_method == "full_payment"
    assert result.amount_safe_to_pay == 1000.0
    assert result.payments[0][1] == 1000.0

