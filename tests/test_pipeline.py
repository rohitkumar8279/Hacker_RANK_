"""
tests/test_pipeline.py
"""
import pytest
import math
from datetime import date
from src.pipeline import process_request
from src.data.join import RequestContext, Message, ImageRef
from src.finance.events import FinancialEvent
import config

def test_process_request_offline_mode(monkeypatch):
    monkeypatch.setattr(config, "LLM_ENABLED", False)
    
    evt = FinancialEvent(
        event_id="e1", user_id="u1", event_type="expense",
        description="", category="rent", direction="debit",
        amount=float('nan'), home_currency="ZAR", event_date=date(2024, 1, 1),
        settlement_date=date(2024, 1, 1), status="pending",
        linked_event_id="", flexibility="fixed", minimum_allowed_amount=None
    )
    
    img = ImageRef(image_id="img1", user_id="u1", request_id="", related_event_id="e1", local_path="")
    
    ctx = RequestContext(
        request_id="r1", user_id="u1", request_date=date(2024, 1, 1),
        request_type="purchase", requested_amount=100.0,
        desired_completion_date=None, allows_partial_payment=False,
        request_text="", profile=None, resolved_events=[evt],
        payment_options=[], messages_for_user=[], messages_for_request=[],
        messages_for_events={}, images_for_user=[], images_for_request=[],
        images_for_events={"e1": [img]}
    )
    
    # Needs a mock profile to avoid NoneType errors in planner/forecast
    class MockProfile:
        home_currency = "ZAR"
        current_available_balance = 500.0
        minimum_balance_to_keep = 0.0
        financial_priorities = []
        expense_categories_to_protect = []
        expense_categories_user_is_willing_to_reduce = []
        expense_categories_user_is_willing_to_stop = []
        payment_methods_user_will_consider = ["full_payment"]
        max_installment_months = None
        
    ctx.profile = MockProfile()
    
    plan, ext = process_request(ctx)
    
    # Offline mode -> fact extraction fails safely -> error is recorded
    assert "extraction unavailable" in ext.errors[0]
    
    # Unresolved event remains NaN, so it should have been filtered out of resolved_events
    assert len(ctx.resolved_events) == 0
    assert len(ctx.unresolved_events) == 0
    
    # Plan is generated using empty events list, starting balance 500
    # Requested 100
    assert plan.amount_safe_to_pay == 100.0
    assert plan.affordability_status == "affordable_now"
    assert plan.recommended_payment_method == "full_payment"

def test_pipeline_crash_fallback():
    # If we pass an absolutely broken context that crashes the planner,
    # the pipeline should catch it and return a fallback result.
    ctx = RequestContext(
        request_id="crash1", user_id="u1", request_date=date(2024, 1, 1),
        request_type="purchase", requested_amount=100.0,
        desired_completion_date=None, allows_partial_payment=False,
        request_text="", profile=None, resolved_events=[],
        payment_options=[], messages_for_user=[],
        messages_for_request=[], messages_for_events={}, images_for_user=[],
        images_for_request=[], images_for_events={}
    )
    # Profile is None, which will crash the planner
    plan, ext = process_request(ctx)
    
    assert plan.request_id == "crash1"
    assert plan.affordability_status == "not_affordable"
    assert plan.recommended_payment_method == "not_recommended"
    assert "Pipeline crash" in ext.errors[0]
