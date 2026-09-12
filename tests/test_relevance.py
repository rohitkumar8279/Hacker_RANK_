"""
tests/test_relevance.py
"""
from datetime import date
from src.data.join import RequestContext, Message, ImageRef
from src.extraction.relevance import get_relevant_evidence
from src.finance.events import FinancialEvent

def test_no_relevant_evidence():
    ctx = RequestContext(
        request_id="r1", user_id="u1", request_date=date(2024, 1, 1),
        request_type="purchase", requested_amount=100.0,
        desired_completion_date=None, allows_partial_payment=False,
        request_text="", profile=None, resolved_events=[],
        payment_options=[], messages_for_user=[], messages_for_request=[],
        messages_for_events={}, images_for_user=[], images_for_request=[],
        images_for_events={}
    )
    ev = get_relevant_evidence(ctx)
    assert len(ev.messages) == 0
    assert len(ev.images) == 0

def test_unresolved_event_evidence():
    evt = FinancialEvent(
        event_id="e1", user_id="u1", event_type="expense",
        description="", category="rent", direction="debit",
        amount=float('nan'), home_currency="ZAR", event_date=date(2024, 1, 1),
        settlement_date=date(2024, 1, 1), status="pending",
        linked_event_id="", flexibility="fixed", minimum_allowed_amount=None
    )
    
    msg = Message(message_id="m1", user_id="u1", request_id="", related_event_id="e1", sent_at="2024-01-01", source_type="email", message_text="rent")
    img = ImageRef(image_id="img1", user_id="u1", request_id="", related_event_id="e1", local_path="")
    
    ctx = RequestContext(
        request_id="r1", user_id="u1", request_date=date(2024, 1, 1),
        request_type="purchase", requested_amount=100.0,
        desired_completion_date=None, allows_partial_payment=False,
        request_text="", profile=None, resolved_events=[evt],
        payment_options=[], messages_for_user=[], messages_for_request=[],
        messages_for_events={"e1": [msg]}, images_for_user=[], images_for_request=[],
        images_for_events={"e1": [img]}
    )
    
    ev = get_relevant_evidence(ctx)
    assert len(ev.messages) == 1
    assert ev.messages[0].message_id == "m1"
    assert len(ev.images) == 1
    assert ev.images[0].image_id == "img1"

def test_request_specific_evidence():
    msg = Message(message_id="m1", user_id="u1", request_id="r1", related_event_id="", sent_at="2024-01-01", source_type="email", message_text="hello")
    ctx = RequestContext(
        request_id="r1", user_id="u1", request_date=date(2024, 1, 1),
        request_type="purchase", requested_amount=100.0,
        desired_completion_date=None, allows_partial_payment=False,
        request_text="", profile=None, resolved_events=[],
        payment_options=[], messages_for_user=[], messages_for_request=[msg],
        messages_for_events={}, images_for_user=[], images_for_request=[],
        images_for_events={}
    )
    ev = get_relevant_evidence(ctx)
    assert len(ev.messages) == 1

def test_irrelevant_user_message_ignored():
    msg = Message(message_id="m1", user_id="u1", request_id="", related_event_id="", sent_at="2024-01-01", source_type="email", message_text="hello")
    ctx = RequestContext(
        request_id="r1", user_id="u1", request_date=date(2024, 1, 1),
        request_type="purchase", requested_amount=100.0,
        desired_completion_date=None, allows_partial_payment=False,
        request_text="", profile=None, resolved_events=[],
        payment_options=[], messages_for_user=[msg], messages_for_request=[],
        messages_for_events={}, images_for_user=[], images_for_request=[],
        images_for_events={}
    )
    ev = get_relevant_evidence(ctx)
    assert len(ev.messages) == 0
