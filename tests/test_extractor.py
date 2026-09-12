"""
tests/test_extractor.py
"""
import pytest
from datetime import date
import math
from src.extraction.extractor import validate_fact, enrich_events
from src.extraction.models import ExtractionFact
from src.data.join import RequestContext
from src.finance.events import FinancialEvent

def test_validate_fact():
    ctx = RequestContext(
        request_id="r1", user_id="u1", request_date=date(2024, 1, 1),
        request_type="purchase", requested_amount=100.0,
        desired_completion_date=None, allows_partial_payment=False,
        request_text="", profile=None, resolved_events=[
            FinancialEvent(event_id="e1", user_id="u1", event_type="expense",
                           description="", category="", direction="debit",
                           amount=10.0, home_currency="ZAR", event_date=date(2024, 1, 1),
                           settlement_date=date(2024, 1, 1), status="pending",
                           linked_event_id="", flexibility="fixed", minimum_allowed_amount=None)
        ],
        payment_options=[], messages_for_user=[], messages_for_request=[],
        messages_for_events={}, images_for_user=[], images_for_request=[],
        images_for_events={}
    )
    
    valid_msgs = {"m1"}
    valid_imgs = {"img1"}
    
    # Valid fact
    f1 = ExtractionFact(event_id="e1", fact_type="amount_correction", value=20.0, source_id="m1", source_type="message", confidence=0.9, currency="ZAR", effective_date="2024-01-01")
    assert validate_fact(f1, ctx, valid_msgs, valid_imgs) == True
    
    # Invalid fact type
    f2 = ExtractionFact(event_id="e1", fact_type="approve_transaction", value=None, source_id="m1", source_type="message", confidence=0.9)
    assert validate_fact(f2, ctx, valid_msgs, valid_imgs) == False
    
    # Invalid source id
    f3 = ExtractionFact(event_id="e1", fact_type="amount_correction", value=20.0, source_id="m2", source_type="message", confidence=0.9)
    assert validate_fact(f3, ctx, valid_msgs, valid_imgs) == False
    
    # Invalid event id
    f4 = ExtractionFact(event_id="e2", fact_type="amount_correction", value=20.0, source_id="m1", source_type="message", confidence=0.9)
    assert validate_fact(f4, ctx, valid_msgs, valid_imgs) == False
    
    # Invalid amount (negative)
    f5 = ExtractionFact(event_id="e1", fact_type="amount_correction", value=-20.0, source_id="m1", source_type="message", confidence=0.9)
    assert validate_fact(f5, ctx, valid_msgs, valid_imgs) == False
    
    # Invalid currency
    f6 = ExtractionFact(event_id="e1", fact_type="amount_correction", value=20.0, source_id="m1", source_type="message", confidence=0.9, currency="ABC")
    assert validate_fact(f6, ctx, valid_msgs, valid_imgs) == False

    # Valid lowercase currency (P1-B)
    f_currency_lower = ExtractionFact(event_id="e1", fact_type="amount_correction", value=20.0, source_id="m1", source_type="message", confidence=0.9, currency="zar")
    assert validate_fact(f_currency_lower, ctx, valid_msgs, valid_imgs) == True
    assert f_currency_lower.currency == "ZAR"
    
    # Invalid date
    f7 = ExtractionFact(event_id="e1", fact_type="delay", value=None, source_id="m1", source_type="message", confidence=0.9, effective_date="not-a-date")
    assert validate_fact(f7, ctx, valid_msgs, valid_imgs) == False

def test_enrich_events():
    e1 = FinancialEvent(event_id="e1", user_id="u1", event_type="expense",
                           description="", category="", direction="debit",
                           amount=float('nan'), home_currency="ZAR", event_date=date(2024, 1, 1),
                           settlement_date=date(2024, 1, 1), status="pending",
                           linked_event_id="", flexibility="fixed", minimum_allowed_amount=None)
                           
    e2 = FinancialEvent(event_id="e2", user_id="u1", event_type="expense",
                           description="", category="", direction="debit",
                           amount=50.0, home_currency="ZAR", event_date=date(2024, 1, 1),
                           settlement_date=date(2024, 1, 1), status="scheduled",
                           linked_event_id="", flexibility="fixed", minimum_allowed_amount=None)
                           
    facts = [
        ExtractionFact(event_id="e1", fact_type="amount_correction", value=100.0, source_id="img1", source_type="image", confidence=0.9),
        ExtractionFact(event_id="e2", fact_type="cancellation", value=None, source_id="m1", source_type="message", confidence=0.9),
    ]
    
    enriched = enrich_events([e1, e2], facts)
    
    # Check mutations on copies
    assert math.isnan(e1.amount)
    assert e2.status == "scheduled"
    
    assert enriched[0].amount == 100.0
    assert enriched[1].status == "cancelled"
    
def test_enrich_events_ignores_known_amounts():
    e1 = FinancialEvent(event_id="e1", user_id="u1", event_type="expense",
                           description="", category="", direction="debit",
                           amount=50.0, home_currency="ZAR", event_date=date(2024, 1, 1),
                           settlement_date=date(2024, 1, 1), status="pending",
                           linked_event_id="", flexibility="fixed", minimum_allowed_amount=None)
                           
    facts = [
        ExtractionFact(event_id="e1", fact_type="amount_correction", value=100.0, source_id="img1", source_type="image", confidence=0.9),
    ]
    
    enriched = enrich_events([e1], facts)
    
    # Should not overwrite because it's an amount_correction on a known amount (50.0)
    assert enriched[0].amount == 50.0

def test_message_cache_isolation():
    from src.extraction.extractor import extract_facts_from_context, clear_extraction_cache, _MESSAGE_CACHE
    from src.data.join import Message
    
    clear_extraction_cache()
    
    # We mock extract_messages internally or just observe behavior when it's not actually running LLMs 
    # since LLM_ENABLED might be false in tests or we can inject fake batch results.
    # However, since we're in an isolated environment without LLM_ENABLED, extract_messages returns empty/error.
    # Let's directly populate the cache simulating a batch extraction of [m1, m2]
    
    from src.extraction.models import ExtractionResult, ExtractionUsage
    
    # Simulate batch result containing facts for m1 and m2
    batch_res = ExtractionResult(
        facts=[
            ExtractionFact(event_id="e1", fact_type="cancellation", value=None, source_id="m1", source_type="message", confidence=0.9),
            ExtractionFact(event_id="e2", fact_type="amendment", value=50.0, source_id="m2", source_type="message", confidence=0.9)
        ]
    )
    
    # The actual bug was in extract_facts_from_context, so we need to mock extract_messages and call it.
    import src.extraction.extractor
    original_extract_messages = src.extraction.extractor.extract_messages
    src.extraction.extractor.extract_messages = lambda msgs: batch_res
    
    ctx = RequestContext(
        request_id="r1", user_id="u1", request_date=date(2024, 1, 1),
        request_type="purchase", requested_amount=100.0,
        desired_completion_date=None, allows_partial_payment=False,
        request_text="", profile=None, resolved_events=[],
        payment_options=[], messages_for_user=[], messages_for_request=[],
        messages_for_events={}, images_for_user=[], images_for_request=[],
        images_for_events={}
    )
    
    m1 = Message(message_id="m1", user_id="u1", request_id="r1", related_event_id="e1", sent_at="2024-01-01T00:00:00Z", source_type="employer", message_text="cancel")
    m2 = Message(message_id="m2", user_id="u1", request_id="r1", related_event_id="e2", sent_at="2024-01-01T00:00:00Z", source_type="employer", message_text="amend")
    
    # Mock evidence
    import src.extraction.relevance
    class DummyEvidence:
        messages = [m1, m2]
        images = []
    
    original_get_evidence = src.extraction.extractor.get_relevant_evidence
    src.extraction.extractor.get_relevant_evidence = lambda c: DummyEvidence()
    
    # First call caches both
    extract_facts_from_context(ctx)
    
    # Check cache isolation (P1-A)
    assert "m1" in _MESSAGE_CACHE
    assert "m2" in _MESSAGE_CACHE
    
    # m1 should ONLY have m1's fact
    m1_cached_facts = _MESSAGE_CACHE["m1"].facts
    assert len(m1_cached_facts) == 1
    assert m1_cached_facts[0].source_id == "m1"
    
    # m2 should ONLY have m2's fact
    m2_cached_facts = _MESSAGE_CACHE["m2"].facts
    assert len(m2_cached_facts) == 1
    assert m2_cached_facts[0].source_id == "m2"
    
    # Restore mocks
    src.extraction.extractor.extract_messages = original_extract_messages
    src.extraction.extractor.get_relevant_evidence = original_get_evidence

