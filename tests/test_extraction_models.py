"""
tests/test_extraction_models.py
"""
from src.extraction.models import ExtractionFact, ExtractionResult

def test_extraction_fact_creation():
    f = ExtractionFact(
        event_id="e1",
        fact_type="amount_correction",
        value=100.0,
        source_id="img1",
        source_type="image",
        confidence=0.9
    )
    assert f.event_id == "e1"
    assert f.fact_type == "amount_correction"

def test_extraction_result_creation():
    r = ExtractionResult(errors=["err1"])
    assert r.errors == ["err1"]
    assert len(r.facts) == 0
