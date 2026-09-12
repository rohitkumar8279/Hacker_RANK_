import pytest
from src.extraction.extractor import global_pre_extract, _MESSAGE_CACHE, _IMAGE_CACHE, clear_extraction_cache, _PRE_EXTRACTION_STATS
from src.extraction.models import ExtractionResult, ExtractionFact
from src.data.join import Message, ImageRef
import config

class MockEvidence:
    def __init__(self, messages, images):
        self.messages = messages
        self.images = images

class MockContext:
    def __init__(self, request_id, messages, images):
        self.request_id = request_id
        self.messages = messages
        self.images = images
        
class MockBuilder:
    def __init__(self, contexts):
        self.contexts = {c.request_id: c for c in contexts}
        
    def build(self, request_id):
        return self.contexts[request_id]


def mock_get_relevant_evidence(ctx):
    return MockEvidence(ctx.messages, ctx.images)

def mock_extract_messages(messages):
    facts = []
    for m in messages:
        if m.message_text == "error":
            continue
        facts.append(ExtractionFact(source_id=m.message_id, source_type="message", event_id="e1", fact_type="cancellation", value=None, confidence=1.0))
        if m.message_id == "msg_1":
            facts.append(ExtractionFact(source_id="invalid_msg_id", source_type="message", event_id="e1", fact_type="cancellation", value=None, confidence=1.0))
    return ExtractionResult(facts=facts)

def mock_extract_images(images):
    facts = []
    for i in images:
        facts.append(ExtractionFact(source_id=i.image_id, source_type="image", event_id="e1", fact_type="amount_correction", value=100.0, confidence=1.0))
        if i.image_id == "img_1":
            facts.append(ExtractionFact(source_id="invalid_img_id", source_type="image", event_id="e1", fact_type="amount_correction", value=99.0, confidence=1.0))
    return ExtractionResult(facts=facts)


def test_global_pre_extract_deduplicates_and_batches(monkeypatch):
    monkeypatch.setattr("src.extraction.extractor.get_relevant_evidence", mock_get_relevant_evidence)
    monkeypatch.setattr("src.extraction.message_extractor.extract_messages", mock_extract_messages)
    monkeypatch.setattr("src.extraction.image_extractor.extract_images", mock_extract_images)
    clear_extraction_cache()
    
    # 3 requests, but overlapping messages
    m1 = Message("m1", "u1", "r1", "e1", "2024-01-01", "employer", "cancel")
    m2 = Message("m2", "u1", "r1", "e1", "2024-01-01", "employer", "cancel")
    m3 = Message("m1", "u1", "r2", "e1", "2024-01-01", "employer", "cancel") # m1 reused
    
    i1 = ImageRef("i1", "u1", "r1", "e1", "path/to/i1.png")
    i2 = ImageRef("i1", "u1", "r2", "e1", "path/to/i1.png") # i1 reused
    
    contexts = [
        MockContext("r1", [m1, m2], [i1]),
        MockContext("r2", [m3], [i2])
    ]
    
    builder = MockBuilder(contexts)
    config.MESSAGE_BATCH_SIZE = 1
    config.IMAGE_BATCH_SIZE = 1
    
    global_pre_extract(["r1", "r2"], builder)
    
    assert _PRE_EXTRACTION_STATS["unique_messages"] == 2
    assert _PRE_EXTRACTION_STATS["unique_images"] == 1
    
    # Check caches populated
    assert "m1" in _MESSAGE_CACHE
    assert "m2" in _MESSAGE_CACHE
    assert "i1" in _IMAGE_CACHE
    
    # Check invalid attribution rejected
    assert _PRE_EXTRACTION_STATS["invalid_attribution_count"] == 0

def test_global_pre_extract_invalid_attribution(monkeypatch):
    monkeypatch.setattr("src.extraction.extractor.get_relevant_evidence", mock_get_relevant_evidence)
    monkeypatch.setattr("src.extraction.message_extractor.extract_messages", mock_extract_messages)
    monkeypatch.setattr("src.extraction.image_extractor.extract_images", mock_extract_images)

    clear_extraction_cache()
    
    m1 = Message("msg_1", "u1", "r1", "e1", "2024-01-01", "employer", "cancel")
    i1 = ImageRef("img_1", "u1", "r1", "e1", "path/to/img_1.png")
    
    builder = MockBuilder([MockContext("r1", [m1], [i1])])
    global_pre_extract(["r1"], builder)
    
    # The mocks return "invalid_msg_id" and "invalid_img_id". These should be logged as invalid.
    assert _PRE_EXTRACTION_STATS["invalid_attribution_count"] == 2
    
    # The invalid facts should NOT be assigned to msg_1 or img_1
    msg_facts = _MESSAGE_CACHE["msg_1"].facts
    assert len(msg_facts) == 1
    assert msg_facts[0].source_id == "msg_1"
    
    img_facts = _IMAGE_CACHE["img_1"].facts
    assert len(img_facts) == 1
    assert img_facts[0].source_id == "img_1"

def test_global_pre_extract_no_cross_contamination(monkeypatch):
    monkeypatch.setattr("src.extraction.extractor.get_relevant_evidence", mock_get_relevant_evidence)
    monkeypatch.setattr("src.extraction.message_extractor.extract_messages", mock_extract_messages)
    monkeypatch.setattr("src.extraction.image_extractor.extract_images", mock_extract_images)

    clear_extraction_cache()
    # Batch size 10 to ensure they go in same batch
    config.MESSAGE_BATCH_SIZE = 10
    
    m1 = Message("m1", "u1", "r1", "e1", "2024-01-01", "employer", "cancel")
    m2 = Message("m2", "u1", "r1", "e1", "2024-01-01", "employer", "cancel")
    
    builder = MockBuilder([MockContext("r1", [m1, m2], [])])
    global_pre_extract(["r1"], builder)
    
    # Ensure m1 cache ONLY has m1's facts, not m2's
    assert len(_MESSAGE_CACHE["m1"].facts) == 1
    assert _MESSAGE_CACHE["m1"].facts[0].source_id == "m1"
    
    assert len(_MESSAGE_CACHE["m2"].facts) == 1
    assert _MESSAGE_CACHE["m2"].facts[0].source_id == "m2"
