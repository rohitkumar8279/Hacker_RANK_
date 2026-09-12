"""
tests/test_message_extractor.py
"""
import pytest
from src.extraction.message_extractor import extract_messages
from src.data.join import Message
import config

def test_extract_messages_offline(monkeypatch):
    monkeypatch.setattr(config, "LLM_ENABLED", False)
    msg = Message(message_id="m1", user_id="u1", request_id="", related_event_id="", sent_at="2024-01-01", source_type="email", message_text="hello")
    res = extract_messages([msg])
    assert len(res.facts) == 0
    assert "extraction unavailable (offline or missing key)" in res.errors[0]

def test_extract_messages_empty():
    res = extract_messages([])
    assert len(res.facts) == 0
    assert len(res.errors) == 0
