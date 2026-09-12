"""
tests/test_image_extractor.py
"""
import pytest
from src.extraction.image_extractor import extract_image
from src.data.join import ImageRef
import config

def test_extract_image_offline(monkeypatch):
    monkeypatch.setattr(config, "LLM_ENABLED", False)
    img = ImageRef(image_id="img1", user_id="u1", request_id="", related_event_id="", local_path="")
    res = extract_image(img)
    assert len(res.facts) == 0
    assert "extraction unavailable (offline or missing key)" in res.errors[0]

def test_extract_image_not_found(monkeypatch):
    monkeypatch.setattr(config, "LLM_ENABLED", True)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake")
    import src.extraction.image_extractor as ie
    monkeypatch.setattr(ie, "genai", True)
    img = ImageRef(image_id="does_not_exist", user_id="u1", request_id="", related_event_id="", local_path="")
    res = extract_image(img)
    assert len(res.facts) == 0
    assert "Image file not found" in res.errors[0]
