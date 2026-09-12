"""
src/extraction/models.py
------------------------
Data models for Stage 5 (Evidence Extraction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, List, Optional


@dataclass
class ExtractionFact:
    event_id: str
    fact_type: str  # amount_correction | cancellation | amendment | settlement | delay
    value: Any
    source_id: str
    source_type: str  # message | image
    confidence: float
    currency: Optional[str] = None
    effective_date: Optional[date] = None


@dataclass
class ExtractionUsage:
    provider: str
    model: str
    operation: str
    source_type: str
    source_id: str
    request_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: Optional[float] = None


@dataclass
class ExtractionResult:
    facts: List[ExtractionFact] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    usage: Optional[ExtractionUsage] = None

