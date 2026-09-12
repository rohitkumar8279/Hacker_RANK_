"""
src/extraction/extractor.py
---------------------------
Orchestrator for the extraction pipeline.
"""

from __future__ import annotations

import math
from datetime import date
from typing import List, Dict, Tuple
from copy import deepcopy

from src.data.join import RequestContext
from src.finance.events import FinancialEvent
from src.extraction.models import ExtractionFact, ExtractionResult, ExtractionUsage
from src.extraction.relevance import get_relevant_evidence
from src.extraction.message_extractor import extract_messages
from src.extraction.image_extractor import extract_image


# Global in-memory cache for the run
_MESSAGE_CACHE: Dict[str, ExtractionResult] = {}
_IMAGE_CACHE: Dict[str, ExtractionResult] = {}

# Global pre-extraction statistics
_PRE_EXTRACTION_STATS = {
    "enabled": False,
    "unique_messages": 0,
    "unique_images": 0,
    "message_batches": 0,
    "image_batches": 0,
    "invalid_attribution_count": 0,
    "cache_hits": 0,
    "cache_misses": 0
}


def clear_extraction_cache():
    _MESSAGE_CACHE.clear()
    _IMAGE_CACHE.clear()
    _PRE_EXTRACTION_STATS.update({
        "enabled": False,
        "unique_messages": 0,
        "unique_images": 0,
        "message_batches": 0,
        "image_batches": 0,
        "invalid_attribution_count": 0,
        "cache_hits": 0,
        "cache_misses": 0
    })

def global_pre_extract(all_request_ids: List[str], builder) -> None:
    """
    Globally pre-extracts facts from all relevant messages and images across the dataset.
    This minimizes Gemini API calls by batching evidence globally instead of per-request.
    """
    import config
    from src.extraction.message_extractor import extract_messages
    from src.extraction.image_extractor import extract_images
    
    unique_messages = {}
    unique_images = {}
    
    # 1. Gather all unique relevant evidence
    for rid in all_request_ids:
        ctx = builder.build(rid)
        evidence = get_relevant_evidence(ctx)
        for msg in evidence.messages:
            unique_messages[msg.message_id] = msg
        for img in evidence.images:
            unique_images[img.image_id] = img
            
    msgs_list = list(unique_messages.values())
    imgs_list = list(unique_images.values())
    
    _PRE_EXTRACTION_STATS["enabled"] = True
    _PRE_EXTRACTION_STATS["unique_messages"] = len(msgs_list)
    _PRE_EXTRACTION_STATS["unique_images"] = len(imgs_list)
    
    # 2. Batch messages (target 2 batches for this dataset size, meaning ~60-100 per batch)
    msg_batch_size = getattr(config, "MESSAGE_BATCH_SIZE", 80)
    for i in range(0, len(msgs_list), msg_batch_size):
        batch = msgs_list[i:i+msg_batch_size]
        if not batch: continue
        _PRE_EXTRACTION_STATS["message_batches"] += 1
        
        batch_res = extract_messages(batch)
        valid_source_ids = {m.message_id for m in batch}
        
        # Populate cache individually per message
        for msg in batch:
            msg_facts = []
            for f in batch_res.facts:
                if f.source_id == msg.message_id:
                    msg_facts.append(f)
                elif f.source_id not in valid_source_ids:
                    # Invalid attribution
                    _PRE_EXTRACTION_STATS["invalid_attribution_count"] += 1
                    
            _MESSAGE_CACHE[msg.message_id] = ExtractionResult(
                facts=msg_facts,
                errors=batch_res.errors,
                usage=batch_res.usage if msg == batch[-1] else None # Only attribute usage once per batch to avoid duplicate counts in reporting
            )

    # 3. Batch images (target 1 batch for small numbers, safely capped)
    img_batch_size = getattr(config, "IMAGE_BATCH_SIZE", 15)
    for i in range(0, len(imgs_list), img_batch_size):
        batch = imgs_list[i:i+img_batch_size]
        if not batch: continue
        _PRE_EXTRACTION_STATS["image_batches"] += 1
        
        batch_res = extract_images(batch)
        valid_source_ids = {img.image_id for img in batch}
        
        # Populate cache individually per image
        for img in batch:
            img_facts = []
            for f in batch_res.facts:
                if f.source_id == img.image_id:
                    img_facts.append(f)
                elif f.source_id not in valid_source_ids:
                    # Invalid attribution
                    _PRE_EXTRACTION_STATS["invalid_attribution_count"] += 1
                    
            _IMAGE_CACHE[img.image_id] = ExtractionResult(
                facts=img_facts,
                errors=batch_res.errors,
                usage=batch_res.usage if img == batch[-1] else None
            )


def validate_fact(fact: ExtractionFact, context: RequestContext, valid_msg_ids: set, valid_img_ids: set) -> bool:
    """
    Deterministically validate a fact returned by the LLM.
    """
    # Source type
    if fact.source_type not in ("message", "image"):
        return False
        
    # Source ID
    if fact.source_type == "message" and fact.source_id not in valid_msg_ids:
        return False
    if fact.source_type == "image" and fact.source_id not in valid_img_ids:
        return False
        
    # Fact type
    valid_types = {"amount_correction", "cancellation", "amendment", "settlement", "delay"}
    if fact.fact_type not in valid_types:
        return False
        
    # Event ID
    if fact.event_id:
        all_event_ids = {e.event_id for e in context.resolved_events + context.unresolved_events}
        if fact.event_id not in all_event_ids:
            return False

    # Amount validation
    if fact.value is not None:
        try:
            val = float(fact.value)
            if val < 0:
                return False
        except (ValueError, TypeError):
            return False
            
    # Currency
    if fact.currency:
        fact.currency = fact.currency.upper()
        if fact.currency not in {"INR", "ZAR", "IDR", "USD", "EUR"}:
            return False
            
    # Date validation
    if fact.effective_date:
        if isinstance(fact.effective_date, str):
            try:
                date.fromisoformat(fact.effective_date)
            except ValueError:
                return False
                
    return True


def extract_facts_from_context(context: RequestContext) -> ExtractionResult:
    """
    Identify evidence, check cache, call extractors, and validate.
    """
    evidence = get_relevant_evidence(context)
    
    if not evidence.messages and not evidence.images:
        return ExtractionResult()

    valid_msg_ids = {m.message_id for m in evidence.messages}
    valid_img_ids = {i.image_id for i in evidence.images}
    
    all_facts = []
    all_errors = []
    
    # 1. Image Extraction (Individual calls, cached)
    for img in evidence.images:
        if img.image_id in _IMAGE_CACHE:
            _PRE_EXTRACTION_STATS["cache_hits"] += 1
            res = _IMAGE_CACHE[img.image_id]
        else:
            _PRE_EXTRACTION_STATS["cache_misses"] += 1
            res = extract_image(img)
            _IMAGE_CACHE[img.image_id] = res
            
        all_facts.extend(res.facts)
        all_errors.extend(res.errors)

    # 2. Message Extraction (Batch uncached messages)
    uncached_messages = []
    for msg in evidence.messages:
        if msg.message_id in _MESSAGE_CACHE:
            _PRE_EXTRACTION_STATS["cache_hits"] += 1
            res = _MESSAGE_CACHE[msg.message_id]
            all_facts.extend(res.facts)
            all_errors.extend(res.errors)
        else:
            _PRE_EXTRACTION_STATS["cache_misses"] += 1
            uncached_messages.append(msg)
            
    if uncached_messages:
        # Batch call
        batch_res = extract_messages(uncached_messages)
        all_facts.extend(batch_res.facts)
        all_errors.extend(batch_res.errors)
        # Store in cache using individual message ids
        # (Though we can't easily map which fact came from which, the cache design 
        # for batches means we might just cache the whole batch result for all its msg IDs)
        for msg in uncached_messages:
            msg_facts = [f for f in batch_res.facts if f.source_id == msg.message_id]
            _MESSAGE_CACHE[msg.message_id] = ExtractionResult(
                facts=msg_facts,
                errors=batch_res.errors,
                usage=batch_res.usage
            )

    # 3. Validation
    validated_facts = []
    for fact in all_facts:
        if validate_fact(fact, context, valid_msg_ids, valid_img_ids):
            validated_facts.append(fact)
            
    # 4. Resolve simple conflicts (highest confidence or explicit rules)
    # Group by event_id + fact_type
    grouped = {}
    for fact in validated_facts:
        key = (fact.event_id, fact.fact_type)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(fact)
        
    final_facts = []
    for key, facts in grouped.items():
        # Prefer message over image for the same fact type (messages are explicit instructions)
        msgs = [f for f in facts if f.source_type == "message"]
        imgs = [f for f in facts if f.source_type == "image"]
        
        candidates = msgs if msgs else imgs
        # Pick the one with highest confidence
        best_fact = sorted(candidates, key=lambda f: f.confidence, reverse=True)[0]
        final_facts.append(best_fact)
        
    return ExtractionResult(facts=final_facts, errors=all_errors)


def enrich_events(events: List[FinancialEvent], facts: List[ExtractionFact]) -> List[FinancialEvent]:
    """
    Applies extracted facts to the events deterministically.
    Returns a new list of events (does not mutate original).
    """
    # Make a deep copy to avoid mutating original state
    enriched_events = [deepcopy(e) for e in events]
    event_map = {e.event_id: e for e in enriched_events}
    
    for fact in facts:
        evt = event_map.get(fact.event_id)
        if not evt:
            continue
            
        if fact.fact_type == "amount_correction":
            # Only apply if it's NaN or None. We do not blindly overwrite known amounts.
            if evt.amount is None or (isinstance(evt.amount, float) and math.isnan(evt.amount)):
                evt.amount = float(fact.value)
                
        elif fact.fact_type == "cancellation":
            evt.status = "cancelled"
            
        elif fact.fact_type == "amendment":
            if fact.value is not None:
                evt.amount = float(fact.value)
                
        elif fact.fact_type == "settlement":
            evt.status = "settled"
            
        elif fact.fact_type == "delay":
            if fact.effective_date:
                d = fact.effective_date
                if isinstance(d, str):
                    d = date.fromisoformat(d)
                evt.event_date = d
                evt.settlement_date = d
                
    return enriched_events
