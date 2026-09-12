"""
src/pipeline.py
---------------
Deterministic orchestration layer for processing a single request.
"""

import math
from typing import Tuple

from src.data.join import RequestContext
from src.extraction.extractor import extract_facts_from_context, enrich_events
from src.extraction.models import ExtractionResult
from src.planning.plan_builder import build_plan
from src.planning.models import PlanResult


def process_request(context: RequestContext) -> Tuple[PlanResult, ExtractionResult]:
    """
    Process a single request end-to-end, isolating extraction and planning logic.
    """
    try:
        # 1. Extraction
        extraction_res = extract_facts_from_context(context)
        
        # 2. Enrichment
        all_events = context.resolved_events + context.unresolved_events
        enriched_events = enrich_events(all_events, extraction_res.facts)
        
        # 3. Filter fully resolved events for the deterministic engine
        # Any event that STILL has amount=NaN after extraction must be skipped 
        # (or it would cause safety/planner to crash/produce NaN bounds).
        final_resolved = []
        for evt in enriched_events:
            if evt.amount is not None and not math.isnan(evt.amount):
                # Ensure status is not cancelled if we want to forecast it, 
                # but forecast.py might handle 'cancelled' status itself.
                # Actually, forecast.py skips events if not 'pending' or 'scheduled' 
                # for future, so keeping 'cancelled' events in the list is fine.
                final_resolved.append(evt)
                
        # 4. Inject enriched, resolved events back into context
        context.resolved_events = final_resolved
        
        # 5. Plan Generation
        plan_res = build_plan(context)
        
        return plan_res, extraction_res
        
    except Exception as e:
        # In case of total failure, we return a fallback plan result 
        # to ensure one request failing doesn't crash the pipeline.
        err_plan = PlanResult(
            request_id=context.request_id,
            amount_safe_to_pay=0.0,
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payments=[],
            earliest_date_for_full_payment=None,
            spending_changes=[]
        )
        # Note: If extraction itself threw an exception instead of returning safely, we catch it here.
        err_ext = ExtractionResult(errors=[f"Pipeline crash: {str(e)}"])
        return err_plan, err_ext
