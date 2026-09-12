import pytest
from datetime import date
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_pipeline import format_earliest_date
from src.planning.models import PlanResult

def test_format_earliest_date_suppression():
    # Test P2: Suppress earliest_date for not_affordable
    
    # 1. Affordable later: should output date
    pr1 = PlanResult(
        request_id="r1", amount_safe_to_pay=0, affordability_status="affordable_later",
        recommended_payment_method="wait", payments=[], 
        earliest_date_for_full_payment=date(2025, 1, 1)
    )
    assert format_earliest_date(pr1) == "2025-01-01"
    
    # 2. Not affordable: should suppress date
    pr2 = PlanResult(
        request_id="r2", amount_safe_to_pay=0, affordability_status="not_affordable",
        recommended_payment_method="not_recommended", payments=[], 
        earliest_date_for_full_payment=date(2025, 1, 1)
    )
    assert format_earliest_date(pr2) == ""
    
    # 3. None date: should be empty
    pr3 = PlanResult(
        request_id="r3", amount_safe_to_pay=0, affordability_status="affordable_now",
        recommended_payment_method="full_payment", payments=[], 
        earliest_date_for_full_payment=None
    )
    assert format_earliest_date(pr3) == ""
