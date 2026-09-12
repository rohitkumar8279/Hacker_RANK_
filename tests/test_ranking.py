"""
tests/test_ranking.py
---------------------
Tests for deterministic plan ranking.
"""

from __future__ import annotations

from datetime import date

from src.planning.models import CandidatePlan
from src.planning.ranking import rank_plans
from src.finance.forecast import SpendingChange


def test_ranking_empty():
    assert rank_plans([]) is None


def test_ranking_completion_date():
    # p1 completes earlier than p2
    p1 = CandidatePlan("full_payment", [(date(2024, 1, 1), 100)], 100)
    p2 = CandidatePlan("full_payment", [(date(2024, 1, 2), 100)], 100)
    
    # Actually priority 1 is complete by desired_completion_date
    # the ranking function currently uses completion_date ascending
    assert rank_plans([p2, p1]) == p1


def test_ranking_spending_changes():
    # p1 requires no changes, p2 requires changes
    p1 = CandidatePlan("full_payment", [(date(2024, 1, 1), 100)], 100)
    p2 = CandidatePlan(
        "full_payment", 
        [(date(2024, 1, 1), 100)], 
        100, 
        spending_changes=[SpendingChange("e1", "stop", None)]
    )
    
    assert rank_plans([p2, p1]) == p1


def test_ranking_total_cost():
    # p1 cost < p2 cost
    p1 = CandidatePlan("installments", [(date(2024, 1, 1), 50), (date(2024, 2, 1), 50)], 100)
    p2 = CandidatePlan("installments", [(date(2024, 1, 1), 50), (date(2024, 2, 1), 55)], 105)
    
    assert rank_plans([p2, p1]) == p1


def test_ranking_start_date():
    # p1 starts earlier
    p1 = CandidatePlan("installments", [(date(2024, 1, 1), 50), (date(2024, 2, 1), 50)], 100)
    p2 = CandidatePlan("installments", [(date(2024, 1, 2), 50), (date(2024, 2, 2), 50)], 100)
    
    assert rank_plans([p2, p1]) == p1


def test_ranking_number_of_payments():
    # p1 has fewer payments
    p1 = CandidatePlan("partial_payment", [(date(2024, 1, 1), 100)], 100)
    p2 = CandidatePlan("installments", [(date(2024, 1, 1), 50), (date(2024, 1, 1), 50)], 100)
    
    assert rank_plans([p2, p1]) == p1


def test_ranking_payment_option_id():
    # p1 has lower option id string
    p1 = CandidatePlan("installments", [(date(2024, 1, 1), 100)], 100, payment_option_id="opt1")
    p2 = CandidatePlan("installments", [(date(2024, 1, 1), 100)], 100, payment_option_id="opt2")
    
    assert rank_plans([p2, p1]) == p1
