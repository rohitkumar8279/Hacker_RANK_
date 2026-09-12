"""
src/planning/ranking.py
-----------------------
Deterministic ranking of safe, eligible candidate plans.
"""

from __future__ import annotations

from typing import List, Optional

from src.planning.models import CandidatePlan


def rank_plans(plans: List[CandidatePlan]) -> Optional[CandidatePlan]:
    """
    Ranks eligible safe plans and returns the best one.
    Returns None if no plans are provided.

    Priority 1: Complete by desired_completion_date (should be pre-filtered, but safe to sort).
    Priority 2: Require no spending changes.
    Priority 3: Minimize total amount paid.
    Priority 4: Start payment earlier.
    Priority 5: Use fewer payments.
    Priority 6: Lowest payment_option_id (for tie-breaking installments).
    """
    if not plans:
        return None

    def sort_key(plan: CandidatePlan):
        # Priority 1: Completion date (we just need them to be sorted ascending if there are ties
        # but realistically all plans passed in should already meet the deadline).
        # We use a large default date if somehow completion_date is None (shouldn't happen for valid plans).
        from datetime import date
        comp_date = plan.completion_date or date.max

        # Priority 2: Require no spending changes (True sorts before False, so we use integer 0/1)
        has_spending_changes = 1 if plan.spending_changes else 0

        # Priority 3: Minimize total amount paid
        total_cost = plan.total_amount

        # Priority 4: Start payment earlier
        first_date = plan.first_payment_date or date.max

        # Priority 5: Use fewer payments
        num_payments = plan.number_of_payments

        # Priority 6: Lowest payment_option_id (empty string sorts before anything)
        opt_id = plan.payment_option_id or ""

        return (
            has_spending_changes,
            total_cost,
            first_date,
            num_payments,
            opt_id,
        )

    return sorted(plans, key=sort_key)[0]
