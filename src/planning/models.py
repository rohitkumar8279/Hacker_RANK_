"""
src/planning/models.py
----------------------
Data models for Stage 4 (Payment Planning and Ranking).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple

from src.finance.forecast import SpendingChange
from src.finance.safety import SafetyResult


@dataclass
class CandidatePlan:
    """A proposed payment schedule to evaluate."""
    payment_method: str                      # full_payment | partial_payment | installments | wait
    payments: List[Tuple[date, float]]       # list of (payment_date, amount)
    total_amount: float                      # total sum paid across all payments
    payment_option_id: Optional[str] = None  # None unless using an explicit option from request_payment_options.csv
    spending_changes: Optional[List[SpendingChange]] = None

    # Populated after validation
    is_safe: bool = False
    safety_result: Optional[SafetyResult] = None

    @property
    def completion_date(self) -> Optional[date]:
        """Date of the final payment in the plan."""
        if not self.payments:
            return None
        return max(d for d, _ in self.payments)

    @property
    def first_payment_date(self) -> Optional[date]:
        """Date of the first payment in the plan."""
        if not self.payments:
            return None
        return min(d for d, _ in self.payments)
        
    @property
    def number_of_payments(self) -> int:
        return len(self.payments)


@dataclass
class PlanResult:
    """The final selected plan and its associated metadata."""
    request_id: str
    amount_safe_to_pay: float
    affordability_status: str                # affordable_now | affordable_with_plan | affordable_later | not_affordable
    recommended_payment_method: str          # full_payment | partial_payment | installments | wait | not_recommended
    payments: List[Tuple[date, float]]
    earliest_date_for_full_payment: Optional[date]
    spending_changes: Optional[List[SpendingChange]] = None
    selected_payment_option_id: Optional[str] = None
    total_amount_paid: float = 0.0
    safety_result: Optional[SafetyResult] = None
