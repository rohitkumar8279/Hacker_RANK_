"""
src/finance/safety.py
---------------------
Deterministic financial safety calculations.

This module answers the core questions the rest of the pipeline needs:

  1. amount_safe_to_pay(forecast, requested_amount) → float
     The maximum amount safe to pay on request_date.

  2. earliest_safe_date(forecast, requested_amount) → Optional[date]
     The earliest date in the 90-day window on which requested_amount
     can be paid in full without breaching minimum_balance_to_keep.

  3. is_plan_safe(forecast, payments) → SafetyResult
     Given a list of (date, amount) payment pairs, verify the plan is safe
     across the entire 90-day window.

KEY INVARIANT (from spec):
  A plan is safe only if balance(d) >= minimum_balance_to_keep for EVERY day d
  in the forecast window, after subtracting all planned payments.

ALGORITHM (amount_safe_to_pay):
  Because the forecast is already built without the payment, we compute:
    headroom = min_future_balance_from(request_date) - minimum_balance_to_keep
  where min_future_balance_from is the lowest closing balance on any day
  in [request_date, day90].
  Then: amount_safe_to_pay = clamp(min(requested_amount, headroom), 0, requested_amount)
  This is O(n) in the number of forecast days and needs no search.

ALGORITHM (earliest_safe_date):
  For each day d in [request_date, day90] in order:
    if min_closing_balance_from(d) - requested_amount >= minimum_balance_to_keep:
        return d
  Return None if no such day exists.

  This is O(n²) worst case but n=91, so it's trivial.
  (We precompute a suffix-min array to make it O(n).)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

from src.finance.forecast import Forecast


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class SafetyResult:
    """
    The result of is_plan_safe().  Provides structured failure information
    instead of a bare True/False.
    """
    safe: bool
    minimum_balance_observed: float
    minimum_balance_date: date
    first_failure_date: Optional[date]   # first day balance < minimum_balance_to_keep
    failure_reason: Optional[str]        # human-readable failure description


# ---------------------------------------------------------------------------
# amount_safe_to_pay
# ---------------------------------------------------------------------------

def max_safe_amount(forecast: Forecast, requested_amount: float) -> float:
    """
    The maximum amount that can safely be debited on request_date without
    causing the balance to drop below minimum_balance_to_keep on any future day.

    Formula (O(n)):
      headroom = min_closing_balance_from(request_date) - minimum_balance_to_keep
      amount_safe_to_pay = clamp(min(requested_amount, headroom), 0, requested_amount)

    The headroom is computed from the BASE forecast (without the payment applied).
    The caller is responsible for passing a forecast that does NOT yet include the
    candidate payment.

    Returns a value in [0, requested_amount].
    """
    if requested_amount < 0:
        raise ValueError("requested_amount must be non-negative")
    if requested_amount == 0:
        return 0.0

    min_future = _suffix_min(forecast)[0]  # minimum across all days from request_date
    headroom = min_future - forecast.minimum_balance

    safe = min(requested_amount, headroom)
    return max(0.0, safe)


# ---------------------------------------------------------------------------
# earliest_safe_date
# ---------------------------------------------------------------------------

def earliest_safe_date(
    forecast: Forecast,
    requested_amount: float,
) -> Optional[date]:
    """
    The earliest date d in [request_date, day90] such that subtracting
    requested_amount from the closing balance of every day in [d, day90]
    still keeps all those balances >= minimum_balance_to_keep.

    Equivalently: the first d where min_closing_from(d) - requested_amount
    >= minimum_balance_to_keep.

    Uses a precomputed suffix-min array for O(n) performance.

    Returns None if no such date exists in the 90-day window.
    """
    if requested_amount <= 0:
        return forecast.request_date  # zero-cost payment is always safe immediately

    suffix_mins = _suffix_min(forecast)
    min_bal = forecast.minimum_balance

    for i, day in enumerate(forecast.days):
        min_from_here = suffix_mins[i]
        if min_from_here - requested_amount >= min_bal:
            return day.day_date

    return None  # Not achievable in the 90-day window


# ---------------------------------------------------------------------------
# Plan safety check
# ---------------------------------------------------------------------------

def is_plan_safe(
    base_forecast: Forecast,
    payments: List[Tuple[date, float]],
) -> SafetyResult:
    """
    Given the BASE forecast (no payments applied) and a list of planned
    (date, amount) payments, determine whether the plan is safe.

    The check re-simulates by walking each day and deducting each payment
    on its payment date.  This is an independent verification — not a reuse
    of intermediate values — which catches planner bugs.

    payments: list of (pay_date, amount) — amount is positive (it's a debit)
    """
    min_bal_threshold = base_forecast.minimum_balance

    # Build payment index
    pay_index: dict = {}
    for pay_date, amount in payments:
        pay_index[pay_date] = pay_index.get(pay_date, 0.0) + amount

    # Walk the forecast and subtract payments
    running_balance = base_forecast.start_balance
    observed_min = math.inf
    observed_min_date = base_forecast.request_date
    first_failure_date = None

    for day in base_forecast.days:
        # Apply the day's existing net change (income - expenses)
        running_balance += day.net_change

        # Deduct any planned payment due on this day
        if day.day_date in pay_index:
            running_balance -= pay_index[day.day_date]

        if running_balance < observed_min:
            observed_min = running_balance
            observed_min_date = day.day_date

        if running_balance < min_bal_threshold and first_failure_date is None:
            first_failure_date = day.day_date

    safe = first_failure_date is None

    return SafetyResult(
        safe=safe,
        minimum_balance_observed=observed_min,
        minimum_balance_date=observed_min_date,
        first_failure_date=first_failure_date,
        failure_reason=(
            None
            if safe
            else (
                f"Balance dropped to {observed_min:.2f} on {first_failure_date} "
                f"(minimum required: {min_bal_threshold:.2f})"
            )
        ),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _suffix_min(forecast: Forecast) -> List[float]:
    """
    Returns a list suffix_min[i] = min(closing_balance for days[i:]).
    Length equals len(forecast.days).
    """
    balances = [d.closing_balance for d in forecast.days]
    n = len(balances)
    suffix_mins = [0.0] * n
    running_min = math.inf
    for i in range(n - 1, -1, -1):
        running_min = min(running_min, balances[i])
        suffix_mins[i] = running_min
    return suffix_mins


# ---------------------------------------------------------------------------
# Installment plan safety check
# ---------------------------------------------------------------------------

def is_installment_plan_safe(
    base_forecast: Forecast,
    installment_schedule: List[Tuple[date, float]],
) -> SafetyResult:
    """
    Thin wrapper around is_plan_safe for installment plans.
    The installment_schedule is a list of (payment_date, payment_amount).
    """
    return is_plan_safe(base_forecast, installment_schedule)
