"""
src/finance/forecast.py
-----------------------
Deterministic 90-day daily balance forecast engine.

Design principles:
- Pure Python/pandas — no LLM calls.
- All amounts are in the user's home_currency (conversion done upstream in events.py).
- The forecast starts from request_date and covers the next FORECAST_HORIZON_DAYS calendar days.
- Starting balance comes from financial_profiles.current_available_balance.
- Events are applied on their event_date (not settlement_date); this matches the intent
  of "what does the user's *effective* balance look like day by day."
- The result is a list of DayBalance named tuples indexed by date.
- An optional payment (amount, date) can be overlaid to test a candidate plan without
  rebuilding the entire forecast from scratch.

Terminology:
  day 0  = request_date
  day 1  = request_date + 1 day
  ...
  day 90 = request_date + 90 days  (inclusive, so 91 rows total)

Projected recurring events (from detect_and_project_recurring) are included in the
forecast by the caller by merging them into the events list before passing here.

Spending changes:
  The caller may pass a list of SpendingChange objects.  These are applied to eligible
  recurring events before the simulation:
    - stop:<event_id>  → remove that event and all projected variants of it
    - reduce_to:<event_id>:<amount> → cap the event (and its projections) at new_amount
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from src.finance.events import FinancialEvent
from config import FORECAST_HORIZON_DAYS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DayBalance:
    """Balance snapshot for a single calendar day in the forecast."""
    day_date: date
    opening_balance: float    # balance at start of day (before events applied)
    total_income: float       # sum of credits on this day
    total_expenses: float     # sum of debits on this day (positive number)
    net_change: float         # total_income - total_expenses
    closing_balance: float    # balance at end of day
    events_applied: List[str] = field(default_factory=list)  # event_ids applied on this day


@dataclass
class SpendingChange:
    """A single spending-change instruction to overlay onto the forecast."""
    event_id: str             # the specific event_id to modify (or its projection root)
    action: str               # "stop" | "reduce_to"
    new_amount: Optional[float] = None  # only set for "reduce_to"


@dataclass
class Forecast:
    """Full 90-day forecast result."""
    user_id: str
    request_date: date
    start_balance: float      # available_balance on request_date
    minimum_balance: float    # minimum_balance_to_keep (never breached by any safe plan)
    days: List[DayBalance]    # day 0 … day 90 (inclusive)

    # Convenience properties
    @property
    def balances(self) -> List[float]:
        """Closing balance for each day, aligned with self.days."""
        return [d.closing_balance for d in self.days]

    @property
    def min_balance_over_window(self) -> float:
        """The lowest closing balance across the entire 90-day window."""
        return min(self.balances)

    @property
    def min_balance_date(self) -> date:
        """The date on which the minimum balance occurs."""
        balances = self.balances
        idx = balances.index(min(balances))
        return self.days[idx].day_date

    def balance_on(self, target_date: date) -> Optional[float]:
        """Return the closing balance on a specific date, or None if out of window."""
        for day in self.days:
            if day.day_date == target_date:
                return day.closing_balance
        return None

    def min_balance_from(self, from_date: date) -> float:
        """
        Minimum closing balance across [from_date, end_of_window].
        Useful for checking whether a payment on from_date is safe for the rest
        of the window.
        """
        vals = [
            d.closing_balance
            for d in self.days
            if d.day_date >= from_date
        ]
        return min(vals) if vals else self.start_balance


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate(
    user_id: str,
    request_date: date,
    start_balance: float,
    minimum_balance: float,
    events: List[FinancialEvent],
    override_payment: Optional[Tuple[date, float]] = None,
    spending_changes: Optional[List[SpendingChange]] = None,
) -> Forecast:
    """
    Simulate the daily balance for the 90-day window starting at request_date.

    Parameters
    ----------
    user_id : str
    request_date : date
        Day 0 of the forecast.
    start_balance : float
        The user's available balance on request_date (in home_currency).
    minimum_balance : float
        Protected floor — never used in the simulation itself, only stored in
        the Forecast for downstream safety checks.
    events : list[FinancialEvent]
        All forecast-eligible events for this user (historical + projected),
        already in home_currency.  Events before request_date are IGNORED
        (they are already reflected in start_balance).
    override_payment : (date, amount) or None
        A candidate payment to overlay. The amount is subtracted (as a debit)
        on the given date.  Used to test whether a payment plan is safe.
    spending_changes : list[SpendingChange] or None
        Spending changes to apply before simulating.

    Returns
    -------
    Forecast
    """
    forecast_end = request_date + timedelta(days=FORECAST_HORIZON_DAYS)

    # Apply spending changes to events
    adjusted_events = _apply_spending_changes(events, spending_changes or [])

    # Build a date → list[events] index for events in the forecast window
    event_index: Dict[date, List[FinancialEvent]] = defaultdict(list)
    for evt in adjusted_events:
        if request_date <= evt.event_date <= forecast_end:
            event_index[evt.event_date].append(evt)

    # Add override payment
    override_by_date: Dict[date, float] = {}
    if override_payment:
        pay_date, pay_amount = override_payment
        if request_date <= pay_date <= forecast_end:
            override_by_date[pay_date] = pay_amount

    # Walk through each day
    days: List[DayBalance] = []
    running_balance = start_balance

    for day_offset in range(FORECAST_HORIZON_DAYS + 1):  # inclusive
        current_date = request_date + timedelta(days=day_offset)

        opening = running_balance
        income = 0.0
        expenses = 0.0
        applied_ids: List[str] = []

        # Apply financial events for this day
        for evt in event_index.get(current_date, []):
            if math.isnan(evt.amount):
                # Unresolved amount — excluded from forecast (spec: do not invent)
                continue
            if evt.direction == "credit":
                income += evt.amount
            else:
                expenses += evt.amount
            applied_ids.append(evt.event_id)

        # Apply override payment (debit) on this day
        if current_date in override_by_date:
            expenses += override_by_date[current_date]
            applied_ids.append("__override_payment__")

        net = income - expenses
        closing = opening + net

        days.append(DayBalance(
            day_date=current_date,
            opening_balance=opening,
            total_income=income,
            total_expenses=expenses,
            net_change=net,
            closing_balance=closing,
            events_applied=applied_ids,
        ))

        running_balance = closing

    return Forecast(
        user_id=user_id,
        request_date=request_date,
        start_balance=start_balance,
        minimum_balance=minimum_balance,
        days=days,
    )


# ---------------------------------------------------------------------------
# Spending change application
# ---------------------------------------------------------------------------

def _apply_spending_changes(
    events: List[FinancialEvent],
    changes: List[SpendingChange],
) -> List[FinancialEvent]:
    """
    Apply spending changes to a list of events.
    Returns a new list (does not mutate originals).

    Rules:
    - "stop" removes the event and all projections of it.
    - "reduce_to" caps the event (and its projections) at new_amount.
    - Only events marked is_flexible may be changed (guard rail).
    - A projected event's root event_id is embedded as the prefix in its event_id
      (e.g. "projected_event_476_0" roots to "event_476").
    """
    if not changes:
        return events

    stop_ids: set = set()
    reduce_map: Dict[str, float] = {}

    for change in changes:
        if change.action == "stop":
            stop_ids.add(change.event_id)
        elif change.action == "reduce_to" and change.new_amount is not None:
            reduce_map[change.event_id] = change.new_amount

    result = []
    for evt in events:
        root_id = _root_event_id(evt)

        if root_id in stop_ids:
            if not evt.is_flexible:
                # Safety guard: never stop a fixed event
                result.append(evt)
            else:
                continue  # drop the event

        elif root_id in reduce_map:
            if not evt.is_flexible:
                result.append(evt)
            else:
                new_amount = reduce_map[root_id]
                # Respect minimum_allowed_amount
                if evt.minimum_allowed_amount is not None:
                    new_amount = max(new_amount, evt.minimum_allowed_amount)
                import copy
                new_evt = copy.copy(evt)
                new_evt.amount = new_amount
                result.append(new_evt)

        else:
            result.append(evt)

    return result


def _root_event_id(evt: FinancialEvent) -> str:
    """
    For a projected event ("projected_event_476_0"), return the root event_id
    ("event_476") so spending changes can apply to it.
    For a real event, return its own event_id.
    """
    if evt.is_projected and evt.event_id.startswith("projected_"):
        # format: projected_<original_event_id>_<count>
        without_prefix = evt.event_id[len("projected_"):]
        # strip trailing _<count>
        parts = without_prefix.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
    return evt.event_id
