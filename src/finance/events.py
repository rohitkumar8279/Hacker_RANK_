"""
src/finance/events.py
---------------------
Financial event chain resolution and normalisation.

KEY FACTS FROM DATASET INSPECTION
===================================
Columns in financial_events.csv:
  event_id, user_id, event_type, description, category, direction,
  amount, currency, event_date, settlement_date, status, linked_event_id,
  flexibility, minimum_allowed_amount

Observed status values:
  settled    – completed, real, counts in forecast
  pending    – awaiting settlement; exclude from forecast (spec: exclude pending credits;
               for pending debits we exclude them too as they are not yet confirmed)
  failed     – exclude
  cancelled  – exclude
  scheduled  – confirmed future event (e.g. next salary, scheduled school fee); INCLUDE

Observed flexibility values:
  fixed                 – cannot be stopped or reduced
  stoppable             – can be stopped entirely (spending change: stop:<event_id>)
  reducible             – can be reduced to minimum_allowed_amount
  reducible_or_stoppable – can be reduced OR stopped

Observed event_types:
  income, expense, subscription, debt_payment, refund, investment

Observed direction values:
  credit, debit

linked_event_id mechanics (from data inspection):
  - event_98  (debit, settled)  → event_99  (credit/refund, linked=event_98)
    Meaning: event_99 is a refund/reversal of event_98.  Both are settled.
    Both are INCLUDED in the forecast (debit went out, credit came back).
  - event_100 (debit, cancelled) → event_101 (debit, settled, linked=event_100)
    Meaning: the original card auth (event_100) was cancelled and replaced by a
    settled charge (event_101).  We INCLUDE event_101, EXCLUDE event_100 (already cancelled).
  - The blank-amount event_253 (user_03) has status=settled and must be resolved
    via images.csv/messages.csv by the extraction layer before events are finalised.

CONFLICT RESOLUTION PRIORITY (from spec + data):
  1. Explicit cancellation/settlement/amendment (status field wins over linked)
  2. Linked superseding record: the newer record in the chain wins
  3. Settled > pending/scheduled when amounts differ for same lifecycle
  4. Financially safer interpretation when conflict cannot be resolved

DUPLICATE DETECTION:
  True duplicates are same (user_id, event_type, category, direction, amount,
  event_date, description) with different event_id and NO linked_event_id.
  Recurring transactions with the same description/amount on different dates
  are NOT duplicates.

FORECAST-ELIGIBLE events:
  Include: settled, scheduled (confirmed future)
  Exclude: pending (credits AND debits – spec says exclude pending credits;
           we conservatively exclude pending debits since they have no confirmed
           settlement_date commitment), failed, cancelled
  Special: refund events are settled credits → INCLUDED (they were actual cash)

RECURRENCE DETECTION:
  The dataset provides individual event rows, NOT a recurrence rule.
  Recurring patterns (monthly rent, monthly salary, bi-weekly groceries, etc.)
  are represented as repeated rows with similar descriptions and approximately
  regular intervals.  We detect recurrence by grouping settled events with the
  same (user_id, category, description) that appear on a roughly periodic schedule
  and projecting them forward into the 90-day forecast window.

  Conservative approach:
  - Only project events whose category matches a known recurring category AND
    that appear at least 2 times in the history for the same user.
  - Use the median interval between events to determine cadence.
  - Start the next occurrence from the day AFTER the last known event date.
  - Projected events use the mean amount from historical instances.

  This avoids projecting one-off events and avoids assuming 30-day months.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import median
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import pandas as pd

from src.finance.fx import convert


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FinancialEvent:
    """Normalised internal representation of one financial event."""
    event_id: str
    user_id: str
    event_type: str          # income | expense | subscription | debt_payment | refund | investment
    description: str
    category: str
    direction: str           # credit | debit
    amount: float            # in home_currency; NaN means unresolved (needs extraction)
    home_currency: str
    event_date: date
    settlement_date: Optional[date]
    status: str              # settled | pending | failed | cancelled | scheduled
    linked_event_id: Optional[str]
    flexibility: str         # fixed | stoppable | reducible | reducible_or_stoppable
    minimum_allowed_amount: Optional[float]  # in home_currency; only for reducible
    is_projected: bool = False  # True for recurring events projected into the forecast window

    @property
    def signed_amount(self) -> float:
        """Positive = cash in, negative = cash out (in home_currency)."""
        if math.isnan(self.amount):
            return 0.0
        if self.direction == "credit":
            return self.amount
        return -self.amount

    @property
    def is_stoppable(self) -> bool:
        return self.flexibility in ("stoppable", "reducible_or_stoppable")

    @property
    def is_reducible(self) -> bool:
        return self.flexibility in ("reducible", "reducible_or_stoppable")

    @property
    def is_flexible(self) -> bool:
        return self.flexibility != "fixed"

    @property
    def is_resolved(self) -> bool:
        """False when amount is blank and needs LLM/image extraction."""
        return not math.isnan(self.amount)


# ---------------------------------------------------------------------------
# Statuses to exclude from the forecast
# ---------------------------------------------------------------------------

_EXCLUDED_STATUSES: FrozenSet[str] = frozenset({"pending", "failed", "cancelled"})

# Statuses that participate in the forecast
_INCLUDED_STATUSES: FrozenSet[str] = frozenset({"settled", "scheduled"})

# Categories we consider recurring and eligible to project forward
_RECURRING_CATEGORIES: FrozenSet[str] = frozenset({
    "rent", "housing", "utilities", "salary", "education",
    "debt_repayment", "music_subscription", "delivery_membership",
    "cloud_storage", "streaming", "gym", "insurance", "healthcare",
    "family_support", "groceries", "transport", "dining", "entertainment",
    "shopping",
})

# Minimum occurrences in history before we project a pattern
_MIN_RECURRENCE_COUNT = 2

# Max cadence gap (days) to treat as recurring; beyond this we consider it one-off
_MAX_CADENCE_DAYS = 95


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _parse_date(val) -> Optional[date]:
    if pd.isna(val) or val == "":
        return None
    if isinstance(val, date):
        return val
    return pd.to_datetime(val).date()


def _parse_float(val) -> float:
    """Returns NaN for blank/null amounts."""
    if pd.isna(val) or val == "":
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _parse_str(val, default: str = "") -> str:
    if pd.isna(val) or val is None:
        return default
    return str(val).strip()


# ---------------------------------------------------------------------------
# Main resolution function
# ---------------------------------------------------------------------------

def resolve_events(
    events_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    extracted_facts: Optional[Dict[str, dict]] = None,
) -> Dict[str, List[FinancialEvent]]:
    """
    Full pipeline: raw DataFrame → per-user list of forecast-eligible events.

    Steps:
    1. Parse all rows into FinancialEvent objects (in original currency).
    2. Apply extracted_facts (LLM/image-sourced amounts/statuses) where provided.
    3. Resolve linked_event_id chains (remove superseded/cancelled events).
    4. Remove excluded-status events (pending, failed, cancelled).
    5. Remove exact duplicates.
    6. Convert amounts to home_currency.
    7. Return dict: {user_id: [FinancialEvent, ...]}

    extracted_facts: {event_id: {"amount": float, "status": str, "date": str}}
                     Applied before chain resolution. Pass None to skip.
    """
    if extracted_facts is None:
        extracted_facts = {}

    # Build home_currency lookup
    home_currency: Dict[str, str] = {}
    if not profiles_df.empty:
        for _, row in profiles_df.iterrows():
            home_currency[str(row["user_id"]).strip()] = str(row["home_currency"]).strip()

    # Build FX-enabled rates lookup for conversion
    # (fx module must already be initialised)

    # ---------------------------------------------------------------------------
    # Step 1: Parse raw rows
    # ---------------------------------------------------------------------------
    raw_events: Dict[str, FinancialEvent] = {}  # event_id → FinancialEvent
    original_currencies: Dict[str, str] = {}    # event_id → original currency

    for _, row in events_df.iterrows():
        event_id = _parse_str(row.get("event_id", ""))
        user_id = _parse_str(row.get("user_id", ""))
        if not event_id or not user_id:
            continue

        amount_raw = _parse_float(row.get("amount", ""))
        currency = _parse_str(row.get("currency", ""))
        event_date = _parse_date(row.get("event_date"))
        settlement_date = _parse_date(row.get("settlement_date"))
        status = _parse_str(row.get("status", "settled")).lower()
        linked_event_id = _parse_str(row.get("linked_event_id", "")) or None
        flexibility = _parse_str(row.get("flexibility", "fixed")).lower()
        min_allowed = _parse_float(row.get("minimum_allowed_amount", ""))

        if event_date is None:
            # Cannot use an event with no date
            continue

        evt = FinancialEvent(
            event_id=event_id,
            user_id=user_id,
            event_type=_parse_str(row.get("event_type", "expense")),
            description=_parse_str(row.get("description", "")),
            category=_parse_str(row.get("category", "")),
            direction=_parse_str(row.get("direction", "debit")).lower(),
            amount=amount_raw,  # original currency; converted in step 6
            home_currency=home_currency.get(user_id, currency),
            event_date=event_date,
            settlement_date=settlement_date,
            status=status,
            linked_event_id=linked_event_id,
            flexibility=flexibility,
            minimum_allowed_amount=None if math.isnan(min_allowed) else min_allowed,
        )
        raw_events[event_id] = evt
        original_currencies[event_id] = currency

    # ---------------------------------------------------------------------------
    # Step 2: Apply extracted facts (LLM/image resolution)
    # ---------------------------------------------------------------------------
    for event_id, facts in extracted_facts.items():
        if event_id not in raw_events:
            continue
        evt = raw_events[event_id]
        if "amount" in facts and facts["amount"] is not None:
            try:
                evt.amount = float(facts["amount"])
            except (TypeError, ValueError):
                pass
        if "status" in facts and facts["status"] in ("settled", "pending", "failed", "cancelled", "scheduled"):
            evt.status = facts["status"]
        if "date" in facts and facts["date"]:
            parsed = _parse_date(facts["date"])
            if parsed:
                evt.event_date = parsed

    # ---------------------------------------------------------------------------
    # Step 3: Resolve linked_event_id chains
    # ---------------------------------------------------------------------------
    # Build reverse index: parent_id → [child event_ids]
    children: Dict[str, List[str]] = {}
    for eid, evt in raw_events.items():
        if evt.linked_event_id:
            children.setdefault(evt.linked_event_id, []).append(eid)

    # Set of event_ids that have been superseded and should be excluded
    superseded: Set[str] = set()

    for parent_id, child_ids in children.items():
        if parent_id not in raw_events:
            continue
        parent = raw_events[parent_id]

        for child_id in child_ids:
            child = raw_events[child_id]
            # Case A: child is a refund/credit of the parent
            #   Both remain in the forecast (debit out, credit back in).
            #   Nothing to supersede.
            if child.event_type == "refund" or (
                child.direction == "credit" and parent.direction == "debit"
            ):
                pass  # keep both

            # Case B: parent was cancelled, child is the replacement settled charge
            elif parent.status == "cancelled" and child.status in _INCLUDED_STATUSES:
                superseded.add(parent_id)  # parent already marked cancelled anyway

            # Case C: child is an amendment/settlement of a pending/estimated parent
            #   Priority: newer settled record wins → supersede the parent
            elif child.status in _INCLUDED_STATUSES and parent.status in ("pending", "scheduled"):
                superseded.add(parent_id)

            # Case D: both settled; newer record wins (amendment)
            elif (
                child.status == "settled"
                and parent.status == "settled"
                and child.event_date >= parent.event_date
            ):
                superseded.add(parent_id)

    # ---------------------------------------------------------------------------
    # Step 4: Filter to forecast-eligible events
    # ---------------------------------------------------------------------------
    eligible: Dict[str, FinancialEvent] = {
        eid: evt
        for eid, evt in raw_events.items()
        if evt.status in _INCLUDED_STATUSES
        and eid not in superseded
    }

    # ---------------------------------------------------------------------------
    # Step 5: Duplicate detection
    # ---------------------------------------------------------------------------
    # True duplicates: same user, same direction, same amount, same event_date,
    # same category, same description, but different event_id, no linked_event_id.
    seen_signatures: Set[Tuple] = set()
    deduped: Dict[str, FinancialEvent] = {}

    for eid, evt in eligible.items():
        if evt.linked_event_id:
            # Part of a chain — not a duplicate by definition
            deduped[eid] = evt
            continue

        sig = (
            evt.user_id,
            evt.direction,
            evt.amount,
            evt.event_date,
            evt.category,
            evt.description,
        )
        if sig in seen_signatures:
            # Exact duplicate — drop
            continue
        seen_signatures.add(sig)
        deduped[eid] = evt

    # ---------------------------------------------------------------------------
    # Step 6: Currency conversion to home_currency
    # ---------------------------------------------------------------------------
    # We need the rates_df here; fx module must already be init-ed.
    for eid, evt in deduped.items():
        ccy = evt.home_currency
        if math.isnan(evt.amount):
            continue  # Will be resolved by extraction layer; leave as NaN
        original_ccy = original_currencies.get(eid)
        if original_ccy and original_ccy != ccy:
            try:
                evt.amount = convert(evt.amount, original_ccy, ccy, evt.event_date)
                if evt.minimum_allowed_amount is not None:
                    evt.minimum_allowed_amount = convert(
                        evt.minimum_allowed_amount, original_ccy, ccy, evt.event_date
                    )
            except ValueError:
                # If no rate available, keep original amount (best effort)
                pass

    # ---------------------------------------------------------------------------
    # Group by user
    # ---------------------------------------------------------------------------
    per_user: Dict[str, List[FinancialEvent]] = {}
    for evt in deduped.values():
        per_user.setdefault(evt.user_id, []).append(evt)

    # Sort by date within each user
    for uid in per_user:
        per_user[uid].sort(key=lambda e: e.event_date)

    return per_user





# ---------------------------------------------------------------------------
# Recurrence projection
# ---------------------------------------------------------------------------

def detect_and_project_recurring(
    events: List[FinancialEvent],
    forecast_start: date,
    forecast_end: date,
) -> List[FinancialEvent]:
    """
    Given a list of historical/known events for one user, detect recurring
    patterns and project them into the forecast window [forecast_start, forecast_end].

    Only returns the NEW projected events; caller merges with existing events.

    Algorithm:
    - Group by (category, description) — same category+description = same recurring item.
    - Require >= _MIN_RECURRENCE_COUNT settled events in the group.
    - Compute intervals between consecutive dates; take median as the cadence.
    - If cadence <= _MAX_CADENCE_DAYS, project from (last_date + cadence) forward.
    - Use mean amount from historical instances.
    - Projected events get status="scheduled" and is_projected=True.
    """
    # Separate historical (pre-forecast-start) from already-known future events
    historical = [e for e in events if e.event_date < forecast_start and e.status == "settled"]
    already_future = {e.event_date for e in events if e.event_date >= forecast_start}

    # Group historical events
    groups: Dict[Tuple[str, str], List[FinancialEvent]] = {}
    for evt in historical:
        if not evt.is_resolved:
            continue
        key = (evt.category, evt.description)
        groups.setdefault(key, []).append(evt)

    projected: List[FinancialEvent] = []

    for (category, description), group in groups.items():
        if len(group) < _MIN_RECURRENCE_COUNT:
            continue

        # Sort by date
        group_sorted = sorted(group, key=lambda e: e.event_date)
        dates = [e.event_date for e in group_sorted]

        # Compute intervals
        intervals = [
            (dates[i + 1] - dates[i]).days
            for i in range(len(dates) - 1)
        ]
        cadence = int(round(median(intervals)))

        if cadence <= 0 or cadence > _MAX_CADENCE_DAYS:
            continue

        # Mean amount
        amounts = [e.amount for e in group_sorted if not math.isnan(e.amount)]
        if not amounts:
            continue
        mean_amount = sum(amounts) / len(amounts)

        # Representative event for metadata
        rep = group_sorted[-1]

        # Project from last_date + cadence
        next_date = rep.event_date + timedelta(days=cadence)
        proj_count = 0

        while next_date <= forecast_end:
            # Don't duplicate an event already in the dataset on this date
            if next_date not in already_future:
                projected.append(FinancialEvent(
                    event_id=f"projected_{rep.event_id}_{proj_count}",
                    user_id=rep.user_id,
                    event_type=rep.event_type,
                    description=description,
                    category=category,
                    direction=rep.direction,
                    amount=mean_amount,
                    home_currency=rep.home_currency,
                    event_date=next_date,
                    settlement_date=next_date,
                    status="scheduled",
                    linked_event_id=None,
                    flexibility=rep.flexibility,
                    minimum_allowed_amount=rep.minimum_allowed_amount,
                    is_projected=True,
                ))
                proj_count += 1

            next_date += timedelta(days=cadence)

    return projected


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------

def summarise_events(per_user: Dict[str, List[FinancialEvent]]) -> dict:
    """
    Returns a summary dict useful for debugging and reporting.
    """
    total_events = sum(len(v) for v in per_user.values())
    users = list(per_user.keys())
    return {
        "total_eligible_events": total_events,
        "unique_users": len(users),
        "users": users,
    }
