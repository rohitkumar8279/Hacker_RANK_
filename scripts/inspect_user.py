"""
scripts/inspect_user.py
-----------------------
Debug script: display a single user's financial timeline.

Usage:
    python -m scripts.inspect_user --user-id user_01 --dataset-dir dataset
    python -m scripts.inspect_user --user-id user_06 --request-date 2026-01-03

Output:
    User overview
    Financial events (resolved)
    90-day forecast table
"""

from __future__ import annotations

import argparse
import sys
import os
from datetime import date, timedelta
from typing import Optional

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data.loader import load_all
from src.data.normalize import normalize
from src.finance.events import resolve_events, detect_and_project_recurring
from src.finance.forecast import simulate


def inspect(user_id: str, dataset_dir: str, request_date: Optional[date]) -> None:
    # Load & normalize
    datasets = load_all(dataset_dir)
    normalize(datasets)

    profiles_df = datasets["financial_profiles"]
    events_df = datasets["financial_events"]
    rates_df = datasets["exchange_rates"]

    # Resolve events for all users (once)
    per_user = resolve_events(events_df, profiles_df, rates_df)

    if user_id not in per_user:
        print(f"User {user_id!r} not found in financial_events. Known users: {list(per_user.keys())}")
        return

    user_events = per_user[user_id]

    # Get profile
    prof_row = profiles_df[profiles_df["user_id"] == user_id]
    if prof_row.empty:
        print(f"No profile found for {user_id}")
        return

    home_currency = prof_row.iloc[0]["home_currency"]
    start_balance = float(prof_row.iloc[0]["current_available_balance"])
    min_balance = float(prof_row.iloc[0]["minimum_balance_to_keep"])

    if request_date is None:
        request_date = date.today()

    forecast_end = request_date + timedelta(days=90)

    # Project recurring events into the forecast window
    projected = detect_and_project_recurring(user_events, request_date, forecast_end)
    all_events = user_events + projected

    # Print header
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  USER: {user_id}  |  Currency: {home_currency}")
    print(f"  Balance: {start_balance:,.2f}  |  Minimum: {min_balance:,.2f}")
    print(f"  Request date: {request_date}  |  Forecast end: {forecast_end}")
    print(sep)

    # Financial events
    print(f"\n{'FINANCIAL EVENTS (resolved)':^72}")
    print(f"{'DATE':<12} {'TYPE':<14} {'DIR':<7} {'AMOUNT':>12} {'STATUS':<12} {'EVENT_ID':<12} {'LINKED':<12}")
    print("-" * 72)
    for evt in sorted(user_events, key=lambda e: e.event_date):
        amount_str = f"{evt.amount:>12,.2f}" if not __import__('math').isnan(evt.amount) else "      BLANK"
        linked = evt.linked_event_id or ""
        print(
            f"{evt.event_date!s:<12} {evt.event_type:<14} {evt.direction:<7} "
            f"{amount_str} {evt.status:<12} {evt.event_id:<12} {linked:<12}"
        )

    # 90-Day Forecast
    forecast = simulate(
        user_id=user_id,
        request_date=request_date,
        start_balance=start_balance,
        minimum_balance=min_balance,
        events=all_events,
    )

    print(f"\n{'90-DAY FORECAST':^72}")
    print(f"{'DATE':<12} {'INCOME':>12} {'EXPENSES':>12} {'NET':>12} {'BALANCE':>12}")
    print("-" * 72)
    for day in forecast.days:
        if day.total_income == 0.0 and day.total_expenses == 0.0:
            continue  # skip quiet days to reduce noise
        marker = " ⚠" if day.closing_balance < min_balance else ""
        print(
            f"{day.day_date!s:<12} {day.total_income:>12,.2f} {day.total_expenses:>12,.2f} "
            f"{day.net_change:>12,.2f} {day.closing_balance:>12,.2f}{marker}"
        )

    print(f"\n  Min balance over window: {forecast.min_balance_over_window:,.2f} "
          f"on {forecast.min_balance_date}")
    print(f"  Minimum required:        {min_balance:,.2f}")
    print(f"  Projected events added:  {len(projected)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect a user's financial timeline")
    parser.add_argument("--user-id", required=True, help="User ID (e.g. user_01)")
    parser.add_argument("--dataset-dir", default="dataset", help="Path to dataset directory")
    parser.add_argument("--request-date", default=None, help="Request date (YYYY-MM-DD)")
    args = parser.parse_args()

    req_date = None
    if args.request_date:
        req_date = date.fromisoformat(args.request_date)

    inspect(args.user_id, args.dataset_dir, req_date)


if __name__ == "__main__":
    main()
