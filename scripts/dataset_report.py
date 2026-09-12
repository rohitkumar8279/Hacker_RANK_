"""
scripts/dataset_report.py
--------------------------
Loads the real dataset and prints a summary for validation/debugging.

Usage:
    python -m scripts.dataset_report --dataset-dir dataset
"""

from __future__ import annotations

import argparse
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data.loader import load_all
from src.data.normalize import normalize
from src.finance.events import resolve_events, summarise_events
from src.data.join import DataBundle, RequestContextBuilder

def report(dataset_dir: str) -> None:
    print(f"\nLoading dataset from: {dataset_dir}")
    datasets = load_all(dataset_dir)
    normalize(datasets)

    events_df = datasets.get("financial_events", pd.DataFrame())
    profiles_df = datasets.get("financial_profiles", pd.DataFrame())
    rates_df = datasets.get("exchange_rates", pd.DataFrame())

    if events_df.empty:
        print("ERROR: financial_events.csv is empty or missing.")
        return

    # Raw stats
    print(f"\n{'='*60}")
    print("RAW DATASET STATS")
    print(f"{'='*60}")
    print(f"  Total financial event rows: {len(events_df)}")
    print(f"  Unique users in events:     {events_df['user_id'].nunique()}")
    print(f"  Unique event types:         {sorted(events_df['event_type'].unique())}")
    print(f"  Unique statuses:            {sorted(events_df['status'].unique())}")
    print(f"  Unique flexibility values:  {sorted(events_df['flexibility'].dropna().unique())}")
    print(f"  Unique categories:          {sorted(events_df['category'].unique())}")

    # Linked event chains
    linked = events_df[events_df["linked_event_id"].notna() & (events_df["linked_event_id"] != "")]
    print(f"\n  Rows with linked_event_id:  {len(linked)}")
    print(f"  Unique chain parents:       {linked['linked_event_id'].nunique()}")

    # Status breakdown
    print("\n  Status breakdown:")
    for status, count in events_df["status"].value_counts().items():
        print(f"    {status:<15} {count}")

    # Blank amounts
    blank_amounts = events_df[events_df["amount"].isna()]
    print(f"\n  Events with blank amount:   {len(blank_amounts)}")
    for _, row in blank_amounts.iterrows():
        print(f"    {row['event_id']} | user={row['user_id']} | status={row['status']}")

    # Resolve events
    print(f"\n{'='*60}")
    print("RESOLVED EVENTS (forecast-eligible)")
    print(f"{'='*60}")
    per_user = resolve_events(events_df, profiles_df, rates_df)
    summary = summarise_events(per_user)

    print(f"  Total eligible events:      {summary['total_eligible_events']}")
    print(f"  Unique users resolved:      {summary['unique_users']}")

    # Per-user breakdown
    print(f"\n  {'User':<12} {'Events':>8} {'Currency':<10} {'Balance':>14} {'Min Balance':>14}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*14} {'-'*14}")
    prof_indexed = profiles_df.set_index("user_id") if not profiles_df.empty else pd.DataFrame()
    for uid in sorted(per_user.keys()):
        n = len(per_user[uid])
        ccy = ""
        bal = ""
        min_bal = ""
        if not prof_indexed.empty and uid in prof_indexed.index:
            row = prof_indexed.loc[uid]
            ccy = str(row.get("home_currency", ""))
            bal = f"{float(row.get('current_available_balance', 0)):>14,.2f}"
            min_bal = f"{float(row.get('minimum_balance_to_keep', 0)):>14,.2f}"
        print(f"  {uid:<12} {n:>8} {ccy:<10} {bal} {min_bal}")

    # Recurring and cancellation patterns
    cancelled = [e for evts in per_user.values() for e in evts if e.status == "cancelled"]
    # (should be empty since we filter them out — but useful to verify)
    projected_count = 0  # can't know without running detect_and_project_recurring

    print(f"\n{'='*60}")
    print("DATASET OBSERVATIONS")
    print(f"{'='*60}")
    # Cancelled in resolved (should be 0)
    print(f"  Cancelled events in resolved output: {len(cancelled)} (expected 0)")
    # Failed
    failed_raw = events_df[events_df["status"] == "failed"]
    print(f"  Failed events (raw):                 {len(failed_raw)}")
    # Pending
    pending_raw = events_df[events_df["status"] == "pending"]
    print(f"  Pending events (raw):                {len(pending_raw)}")
    # Scheduled
    scheduled_raw = events_df[events_df["status"] == "scheduled"]
    print(f"  Scheduled events (raw):              {len(scheduled_raw)}")
    
    # -----------------------------------------------------------------------
    # Stage 3 Report: Join & Context Builder
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STAGE 3 CONTEXT BUILDER STATS")
    print(f"{'='*60}")

    requests_df = datasets.get("requests", pd.DataFrame())
    if requests_df.empty:
        print("ERROR: requests.csv is empty or missing.")
        return

    bundle = DataBundle.from_datasets(datasets, dataset_dir=dataset_dir)
    builder = RequestContextBuilder(bundle)

    num_requests = len(requests_df)
    unique_users = requests_df["user_id"].nunique() if "user_id" in requests_df else 0
    requests_per_user = requests_df.groupby("user_id").size().describe() if "user_id" in requests_df else pd.Series()

    options_df = datasets.get("request_payment_options", pd.DataFrame())
    requests_with_options = options_df["request_id"].nunique() if not options_df.empty else 0
    requests_without_options = num_requests - requests_with_options

    msgs_df = datasets.get("messages", pd.DataFrame())
    msg_user_linked = len(msgs_df[(msgs_df["user_id"] != "") & (msgs_df["request_id"] == "") & (msgs_df["related_event_id"] == "")]) if not msgs_df.empty else 0
    msg_request_linked = len(msgs_df[msgs_df["request_id"] != ""]) if not msgs_df.empty else 0
    msg_event_linked = len(msgs_df[msgs_df["related_event_id"] != ""]) if not msgs_df.empty else 0

    imgs_df = datasets.get("images", pd.DataFrame())
    img_user_linked = len(imgs_df[(imgs_df["user_id"] != "") & (imgs_df["request_id"] == "") & (imgs_df["related_event_id"] == "")]) if not imgs_df.empty else 0
    img_request_linked = len(imgs_df[imgs_df["request_id"] != ""]) if not imgs_df.empty else 0
    img_event_linked = len(imgs_df[imgs_df["related_event_id"] != ""]) if not imgs_df.empty else 0

    unresolved_contexts = 0
    unresolved_events_seen = set()
    for ctx_id in builder.all_request_ids():
        try:
            ctx = builder.build(ctx_id)
            if ctx.has_unresolved_events:
                unresolved_contexts += 1
                for e in ctx.unresolved_events:
                    unresolved_events_seen.add(e.event_id)
        except Exception:
            pass

    print(f"  Total requests:                      {num_requests}")
    print(f"  Unique users with requests:          {unique_users}")
    if not requests_per_user.empty:
        print(f"  Requests per user (mean):            {requests_per_user['mean']:.2f}")
        print(f"  Requests per user (max):             {int(requests_per_user['max'])}")
    print(f"\n  Requests with payment options:       {requests_with_options}")
    print(f"  Requests without payment options:    {requests_without_options}")
    
    print("\n  Messages:")
    print(f"    User-linked only:                  {msg_user_linked}")
    print(f"    Request-linked:                    {msg_request_linked}")
    print(f"    Event-linked:                      {msg_event_linked}")
    
    print("\n  Images:")
    print(f"    User-linked only:                  {img_user_linked}")
    print(f"    Request-linked:                    {img_request_linked}")
    print(f"    Event-linked:                      {img_event_linked}")

    print(f"\n  Events with unresolved amounts:      {len(unresolved_events_seen)}")
    print(f"  Contexts containing unresolved:      {unresolved_contexts}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Dataset validation report")
    parser.add_argument("--dataset-dir", default="dataset", help="Path to dataset directory")
    args = parser.parse_args()
    report(args.dataset_dir)


if __name__ == "__main__":
    main()
