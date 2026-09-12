"""
src/data/normalize.py
---------------------
Normalise raw DataFrames to correct Python types.

Responsibilities:
1. Parse date columns to datetime.date.
2. Parse numeric columns to float.
3. Strip whitespace from string columns.
4. Initialise the FX store from exchange_rates.
5. Add home_currency column to financial_events so downstream code
   can do currency conversion without rejoining profiles every time.

NOTE: Currency conversion of event amounts happens inside events.py
      (resolve_events), not here.  This module only sets up the FX store
      and propagates home_currency metadata.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from src.finance import fx as fx_module


def normalize(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Normalise all datasets in-place (returns the same dict).
    Must be called after load_all() and before resolve_events().
    """
    # Initialise FX store first (everything else may depend on it)
    rates_df = datasets.get("exchange_rates", pd.DataFrame())
    if not rates_df.empty:
        _parse_exchange_rates(rates_df)
        fx_module.init_fx(rates_df)

    # financial_profiles
    profiles = datasets.get("financial_profiles", pd.DataFrame())
    if not profiles.empty:
        profiles["current_available_balance"] = pd.to_numeric(
            profiles["current_available_balance"], errors="coerce"
        )
        profiles["minimum_balance_to_keep"] = pd.to_numeric(
            profiles["minimum_balance_to_keep"], errors="coerce"
        )
        if "max_installment_months" in profiles.columns:
            profiles["max_installment_months"] = pd.to_numeric(
                profiles["max_installment_months"], errors="coerce"
            )

    # financial_events
    events = datasets.get("financial_events", pd.DataFrame())
    if not events.empty:
        events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
        events["settlement_date"] = pd.to_datetime(events["settlement_date"], errors="coerce")
        events["amount"] = pd.to_numeric(events["amount"], errors="coerce")
        events["minimum_allowed_amount"] = pd.to_numeric(
            events["minimum_allowed_amount"], errors="coerce"
        )
        # Propagate home_currency from profiles
        if not profiles.empty:
            ccy_map = profiles.set_index("user_id")["home_currency"].to_dict()
            events["home_currency"] = events["user_id"].map(ccy_map)

    # requests
    requests = datasets.get("requests", pd.DataFrame())
    if not requests.empty:
        requests["request_date"] = pd.to_datetime(requests["request_date"], errors="coerce")
        requests["desired_completion_date"] = pd.to_datetime(
            requests["desired_completion_date"], errors="coerce"
        )
        requests["requested_amount"] = pd.to_numeric(
            requests["requested_amount"], errors="coerce"
        )
        requests["allows_partial_payment"] = (
            requests["allows_partial_payment"].str.strip().str.lower() == "true"
        )

    # request_payment_options
    options = datasets.get("request_payment_options", pd.DataFrame())
    if not options.empty:
        for col in ("first_payment_date", "payment_end_date"):
            if col in options.columns:
                options[col] = pd.to_datetime(options[col], errors="coerce")
        for col in ("number_of_payments", "days_between_payments", "financing_fee_pct",
                    "total_payable_amount", "installment_amount"):
            if col in options.columns:
                options[col] = pd.to_numeric(options[col], errors="coerce")

    # sample_requests (if present)
    sample = datasets.get("sample_requests", pd.DataFrame())
    if not sample.empty:
        sample["request_date"] = pd.to_datetime(sample["request_date"], errors="coerce")
        sample["desired_completion_date"] = pd.to_datetime(
            sample["desired_completion_date"], errors="coerce"
        )
        sample["requested_amount"] = pd.to_numeric(
            sample["requested_amount"], errors="coerce"
        )
        sample["amount_safe_to_pay"] = pd.to_numeric(
            sample["amount_safe_to_pay"], errors="coerce"
        )

    return datasets


def _parse_exchange_rates(rates_df: pd.DataFrame) -> None:
    """Parse rate_date and rate columns in-place."""
    rates_df["rate_date"] = pd.to_datetime(rates_df["rate_date"], errors="coerce")
    rates_df["rate"] = pd.to_numeric(rates_df["rate"], errors="coerce")
