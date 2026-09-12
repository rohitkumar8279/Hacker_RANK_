"""
src/finance/fx.py
-----------------
Deterministic currency conversion using dataset/exchange_rates.csv only.

Design:
- Exchange rates are loaded once and indexed by (from_currency, to_currency, rate_date).
- Lookup uses the rate dated on or before the target date (nearest prior date).
- No live rates, no fallback APIs.
- Same-currency conversion is always a no-op.

All rates in exchange_rates.csv are expressed relative to USD as the bridge currency.
We build a bidirectional rate graph so any (from → to) conversion can be resolved
via at most one intermediate step (from → USD → to).
"""

from __future__ import annotations

import math
from datetime import date
from functools import lru_cache
from typing import Dict, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Internal rate store (populated once at load time)
# ---------------------------------------------------------------------------

class FXStore:
    """
    Holds dated exchange rates and exposes a single convert() method.

    Rate structure from CSV: (rate_date, from_currency, to_currency, rate)
    where rate = how many to_currency units per 1 from_currency unit.
    """

    def __init__(self, rates_df: pd.DataFrame) -> None:
        # Store as {(from_ccy, to_ccy): sorted list of (date, rate)}
        self._rates: Dict[Tuple[str, str], list] = {}

        if rates_df.empty:
            return

        df = rates_df.copy()
        df["rate_date"] = pd.to_datetime(df["rate_date"]).dt.date

        for _, row in df.iterrows():
            key = (str(row["from_currency"]).strip(), str(row["to_currency"]).strip())
            entry = (row["rate_date"], float(row["rate"]))
            self._rates.setdefault(key, []).append(entry)

        # Sort each list by date ascending
        for key in self._rates:
            self._rates[key].sort(key=lambda x: x[0])

    def _lookup_rate(
        self, from_ccy: str, to_ccy: str, on_date: date
    ) -> Optional[float]:
        """
        Returns the direct rate for (from_ccy → to_ccy) using the rate dated
        on or before on_date.  Returns None if no such direct rate exists.
        """
        key = (from_ccy, to_ccy)
        entries = self._rates.get(key)
        if not entries:
            return None

        if hasattr(on_date, "date"):
            on_date = on_date.date()

        # Find the last entry whose date <= on_date
        rate = None
        for entry_date, entry_rate in entries:
            if hasattr(entry_date, "date"):
                entry_date = entry_date.date()
            if entry_date <= on_date:
                rate = entry_rate
            else:
                break
        return rate

    def convert(self, amount: float, from_ccy: str, to_ccy: str, on_date: date) -> float:
        """
        Convert amount from from_ccy to to_ccy using the rate on or before on_date.

        Resolution order:
        1. Same currency → no-op.
        2. Direct rate exists → use it.
        3. Inverse direct rate exists → use 1/rate.
        4. Bridge via USD → (from→USD) then (USD→to).
        5. Bridge via USD inverse → try combinations.
        6. No rate found → raise ValueError (not silently zero).
        """
        if from_ccy == to_ccy:
            return amount

        if amount == 0.0 or math.isnan(amount):
            return amount

        rate = self._resolve_rate(from_ccy, to_ccy, on_date)
        if rate is None:
            raise ValueError(
                f"No exchange rate found for {from_ccy} → {to_ccy} on or before {on_date}"
            )
        return amount * rate

    def _resolve_rate(
        self, from_ccy: str, to_ccy: str, on_date: date
    ) -> Optional[float]:
        # 1. Direct
        r = self._lookup_rate(from_ccy, to_ccy, on_date)
        if r is not None:
            return r

        # 2. Inverse
        r_inv = self._lookup_rate(to_ccy, from_ccy, on_date)
        if r_inv is not None and r_inv != 0:
            return 1.0 / r_inv

        # 3. USD bridge: from_ccy → USD → to_ccy
        # from_ccy → USD (direct or inverse)
        from_to_usd = self._lookup_rate(from_ccy, "USD", on_date)
        if from_to_usd is None:
            r_usd_from = self._lookup_rate("USD", from_ccy, on_date)
            if r_usd_from and r_usd_from != 0:
                from_to_usd = 1.0 / r_usd_from

        # USD → to_ccy (direct or inverse)
        usd_to_to = self._lookup_rate("USD", to_ccy, on_date)
        if usd_to_to is None:
            r_to_usd = self._lookup_rate(to_ccy, "USD", on_date)
            if r_to_usd and r_to_usd != 0:
                usd_to_to = 1.0 / r_to_usd

        if from_to_usd is not None and usd_to_to is not None:
            return from_to_usd * usd_to_to

        return None


# Module-level singleton, populated by init_fx()
_fx_store: Optional[FXStore] = None


def init_fx(rates_df: pd.DataFrame) -> None:
    """Call once at startup with the exchange_rates DataFrame."""
    global _fx_store
    _fx_store = FXStore(rates_df)


def convert(amount: float, from_ccy: str, to_ccy: str, on_date: date) -> float:
    """
    Top-level conversion function.  Must call init_fx() before using this.
    """
    if _fx_store is None:
        raise RuntimeError("FX store not initialised. Call init_fx() first.")
    return _fx_store.convert(amount, from_ccy, to_ccy, on_date)
