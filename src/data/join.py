"""
src/data/join.py
----------------
RequestContext Builder — Stage 3.

Responsibility:
  Build a complete, deterministic RequestContext for a single (user_id, request_id).
  Downstream components (planner, extractor, validator) consume only RequestContext;
  they never query raw DataFrames.

Pipeline position:
  load_all() → normalize() → RequestContextBuilder(datasets) → .build(request_id)

Design:
  - All indexes built ONCE at construction time: O(n) setup, O(1) per-request lookup.
  - Per-user event resolution is cached in-memory (dict); cleared explicitly in tests.
  - No LLM calls.  No live FX.  No network.
  - Messages and images are stored as raw structured data (untrusted evidence boundary).

Column names verified from live dataset inspection 2026-09-12:

  requests.csv:
    request_id, user_id, request_date, request_type, requested_amount,
    desired_completion_date, allows_partial_payment, request_text

  financial_profiles.csv:
    user_id, home_currency, current_available_balance, minimum_balance_to_keep,
    financial_priorities, expense_categories_to_protect,
    expense_categories_user_is_willing_to_reduce,
    expense_categories_user_is_willing_to_stop,
    payment_methods_user_will_consider, max_installment_months

  request_payment_options.csv:
    payment_option_id, request_id, payment_method, payment_amount,
    number_of_payments, first_payment_date, payment_frequency_days,
    financing_fee, total_payable_amount

  messages.csv:
    message_id, user_id, request_id, related_event_id, sent_at, source_type, message_text

  images.csv:
    image_id, user_id, request_id, related_event_id
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Set

import pandas as pd

from src.finance.events import (
    FinancialEvent,
    resolve_events,
    detect_and_project_recurring,
)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class ContextBuildError(ValueError):
    """Raised when a RequestContext cannot be built due to invalid/missing data."""


# ---------------------------------------------------------------------------
# Sub-models (all typed, no raw DataFrames exposed downstream)
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    """Parsed financial profile for one user."""
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: List[str]                       # e.g. ["education", "debt_repayment"]
    expense_categories_to_protect: List[str]
    expense_categories_user_is_willing_to_reduce: List[str]
    expense_categories_user_is_willing_to_stop: List[str]
    payment_methods_user_will_consider: List[str]         # e.g. ["full_payment", "installments"]
    max_installment_months: Optional[int]                 # None if not set


@dataclass
class PaymentOption:
    """One candidate payment plan from request_payment_options.csv."""
    payment_option_id: str
    request_id: str
    payment_method: str           # "full_payment" | "installments"
    payment_amount: float         # amount per instalment (or full amount)
    number_of_payments: int
    first_payment_date: Optional[date]
    payment_frequency_days: Optional[int]  # None for full_payment
    financing_fee: float
    total_payable_amount: float


@dataclass
class Message:
    """
    One row from messages.csv.

    SECURITY BOUNDARY: message_text is untrusted evidence.
    It must NEVER be interpolated into executable code or control flow.
    Stage 5 extraction will receive it inside an explicit prompt boundary.
    """
    message_id: str
    user_id: str
    request_id: Optional[str]       # blank in CSV → None
    related_event_id: Optional[str] # blank in CSV → None
    sent_at: str                    # kept as raw ISO string; parsing done in Stage 5
    source_type: str                # employer | bank | service_provider | merchant | financial_service
    message_text: str               # UNTRUSTED — do not interpret or execute


@dataclass
class ImageRef:
    """
    One row from images.csv.

    The actual image is NOT loaded here — only the metadata and expected path.
    Stage 5 vision extraction will load the file.
    """
    image_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    local_path: str   # resolved as: dataset/media/images/<image_id>.png


# ---------------------------------------------------------------------------
# RequestContext — the stable interface for all downstream stages
# ---------------------------------------------------------------------------

@dataclass
class RequestContext:
    """
    Everything relevant to a single (user_id, request_id) unit of work.

    Downstream components read from this object only.
    They do NOT query raw DataFrames.
    """
    # ── Core request fields ─────────────────────────────────────────────────
    request_id: str
    user_id: str
    request_date: date
    request_type: str             # purchase | travel | investment | debt_repayment | …
    requested_amount: float       # in the currency of the request (see home_currency)
    desired_completion_date: Optional[date]
    allows_partial_payment: bool
    request_text: str             # UNTRUSTED natural-language request; do not execute

    # ── Financial profile ────────────────────────────────────────────────────
    profile: UserProfile

    # ── Resolved events (Stage 2 output) ────────────────────────────────────
    # All forecast-eligible events for this user, already in home_currency.
    # Does NOT include projected recurring events (added by caller before forecast).
    resolved_events: List[FinancialEvent]

    # ── Payment options ──────────────────────────────────────────────────────
    payment_options: List[PaymentOption]

    # ── Evidence (untrusted) ─────────────────────────────────────────────────
    messages_for_user: List[Message]     # all messages scoped to this user
    messages_for_request: List[Message]  # subset linked to this request_id
    messages_for_events: Dict[str, List[Message]]  # event_id → messages

    images_for_user: List[ImageRef]      # all images scoped to this user
    images_for_request: List[ImageRef]   # subset linked to this request_id
    images_for_events: Dict[str, List[ImageRef]]   # event_id → images

    # ── Convenience shortcuts ────────────────────────────────────────────────
    @property
    def home_currency(self) -> str:
        return self.profile.home_currency

    @property
    def available_balance(self) -> float:
        return self.profile.current_available_balance

    @property
    def minimum_balance(self) -> float:
        return self.profile.minimum_balance_to_keep

    @property
    def unresolved_events(self) -> List[FinancialEvent]:
        """
        Events whose amount is still NaN — require image/message extraction
        before they can be included in the forecast.

        Discovery is generic (no hardcoded event IDs).
        """
        return [e for e in self.resolved_events if not e.is_resolved]

    @property
    def has_unresolved_events(self) -> bool:
        return len(self.unresolved_events) > 0

    @property
    def all_evidence_images(self) -> List[ImageRef]:
        """Flat list of all images associated with this context (no duplicates)."""
        seen: Set[str] = set()
        result: List[ImageRef] = []
        for img in self.images_for_user:
            if img.image_id not in seen:
                seen.add(img.image_id)
                result.append(img)
        return result

    @property
    def all_evidence_messages(self) -> List[Message]:
        """Flat list of all messages associated with this context (no duplicates)."""
        seen: Set[str] = set()
        result: List[Message] = []
        for msg in self.messages_for_user:
            if msg.message_id not in seen:
                seen.add(msg.message_id)
                result.append(msg)
        return result


# ---------------------------------------------------------------------------
# DataBundle — thin wrapper around normalized datasets
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    """
    Holds all normalized DataFrames from a single load_all() + normalize() call.
    Passed to RequestContextBuilder at construction time.
    """
    requests: pd.DataFrame
    financial_profiles: pd.DataFrame
    financial_events: pd.DataFrame
    exchange_rates: pd.DataFrame
    request_payment_options: pd.DataFrame
    messages: pd.DataFrame
    images: pd.DataFrame
    dataset_dir: str = "dataset"   # used to resolve image local paths

    @classmethod
    def from_datasets(cls, datasets: Dict[str, pd.DataFrame], dataset_dir: str = "dataset") -> "DataBundle":
        """Build a DataBundle from the dict returned by load_all()."""
        return cls(
            requests=datasets.get("requests", pd.DataFrame()),
            financial_profiles=datasets.get("financial_profiles", pd.DataFrame()),
            financial_events=datasets.get("financial_events", pd.DataFrame()),
            exchange_rates=datasets.get("exchange_rates", pd.DataFrame()),
            request_payment_options=datasets.get("request_payment_options", pd.DataFrame()),
            messages=datasets.get("messages", pd.DataFrame()),
            images=datasets.get("images", pd.DataFrame()),
            dataset_dir=dataset_dir,
        )


# ---------------------------------------------------------------------------
# RequestContextBuilder
# ---------------------------------------------------------------------------

class RequestContextBuilder:
    """
    Builds RequestContext objects from a DataBundle.

    Indexes are built ONCE at __init__ time.
    Per-user event resolution is cached across calls (same builder instance).

    Usage:
        datasets = load_all("dataset")
        normalize(datasets)
        bundle  = DataBundle.from_datasets(datasets)
        builder = RequestContextBuilder(bundle)

        ctx = builder.build("request_01")

        for ctx in builder.iter_requests():
            ...

    Cache management (tests):
        builder.clear_event_cache()
    """

    def __init__(self, bundle: DataBundle) -> None:
        self._bundle = bundle

        # ── Pre-build indexes ────────────────────────────────────────────────
        self._profiles_by_user: Dict[str, pd.Series] = self._index_profiles()
        self._requests_by_id: Dict[str, pd.Series] = self._index_requests()
        self._options_by_request: Dict[str, List[pd.Series]] = self._index_payment_options()
        self._messages_by_user: Dict[str, List[Message]] = {}
        self._messages_by_request: Dict[str, List[Message]] = {}
        self._messages_by_event: Dict[str, List[Message]] = {}
        self._images_by_user: Dict[str, List[ImageRef]] = {}
        self._images_by_request: Dict[str, List[ImageRef]] = {}
        self._images_by_event: Dict[str, List[ImageRef]] = {}
        self._build_message_indexes()
        self._build_image_indexes()

        # ── Per-user event resolution cache ──────────────────────────────────
        # {user_id: List[FinancialEvent]}  — populated lazily on first request per user
        self._event_cache: Dict[str, List[FinancialEvent]] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def build(self, request_id: str) -> RequestContext:
        """
        Build a complete RequestContext for the given request_id.

        Raises ContextBuildError if:
          - request_id not found
          - user_id not found in events/profiles
          - invalid/missing core fields
        """
        # 1. Fetch and validate the request row
        request = self._get_request(request_id)
        user_id = str(request["user_id"]).strip()

        # 2. Fetch and validate the user profile
        profile = self._get_profile(user_id, request_id)

        # 3. Get resolved events (cached per user)
        resolved_events = self._get_resolved_events(user_id)

        # Validate that all resolved events belong to this user
        cross_user = [e for e in resolved_events if e.user_id != user_id]
        if cross_user:
            raise ContextBuildError(
                f"[{request_id}] Cross-user event leak: {[e.event_id for e in cross_user]}"
            )

        # 4. Payment options
        payment_options = self._get_payment_options(request_id)

        # 5. Messages
        msgs_user = self._messages_by_user.get(user_id, [])
        msgs_request = self._messages_by_request.get(request_id, [])

        # Validate: messages for this request must belong to this user
        for msg in msgs_request:
            if msg.user_id != user_id:
                raise ContextBuildError(
                    f"[{request_id}] Message {msg.message_id} belongs to user "
                    f"{msg.user_id}, not {user_id}"
                )

        # Event-specific messages for events belonging to this user
        user_event_ids = {e.event_id for e in resolved_events}
        msgs_events: Dict[str, List[Message]] = {}
        for eid, msgs in self._messages_by_event.items():
            for msg in msgs:
                if msg.user_id == user_id:
                    msgs_events.setdefault(eid, []).append(msg)

        # 6. Images
        imgs_user = self._images_by_user.get(user_id, [])
        imgs_request = self._images_by_request.get(request_id, [])

        # Validate: images for this request must belong to this user
        for img in imgs_request:
            if img.user_id != user_id:
                raise ContextBuildError(
                    f"[{request_id}] Image {img.image_id} belongs to user "
                    f"{img.user_id}, not {user_id}"
                )

        # Event-specific images for events belonging to this user
        imgs_events: Dict[str, List[ImageRef]] = {}
        for eid, imgs in self._images_by_event.items():
            for img in imgs:
                if img.user_id == user_id:
                    imgs_events.setdefault(eid, []).append(img)

        # 7. Parse core request fields (validated)
        req_date = _parse_date_strict(request.get("request_date", ""), request_id, "request_date")
        requested_amount = _parse_amount_strict(request.get("requested_amount", ""), request_id, "requested_amount")
        desired_completion_date = _parse_date_optional(request.get("desired_completion_date", ""))
        allows_partial = _parse_bool(request.get("allows_partial_payment", "false"))
        request_type = _parse_str_strict(request.get("request_type", ""), request_id, "request_type")
        request_text = str(request.get("request_text", "")).strip()

        return RequestContext(
            request_id=request_id,
            user_id=user_id,
            request_date=req_date,
            request_type=request_type,
            requested_amount=requested_amount,
            desired_completion_date=desired_completion_date,
            allows_partial_payment=allows_partial,
            request_text=request_text,
            profile=profile,
            resolved_events=resolved_events,
            payment_options=payment_options,
            messages_for_user=msgs_user,
            messages_for_request=msgs_request,
            messages_for_events=msgs_events,
            images_for_user=imgs_user,
            images_for_request=imgs_request,
            images_for_events=imgs_events,
        )

    def iter_requests(self):
        """
        Iterate through all requests in CSV order, yielding RequestContext objects.
        Skips requests that fail validation (logs the error).

        Order matches requests.csv row order (deterministic for output.csv).
        """
        if self._bundle.requests.empty:
            return

        for _, row in self._bundle.requests.iterrows():
            rid = str(row.get("request_id", "")).strip()
            if not rid:
                continue
            try:
                yield self.build(rid)
            except ContextBuildError as exc:
                print(f"[join] SKIP {rid}: {exc}")

    def all_request_ids(self) -> List[str]:
        """Return request IDs in CSV order."""
        if self._bundle.requests.empty:
            return []
        return [str(r).strip() for r in self._bundle.requests["request_id"] if str(r).strip()]

    def clear_event_cache(self) -> None:
        """Clear the per-user event resolution cache. Useful in tests."""
        self._event_cache.clear()

    # ── Private: index builders ──────────────────────────────────────────────

    def _index_profiles(self) -> Dict[str, pd.Series]:
        """user_id → profile row"""
        index: Dict[str, pd.Series] = {}
        if self._bundle.financial_profiles.empty:
            return index
        for _, row in self._bundle.financial_profiles.iterrows():
            uid = str(row.get("user_id", "")).strip()
            if uid:
                index[uid] = row
        return index

    def _index_requests(self) -> Dict[str, pd.Series]:
        """request_id → request row"""
        index: Dict[str, pd.Series] = {}
        if self._bundle.requests.empty:
            return index
        seen: Set[str] = set()
        for _, row in self._bundle.requests.iterrows():
            rid = str(row.get("request_id", "")).strip()
            if not rid:
                continue
            if rid in seen:
                raise ContextBuildError(f"Duplicate request_id in requests.csv: {rid}")
            seen.add(rid)
            index[rid] = row
        return index

    def _index_payment_options(self) -> Dict[str, List[pd.Series]]:
        """request_id → [payment_option_row, ...]"""
        index: Dict[str, List[pd.Series]] = {}
        if self._bundle.request_payment_options.empty:
            return index
        for _, row in self._bundle.request_payment_options.iterrows():
            rid = str(row.get("request_id", "")).strip()
            if rid:
                index.setdefault(rid, []).append(row)
        return index

    def _build_message_indexes(self) -> None:
        """Build user / request / event indexes for messages."""
        if self._bundle.messages.empty:
            return
        for _, row in self._bundle.messages.iterrows():
            msg = _parse_message_row(row)
            if not msg:
                continue
            self._messages_by_user.setdefault(msg.user_id, []).append(msg)
            if msg.request_id:
                self._messages_by_request.setdefault(msg.request_id, []).append(msg)
            if msg.related_event_id:
                self._messages_by_event.setdefault(msg.related_event_id, []).append(msg)

    def _build_image_indexes(self) -> None:
        """Build user / request / event indexes for images."""
        if self._bundle.images.empty:
            return
        for _, row in self._bundle.images.iterrows():
            img = _parse_image_row(row, self._bundle.dataset_dir)
            if not img:
                continue
            self._images_by_user.setdefault(img.user_id, []).append(img)
            if img.request_id:
                self._images_by_request.setdefault(img.request_id, []).append(img)
            if img.related_event_id:
                self._images_by_event.setdefault(img.related_event_id, []).append(img)

    # ── Private: row fetchers ────────────────────────────────────────────────

    def _get_request(self, request_id: str) -> pd.Series:
        if request_id not in self._requests_by_id:
            raise ContextBuildError(f"request_id not found: {request_id!r}")
        return self._requests_by_id[request_id]

    def _get_profile(self, user_id: str, request_id: str) -> UserProfile:
        if user_id not in self._profiles_by_user:
            raise ContextBuildError(
                f"[{request_id}] No financial profile found for user_id={user_id!r}"
            )
        row = self._profiles_by_user[user_id]
        return _parse_profile_row(row, request_id)

    def _get_resolved_events(self, user_id: str) -> List[FinancialEvent]:
        """
        Return the resolved (forecast-eligible) events for user_id.
        Results are cached: multiple requests for the same user resolve only once.
        """
        if user_id not in self._event_cache:
            per_user = resolve_events(
                self._bundle.financial_events,
                self._bundle.financial_profiles,
                self._bundle.exchange_rates,
            )
            # Cache ALL users from this resolve call (not just the current user)
            for uid, evts in per_user.items():
                self._event_cache[uid] = evts
            # If the user has no events at all, store empty list
            if user_id not in self._event_cache:
                self._event_cache[user_id] = []
        return self._event_cache[user_id]

    def _get_payment_options(self, request_id: str) -> List[PaymentOption]:
        rows = self._options_by_request.get(request_id, [])
        options = []
        for row in rows:
            opt = _parse_payment_option_row(row, request_id)
            if opt:
                options.append(opt)
        return options


# ---------------------------------------------------------------------------
# Row parsers — strict where required, lenient for optional fields
# ---------------------------------------------------------------------------

def _parse_profile_row(row: pd.Series, request_id: str) -> UserProfile:
    """Parse one financial_profiles.csv row into a UserProfile."""
    uid = str(row.get("user_id", "")).strip()

    # Required numeric fields
    balance = _parse_amount_strict(row.get("current_available_balance", ""), request_id, "current_available_balance")
    min_balance = _parse_amount_strict(row.get("minimum_balance_to_keep", ""), request_id, "minimum_balance_to_keep")

    # Required string fields
    home_currency = _parse_str_strict(row.get("home_currency", ""), request_id, "home_currency")

    # Pipe-delimited list fields (empty string → empty list)
    def parse_list(val) -> List[str]:
        s = str(val).strip() if val else ""
        return [x.strip() for x in s.split("|") if x.strip()]

    # Optional integer
    max_months_raw = str(row.get("max_installment_months", "")).strip()
    max_months: Optional[int] = None
    if max_months_raw and max_months_raw.isdigit():
        max_months = int(max_months_raw)

    return UserProfile(
        user_id=uid,
        home_currency=home_currency,
        current_available_balance=balance,
        minimum_balance_to_keep=min_balance,
        financial_priorities=parse_list(row.get("financial_priorities", "")),
        expense_categories_to_protect=parse_list(row.get("expense_categories_to_protect", "")),
        expense_categories_user_is_willing_to_reduce=parse_list(row.get("expense_categories_user_is_willing_to_reduce", "")),
        expense_categories_user_is_willing_to_stop=parse_list(row.get("expense_categories_user_is_willing_to_stop", "")),
        payment_methods_user_will_consider=parse_list(row.get("payment_methods_user_will_consider", "")),
        max_installment_months=max_months,
    )


def _parse_payment_option_row(row: pd.Series, request_id: str) -> Optional[PaymentOption]:
    """Parse one request_payment_options.csv row. Returns None if critically invalid."""
    option_id = str(row.get("payment_option_id", "")).strip()
    if not option_id:
        return None

    # Validate request_id association
    row_request_id = str(row.get("request_id", "")).strip()
    if row_request_id != request_id:
        raise ContextBuildError(
            f"[{request_id}] Payment option {option_id} has request_id={row_request_id!r} "
            f"(expected {request_id!r}) — cross-request leak"
        )

    payment_method = str(row.get("payment_method", "")).strip()
    payment_amount = _parse_float_safe(row.get("payment_amount", ""))
    number_of_payments = _parse_int_safe(row.get("number_of_payments", ""), default=1)
    first_payment_date = _parse_date_optional(row.get("first_payment_date", ""))
    payment_frequency_days = _parse_int_optional(row.get("payment_frequency_days", ""))
    financing_fee = _parse_float_safe(row.get("financing_fee", ""), default=0.0)
    total_payable_amount = _parse_float_safe(row.get("total_payable_amount", ""))

    if math.isnan(payment_amount) or math.isnan(total_payable_amount):
        return None  # critically malformed option

    return PaymentOption(
        payment_option_id=option_id,
        request_id=request_id,
        payment_method=payment_method,
        payment_amount=payment_amount,
        number_of_payments=number_of_payments,
        first_payment_date=first_payment_date,
        payment_frequency_days=payment_frequency_days,
        financing_fee=financing_fee,
        total_payable_amount=total_payable_amount,
    )


def _parse_message_row(row: pd.Series) -> Optional[Message]:
    """Parse one messages.csv row. Returns None if message_id or user_id missing."""
    msg_id = str(row.get("message_id", "")).strip()
    user_id = str(row.get("user_id", "")).strip()
    if not msg_id or not user_id:
        return None

    request_id = _str_or_none(row.get("request_id", ""))
    related_event_id = _str_or_none(row.get("related_event_id", ""))
    sent_at = str(row.get("sent_at", "")).strip()
    source_type = str(row.get("source_type", "")).strip()
    message_text = str(row.get("message_text", "")).strip()  # UNTRUSTED

    return Message(
        message_id=msg_id,
        user_id=user_id,
        request_id=request_id,
        related_event_id=related_event_id,
        sent_at=sent_at,
        source_type=source_type,
        message_text=message_text,
    )


def _parse_image_row(row: pd.Series, dataset_dir: str) -> Optional[ImageRef]:
    """Parse one images.csv row. Returns None if image_id or user_id missing."""
    image_id = str(row.get("image_id", "")).strip()
    user_id = str(row.get("user_id", "")).strip()
    if not image_id or not user_id:
        return None

    request_id = _str_or_none(row.get("request_id", ""))
    related_event_id = _str_or_none(row.get("related_event_id", ""))
    local_path = os.path.join(dataset_dir, "media", "images", f"{image_id}.png")

    return ImageRef(
        image_id=image_id,
        user_id=user_id,
        request_id=request_id,
        related_event_id=related_event_id,
        local_path=local_path,
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_date_strict(val, request_id: str, field_name: str) -> date:
    """Parse a date, raising ContextBuildError on failure."""
    s = str(val).strip() if val else ""
    if not s:
        raise ContextBuildError(f"[{request_id}] Missing required field: {field_name!r}")
    try:
        return pd.to_datetime(s).date()
    except Exception:
        raise ContextBuildError(f"[{request_id}] Invalid date for {field_name!r}: {s!r}")


def _parse_date_optional(val) -> Optional[date]:
    """Parse a date; return None if blank or unparseable."""
    s = str(val).strip() if val else ""
    if not s:
        return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def _parse_amount_strict(val, request_id: str, field_name: str) -> float:
    """Parse a float amount, raising ContextBuildError on failure."""
    s = str(val).strip() if val else ""
    if not s:
        raise ContextBuildError(f"[{request_id}] Missing required field: {field_name!r}")
    try:
        result = float(s)
        if math.isnan(result):
            raise ValueError("NaN")
        if result < 0:
            raise ContextBuildError(
                f"[{request_id}] {field_name!r} must be non-negative, got {result}"
            )
        return result
    except ContextBuildError:
        raise
    except Exception:
        raise ContextBuildError(f"[{request_id}] Invalid amount for {field_name!r}: {s!r}")


def _parse_str_strict(val, request_id: str, field_name: str) -> str:
    """Parse a required string, raising ContextBuildError if blank."""
    s = str(val).strip() if val else ""
    if not s:
        raise ContextBuildError(f"[{request_id}] Missing required field: {field_name!r}")
    return s


def _parse_bool(val) -> bool:
    """Parse allows_partial_payment to bool."""
    s = str(val).strip().lower() if val else "false"
    return s in ("true", "1", "yes")


def _parse_float_safe(val, default: float = float("nan")) -> float:
    s = str(val).strip() if val else ""
    if not s:
        return default
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def _parse_int_safe(val, default: int = 1) -> int:
    s = str(val).strip() if val else ""
    if not s:
        return default
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return default


def _parse_int_optional(val) -> Optional[int]:
    s = str(val).strip() if val else ""
    if not s:
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _str_or_none(val) -> Optional[str]:
    s = str(val).strip() if val else ""
    return s if s else None
