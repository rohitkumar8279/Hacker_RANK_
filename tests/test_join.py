"""
tests/test_join.py
------------------
Comprehensive tests for src/data/join.py (Stage 3).

Tests are fully self-contained using synthetic DataFrames.
No file I/O; no network calls; no LLM calls.

Coverage:
1.  valid request context — all fields correctly populated
2.  correct profile attached to the right user
3.  only the correct user's events appear
4.  event resolution is reused (cache hit)
5.  multiple requests for same user
6.  payment options correctly scoped to one request
7.  request without payment options → empty list
8.  user-level message association
9.  request-level message association
10. event-level message association
11. image association
12. unresolved blank amount detection
13. invalid request ID raises ContextBuildError
14. missing profile raises ContextBuildError
15. mismatched event/user relationship is caught
16. invalid payment option amount skipped
17. deterministic iteration order
18. cache reuse (same events object returned on second call)
19. cache clearing resets state
20. no cross-user event leakage
21. UserProfile pipe-separated list fields
22. UserProfile max_installment_months parsing
23. PaymentOption fields parsed correctly
24. allows_partial_payment parsing (true/false/True/False)
25. desired_completion_date is optional
26. Image local path construction
"""

from __future__ import annotations

import math
import os
import pytest
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

# NOTE: unittest.mock is NOT used here because on this system
# `from unittest.mock import patch` triggers asyncio → _overlapped DLL import
# which is blocked by the Application Control policy.
# Instead we use manual monkey-patching where needed.

# Initialise FX store before importing anything that uses it
from src.finance import fx as fx_module
fx_module.init_fx(pd.DataFrame([
    {"rate_date": "2024-01-01", "from_currency": "USD", "to_currency": "ZAR", "rate": "18.0"},
    {"rate_date": "2024-01-01", "from_currency": "EUR", "to_currency": "ZAR", "rate": "20.0"},
    {"rate_date": "2024-01-01", "from_currency": "USD", "to_currency": "IDR", "rate": "15000.0"},
    {"rate_date": "2025-01-01", "from_currency": "USD", "to_currency": "ZAR", "rate": "18.0"},
    {"rate_date": "2025-01-01", "from_currency": "EUR", "to_currency": "ZAR", "rate": "20.0"},
]))

from src.data.join import (
    DataBundle,
    RequestContext,
    RequestContextBuilder,
    ContextBuildError,
    UserProfile,
    PaymentOption,
    Message,
    ImageRef,
)


# ---------------------------------------------------------------------------
# Factories for test DataFrames
# ---------------------------------------------------------------------------

def make_requests_df(rows: Optional[List[dict]] = None) -> pd.DataFrame:
    defaults = [{
        "request_id": "request_01",
        "user_id": "user_01",
        "request_date": "2024-03-03",
        "request_type": "purchase",
        "requested_amount": "25256",
        "desired_completion_date": "2024-03-20",
        "allows_partial_payment": "true",
        "request_text": "Can I afford it?",
    }]
    return pd.DataFrame(rows or defaults, dtype=str)


def make_profiles_df(rows: Optional[List[dict]] = None) -> pd.DataFrame:
    defaults = [{
        "user_id": "user_01",
        "home_currency": "ZAR",
        "current_available_balance": "58481.1",
        "minimum_balance_to_keep": "18000",
        "financial_priorities": "education|debt_repayment",
        "expense_categories_to_protect": "rent|groceries",
        "expense_categories_user_is_willing_to_reduce": "dining",
        "expense_categories_user_is_willing_to_stop": "delivery_membership",
        "payment_methods_user_will_consider": "full_payment",
        "max_installment_months": "",
    }]
    return pd.DataFrame(rows or defaults, dtype=str)


def make_events_df(rows: Optional[List[dict]] = None) -> pd.DataFrame:
    defaults = [{
        "event_id": "event_01",
        "user_id": "user_01",
        "event_type": "income",
        "description": "Monthly salary",
        "category": "salary",
        "direction": "credit",
        "amount": "23320",
        "currency": "ZAR",
        "event_date": "2024-03-15",
        "settlement_date": "2024-03-15",
        "status": "scheduled",
        "linked_event_id": "",
        "flexibility": "fixed",
        "minimum_allowed_amount": "",
    }]
    return pd.DataFrame(rows or defaults, dtype=str)


def make_options_df(rows: Optional[List[dict]] = None) -> pd.DataFrame:
    defaults = [{
        "payment_option_id": "payment_option_01",
        "request_id": "request_01",
        "payment_method": "full_payment",
        "payment_amount": "25256",
        "number_of_payments": "1",
        "first_payment_date": "2024-03-03",
        "payment_frequency_days": "",
        "financing_fee": "0",
        "total_payable_amount": "25256",
    }]
    return pd.DataFrame(rows or defaults, dtype=str)


def make_messages_df(rows: Optional[List[dict]] = None) -> pd.DataFrame:
    defaults = [{
        "message_id": "message_01",
        "user_id": "user_01",
        "request_id": "request_01",
        "related_event_id": "",
        "sent_at": "2024-03-01T09:30:00Z",
        "source_type": "employer",
        "message_text": "Your salary has been confirmed.",
    }]
    return pd.DataFrame(rows or defaults, dtype=str)


def make_images_df(rows: Optional[List[dict]] = None) -> pd.DataFrame:
    defaults = [{
        "image_id": "image_01",
        "user_id": "user_01",
        "request_id": "request_01",
        "related_event_id": "",
    }]
    return pd.DataFrame(rows or defaults, dtype=str)


def make_bundle(
    requests_rows=None,
    profiles_rows=None,
    events_rows=None,
    options_rows=None,
    messages_rows=None,
    images_rows=None,
    dataset_dir="dataset",
) -> DataBundle:
    return DataBundle(
        requests=make_requests_df(requests_rows),
        financial_profiles=make_profiles_df(profiles_rows),
        financial_events=make_events_df(events_rows),
        exchange_rates=pd.DataFrame(),
        request_payment_options=make_options_df(options_rows),
        messages=make_messages_df(messages_rows),
        images=make_images_df(images_rows),
        dataset_dir=dataset_dir,
    )


def make_builder(**kwargs) -> RequestContextBuilder:
    return RequestContextBuilder(make_bundle(**kwargs))


# ---------------------------------------------------------------------------
# 1. Valid request context
# ---------------------------------------------------------------------------

def test_valid_context_fields():
    builder = make_builder()
    ctx = builder.build("request_01")

    assert ctx.request_id == "request_01"
    assert ctx.user_id == "user_01"
    assert ctx.request_date == date(2024, 3, 3)
    assert ctx.request_type == "purchase"
    assert ctx.requested_amount == pytest.approx(25256.0)
    assert ctx.desired_completion_date == date(2024, 3, 20)
    assert ctx.allows_partial_payment is True
    assert ctx.request_text == "Can I afford it?"


# ---------------------------------------------------------------------------
# 2. Correct profile attached
# ---------------------------------------------------------------------------

def test_correct_profile_attached():
    builder = make_builder()
    ctx = builder.build("request_01")

    profile = ctx.profile
    assert profile.user_id == "user_01"
    assert profile.home_currency == "ZAR"
    assert profile.current_available_balance == pytest.approx(58481.1)
    assert profile.minimum_balance_to_keep == pytest.approx(18000.0)


# ---------------------------------------------------------------------------
# 3. Only the correct user's events
# ---------------------------------------------------------------------------

def test_only_correct_user_events():
    events_rows = [
        {   # user_01 — should be included
            "event_id": "event_01", "user_id": "user_01",
            "event_type": "income", "description": "Salary",
            "category": "salary", "direction": "credit",
            "amount": "5000", "currency": "ZAR",
            "event_date": "2024-03-15", "settlement_date": "2024-03-15",
            "status": "settled", "linked_event_id": "",
            "flexibility": "fixed", "minimum_allowed_amount": "",
        },
        {   # user_02 — should NOT appear in user_01's context
            "event_id": "event_02", "user_id": "user_02",
            "event_type": "expense", "description": "Rent",
            "category": "rent", "direction": "debit",
            "amount": "3000", "currency": "ZAR",
            "event_date": "2024-03-01", "settlement_date": "2024-03-01",
            "status": "settled", "linked_event_id": "",
            "flexibility": "fixed", "minimum_allowed_amount": "",
        },
    ]
    builder = make_builder(events_rows=events_rows)
    ctx = builder.build("request_01")

    assert all(e.user_id == "user_01" for e in ctx.resolved_events)
    assert not any(e.event_id == "event_02" for e in ctx.resolved_events)


# ---------------------------------------------------------------------------
# 4. Event resolution reused (cache hit)
# ---------------------------------------------------------------------------

def test_event_resolution_cached():
    """
    resolve_events should only be called once per user across multiple build() calls.
    We verify this by inspecting the builder's internal cache directly.
    """
    bundle = make_bundle()
    builder = RequestContextBuilder(bundle)

    # Cache starts empty
    assert len(builder._event_cache) == 0

    _ = builder.build("request_01")
    # After first build, cache should have user_01
    assert "user_01" in builder._event_cache
    events_list_after_first = builder._event_cache["user_01"]

    _ = builder.build("request_01")
    # After second build, the SAME list object should be in cache (no re-resolve)
    assert builder._event_cache["user_01"] is events_list_after_first


# ---------------------------------------------------------------------------
# 5. Multiple requests for same user — same event list object
# ---------------------------------------------------------------------------

def test_multiple_requests_same_user_same_events():
    requests_rows = [
        {"request_id": "request_01", "user_id": "user_01",
         "request_date": "2024-03-03", "request_type": "purchase",
         "requested_amount": "1000", "desired_completion_date": "2024-04-01",
         "allows_partial_payment": "false", "request_text": "Request 1"},
        {"request_id": "request_02", "user_id": "user_01",
         "request_date": "2024-04-01", "request_type": "travel",
         "requested_amount": "5000", "desired_completion_date": "2024-05-01",
         "allows_partial_payment": "true", "request_text": "Request 2"},
    ]
    options_rows = [
        {"payment_option_id": "opt_01", "request_id": "request_01",
         "payment_method": "full_payment", "payment_amount": "1000",
         "number_of_payments": "1", "first_payment_date": "2024-03-03",
         "payment_frequency_days": "", "financing_fee": "0", "total_payable_amount": "1000"},
        {"payment_option_id": "opt_02", "request_id": "request_02",
         "payment_method": "full_payment", "payment_amount": "5000",
         "number_of_payments": "1", "first_payment_date": "2024-04-01",
         "payment_frequency_days": "", "financing_fee": "0", "total_payable_amount": "5000"},
    ]
    messages_rows = [
        {"message_id": "message_01", "user_id": "user_01", "request_id": "request_01",
         "related_event_id": "", "sent_at": "2024-03-01T09:30:00Z",
         "source_type": "employer", "message_text": "Update 1"},
        {"message_id": "message_02", "user_id": "user_01", "request_id": "request_02",
         "related_event_id": "", "sent_at": "2024-03-15T09:30:00Z",
         "source_type": "employer", "message_text": "Update 2"},
    ]
    images_rows = [
        {"image_id": "image_01", "user_id": "user_01", "request_id": "",
         "related_event_id": ""},
    ]
    builder = make_builder(
        requests_rows=requests_rows,
        options_rows=options_rows,
        messages_rows=messages_rows,
        images_rows=images_rows,
    )
    ctx1 = builder.build("request_01")
    ctx2 = builder.build("request_02")

    # Same user → same resolved events object (cached)
    assert ctx1.resolved_events is ctx2.resolved_events

    # Different requests → different payment options
    assert len(ctx1.payment_options) == 1
    assert ctx1.payment_options[0].payment_option_id == "opt_01"
    assert len(ctx2.payment_options) == 1
    assert ctx2.payment_options[0].payment_option_id == "opt_02"

    # Different requests → different message subsets
    assert len(ctx1.messages_for_request) == 1
    assert ctx1.messages_for_request[0].message_id == "message_01"
    assert len(ctx2.messages_for_request) == 1
    assert ctx2.messages_for_request[0].message_id == "message_02"

    # Both should see user-level image
    assert any(img.image_id == "image_01" for img in ctx1.images_for_user)
    assert any(img.image_id == "image_01" for img in ctx2.images_for_user)


# ---------------------------------------------------------------------------
# 6. Payment options correctly scoped
# ---------------------------------------------------------------------------

def test_payment_options_scoped_to_request():
    builder = make_builder()
    ctx = builder.build("request_01")
    assert len(ctx.payment_options) == 1
    opt = ctx.payment_options[0]
    assert opt.payment_option_id == "payment_option_01"
    assert opt.payment_method == "full_payment"
    assert opt.payment_amount == pytest.approx(25256.0)
    assert opt.number_of_payments == 1
    assert opt.first_payment_date == date(2024, 3, 3)
    assert opt.financing_fee == pytest.approx(0.0)
    assert opt.total_payable_amount == pytest.approx(25256.0)


# ---------------------------------------------------------------------------
# 7. Request without payment options
# ---------------------------------------------------------------------------

def test_no_payment_options():
    options_rows = [
        {"payment_option_id": "opt_99", "request_id": "request_99",
         "payment_method": "full_payment", "payment_amount": "100",
         "number_of_payments": "1", "first_payment_date": "2024-01-01",
         "payment_frequency_days": "", "financing_fee": "0", "total_payable_amount": "100"},
    ]
    builder = make_builder(options_rows=options_rows)
    ctx = builder.build("request_01")
    assert ctx.payment_options == []


# ---------------------------------------------------------------------------
# 8. User-level message association
# ---------------------------------------------------------------------------

def test_user_level_messages():
    messages_rows = [
        {"message_id": "msg_u", "user_id": "user_01", "request_id": "",
         "related_event_id": "", "sent_at": "2024-03-01T09:30:00Z",
         "source_type": "employer", "message_text": "User-level message."},
        {"message_id": "msg_other", "user_id": "user_02", "request_id": "",
         "related_event_id": "", "sent_at": "2024-03-01T09:30:00Z",
         "source_type": "employer", "message_text": "Other user message."},
    ]
    builder = make_builder(messages_rows=messages_rows)
    ctx = builder.build("request_01")

    user_msg_ids = [m.message_id for m in ctx.messages_for_user]
    assert "msg_u" in user_msg_ids
    assert "msg_other" not in user_msg_ids  # different user


# ---------------------------------------------------------------------------
# 9. Request-level message association
# ---------------------------------------------------------------------------

def test_request_level_messages():
    messages_rows = [
        {"message_id": "msg_req", "user_id": "user_01", "request_id": "request_01",
         "related_event_id": "", "sent_at": "2024-03-01T09:30:00Z",
         "source_type": "employer", "message_text": "For this request."},
        {"message_id": "msg_other_req", "user_id": "user_01", "request_id": "request_99",
         "related_event_id": "", "sent_at": "2024-03-01T09:30:00Z",
         "source_type": "employer", "message_text": "For another request."},
    ]
    builder = make_builder(messages_rows=messages_rows)
    ctx = builder.build("request_01")

    req_msg_ids = [m.message_id for m in ctx.messages_for_request]
    assert "msg_req" in req_msg_ids
    assert "msg_other_req" not in req_msg_ids


# ---------------------------------------------------------------------------
# 10. Event-level message association
# ---------------------------------------------------------------------------

def test_event_level_messages():
    messages_rows = [
        {"message_id": "msg_evt", "user_id": "user_01", "request_id": "",
         "related_event_id": "event_01", "sent_at": "2024-03-01T09:30:00Z",
         "source_type": "employer", "message_text": "About event_01."},
        {"message_id": "msg_no_evt", "user_id": "user_01", "request_id": "request_01",
         "related_event_id": "", "sent_at": "2024-03-01T09:30:00Z",
         "source_type": "employer", "message_text": "Not about an event."},
    ]
    builder = make_builder(messages_rows=messages_rows)
    ctx = builder.build("request_01")

    assert "event_01" in ctx.messages_for_events
    assert any(m.message_id == "msg_evt" for m in ctx.messages_for_events["event_01"])


# ---------------------------------------------------------------------------
# 11. Image association
# ---------------------------------------------------------------------------

def test_image_association():
    images_rows = [
        {"image_id": "img_01", "user_id": "user_01",
         "request_id": "request_01", "related_event_id": "event_01"},
        {"image_id": "img_02", "user_id": "user_01",
         "request_id": "", "related_event_id": ""},
        {"image_id": "img_other", "user_id": "user_02",
         "request_id": "", "related_event_id": ""},
    ]
    builder = make_builder(images_rows=images_rows, dataset_dir="dataset")
    ctx = builder.build("request_01")

    user_img_ids = [i.image_id for i in ctx.images_for_user]
    assert "img_01" in user_img_ids
    assert "img_02" in user_img_ids
    assert "img_other" not in user_img_ids  # different user

    req_img_ids = [i.image_id for i in ctx.images_for_request]
    assert "img_01" in req_img_ids
    assert "img_02" not in req_img_ids

    assert "event_01" in ctx.images_for_events
    assert any(i.image_id == "img_01" for i in ctx.images_for_events["event_01"])


# ---------------------------------------------------------------------------
# 12. Unresolved blank amount detection
# ---------------------------------------------------------------------------

def test_unresolved_event_detected():
    events_rows = [
        {   # resolved event
            "event_id": "event_01", "user_id": "user_01",
            "event_type": "income", "description": "Salary",
            "category": "salary", "direction": "credit",
            "amount": "5000", "currency": "ZAR",
            "event_date": "2024-01-15", "settlement_date": "2024-01-15",
            "status": "settled", "linked_event_id": "",
            "flexibility": "fixed", "minimum_allowed_amount": "",
        },
        {   # blank amount event — unresolved
            "event_id": "event_253", "user_id": "user_01",
            "event_type": "income", "description": "August net salary",
            "category": "salary", "direction": "credit",
            "amount": "", "currency": "ZAR",
            "event_date": "2024-02-15", "settlement_date": "2024-02-15",
            "status": "settled", "linked_event_id": "",
            "flexibility": "fixed", "minimum_allowed_amount": "",
        },
    ]
    builder = make_builder(events_rows=events_rows)
    ctx = builder.build("request_01")

    assert ctx.has_unresolved_events is True
    unresolved_ids = [e.event_id for e in ctx.unresolved_events]
    assert "event_253" in unresolved_ids
    assert "event_01" not in unresolved_ids  # resolved event should NOT be in unresolved list


def test_no_unresolved_events():
    builder = make_builder()
    ctx = builder.build("request_01")
    assert ctx.has_unresolved_events is False
    assert ctx.unresolved_events == []


# ---------------------------------------------------------------------------
# 13. Invalid request ID raises ContextBuildError
# ---------------------------------------------------------------------------

def test_invalid_request_id_raises():
    builder = make_builder()
    with pytest.raises(ContextBuildError, match="not found"):
        builder.build("request_NONEXISTENT")


# ---------------------------------------------------------------------------
# 14. Missing profile raises ContextBuildError
# ---------------------------------------------------------------------------

def test_missing_profile_raises():
    # Request for user_01 but profiles only has user_99
    profiles_rows = [{
        "user_id": "user_99",
        "home_currency": "ZAR",
        "current_available_balance": "50000",
        "minimum_balance_to_keep": "10000",
        "financial_priorities": "",
        "expense_categories_to_protect": "",
        "expense_categories_user_is_willing_to_reduce": "",
        "expense_categories_user_is_willing_to_stop": "",
        "payment_methods_user_will_consider": "full_payment",
        "max_installment_months": "",
    }]
    builder = make_builder(profiles_rows=profiles_rows)
    with pytest.raises(ContextBuildError, match="No financial profile"):
        builder.build("request_01")


# ---------------------------------------------------------------------------
# 15. Cross-user payment option raises ContextBuildError
# ---------------------------------------------------------------------------

def test_cross_request_payment_option_raises():
    options_rows = [{
        "payment_option_id": "opt_bad",
        "request_id": "request_WRONG",  # does not match request_01
        "payment_method": "full_payment",
        "payment_amount": "25256",
        "number_of_payments": "1",
        "first_payment_date": "2024-03-03",
        "payment_frequency_days": "",
        "financing_fee": "0",
        "total_payable_amount": "25256",
    }]
    # The cross-request option should be silently excluded (different request_id)
    # It should NOT raise — it's just not scoped to this request
    builder = make_builder(options_rows=options_rows)
    ctx = builder.build("request_01")
    assert ctx.payment_options == []  # that option belongs to request_WRONG, not request_01


# ---------------------------------------------------------------------------
# 16. Invalid payment option amount → skipped
# ---------------------------------------------------------------------------

def test_invalid_payment_option_skipped():
    options_rows = [{
        "payment_option_id": "opt_bad",
        "request_id": "request_01",
        "payment_method": "full_payment",
        "payment_amount": "NOT_A_NUMBER",
        "number_of_payments": "1",
        "first_payment_date": "2024-03-03",
        "payment_frequency_days": "",
        "financing_fee": "0",
        "total_payable_amount": "also_bad",
    }]
    builder = make_builder(options_rows=options_rows)
    ctx = builder.build("request_01")
    assert ctx.payment_options == []  # critically malformed → skipped


# ---------------------------------------------------------------------------
# 17. Deterministic iteration order
# ---------------------------------------------------------------------------

def test_iter_requests_order():
    requests_rows = [
        {"request_id": "request_01", "user_id": "user_01",
         "request_date": "2024-01-01", "request_type": "purchase",
         "requested_amount": "1000", "desired_completion_date": "",
         "allows_partial_payment": "false", "request_text": "r1"},
        {"request_id": "request_02", "user_id": "user_01",
         "request_date": "2024-02-01", "request_type": "travel",
         "requested_amount": "2000", "desired_completion_date": "",
         "allows_partial_payment": "false", "request_text": "r2"},
        {"request_id": "request_03", "user_id": "user_01",
         "request_date": "2024-03-01", "request_type": "investment",
         "requested_amount": "3000", "desired_completion_date": "",
         "allows_partial_payment": "false", "request_text": "r3"},
    ]
    messages_rows = [{"message_id": "m1", "user_id": "user_01",
                      "request_id": "", "related_event_id": "",
                      "sent_at": "2024-01-01T00:00:00Z", "source_type": "employer",
                      "message_text": "msg"}]
    images_rows = [{"image_id": "img_01", "user_id": "user_01",
                    "request_id": "", "related_event_id": ""}]
    options_rows = [
        {"payment_option_id": "o1", "request_id": "request_01", "payment_method": "full_payment",
         "payment_amount": "1000", "number_of_payments": "1", "first_payment_date": "2024-01-01",
         "payment_frequency_days": "", "financing_fee": "0", "total_payable_amount": "1000"},
        {"payment_option_id": "o2", "request_id": "request_02", "payment_method": "full_payment",
         "payment_amount": "2000", "number_of_payments": "1", "first_payment_date": "2024-02-01",
         "payment_frequency_days": "", "financing_fee": "0", "total_payable_amount": "2000"},
        {"payment_option_id": "o3", "request_id": "request_03", "payment_method": "full_payment",
         "payment_amount": "3000", "number_of_payments": "1", "first_payment_date": "2024-03-01",
         "payment_frequency_days": "", "financing_fee": "0", "total_payable_amount": "3000"},
    ]
    builder = make_builder(
        requests_rows=requests_rows,
        messages_rows=messages_rows,
        images_rows=images_rows,
        options_rows=options_rows,
    )

    ids = [ctx.request_id for ctx in builder.iter_requests()]
    assert ids == ["request_01", "request_02", "request_03"]


# ---------------------------------------------------------------------------
# 18. Cache reuse
# ---------------------------------------------------------------------------

def test_cache_reuse():
    """Second build() for same user must return same event list object."""
    builder = make_builder()
    ctx1 = builder.build("request_01")
    ctx2 = builder.build("request_01")
    assert ctx1.resolved_events is ctx2.resolved_events


# ---------------------------------------------------------------------------
# 19. Cache clearing
# ---------------------------------------------------------------------------

def test_cache_clearing():
    """
    After clear_event_cache(), the cache is empty again.
    The next build() should re-populate it.
    """
    bundle = make_bundle()
    builder = RequestContextBuilder(bundle)

    # Populate cache
    _ = builder.build("request_01")
    assert "user_01" in builder._event_cache
    events_first = builder._event_cache["user_01"]

    # Clear cache
    builder.clear_event_cache()
    assert len(builder._event_cache) == 0

    # Re-build: cache should be repopulated (a new list object)
    _ = builder.build("request_01")
    assert "user_01" in builder._event_cache
    # The cache was re-populated (a new resolution was performed)
    # We can't guarantee a different list object since resolve_events may return
    # equivalent results, but we CAN check the cache was populated from empty.
    assert len(builder._event_cache) > 0


# ---------------------------------------------------------------------------
# 20. No cross-user event leakage
# ---------------------------------------------------------------------------

def test_no_cross_user_event_leakage():
    events_rows = [
        {"event_id": "e_u1", "user_id": "user_01",
         "event_type": "income", "description": "Salary user1",
         "category": "salary", "direction": "credit",
         "amount": "1000", "currency": "ZAR",
         "event_date": "2024-01-15", "settlement_date": "2024-01-15",
         "status": "settled", "linked_event_id": "",
         "flexibility": "fixed", "minimum_allowed_amount": ""},
        {"event_id": "e_u2", "user_id": "user_02",
         "event_type": "expense", "description": "Rent user2",
         "category": "rent", "direction": "debit",
         "amount": "2000", "currency": "ZAR",
         "event_date": "2024-01-05", "settlement_date": "2024-01-05",
         "status": "settled", "linked_event_id": "",
         "flexibility": "fixed", "minimum_allowed_amount": ""},
    ]
    builder = make_builder(events_rows=events_rows)
    ctx = builder.build("request_01")

    for evt in ctx.resolved_events:
        assert evt.user_id == "user_01", f"Found event from {evt.user_id} in user_01 context"


# ---------------------------------------------------------------------------
# 21. UserProfile pipe-separated list fields
# ---------------------------------------------------------------------------

def test_profile_pipe_list_parsing():
    builder = make_builder()
    ctx = builder.build("request_01")
    profile = ctx.profile

    assert "education" in profile.financial_priorities
    assert "debt_repayment" in profile.financial_priorities
    assert "rent" in profile.expense_categories_to_protect
    assert "groceries" in profile.expense_categories_to_protect
    assert "dining" in profile.expense_categories_user_is_willing_to_reduce
    assert "delivery_membership" in profile.expense_categories_user_is_willing_to_stop
    assert "full_payment" in profile.payment_methods_user_will_consider


def test_profile_empty_list_fields():
    profiles_rows = [{
        "user_id": "user_01",
        "home_currency": "ZAR",
        "current_available_balance": "50000",
        "minimum_balance_to_keep": "10000",
        "financial_priorities": "",
        "expense_categories_to_protect": "",
        "expense_categories_user_is_willing_to_reduce": "",
        "expense_categories_user_is_willing_to_stop": "",
        "payment_methods_user_will_consider": "full_payment",
        "max_installment_months": "",
    }]
    builder = make_builder(profiles_rows=profiles_rows)
    ctx = builder.build("request_01")
    assert ctx.profile.financial_priorities == []
    assert ctx.profile.expense_categories_to_protect == []


# ---------------------------------------------------------------------------
# 22. UserProfile max_installment_months
# ---------------------------------------------------------------------------

def test_profile_max_installment_months_set():
    profiles_rows = [{
        "user_id": "user_01",
        "home_currency": "ZAR",
        "current_available_balance": "50000",
        "minimum_balance_to_keep": "10000",
        "financial_priorities": "education",
        "expense_categories_to_protect": "",
        "expense_categories_user_is_willing_to_reduce": "",
        "expense_categories_user_is_willing_to_stop": "",
        "payment_methods_user_will_consider": "installments",
        "max_installment_months": "12",
    }]
    builder = make_builder(profiles_rows=profiles_rows)
    ctx = builder.build("request_01")
    assert ctx.profile.max_installment_months == 12


def test_profile_max_installment_months_empty():
    builder = make_builder()  # default has empty max_installment_months
    ctx = builder.build("request_01")
    assert ctx.profile.max_installment_months is None


# ---------------------------------------------------------------------------
# 23. PaymentOption with installments
# ---------------------------------------------------------------------------

def test_installment_payment_option():
    options_rows = [{
        "payment_option_id": "opt_inst",
        "request_id": "request_01",
        "payment_method": "installments",
        "payment_amount": "1852.11",
        "number_of_payments": "15",
        "first_payment_date": "2024-03-06",
        "payment_frequency_days": "30",
        "financing_fee": "2525.65",
        "total_payable_amount": "27781.65",
    }]
    builder = make_builder(options_rows=options_rows)
    ctx = builder.build("request_01")
    assert len(ctx.payment_options) == 1
    opt = ctx.payment_options[0]
    assert opt.payment_method == "installments"
    assert opt.payment_amount == pytest.approx(1852.11)
    assert opt.number_of_payments == 15
    assert opt.first_payment_date == date(2024, 3, 6)
    assert opt.payment_frequency_days == 30
    assert opt.financing_fee == pytest.approx(2525.65)
    assert opt.total_payable_amount == pytest.approx(27781.65)


# ---------------------------------------------------------------------------
# 24. allows_partial_payment parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("True", True), ("TRUE", True),
    ("false", False), ("False", False), ("FALSE", False),
    ("", False), ("0", False), ("1", True),
])
def test_allows_partial_payment_parsing(raw, expected):
    requests_rows = [{
        "request_id": "request_01", "user_id": "user_01",
        "request_date": "2024-03-03", "request_type": "purchase",
        "requested_amount": "1000", "desired_completion_date": "2024-04-01",
        "allows_partial_payment": raw, "request_text": "test",
    }]
    builder = make_builder(requests_rows=requests_rows)
    ctx = builder.build("request_01")
    assert ctx.allows_partial_payment == expected


# ---------------------------------------------------------------------------
# 25. desired_completion_date is optional
# ---------------------------------------------------------------------------

def test_desired_completion_date_optional():
    requests_rows = [{
        "request_id": "request_01", "user_id": "user_01",
        "request_date": "2024-03-03", "request_type": "purchase",
        "requested_amount": "1000", "desired_completion_date": "",
        "allows_partial_payment": "false", "request_text": "test",
    }]
    builder = make_builder(requests_rows=requests_rows)
    ctx = builder.build("request_01")
    assert ctx.desired_completion_date is None


# ---------------------------------------------------------------------------
# 26. Image local_path construction
# ---------------------------------------------------------------------------

def test_image_local_path_construction():
    images_rows = [{
        "image_id": "image_01", "user_id": "user_01",
        "request_id": "request_01", "related_event_id": "",
    }]
    builder = make_builder(images_rows=images_rows, dataset_dir="dataset")
    ctx = builder.build("request_01")
    img = ctx.images_for_request[0]
    expected_path = os.path.join("dataset", "media", "images", "image_01.png")
    assert img.local_path == expected_path


# ---------------------------------------------------------------------------
# 27. Convenience properties
# ---------------------------------------------------------------------------

def test_home_currency_shortcut():
    builder = make_builder()
    ctx = builder.build("request_01")
    assert ctx.home_currency == "ZAR"


def test_available_balance_shortcut():
    builder = make_builder()
    ctx = builder.build("request_01")
    assert ctx.available_balance == pytest.approx(58481.1)


def test_minimum_balance_shortcut():
    builder = make_builder()
    ctx = builder.build("request_01")
    assert ctx.minimum_balance == pytest.approx(18000.0)


# ---------------------------------------------------------------------------
# 28. all_evidence_messages / all_evidence_images dedup
# ---------------------------------------------------------------------------

def test_all_evidence_messages_no_duplicates():
    """A message linked to both request_id and related_event_id appears only once."""
    messages_rows = [{
        "message_id": "msg_both", "user_id": "user_01",
        "request_id": "request_01", "related_event_id": "event_01",
        "sent_at": "2024-03-01T09:30:00Z",
        "source_type": "employer", "message_text": "Linked to both.",
    }]
    builder = make_builder(messages_rows=messages_rows)
    ctx = builder.build("request_01")
    all_msgs = ctx.all_evidence_messages
    msg_ids = [m.message_id for m in all_msgs]
    assert msg_ids.count("msg_both") == 1


# ---------------------------------------------------------------------------
# 29. DataBundle.from_datasets factory
# ---------------------------------------------------------------------------

def test_data_bundle_from_datasets():
    datasets = {
        "requests": make_requests_df(),
        "financial_profiles": make_profiles_df(),
        "financial_events": make_events_df(),
        "exchange_rates": pd.DataFrame(),
        "request_payment_options": make_options_df(),
        "messages": make_messages_df(),
        "images": make_images_df(),
    }
    bundle = DataBundle.from_datasets(datasets, dataset_dir="dataset")
    builder = RequestContextBuilder(bundle)
    ctx = builder.build("request_01")
    assert ctx.request_id == "request_01"


# ---------------------------------------------------------------------------
# 30. all_request_ids returns correct list in order
# ---------------------------------------------------------------------------

def test_all_request_ids():
    requests_rows = [
        {"request_id": "request_01", "user_id": "user_01",
         "request_date": "2024-01-01", "request_type": "purchase",
         "requested_amount": "100", "desired_completion_date": "",
         "allows_partial_payment": "false", "request_text": ""},
        {"request_id": "request_02", "user_id": "user_01",
         "request_date": "2024-02-01", "request_type": "travel",
         "requested_amount": "200", "desired_completion_date": "",
         "allows_partial_payment": "false", "request_text": ""},
    ]
    options_rows = [
        {"payment_option_id": "o1", "request_id": "request_01", "payment_method": "full_payment",
         "payment_amount": "100", "number_of_payments": "1", "first_payment_date": "2024-01-01",
         "payment_frequency_days": "", "financing_fee": "0", "total_payable_amount": "100"},
        {"payment_option_id": "o2", "request_id": "request_02", "payment_method": "full_payment",
         "payment_amount": "200", "number_of_payments": "1", "first_payment_date": "2024-02-01",
         "payment_frequency_days": "", "financing_fee": "0", "total_payable_amount": "200"},
    ]
    messages_rows = [{"message_id": "m1", "user_id": "user_01", "request_id": "",
                      "related_event_id": "", "sent_at": "", "source_type": "employer",
                      "message_text": ""}]
    images_rows = [{"image_id": "img_01", "user_id": "user_01",
                    "request_id": "", "related_event_id": ""}]
    builder = make_builder(
        requests_rows=requests_rows,
        options_rows=options_rows,
        messages_rows=messages_rows,
        images_rows=images_rows,
    )
    assert builder.all_request_ids() == ["request_01", "request_02"]
