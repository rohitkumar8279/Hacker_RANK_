"""
Validation script: check our engine against sample_request_01.

sample_request_01:
  user_id: user_01
  request_date: 2024-03-03
  request_type: purchase
  requested_amount: 25256 ZAR
  desired_completion_date: 2024-03-20
  allows_partial_payment: true

Expected output:
  amount_safe_to_pay: 25256
  affordability_status: affordable_now
  recommended_payment_method: full_payment
  payment_plan: 2024-03-03:25256
  earliest_date_for_full_payment: 2024-03-03

We'll build a minimal version of user_01's events from the data we observed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import date, timedelta
from src.finance.events import FinancialEvent
from src.finance.forecast import simulate
from src.finance import fx as fx_module
from src.finance.safety import max_safe_amount, earliest_safe_date, is_plan_safe
import pandas as pd

# Init FX with real rates (from exchange_rates.csv inspection)
rates_df = pd.DataFrame([
    {"rate_date": "2024-01-15", "from_currency": "USD", "to_currency": "EUR", "rate": 0.92},
    {"rate_date": "2024-01-15", "from_currency": "USD", "to_currency": "IDR", "rate": 15833.33},
    {"rate_date": "2024-01-15", "from_currency": "USD", "to_currency": "INR", "rate": 83.33},
    {"rate_date": "2024-01-15", "from_currency": "EUR", "to_currency": "ZAR", "rate": 20.0},
    {"rate_date": "2024-02-15", "from_currency": "USD", "to_currency": "EUR", "rate": 0.92},
    {"rate_date": "2024-02-15", "from_currency": "EUR", "to_currency": "ZAR", "rate": 20.0},
    {"rate_date": "2024-03-15", "from_currency": "USD", "to_currency": "EUR", "rate": 0.92},
    {"rate_date": "2024-03-15", "from_currency": "EUR", "to_currency": "ZAR", "rate": 20.0},
])
fx_module.init_fx(rates_df)

# user_01 profile (from financial_profiles.csv):
#   home_currency: ZAR
#   current_available_balance: 58481.1
#   minimum_balance_to_keep: 18000

BALANCE = 58481.1
MIN_BAL = 18000.0
REQUEST_DATE = date(2024, 3, 3)
REQUESTED = 25256.0

def make_evt(event_id, event_date, amount, direction, category="rent", flexibility="fixed", status="settled"):
    return FinancialEvent(
        event_id=event_id, user_id="user_01",
        event_type="expense" if direction == "debit" else "income",
        description="", category=category, direction=direction,
        amount=amount, home_currency="ZAR", event_date=event_date,
        settlement_date=event_date, status=status, linked_event_id=None,
        flexibility=flexibility, minimum_allowed_amount=None,
    )

# Key events from dataset for user_01 in/around the 90-day window from 2024-03-03:
# Rent: 5148 on 2024-03-02 (already settled before request_date)
# Groceries (from local market): ~964 on 2024-03-01
# event_84 local taxi: 339.29 on 2024-03-02
# event_58: local market purchase 964.05 on 2024-03-01 (already past)
# event_102: pending fuel auth 567.6 on 2024-03-02 → EXCLUDED (pending)
# event_103: Next confirmed salary 23320 on 2024-03-15 → INCLUDED (scheduled)

user_events = [
    # March salary (scheduled)
    make_evt("event_103", date(2024, 3, 15), 23320.0, "credit", "salary", status="scheduled"),
    # Pending event_102 excluded → don't add it
]

forecast = simulate(
    user_id="user_01",
    request_date=REQUEST_DATE,
    start_balance=BALANCE,
    minimum_balance=MIN_BAL,
    events=user_events,
)

safe_amt = max_safe_amount(forecast, REQUESTED)
earliest = earliest_safe_date(forecast, REQUESTED)
plan_safety = is_plan_safe(forecast, [(REQUEST_DATE, REQUESTED)])

print(f"\n=== Validation against sample_request_01 ===")
print(f"User:              user_01 (ZAR)")
print(f"Starting balance:  {BALANCE:,.2f}")
print(f"Minimum balance:   {MIN_BAL:,.2f}")
print(f"Requested amount:  {REQUESTED:,.2f}")
print(f"Request date:      {REQUEST_DATE}")
print()
print(f"amount_safe_to_pay:           {safe_amt:,.2f}  (expected: 25256.00)")
print(f"earliest_date_for_full_payment: {earliest}  (expected: 2024-03-03)")
print(f"Plan safe (pay full on date): {plan_safety.safe}  (expected: True)")
print(f"Min balance observed:         {plan_safety.minimum_balance_observed:,.2f}  (must be >= {MIN_BAL:,.2f})")
print()

# After salary on day 12, min_balance_from(salary_day) would be higher
# Before salary: balance stays at 58481.1 - nothing else in window before day 12
# After request payment: 58481.1 - 25256 = 33225.1 > 18000 ✓
# Day 12 (salary): 33225.1 + 23320 = 56545.1
print(f"Balance after payment (day 0): {BALANCE - REQUESTED:,.2f}  (must be >= {MIN_BAL:,.2f})")
print(f"Balance after salary (day 12): {BALANCE - REQUESTED + 23320:,.2f}")

match_amount = abs(safe_amt - REQUESTED) < 0.01
match_date = earliest == REQUEST_DATE
print()
print(f"MATCH amount_safe_to_pay: {'✓' if match_amount else '✗'}")
print(f"MATCH earliest_date:      {'✓' if match_date else '✗'}")
print(f"MATCH plan_safe:          {'✓' if plan_safety.safe else '✗'}")
