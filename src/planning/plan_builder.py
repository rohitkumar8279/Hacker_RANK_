"""
src/planning/plan_builder.py
----------------------------
Builds and evaluates all candidate plans for a RequestContext.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import List

from src.finance.events import detect_and_project_recurring
from src.finance.forecast import simulate
from src.finance.safety import max_safe_amount, earliest_safe_date, is_plan_safe
from src.data.join import RequestContext
from src.planning.models import CandidatePlan, PlanResult
from src.planning.ranking import rank_plans


def build_plan(context: RequestContext) -> PlanResult:
    """
    Evaluates all safe candidate plans, ranks them, and returns the final PlanResult.
    """
    # 1. Base forecast
    # Project recurring events
    forecast_end = context.request_date + timedelta(days=90)
    projected = detect_and_project_recurring(context.resolved_events, context.request_date, forecast_end)
    all_events = context.resolved_events + projected

    forecast = simulate(
        user_id=context.user_id,
        request_date=context.request_date,
        start_balance=context.available_balance,
        minimum_balance=context.minimum_balance,
        events=all_events,
    )

    # 2. Core variables
    amt_safe = max_safe_amount(forecast, context.requested_amount)
    earliest_safe = earliest_safe_date(forecast, context.requested_amount)
    
    allowed_methods = set(context.profile.payment_methods_user_will_consider)
    desired_completion = context.desired_completion_date or date.max

    candidates: List[CandidatePlan] = []

    # -----------------------------------------------------------------------
    # A. Full Payment Candidate
    # -----------------------------------------------------------------------
    if "full_payment" in allowed_methods and amt_safe >= context.requested_amount - 0.01:
        plan = CandidatePlan(
            payment_method="full_payment",
            payments=[(context.request_date, context.requested_amount)],
            total_amount=context.requested_amount,
        )
        plan.safety_result = is_plan_safe(forecast, plan.payments)
        plan.is_safe = plan.safety_result.safe
        
        if plan.is_safe and (plan.completion_date is None or plan.completion_date <= desired_completion):
            candidates.append(plan)

    # -----------------------------------------------------------------------
    # B. Wait Candidate
    # -----------------------------------------------------------------------
    if "full_payment" in allowed_methods and earliest_safe and earliest_safe > context.request_date:
        plan = CandidatePlan(
            payment_method="wait",
            payments=[(earliest_safe, context.requested_amount)],
            total_amount=context.requested_amount,
        )
        plan.safety_result = is_plan_safe(forecast, plan.payments)
        plan.is_safe = plan.safety_result.safe
        
        if plan.is_safe and (plan.completion_date is None or plan.completion_date <= desired_completion):
            candidates.append(plan)

    # -----------------------------------------------------------------------
    # C. Partial Payment Candidate
    # -----------------------------------------------------------------------
    if (
        context.allows_partial_payment 
        and "partial_payment" in allowed_methods 
        and 0 < amt_safe < context.requested_amount 
        and earliest_safe
    ):
        plan = CandidatePlan(
            payment_method="partial_payment",
            payments=[
                (context.request_date, amt_safe),
                (earliest_safe, context.requested_amount - amt_safe),
            ],
            total_amount=context.requested_amount,
        )
        plan.safety_result = is_plan_safe(forecast, plan.payments)
        plan.is_safe = plan.safety_result.safe
        
        if plan.is_safe and (plan.completion_date is None or plan.completion_date <= desired_completion):
            candidates.append(plan)

    # -----------------------------------------------------------------------
    # D. Installment Candidates
    # -----------------------------------------------------------------------
    if "installments" in allowed_methods:
        max_months = context.profile.max_installment_months
        
        for option in context.payment_options:
            if option.payment_method != "installments":
                continue
                
            # Check max installment constraint (approximate by total days / 30)
            if max_months is not None and option.payment_frequency_days:
                total_days = (option.number_of_payments - 1) * option.payment_frequency_days
                if total_days > max_months * 31: # Use 31 to be safe against month length variations
                    continue
            
            # Construct schedule
            schedule = []
            curr_date = option.first_payment_date
            if not curr_date:
                continue
                
            freq = option.payment_frequency_days or 30
            for i in range(option.number_of_payments):
                schedule.append((curr_date, option.payment_amount))
                curr_date = curr_date + timedelta(days=freq)
                
            plan = CandidatePlan(
                payment_method="installments",
                payments=schedule,
                total_amount=option.total_payable_amount,
                payment_option_id=option.payment_option_id,
            )
            
            plan.safety_result = is_plan_safe(forecast, plan.payments)
            plan.is_safe = plan.safety_result.safe
            
            if plan.is_safe and (plan.completion_date is None or plan.completion_date <= desired_completion):
                candidates.append(plan)

    # -----------------------------------------------------------------------
    # Ranking & Status Selection
    # -----------------------------------------------------------------------
    best_plan = rank_plans(candidates)

    status = "not_affordable"
    rec_method = "not_recommended"
    payments = []
    opt_id = None
    tot_amt = 0.0
    s_result = None
    sp_changes = None

    if best_plan:
        rec_method = best_plan.payment_method
        payments = best_plan.payments
        opt_id = best_plan.payment_option_id
        tot_amt = best_plan.total_amount
        s_result = best_plan.safety_result
        sp_changes = best_plan.spending_changes
        
        if rec_method == "full_payment" and best_plan.first_payment_date == context.request_date:
            status = "affordable_now"
        elif rec_method == "wait":
            status = "affordable_later"
        elif rec_method in ("installments", "partial_payment") or sp_changes:
            status = "affordable_with_plan"
        else:
            status = "affordable_now"

    # Edge Case: Full amount is affordable now, but user rejects full_payment, and no other plans exist
    # Then it's not_affordable. The logic above handles this naturally since candidates will be empty.

    return PlanResult(
        request_id=context.request_id,
        amount_safe_to_pay=amt_safe,
        affordability_status=status,
        recommended_payment_method=rec_method,
        payments=payments,
        earliest_date_for_full_payment=earliest_safe if earliest_safe and earliest_safe <= forecast_end else None,
        spending_changes=sp_changes,
        selected_payment_option_id=opt_id,
        total_amount_paid=tot_amt,
        safety_result=s_result,
    )
