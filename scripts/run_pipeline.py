"""
scripts/run_pipeline.py
-----------------------
Runs the full extraction and planning pipeline over the dataset.
"""

import argparse
import csv
import json
import os
import sys
from datetime import date
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from src.data.loader import load_all
from src.data.join import RequestContextBuilder, DataBundle
from src.finance.fx import init_fx
from src.pipeline import process_request
from src.extraction.extractor import clear_extraction_cache, global_pre_extract, _PRE_EXTRACTION_STATS


def format_payment_plan(plan_res) -> str:
    if plan_res.recommended_payment_method == "wait" and plan_res.earliest_date_for_full_payment:
        # Guarantee we format the wait payment explicitly exactly as requested
        amt = plan_res.amount_safe_to_pay
        # Actually the requested amount should be used, but since it's full payment, amt_safe == requested_amount for wait?
        # Let's just rely on plan_res.payments which has the requested amount.
        if plan_res.payments:
            dt, amt = plan_res.payments[0]
            amt_str = f"{int(amt)}" if float(amt).is_integer() else f"{amt:.2f}"
            return f"{dt.isoformat()}:{amt_str}"

    if not plan_res.payments:
        return "none"
    
    formatted = []
    for dt, amt in plan_res.payments:
        amt_str = f"{int(amt)}" if float(amt).is_integer() else f"{amt:.2f}"
        formatted.append(f"{dt.isoformat()}:{amt_str}")
    return "|".join(formatted)

def format_spending_changes(changes: List) -> str:
    if not changes:
        return "none"
    formatted = []
    for c in changes:
        if c.change_type == "stop":
            formatted.append(f"stop:{c.event_id}")
            formatted.append(f"reduce_to:{c.event_id}:{c.new_amount:.2f}")
    return "|".join(formatted)

def format_earliest_date(plan_res) -> str:
    if plan_res.earliest_date_for_full_payment and plan_res.affordability_status != "not_affordable":
        return plan_res.earliest_date_for_full_payment.isoformat()
    return ""



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run a few targeted requests")
    parser.add_argument("--full", action="store_true", help="Run all 250 requests")
    args = parser.parse_args()

    if not args.smoke_test and not args.full:
        print("Please specify --smoke-test or --full")
        sys.exit(1)

    print("Loading datasets...")
    raw_data = load_all(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    raw_data.pop("sample_requests", None)
    raw_data.pop("output", None)
    
    bundle = DataBundle(**raw_data)
    init_fx(bundle.exchange_rates)
    builder = RequestContextBuilder(bundle)
    
    all_reqs = builder.all_request_ids()
    
    if args.smoke_test:
        # Pick specific requests if available
        # request_33 (image), request_01 (sample), and another normal one
        smoke_ids = ["request_33", "request_01", "request_50"]
        reqs_to_run = [r for r in smoke_ids if r in all_reqs]
        if not reqs_to_run:
            reqs_to_run = all_reqs[:3]
        print(f"--- SMOKE TEST MODE --- Running {len(reqs_to_run)} requests: {reqs_to_run}")
    else:
        reqs_to_run = all_reqs
        print(f"--- FULL MODE --- Running {len(reqs_to_run)} requests.")

    clear_extraction_cache()
    
    print("Running Global Pre-Extraction...")
    global_pre_extract(reqs_to_run, builder)
    print("Global Pre-Extraction completed.")

    output_rows = []
    diagnostics = {}
    usages = []

    for idx, rid in enumerate(reqs_to_run):
        if idx % 10 == 0:
            print(f"Processing {idx}/{len(reqs_to_run)}...")
            
        ctx = builder.build(rid)
        
        # Save pre-extraction state for diagnostics
        pre_unresolved = len(ctx.unresolved_events)
        
        plan_res, ext_res = process_request(ctx)
        
        if ext_res.usage:
            usages.append(ext_res.usage)
            
        diagnostics[rid] = {
            "request_id": rid,
            "extracted_facts": len(ext_res.facts),
            "extraction_errors": ext_res.errors,
            "unresolved_events_before": pre_unresolved,
            "status": plan_res.affordability_status,
            "payment_method": plan_res.recommended_payment_method,
            "amount_safe": plan_res.amount_safe_to_pay,
            "num_payments": len(plan_res.payments)
        }
        
        row = {
            "request_id": plan_res.request_id,
            "amount_safe_to_pay": round(plan_res.amount_safe_to_pay, 2),
            "affordability_status": plan_res.affordability_status,
            "recommended_payment_method": plan_res.recommended_payment_method,
            "payment_plan": format_payment_plan(plan_res),
            "earliest_date_for_full_payment": format_earliest_date(plan_res),
            "spending_changes_needed": format_spending_changes(plan_res.spending_changes),
            "decision_explanation": "Decision generated by deterministic engine."
        }
        output_rows.append(row)

    print("Writing output.csv...")
    with open("output.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "request_id", "amount_safe_to_pay", "affordability_status", 
            "recommended_payment_method", "payment_plan", 
            "earliest_date_for_full_payment", "spending_changes_needed", 
            "decision_explanation"
        ])
        writer.writeheader()
        writer.writerows(output_rows)

    print("Writing diagnostics...")
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/run_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    print("Writing usage report...")
    
    total_calls_attempted = sum(1 for d in diagnostics.values() if d.get("extracted_facts") or d.get("extraction_errors"))
    successful_calls = len(usages)
    
    # Calculate failed calls and specifically 429 rate limit failures
    failed_calls = 0
    rate_limit_failures = 0
    for d in diagnostics.values():
        if d.get("extraction_errors"):
            failed_calls += len(d["extraction_errors"])
            for err in d["extraction_errors"]:
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    rate_limit_failures += 1
                    
    total_calls_attempted = successful_calls + failed_calls

    total_in = sum(u.input_tokens for u in usages)
    total_out = sum(u.output_tokens for u in usages)
    total_tokens = total_in + total_out
    total_cost = sum(u.estimated_cost for u in usages if u.estimated_cost)
    
    avg_tokens = (total_tokens / successful_calls) if successful_calls > 0 else 0.0
    
    with open("evaluation/usage_report.md", "w") as f:
        f.write("# Usage Report\n\n")
        f.write("### Provider\nGoogle Gemini\n\n")
        f.write(f"### Model\n{config.GEMINI_MODEL}\n\n")
        
        f.write("### Calls\n")
        f.write(f"- total attempted: {total_calls_attempted}\n")
        f.write(f"- successful calls: {successful_calls}\n")
        f.write(f"- failed calls: {failed_calls}\n")
        f.write(f"- 429 rate-limit failures: {rate_limit_failures}\n\n")
        
        if _PRE_EXTRACTION_STATS["enabled"]:
            f.write("### Global Pre-Extraction\n")
            f.write(f"- unique relevant messages: {_PRE_EXTRACTION_STATS['unique_messages']}\n")
            f.write(f"- unique relevant images: {_PRE_EXTRACTION_STATS['unique_images']}\n")
            f.write(f"- message batches: {_PRE_EXTRACTION_STATS['message_batches']}\n")
            f.write(f"- image batches: {_PRE_EXTRACTION_STATS['image_batches']}\n")
            f.write(f"- invalid attribution count: {_PRE_EXTRACTION_STATS['invalid_attribution_count']}\n")
            f.write(f"- cache hits: {_PRE_EXTRACTION_STATS['cache_hits']}\n")
            f.write(f"- cache misses: {_PRE_EXTRACTION_STATS['cache_misses']}\n\n")
            
        f.write("### Tokens\n")
        f.write(f"- input: {total_in}\n")
        f.write(f"- output: {total_out}\n")
        f.write(f"- total: {total_tokens}\n")
        f.write(f"- average/successful request: {avg_tokens:.2f}\n\n")
        
        f.write("### Cost\n")
        f.write(f"- total estimated cost: ${total_cost:.4f}\n")
        if reqs_to_run:
            f.write(f"- average/request: ${total_cost / len(reqs_to_run):.4f}\n\n")
        
    print("Done!")

if __name__ == "__main__":
    main()
