"""
scripts/validate_output.py
--------------------------
Independent validation script for output.csv to enforce the final submission contract.
"""

import argparse
import csv
import re
import sys


def validate_output(filepath: str):
    try:
        with open(filepath, "r") as f:
            reader = list(csv.DictReader(f))
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
        sys.exit(1)

    if len(reader) != 250:
        print(f"Error: Expected 250 rows, got {len(reader)}")
        sys.exit(1)

    expected_cols = [
        "request_id", "amount_safe_to_pay", "affordability_status", 
        "recommended_payment_method", "payment_plan", 
        "earliest_date_for_full_payment", "spending_changes_needed", 
        "decision_explanation"
    ]
    
    if list(reader[0].keys()) != expected_cols:
        print("Error: Column names/order do not match required contract.")
        print(f"Found: {list(reader[0].keys())}")
        sys.exit(1)

    req_ids = set()
    errors = []

    for idx, row in enumerate(reader, start=2):  # start=2 for header + 1-indexed
        rid = row["request_id"]
        if rid in req_ids:
            errors.append(f"Row {idx}: Duplicate request_id {rid}")
        req_ids.add(rid)

        # Status
        allowed_statuses = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
        if row["affordability_status"] not in allowed_statuses:
            errors.append(f"Row {idx}: Invalid status '{row['affordability_status']}'")

        # Payment method
        allowed_methods = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
        if row["recommended_payment_method"] not in allowed_methods:
            errors.append(f"Row {idx}: Invalid payment method '{row['recommended_payment_method']}'")

        # Amount
        try:
            amt = float(row["amount_safe_to_pay"])
            if amt < 0:
                errors.append(f"Row {idx}: Negative amount_safe_to_pay {amt}")
        except ValueError:
            errors.append(f"Row {idx}: Non-numeric amount_safe_to_pay '{row['amount_safe_to_pay']}'")

        # Plan
        plan = row["payment_plan"]
        if plan != "none" and plan != "":
            # Format: YYYY-MM-DD:amount|YYYY-MM-DD:amount
            for payment in plan.split("|"):
                if not re.match(r"^\d{4}-\d{2}-\d{2}:\d+(\.\d+)?$", payment):
                    errors.append(f"Row {idx}: Invalid payment plan format '{payment}'")

        # Earliest date
        date_str = row["earliest_date_for_full_payment"]
        if date_str:
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
                errors.append(f"Row {idx}: Invalid earliest_date format '{date_str}'")

        # Spending changes
        changes = row["spending_changes_needed"]
        if changes:
            parts = changes.split("|")
            if len(parts) > 3:
                errors.append(f"Row {idx}: Too many spending changes (max 3)")
            for part in parts:
                if not part.startswith("stop:") and not part.startswith("reduce_to:"):
                    errors.append(f"Row {idx}: Invalid spending change '{part}'")

    if errors:
        print("Validation Failed with the following errors:")
        for err in errors[:50]:
            print(err)
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more errors.")
        sys.exit(1)
        
    print("Validation passed successfully! All contracts satisfied.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to output.csv")
    args = parser.parse_args()
    validate_output(args.file)
