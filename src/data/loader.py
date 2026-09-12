"""
src/data/loader.py
------------------
Load all dataset CSVs once into typed DataFrames.
"""

import os
from typing import Dict

import pandas as pd


EXPECTED_FILES = [
    "requests.csv",
    "sample_requests.csv",
    "financial_profiles.csv",
    "financial_events.csv",
    "exchange_rates.csv",
    "request_payment_options.csv",
    "messages.csv",
    "images.csv",
    "output.csv",
]


def load_all(dataset_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load every CSV from dataset_dir into a dict of DataFrames.
    Key = filename without extension (e.g. "financial_events").
    Missing files are stored as empty DataFrames with a warning.
    """
    datasets: Dict[str, pd.DataFrame] = {}

    for filename in EXPECTED_FILES:
        key = filename.replace(".csv", "")
        filepath = os.path.join(dataset_dir, filename)
        if os.path.exists(filepath):
            try:
                datasets[key] = pd.read_csv(filepath, dtype=str, keep_default_na=False)
            except Exception as exc:
                print(f"[loader] ERROR reading {filename}: {exc}")
                datasets[key] = pd.DataFrame()
        else:
            if filename not in ("sample_requests.csv", "output.csv"):
                print(f"[loader] WARNING: {filename} not found in {dataset_dir}")
            datasets[key] = pd.DataFrame()

    print(f"[loader] Loaded {len(datasets)} datasets from {dataset_dir}")
    for key, df in datasets.items():
        if not df.empty:
            print(f"  {key}: {len(df)} rows, columns: {list(df.columns)}")

    return datasets
