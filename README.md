# Affordability Agent

A robust, enterprise-grade deterministic finance engine coupled with an LLM-assisted extraction layer.

## Problem Solved
The Affordability Agent assesses whether users can afford requested purchases or payments, taking into account their current financial profile, historical events, upcoming payments, and unstructured evidence like emails or receipts. 

## Architecture
The system employs a hybrid architecture:
1. **Extraction Layer (LLM):** Parses unstructured data (messages, images) into structured financial facts.
2. **Decision Engine (Deterministic):** Resolves events, calculates affordability bounds, forecasts daily balances, and ranks payment plans securely.

**IMPORTANT ARCHITECTURE MESSAGE:**
> Gemini is used for evidence extraction from unstructured messages/images. It does not make the final financial decision. The deterministic finance engine applies the affordability, forecasting, safety, payment eligibility, and ranking rules.

### Key Features
- **Deterministic Financial Decision Engine**: Enforces strict financial safety boundaries and mathematically guarantees no user will over-draft.
- **Role of Gemini**: Gemini Vision and Text models accurately extract critical context (cancellations, amendments) from untrusted evidence.
- **90-day Safety Forecasting**: Simulates daily account balances up to 90 days in the future against a dynamically generated schedule of resolved events and FX conversions.
- **Payment-Plan Generation & Ranking**: Generates valid payment candidates (full, partial, installments) based on the safe boundaries, ranks them logically according to user preference, and assigns the best fit.
- **Global Evidence Pre-Extraction & Batching**: Groups and extracts all required evidence across the dataset in bulk prior to execution, minimizing API calls and maximizing token efficiency under tight API quotas.
- **Safe Fallback**: If LLM extraction is unavailable (e.g., rate limits, offline mode), the engine safely defaults to deterministic behaviors without crashing or halting the pipeline.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

## Usage

### Running the pipeline
Run the complete pipeline over the dataset:
```bash
python scripts/run_pipeline.py --full
```
Run a small integration smoke test:
```bash
python scripts/run_pipeline.py --smoke-test
```

### Running tests
The project features a comprehensive test suite (177 tests) validating the deterministic engine and LLM mocks.
```bash
pytest -q
```

### Output format
The primary output is written to `output.csv`. It contains the deterministic assessment for each request:
- `request_id`
- `amount_safe_to_pay`
- `affordability_status`
- `recommended_payment_method`
- `payment_plan`
- `earliest_date_for_full_payment`
- `spending_changes_needed`
- `decision_explanation`

### Evaluation / Usage Reporting
Upon completion, the pipeline automatically generates detailed diagnostics in the `evaluation/` directory:
- `run_diagnostics.json`: Request-level debugging info.
- `usage_report.md`: Detailed breakdown of LLM API calls, token counts, quota impacts, and cache hit metrics for the global pre-extraction pass.
