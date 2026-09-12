"""
config.py
---------
Central configuration for the Affordability Agent.
All values are read from environment variables (with sensible defaults).
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables directly

# LLM
LLM_ENABLED: bool = os.getenv("LLM_ENABLED", "true").lower() == "true"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

# Caching
CACHE_DIR: str = os.getenv("CACHE_DIR", "evaluation/.cache")

# Forecast
FORECAST_HORIZON_DAYS: int = 90   # calendar days (inclusive: day 0 … day 90)

# Token pricing table (per 1k tokens, for usage_report.md generation)
PRICING: dict = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gemini-3.6-flash": {"input": 0.0, "output": 0.0},
    "gemini-2.5-flash": {"input": 0.0, "output": 0.0},
    "gemini-1.5-flash": {"input": 0.0, "output": 0.0},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

# Dev/debug
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
DEV_MODE: bool = os.getenv("DEV_MODE", "false").lower() == "true"
