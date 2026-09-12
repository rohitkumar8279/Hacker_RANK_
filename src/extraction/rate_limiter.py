"""
src/extraction/rate_limiter.py
------------------------------
A centralized rate limiter and retry mechanism for Gemini API calls to respect Free Tier quotas.

Two classes of quota errors are handled differently:

1. TEMPORARY per-minute 429 (rate limit on generate_content_free_tier_requests/minute):
   - Retry up to max_retries times with the delay suggested by the API.
   - The per-minute window resets after ~60 seconds.

2. DAILY quota exhaustion (GenerateRequestsPerDayPerProjectPerModel):
   - Do NOT retry. The quota resets at midnight UTC — retrying wastes ~25s per request
     and never succeeds.
   - Immediately re-raise so the extractor records the real failure and falls back
     to the deterministic path without any delay.
"""

import time
import re
from collections import deque
from typing import Callable, Any


# ---------------------------------------------------------------------------
# Daily quota detection
# ---------------------------------------------------------------------------

# Patterns that appear in the Gemini API error body specifically for per-day limits.
# These distinguish a permanent daily exhaustion from a temporary per-minute 429.
_DAILY_QUOTA_PATTERNS = [
    "GenerateRequestsPerDayPerProjectPerModel",  # quotaId field in the structured error
    "requests/day",                               # human-readable quota description
    "daily quota",                                # broader match
    "per_day",                                    # alternative form
]


def is_daily_quota_exhausted(error_str: str) -> bool:
    """
    Returns True if the error indicates the Gemini daily quota is exhausted.

    A daily quota error differs from a temporary per-minute 429:
    - Daily errors reference "GenerateRequestsPerDayPerProjectPerModel",
      "requests/day", "daily quota", or "per_day" in the error body.
    - Temporary per-minute errors carry a short retry-after delay (seconds).

    Both are surfaced as RESOURCE_EXHAUSTED / 429, so the message body is inspected.
    """
    lower = error_str.lower()
    return any(pat.lower() in lower for pat in _DAILY_QUOTA_PATTERNS)


def is_temporary_rate_limit(error_str: str) -> bool:
    """
    Returns True if the error is a temporary per-minute rate limit (safe to retry).
    Returns False for daily quota exhaustion or non-rate-limit errors.
    """
    is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
    if not is_rate_limit:
        return False
    return not is_daily_quota_exhausted(error_str)


# ---------------------------------------------------------------------------
# Rate limiter (per-minute window)
# ---------------------------------------------------------------------------

class GeminiRateLimiter:
    def __init__(self, max_requests: int = 5, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_times = deque()
        # Prefer spacing requests safely to avoid bursting.
        # e.g., 60 seconds / 5 requests = 12 seconds per request. We use 12.5 to be safe.
        self.min_spacing = (window_seconds / max_requests) + 0.5

    def wait(self):
        """
        Blocks until a request is permitted by the rate limit constraints.
        """
        now = time.time()

        # Clean up timestamps older than the window
        while self.request_times and now - self.request_times[0] >= self.window_seconds:
            self.request_times.popleft()

        # Enforce rolling window capacity
        if len(self.request_times) >= self.max_requests:
            wait_time = self.window_seconds - (now - self.request_times[0])
            if wait_time > 0:
                time.sleep(wait_time)
                now = time.time()

        # Enforce safe minimum spacing
        if self.request_times:
            time_since_last = now - self.request_times[-1]
            if time_since_last < self.min_spacing:
                time.sleep(self.min_spacing - time_since_last)
                now = time.time()

        self.request_times.append(now)


# Global singleton
_limiter = GeminiRateLimiter()


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def call_with_retry(func: Callable, *args, **kwargs) -> Any:
    """
    Executes the provided function with rate limiting and robust retry logic.

    Retry behaviour:
    - DAILY quota exhaustion: re-raised immediately with NO sleep and NO retry.
      The caller records the error and falls back to the deterministic engine.
    - TEMPORARY per-minute 429: retried up to max_retries times using the
      delay suggested by the API (default 25s if not specified).
    - Any other exception: re-raised immediately without retry.
    """
    max_retries = 3

    for attempt in range(max_retries):
        _limiter.wait()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)

            # --- Daily quota exhaustion: fail fast, no retry, no sleep ---
            if is_daily_quota_exhausted(error_str):
                # Re-raise immediately. Retrying is futile until midnight UTC.
                raise e

            # --- Temporary per-minute rate limit: backoff and retry ---
            if is_temporary_rate_limit(error_str) and attempt < max_retries - 1:
                # Default backoff
                retry_delay = 25.0

                # Parse the exact retry delay suggested by the API.
                # e.g., "Please retry in 22.752129854s."
                match = re.search(r'retry in ([\d\.]+)s', error_str)
                if match:
                    try:
                        retry_delay = float(match.group(1)) + 1.0  # 1s buffer
                    except ValueError:
                        pass

                time.sleep(retry_delay)
                continue

            # --- All other errors (non-rate-limit, or retries exhausted): re-raise ---
            raise e
