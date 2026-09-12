import time
import pytest
from src.extraction.rate_limiter import (
    is_daily_quota_exhausted,
    is_temporary_rate_limit,
    call_with_retry,
    _limiter,
)


class TestIsDailyQuotaExhausted:

    def test_detects_GenerateRequestsPerDayPerProjectPerModel(self):
        err = "429 RESOURCE_EXHAUSTED: quotaId: GenerateRequestsPerDayPerProjectPerModel"
        assert is_daily_quota_exhausted(err) is True

    def test_detects_requests_per_day(self):
        err = "Free tier limit: 50 requests/day exceeded."
        assert is_daily_quota_exhausted(err) is True

    def test_detects_daily_quota(self):
        err = "API error: daily quota has been reached. Please try again tomorrow."
        assert is_daily_quota_exhausted(err) is True

    def test_detects_per_day_variant(self):
        err = "RESOURCE_EXHAUSTED: limit per_day reached for this model."
        assert is_daily_quota_exhausted(err) is True

    def test_case_insensitive(self):
        err = "RESOURCE_EXHAUSTED: DAILY QUOTA EXCEEDED for model gemini-3.6-flash"
        assert is_daily_quota_exhausted(err) is True

    def test_not_daily_for_per_minute_quota(self):
        err = "429 RESOURCE_EXHAUSTED: quotaId: generate_content_free_tier_requests limit: 5 requests/minute"
        assert is_daily_quota_exhausted(err) is False

    def test_not_daily_for_generic_429(self):
        err = "429 Too Many Requests"
        assert is_daily_quota_exhausted(err) is False

    def test_not_daily_for_resource_exhausted_no_day(self):
        err = "RESOURCE_EXHAUSTED: model overloaded, please retry in 22s."
        assert is_daily_quota_exhausted(err) is False

    def test_not_daily_for_unrelated_error(self):
        err = "404 NOT_FOUND: model not found"
        assert is_daily_quota_exhausted(err) is False


class TestIsTemporaryRateLimit:

    def test_temporary_for_per_minute_quota(self):
        err = "429 RESOURCE_EXHAUSTED: quotaId: generate_content_free_tier_requests"
        assert is_temporary_rate_limit(err) is True

    def test_temporary_for_generic_429(self):
        err = "429 Too Many Requests: please retry in 22s."
        assert is_temporary_rate_limit(err) is True

    def test_temporary_for_RESOURCE_EXHAUSTED_no_day(self):
        err = "RESOURCE_EXHAUSTED: capacity exceeded for this minute."
        assert is_temporary_rate_limit(err) is True

    def test_not_temporary_for_daily_quota(self):
        err = "429 RESOURCE_EXHAUSTED: GenerateRequestsPerDayPerProjectPerModel exceeded"
        assert is_temporary_rate_limit(err) is False

    def test_not_temporary_for_requests_per_day(self):
        err = "Quota exceeded: 50 requests/day limit reached."
        assert is_temporary_rate_limit(err) is False

    def test_not_temporary_for_unrelated_error(self):
        err = "500 Internal Server Error"
        assert is_temporary_rate_limit(err) is False


class TestCallWithRetryDailyQuota:

    def test_daily_quota_raises_immediately_no_retry(self, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
        _limiter.request_times.clear()
        call_count = [0]

        def always_daily_quota():
            call_count[0] += 1
            raise Exception("429 RESOURCE_EXHAUSTED: GenerateRequestsPerDayPerProjectPerModel exceeded")

        with pytest.raises(Exception, match="GenerateRequestsPerDayPerProjectPerModel"):
            call_with_retry(always_daily_quota)

        assert call_count[0] == 1, f"Expected 1 call (no retries), got {call_count[0]}"
        long_sleeps = [s for s in sleep_calls if s >= 10]
        assert long_sleeps == [], f"Expected no retry sleep; got {sleep_calls}"

    def test_requests_per_day_raises_immediately(self, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
        _limiter.request_times.clear()
        call_count = [0]

        def always_day_limit():
            call_count[0] += 1
            raise Exception("Free tier: 50 requests/day exceeded for this model.")

        with pytest.raises(Exception, match="requests/day"):
            call_with_retry(always_day_limit)

        assert call_count[0] == 1
        long_sleeps = [s for s in sleep_calls if s >= 10]
        assert long_sleeps == []


class TestCallWithRetryTemporary429:

    def test_temporary_429_retries_three_times(self, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
        _limiter.request_times.clear()
        call_count = [0]

        def always_temp_429():
            call_count[0] += 1
            raise Exception("429 RESOURCE_EXHAUSTED: generate_content_free_tier_requests limit. retry in 30s.")

        with pytest.raises(Exception, match="generate_content_free_tier_requests"):
            call_with_retry(always_temp_429)

        assert call_count[0] == 3, f"Expected 3 attempts, got {call_count[0]}"
        assert len(sleep_calls) >= 2, f"Expected retry sleeps, got {sleep_calls}"

    def test_temporary_429_succeeds_on_retry(self, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
        _limiter.request_times.clear()
        call_count = [0]

        def fail_once_then_succeed():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("429 RESOURCE_EXHAUSTED: generate_content_free_tier_requests")
            return "success"

        result = call_with_retry(fail_once_then_succeed)
        assert result == "success"
        assert call_count[0] == 2
