"""Pruebas del limitador en memoria, sin red ni consumo de API."""
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from valor_a_la_vista.usage_limits import ApiUsageLimiter, QuotaExceeded


class UsageLimitsTests(unittest.TestCase):
    def test_session_and_global_limits(self):
        limiter = ApiUsageLimiter()
        with patch.dict(os.environ, {"VAV_DAILY_API_LIMIT": "3", "VAV_SESSION_API_LIMIT": "2"}):
            limiter.charge("primera")
            limiter.charge("primera")
            with self.assertRaises(QuotaExceeded):
                limiter.charge("primera")
            limiter.charge("segunda")
            with self.assertRaises(QuotaExceeded):
                limiter.charge("tercera")

    def test_new_day_restores_limits(self):
        limiter = ApiUsageLimiter()
        with patch.dict(os.environ, {"VAV_DAILY_API_LIMIT": "1", "VAV_SESSION_API_LIMIT": "1"}):
            limiter.charge("sesion", now=datetime(2026, 1, 1, tzinfo=timezone.utc))
            limiter.charge("sesion", now=datetime(2026, 1, 2, tzinfo=timezone.utc))
            with self.assertRaises(QuotaExceeded):
                limiter.charge("sesion", now=datetime(2026, 1, 2, tzinfo=timezone.utc))

    def test_invalid_session_is_rejected(self):
        with self.assertRaises(ValueError):
            ApiUsageLimiter().charge("")
