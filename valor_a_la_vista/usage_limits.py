"""Cuotas por proceso para demostración; NO constituyen un tope monetario garantizado.

Las instancias pueden reiniciarse o multiplicarse. Para uso público masivo se
necesita un contador distribuido, autenticación y límites en el proveedor.
"""
from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
import os


class QuotaExceeded(RuntimeError):
    """Se ha alcanzado el presupuesto de operaciones permitido."""


def _limit(name, default):
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return min(100000, max(1, value))


class ApiUsageLimiter:
    def __init__(self):
        self._lock = Lock()
        self._day = None
        self._total = 0
        self._sessions = {}

    def charge(self, session_id, *, now=None):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Sesión inválida.")
        day = (now or datetime.now(timezone.utc)).date().isoformat()
        total_limit = _limit("VAV_DAILY_API_LIMIT", 32)
        session_limit = _limit("VAV_SESSION_API_LIMIT", 16)
        with self._lock:
            if day != self._day:
                self._day = day
                self._total = 0
                self._sessions.clear()
            if self._total >= total_limit:
                raise QuotaExceeded("Límite diario de consultas de demostración alcanzado; vuelve otro día.")
            spent = self._sessions.get(session_id, 0)
            if spent >= session_limit:
                raise QuotaExceeded("La sesión alcanzó su límite diario de consultas a la IA.")
            self._total += 1
            self._sessions[session_id] = spent + 1
            return self._total, spent + 1
