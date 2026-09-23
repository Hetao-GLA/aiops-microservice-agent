"""Small in-memory metrics window for the rule-based experiment baseline."""

from collections import deque
from threading import Lock
import time


class OrderMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._events: deque[tuple[float, int]] = deque()

    def record(self, status_code: int) -> None:
        with self._lock:
            self._events.append((time.monotonic(), status_code))
            self._discard_old_events(max_age_seconds=300)

    def snapshot(self, window_seconds: float) -> dict[str, object]:
        if not 1 <= window_seconds <= 300:
            raise ValueError("window_seconds must be between 1 and 300")

        with self._lock:
            cutoff = time.monotonic() - window_seconds
            statuses = [
                status_code
                for observed_at, status_code in self._events
                if observed_at >= cutoff
            ]

        total = len(statuses)
        http_500_count = sum(status_code == 500 for status_code in statuses)
        failure_count = sum(status_code >= 500 for status_code in statuses)
        return {
            "window_seconds": window_seconds,
            "total_requests": total,
            "successful_requests": sum(status_code < 400 for status_code in statuses),
            "failure_count": failure_count,
            "http_500_count": http_500_count,
            "http_500_rate": round(http_500_count / total, 4) if total else 0.0,
        }

    def _discard_old_events(self, *, max_age_seconds: float) -> None:
        cutoff = time.monotonic() - max_age_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()


order_metrics = OrderMetrics()

