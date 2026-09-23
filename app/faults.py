"""Controlled in-process fault state used only by the experiment platform."""

from datetime import UTC, datetime
from threading import Lock


class FaultState:
    def __init__(self) -> None:
        self._lock = Lock()
        self._http_500_enabled = False
        self._changed_at = datetime.now(UTC)

    def set_http_500(self, enabled: bool) -> dict[str, object]:
        with self._lock:
            self._http_500_enabled = enabled
            self._changed_at = datetime.now(UTC)
            return {
                "fault": "http_500_failure",
                "enabled": self._http_500_enabled,
                "changed_at": self._changed_at.isoformat(),
            }

    def http_500_enabled(self) -> bool:
        with self._lock:
            return self._http_500_enabled

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "fault": "http_500_failure",
                "enabled": self._http_500_enabled,
                "changed_at": self._changed_at.isoformat(),
            }


fault_state = FaultState()

