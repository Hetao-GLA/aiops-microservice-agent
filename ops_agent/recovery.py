"""Allow-listed recovery execution and post-action verification."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
import subprocess
import time
from typing import Any

import httpx

from ops_agent.models import ActionId


CommandRunner = Callable[..., subprocess.CompletedProcess]


class RecoveryExecutor:
    def __init__(
        self,
        *,
        base_url: str,
        project_root: str | Path,
        compose_file: str | Path | None,
        command_timeout_seconds: float = 30.0,
        verification_timeout_seconds: float = 30.0,
        verification_interval_seconds: float = 1.0,
        run_command: CommandRunner = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.project_root = Path(project_root)
        self.compose_file = Path(compose_file) if compose_file else None
        self.command_timeout_seconds = command_timeout_seconds
        self.verification_timeout_seconds = verification_timeout_seconds
        self.verification_interval_seconds = verification_interval_seconds
        self.run_command = run_command
        self.sleep = sleep
        self.monotonic = monotonic

    def _compose_start(self, service: str) -> dict[str, Any]:
        if service not in {"api", "database"}:
            raise ValueError("Compose recovery service is not allow-listed")
        command = ["docker", "compose"]
        if self.compose_file is not None:
            command.extend(["-f", str(self.compose_file)])
        command.extend(["start", service])
        result = self.run_command(
            command,
            cwd=self.project_root,
            check=True,
            text=True,
            capture_output=True,
            timeout=self.command_timeout_seconds,
        )
        return {
            "operation": "compose_start",
            "service": service,
            "stdout": result.stdout.strip()[-2000:],
        }

    def _disable_http_500(self) -> dict[str, Any]:
        with httpx.Client(base_url=self.base_url, timeout=5.0) as client:
            response = client.post(
                "/internal/faults/http-500", json={"enabled": False}
            )
            response.raise_for_status()
            payload = response.json()
        return {
            "operation": "disable_controlled_http_500",
            "enabled": bool(payload["enabled"]),
            "changed_at": payload.get("changed_at"),
        }

    def execute(self, action_id: ActionId) -> dict[str, Any]:
        if action_id == "start_database":
            return self._compose_start("database")
        if action_id == "start_api":
            return self._compose_start("api")
        if action_id == "disable_http_500":
            return self._disable_http_500()
        raise ValueError("Recovery action is not allow-listed")

    def _probe(self, path: str) -> dict[str, Any]:
        try:
            with httpx.Client(base_url=self.base_url, timeout=4.0) as client:
                response = client.get(path)
            try:
                body: Any = response.json()
            except ValueError:
                body = None
            return {"status_code": response.status_code, "body": body}
        except httpx.HTTPError as exc:
            return {"status_code": None, "error_type": type(exc).__name__}

    def _verification_checks(self, action_id: ActionId) -> dict[str, Any]:
        live = self._probe("/health/live")
        database = self._probe("/health/database")
        checks: dict[str, Any] = {
            "liveness": live,
            "database": database,
            "liveness_passed": live.get("status_code") == 200,
            "database_passed": database.get("status_code") == 200,
        }
        if action_id == "disable_http_500":
            fault = self._probe("/internal/faults/http-500")
            enabled = (fault.get("body") or {}).get("enabled")
            checks["http_500_fault"] = fault
            checks["http_500_disabled"] = enabled is False
        return checks

    @staticmethod
    def _passed(action_id: ActionId, checks: dict[str, Any]) -> bool:
        base = bool(checks["liveness_passed"] and checks["database_passed"])
        if action_id == "disable_http_500":
            return base and bool(checks.get("http_500_disabled"))
        return base

    def verify(self, action_id: ActionId) -> dict[str, Any]:
        deadline = self.monotonic() + self.verification_timeout_seconds
        attempts = 0
        checks: dict[str, Any] = {}
        while True:
            attempts += 1
            checks = self._verification_checks(action_id)
            if self._passed(action_id, checks):
                return {
                    "passed": True,
                    "checked_at": datetime.now(UTC).isoformat(),
                    "attempts": attempts,
                    "checks": checks,
                }
            if self.monotonic() >= deadline:
                return {
                    "passed": False,
                    "checked_at": datetime.now(UTC).isoformat(),
                    "attempts": attempts,
                    "checks": checks,
                }
            self.sleep(self.verification_interval_seconds)
