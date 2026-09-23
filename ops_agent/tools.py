"""Read-only operational evidence tools exposed to the planner."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import subprocess
from typing import Any

import httpx

from ops_agent.models import FaultType
from ops_agent.runbooks import RunbookRegistry


CommandRunner = Callable[..., subprocess.CompletedProcess]
ALLOWED_LOG_SERVICES = frozenset({"api", "database", "workload", "detector"})


class EvidenceToolkit:
    def __init__(
        self,
        *,
        base_url: str,
        detection_path: str | Path,
        project_root: str | Path,
        compose_file: str | Path | None,
        command_timeout_seconds: float = 30.0,
        run_command: CommandRunner = subprocess.run,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.detection_path = Path(detection_path)
        self.project_root = Path(project_root)
        self.compose_file = Path(compose_file) if compose_file else None
        self.command_timeout_seconds = command_timeout_seconds
        self.run_command = run_command

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            with httpx.Client(base_url=self.base_url, timeout=4.0) as client:
                response = client.request(method, path, params=params, json=json_body)
            try:
                body: Any = response.json()
            except ValueError:
                body = {"text": response.text[:1000]}
            return {
                "reachable": True,
                "status_code": response.status_code,
                "body": body,
            }
        except httpx.HTTPError as exc:
            return {
                "reachable": False,
                "status_code": None,
                "error_type": type(exc).__name__,
            }

    def collect_health(self) -> dict[str, Any]:
        return {
            "liveness": self._request("GET", "/health/live"),
            "database": self._request("GET", "/health/database"),
        }

    def collect_metrics(self) -> dict[str, Any]:
        return self._request(
            "GET", "/metrics/orders", params={"window_seconds": 5}
        )

    def collect_fault_control(self) -> dict[str, Any]:
        return self._request("GET", "/internal/faults/http-500")

    def recent_detections(self, limit: int = 10) -> list[dict[str, Any]]:
        if not 1 <= limit <= 100:
            raise ValueError("Detection limit must be between 1 and 100")
        if not self.detection_path.is_file():
            return []
        records: list[dict[str, Any]] = []
        for line in self.detection_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"invalid_json_record": True})
        return records[-limit:]

    def recent_logs(self, service: str, tail: int = 80) -> dict[str, Any]:
        if service not in ALLOWED_LOG_SERVICES:
            raise ValueError(f"Log service is not allow-listed: {service}")
        if not 1 <= tail <= 200:
            raise ValueError("Log tail must be between 1 and 200")
        command = ["docker", "compose"]
        if self.compose_file is not None:
            command.extend(["-f", str(self.compose_file)])
        command.extend(["logs", "--no-color", "--tail", str(tail), service])
        try:
            result = self.run_command(
                command,
                cwd=self.project_root,
                check=True,
                text=True,
                capture_output=True,
                timeout=self.command_timeout_seconds,
            )
            output = result.stdout[-12000:]
            return {"service": service, "tail": tail, "logs": output}
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "service": service,
                "tail": tail,
                "logs": "",
                "error_type": type(exc).__name__,
            }

    def collect_snapshot(self) -> dict[str, Any]:
        return {
            "health": self.collect_health(),
            "metrics": self.collect_metrics(),
            "fault_control": self.collect_fault_control(),
            "recent_detections": self.recent_detections(limit=5),
        }

    @staticmethod
    def classify_snapshot(
        snapshot: dict[str, Any],
    ) -> tuple[FaultType | None, float, dict[str, Any]]:
        health = snapshot.get("health", {})
        liveness = health.get("liveness", {})
        if not liveness.get("reachable") or liveness.get("status_code") != 200:
            return "service_stopped", 1.0, {"decisive_signal": "liveness"}

        database = health.get("database", {})
        if not database.get("reachable") or database.get("status_code") != 200:
            return "database_connection_failure", 1.0, {
                "decisive_signal": "database_health"
            }

        metrics = snapshot.get("metrics", {})
        body = metrics.get("body", {}) if metrics.get("status_code") == 200 else {}
        try:
            if (
                int(body.get("total_requests", 0)) >= 2
                and int(body.get("http_500_count", 0)) >= 2
                and float(body.get("http_500_rate", 0.0)) >= 0.5
            ):
                return "http_500_failure", 1.0, {
                    "decisive_signal": "http_500_rate",
                    "http_500_rate": float(body["http_500_rate"]),
                }
        except (TypeError, ValueError):
            pass
        return None, 0.0, {"decisive_signal": "none"}


class ReadOnlyToolExecutor:
    """Dispatch exactly the read-only functions available to an LLM planner."""

    def __init__(
        self, toolkit: EvidenceToolkit, runbooks: RunbookRegistry
    ) -> None:
        self.toolkit = toolkit
        self.runbooks = runbooks

    def __call__(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == "get_current_health":
            return self.toolkit.collect_health()
        if name == "get_current_metrics":
            return self.toolkit.collect_metrics()
        if name == "get_recent_logs":
            return self.toolkit.recent_logs(
                str(arguments["service"]), int(arguments["tail"])
            )
        if name == "get_recent_detections":
            return self.toolkit.recent_detections(int(arguments["limit"]))
        if name == "get_runbook":
            return self.runbooks.as_tool_output(arguments["fault_type"])
        raise ValueError(f"Unknown or non-read-only Agent tool: {name}")
