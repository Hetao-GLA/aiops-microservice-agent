"""Transparent rule-based baseline for database availability incidents."""

from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import httpx

from app.logging_config import configure_logging


configure_logging()
logger = logging.getLogger("rule-detector")


class DetectionRecorder:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)

    def record(self, **fields: Any) -> dict[str, Any]:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "detector": "operations-health-rule-v2",
            **fields,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry


class OperationsHealthRule:
    """Create one detection incident for each observed fault interval."""

    def __init__(self, recorder: DetectionRecorder) -> None:
        self.recorder = recorder
        self.active_detection_id: str | None = None
        self.active_fault: str | None = None

    def process_sample(
        self,
        *,
        observed_fault: str | None,
        evidence: dict[str, Any],
    ) -> None:
        if (
            observed_fault is not None
            and self.active_detection_id is not None
            and observed_fault != self.active_fault
        ):
            self._end_active_detection(
                evidence={"reason": "fault_transition", **evidence}
            )

        if observed_fault is not None and self.active_detection_id is None:
            self.active_detection_id = f"DET-{uuid4().hex[:12].upper()}"
            self.active_fault = observed_fault
            self.recorder.record(
                detection_id=self.active_detection_id,
                event="detection_started",
                predicted_fault=observed_fault,
                predicted_component=_component_for_fault(observed_fault),
                confidence=1.0,
                evidence=evidence,
            )
            logger.warning(
                "operations_failure_detected",
                extra={
                    "service": "rule-detector",
                    "detection_id": self.active_detection_id,
                    "predicted_fault": observed_fault,
                },
            )
            return

        if observed_fault is None and self.active_detection_id is not None:
            self._end_active_detection(evidence=evidence)

    def _end_active_detection(self, *, evidence: dict[str, Any]) -> None:
        detection_id = self.active_detection_id
        predicted_fault = self.active_fault
        self.recorder.record(
            detection_id=detection_id,
            event="detection_ended",
            predicted_fault=predicted_fault,
            predicted_component=_component_for_fault(predicted_fault),
            confidence=1.0,
            evidence=evidence,
        )
        logger.info(
            "operations_recovery_detected",
            extra={
                "service": "rule-detector",
                "detection_id": detection_id,
                "predicted_fault": predicted_fault,
            },
        )
        self.active_detection_id = None
        self.active_fault = None


def _component_for_fault(fault: str | None) -> str:
    return {
        "service_stopped": "service",
        "database_connection_failure": "database",
        "http_500_failure": "application",
    }.get(fault, "unknown")


def probe_operations(client: httpx.Client) -> tuple[str | None, dict[str, Any]]:
    try:
        response = client.get("/health/live")
        if response.status_code != 200:
            return "service_stopped", {
                "probe": "/health/live",
                "status_code": response.status_code,
            }
    except httpx.HTTPError as exc:
        return "service_stopped", {
            "probe": "/health/live",
            "status_code": None,
            "error_type": type(exc).__name__,
        }

    try:
        response = client.get("/health/database")
        if response.status_code != 200:
            return "database_connection_failure", {
                "probe": "/health/database",
                "status_code": response.status_code,
            }
    except httpx.HTTPError as exc:
        return "database_connection_failure", {
            "probe": "/health/database",
            "status_code": None,
            "error_type": type(exc).__name__,
        }

    try:
        response = client.get("/metrics/orders", params={"window_seconds": 5})
        if response.status_code == 200:
            metrics = response.json()
            if (
                metrics["total_requests"] >= 2
                and metrics["http_500_count"] >= 2
                and metrics["http_500_rate"] >= 0.5
            ):
                return "http_500_failure", {
                    "probe": "/metrics/orders",
                    "status_code": response.status_code,
                    "window_seconds": metrics["window_seconds"],
                    "total_requests": metrics["total_requests"],
                    "http_500_count": metrics["http_500_count"],
                    "http_500_rate": metrics["http_500_rate"],
                }
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        return None, {
            "probe": "/metrics/orders",
            "error_type": type(exc).__name__,
        }

    return None, {
        "probe": "/health/live,/health/database,/metrics/orders",
        "status_code": 200,
    }


def run_detector(
    *,
    base_url: str,
    interval_seconds: float,
    output_path: str | Path,
) -> None:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")

    rule = OperationsHealthRule(DetectionRecorder(output_path))
    with httpx.Client(base_url=base_url, timeout=4.0) as client:
        while True:
            observed_fault, evidence = probe_operations(client)
            rule.process_sample(
                observed_fault=observed_fault,
                evidence=evidence,
            )
            time.sleep(interval_seconds)


def main() -> None:
    run_detector(
        base_url=os.getenv("TARGET_BASE_URL", "http://localhost:8000"),
        interval_seconds=float(os.getenv("DETECTION_INTERVAL_SECONDS", "1")),
        output_path=os.getenv(
            "DETECTION_PATH", "data/detections/detections.jsonl"
        ),
    )


if __name__ == "__main__":
    main()
