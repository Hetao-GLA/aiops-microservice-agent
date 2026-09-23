"""Capture host-side probes and Docker logs for one controlled incident."""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import httpx


CommandRunner = Callable[..., subprocess.CompletedProcess]
SleepFunction = Callable[[float], None]

DEFAULT_CONTAINERS: dict[str, str] = {
    "order-service": "aiops-order-service",
    "workload": "aiops-workload-generator",
    "rule-detector": "aiops-rule-detector",
    "database": "aiops-database",
}
PROBE_ENDPOINTS: tuple[tuple[str, str, dict[str, object] | None], ...] = (
    ("liveness", "/health/live", None),
    ("database_health", "/health/database", None),
    ("order_metrics", "/metrics/orders", {"window_seconds": 5}),
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not safe:
        raise ValueError(f"Unsafe empty path component derived from {value!r}")
    return safe


def new_telemetry_run_id(prefix: str) -> str:
    return f"{_safe_component(prefix)}-{_utc_now():%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"


def resolve_probe_timeout(
    interval_seconds: float, configured_timeout: float | None
) -> float:
    if configured_timeout is not None:
        if configured_timeout <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        return configured_timeout
    return max(0.2, min(2.0, interval_seconds * 0.8))


def _rename_directory_with_retry(
    source: Path,
    destination: Path,
    *,
    attempts: int = 8,
    sleep: SleepFunction = time.sleep,
) -> None:
    """Retry transient Windows file locks without replacing existing data."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if source.parent.resolve() != destination.parent.resolve():
        raise ValueError("Rename must stay in the same run directory")
    for attempt in range(attempts):
        if destination.exists():
            raise FileExistsError(destination)
        try:
            source.rename(destination)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33}:
                raise
            if attempt == attempts - 1:
                raise
            sleep(min(0.1 * (2 ** attempt), 1.0))


class JsonlAppendCapture:
    """Archive only JSONL records appended after a run-specific start offset."""

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path)
        self.start_offset: int | None = None

    def start(self) -> None:
        if self.start_offset is not None:
            raise RuntimeError("JSONL append capture has already started")
        self.start_offset = self.source_path.stat().st_size if self.source_path.exists() else 0

    def write_archive(self, output_path: str | Path) -> int:
        if self.start_offset is None:
            raise RuntimeError("JSONL append capture has not started")
        current_size = self.source_path.stat().st_size if self.source_path.exists() else 0
        if current_size < self.start_offset:
            raise RuntimeError("JSONL source was truncated during the experiment")
        if current_size == self.start_offset:
            data = b""
        else:
            with self.source_path.open("rb") as handle:
                handle.seek(self.start_offset)
                data = handle.read()
        if data and not data.endswith(b"\n"):
            raise RuntimeError("JSONL source ended with an incomplete record")

        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid appended JSONL record at line {line_number}"
                ) from exc

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(records)


def collect_probe_sample(
    client: httpx.Client,
    *,
    name: str,
    endpoint: str,
    params: dict[str, object] | None,
) -> dict[str, Any]:
    observed_at = _utc_now()
    started = time.perf_counter()
    sample: dict[str, Any] = {
        "timestamp": _iso(observed_at),
        "probe": name,
        "endpoint": endpoint,
    }
    try:
        response = client.get(endpoint, params=params)
        sample["status_code"] = response.status_code
        sample["available"] = True
        try:
            sample["payload"] = response.json()
        except ValueError:
            sample["payload"] = {"text": response.text[:1000]}
    except httpx.HTTPError as exc:
        sample.update(
            {
                "status_code": None,
                "available": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    sample["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return sample


def parse_docker_log_lines(
    text: str,
    *,
    service: str,
    container: str,
    stream: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line:
            continue
        first, separator, remainder = line.partition(" ")
        has_timestamp = "T" in first and (first.endswith("Z") or "+" in first[10:])
        records.append(
            {
                "timestamp": first if has_timestamp else None,
                "service": service,
                "container": container,
                "stream": stream,
                "message": remainder if has_timestamp and separator else line,
            }
        )
    return records


def capture_docker_logs(
    *,
    started_at: datetime,
    ended_at: datetime,
    containers: dict[str, str],
    run_command: CommandRunner = subprocess.run,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for service, container in containers.items():
        result = run_command(
            [
                "docker",
                "logs",
                "--timestamps",
                "--since",
                _iso(started_at),
                "--until",
                _iso(ended_at),
                container,
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        returncode = int(getattr(result, "returncode", 0))
        stdout = str(getattr(result, "stdout", "") or "")
        stderr = str(getattr(result, "stderr", "") or "")
        if returncode != 0:
            errors.append(
                {
                    "service": service,
                    "container": container,
                    "error": stderr.strip() or f"docker logs exited {returncode}",
                }
            )
            continue
        records.extend(
            parse_docker_log_lines(
                stdout,
                service=service,
                container=container,
                stream="stdout",
            )
        )
        records.extend(
            parse_docker_log_lines(
                stderr,
                service=service,
                container=container,
                stream="stderr",
            )
        )
    records.sort(
        key=lambda record: (
            record["timestamp"] is None,
            record["timestamp"] or "",
            record["service"],
        )
    )
    return records, errors


class LocalTelemetrySession:
    """Background probe sampler plus end-of-incident Docker log export."""

    def __init__(
        self,
        *,
        root: str | Path,
        run_id: str,
        trial_number: int,
        expected_fault: str,
        base_url: str,
        pre_seconds: float = 10.0,
        post_seconds: float = 10.0,
        interval_seconds: float = 1.0,
        request_timeout_seconds: float | None = None,
        containers: dict[str, str] | None = None,
        run_command: CommandRunner = subprocess.run,
        sleep: SleepFunction = time.sleep,
    ) -> None:
        if trial_number < 1:
            raise ValueError("trial_number must be at least 1")
        if pre_seconds < 0 or post_seconds < 0:
            raise ValueError("pre_seconds and post_seconds cannot be negative")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.root = Path(root)
        self.run_id = _safe_component(run_id)
        self.trial_number = trial_number
        self.expected_fault = expected_fault
        self.base_url = base_url
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.interval_seconds = interval_seconds
        self.request_timeout_seconds = resolve_probe_timeout(
            interval_seconds, request_timeout_seconds
        )
        self.containers = dict(containers or DEFAULT_CONTAINERS)
        self.run_command = run_command
        self.sleep = sleep
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None
        self.sample_count = 0
        self.collector_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._working_dir = (
            self.root
            / self.run_id
            / f".trial-{trial_number:03d}-{uuid4().hex[:8]}"
        )
        self._probe_path = self._working_dir / "probes.jsonl"

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Telemetry session has already started")
        self._working_dir.mkdir(parents=True, exist_ok=False)
        self.started_at = _utc_now()
        self._thread = threading.Thread(
            target=self._probe_loop,
            name=f"telemetry-trial-{self.trial_number}",
            daemon=True,
        )
        self._thread.start()
        if self.pre_seconds:
            self.sleep(self.pre_seconds)

    def finish(self, incident_id: str) -> Path:
        if self.started_at is None or self._thread is None:
            raise RuntimeError("Telemetry session has not started")
        if self.post_seconds:
            self.sleep(self.post_seconds)
        self._stop_and_join()
        self.ended_at = _utc_now()

        docker_records, docker_errors = capture_docker_logs(
            started_at=self.started_at,
            ended_at=self.ended_at,
            containers=self.containers,
            run_command=self.run_command,
        )
        docker_path = self._working_dir / "docker-logs.jsonl"
        with docker_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in docker_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        final_dir = self.root / self.run_id / _safe_component(incident_id)
        if final_dir.exists():
            raise FileExistsError(final_dir)
        metadata = {
            "schema_version": 1,
            "run_id": self.run_id,
            "trial_number": self.trial_number,
            "incident_id": incident_id,
            "expected_fault": self.expected_fault,
            "collector_started_at": _iso(self.started_at),
            "collector_ended_at": _iso(self.ended_at),
            "pre_seconds": self.pre_seconds,
            "post_seconds": self.post_seconds,
            "interval_seconds": self.interval_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
            "probe_sample_count": self.sample_count,
            "docker_log_record_count": len(docker_records),
            "containers": self.containers,
            "collector_error": self.collector_error,
            "docker_capture_errors": docker_errors,
        }
        (self._working_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _rename_directory_with_retry(self._working_dir, final_dir)

        if self.collector_error is not None:
            raise RuntimeError(f"Probe collector failed: {self.collector_error}")
        if docker_errors:
            failed = ", ".join(item["container"] for item in docker_errors)
            raise RuntimeError(f"Docker log capture failed for: {failed}")
        return final_dir

    def abort(self, error: BaseException) -> Path | None:
        if self.started_at is None:
            return None
        self._stop_and_join()
        self.ended_at = _utc_now()
        if not self._working_dir.exists():
            return None
        aborted_dir = (
            self.root
            / self.run_id
            / f"aborted-trial-{self.trial_number:03d}-{uuid4().hex[:8]}"
        )
        (self._working_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "trial_number": self.trial_number,
                    "expected_fault": self.expected_fault,
                    "collector_started_at": _iso(self.started_at),
                    "collector_ended_at": _iso(self.ended_at),
                    "probe_sample_count": self.sample_count,
                    "collector_error": self.collector_error,
                    "experiment_error_type": type(error).__name__,
                    "experiment_error": str(error),
                    "aborted": True,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _rename_directory_with_retry(self._working_dir, aborted_dir)
        return aborted_dir

    def _stop_and_join(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 3))
            if self._thread.is_alive():
                raise RuntimeError("Telemetry probe thread did not stop")

    def _probe_loop(self) -> None:
        try:
            with httpx.Client(
                base_url=self.base_url, timeout=self.request_timeout_seconds
            ) as client:
                with self._probe_path.open("a", encoding="utf-8", newline="\n") as handle:
                    while not self._stop.is_set():
                        cycle_started = time.monotonic()
                        for name, endpoint, params in PROBE_ENDPOINTS:
                            sample = collect_probe_sample(
                                client,
                                name=name,
                                endpoint=endpoint,
                                params=params,
                            )
                            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                            self.sample_count += 1
                        handle.flush()
                        elapsed = time.monotonic() - cycle_started
                        self._stop.wait(max(0.0, self.interval_seconds - elapsed))
        except Exception as exc:  # retained in metadata and raised by finish
            self.collector_error = f"{type(exc).__name__}: {exc}"
