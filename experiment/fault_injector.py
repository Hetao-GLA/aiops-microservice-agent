"""Host-side, allow-listed fault injection for repeatable experiments."""

import argparse
import os
from pathlib import Path
import subprocess
import time
from typing import Callable, Sequence

import httpx

from experiment.ground_truth import GroundTruthRecorder


CommandRunner = Callable[..., subprocess.CompletedProcess]
SleepFunction = Callable[[float], None]
FaultToggle = Callable[[bool], str]


def _compose_command(
    action: str,
    compose_file: str | None,
    compose_service: str,
) -> list[str]:
    command = ["docker", "compose"]
    if compose_file:
        command.extend(["-f", compose_file])
    command.extend([action, compose_service])
    return command


def _container_stopped_at(
    container_name: str,
    run_command: CommandRunner,
) -> str:
    result = run_command(
        [
            "docker",
            "inspect",
            "--format",
            "{{.State.FinishedAt}}",
            container_name,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    stopped_at = result.stdout.strip()
    if not stopped_at:
        raise RuntimeError(f"Docker did not return the stop time for {container_name}")
    return stopped_at


def _inject_container_stop(
    *,
    duration_seconds: int,
    recorder: GroundTruthRecorder,
    compose_service: str,
    container_name: str,
    affected_service: str,
    fault_type: str,
    root_cause: str,
    compose_file: str | None,
    run_command: CommandRunner,
    sleep: SleepFunction,
) -> str:
    if not 1 <= duration_seconds <= 600:
        raise ValueError("duration_seconds must be between 1 and 600")

    incident_id = recorder.new_incident_id()
    common = {
        "incident_id": incident_id,
        "service": affected_service,
        "fault_type": fault_type,
        "root_cause": root_cause,
    }

    run_command(
        _compose_command("stop", compose_file, compose_service),
        check=True,
        text=True,
    )
    stopped_at = _container_stopped_at(container_name, run_command)
    recorder.record(
        event="fault_started",
        timestamp=stopped_at,
        duration_seconds=duration_seconds,
        timing_source="docker.State.FinishedAt",
        **common,
    )

    recovery_error: Exception | None = None
    try:
        sleep(duration_seconds)
    finally:
        try:
            run_command(
                _compose_command("start", compose_file, compose_service),
                check=True,
                text=True,
            )
        except Exception as exc:  # preserve truth even if restoration fails
            recovery_error = exc

        recorder.record(
            event="fault_ended" if recovery_error is None else "recovery_failed",
            recovery_succeeded=recovery_error is None,
            **common,
        )

    if recovery_error is not None:
        raise recovery_error

    return incident_id


def inject_database_disconnect(
    *,
    duration_seconds: int,
    recorder: GroundTruthRecorder,
    compose_file: str | None = None,
    run_command: CommandRunner = subprocess.run,
    sleep: SleepFunction = time.sleep,
) -> str:
    """Stop PostgreSQL temporarily and record the known cause independently."""
    return _inject_container_stop(
        duration_seconds=duration_seconds,
        recorder=recorder,
        compose_service="database",
        container_name="aiops-database",
        affected_service="order-service",
        fault_type="database_connection_failure",
        root_cause="postgres_container_stopped",
        compose_file=compose_file,
        run_command=run_command,
        sleep=sleep,
    )


def inject_service_stop(
    *,
    duration_seconds: int,
    recorder: GroundTruthRecorder,
    compose_file: str | None = None,
    run_command: CommandRunner = subprocess.run,
    sleep: SleepFunction = time.sleep,
) -> str:
    """Stop the API container temporarily and record the known cause."""
    return _inject_container_stop(
        duration_seconds=duration_seconds,
        recorder=recorder,
        compose_service="api",
        container_name="aiops-order-service",
        affected_service="order-service",
        fault_type="service_stopped",
        root_cause="api_container_stopped",
        compose_file=compose_file,
        run_command=run_command,
        sleep=sleep,
    )


def _http_500_toggle(base_url: str, enabled: bool) -> str:
    with httpx.Client(base_url=base_url, timeout=5.0) as client:
        response = client.post(
            "/internal/faults/http-500",
            json={"enabled": enabled},
        )
        response.raise_for_status()
        return response.json()["changed_at"]


def inject_http_500(
    *,
    duration_seconds: int,
    recorder: GroundTruthRecorder,
    base_url: str = "http://localhost:8000",
    toggle: FaultToggle | None = None,
    sleep: SleepFunction = time.sleep,
) -> str:
    """Enable the controlled order HTTP 500 mode temporarily."""
    if not 1 <= duration_seconds <= 600:
        raise ValueError("duration_seconds must be between 1 and 600")

    incident_id = recorder.new_incident_id()
    common = {
        "incident_id": incident_id,
        "service": "order-service",
        "fault_type": "http_500_failure",
        "root_cause": "forced_application_error_mode",
    }
    selected_toggle = toggle or (lambda enabled: _http_500_toggle(base_url, enabled))

    started_at = selected_toggle(True)
    recorder.record(
        event="fault_started",
        timestamp=started_at,
        duration_seconds=duration_seconds,
        timing_source="application.changed_at",
        **common,
    )

    recovery_error: Exception | None = None
    ended_at: str | None = None
    try:
        sleep(duration_seconds)
    finally:
        try:
            ended_at = selected_toggle(False)
        except Exception as exc:
            recovery_error = exc

        recorder.record(
            event="fault_ended" if recovery_error is None else "recovery_failed",
            timestamp=ended_at,
            recovery_succeeded=recovery_error is None,
            **common,
        )

    if recovery_error is not None:
        raise recovery_error
    return incident_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inject an allow-listed fault and record ground truth."
    )
    parser.add_argument(
        "fault",
        choices=["database_disconnect", "service_stop", "http_500"],
        help="The controlled fault to inject.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Fault duration in seconds (1-600).",
    )
    parser.add_argument(
        "--compose-file",
        default=None,
        help="Optional path to an alternative Compose file.",
    )
    parser.add_argument(
        "--output",
        default=os.getenv(
            "GROUND_TRUTH_PATH", "data/ground_truth/incidents.jsonl"
        ),
        help="Ground-truth JSONL output path.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Target API base URL for application-level faults.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recorder = GroundTruthRecorder(Path(args.output))

    if args.fault == "database_disconnect":
        incident_id = inject_database_disconnect(
            duration_seconds=args.duration,
            recorder=recorder,
            compose_file=args.compose_file,
        )
    elif args.fault == "service_stop":
        incident_id = inject_service_stop(
            duration_seconds=args.duration,
            recorder=recorder,
            compose_file=args.compose_file,
        )
    else:
        incident_id = inject_http_500(
            duration_seconds=args.duration,
            recorder=recorder,
            base_url=args.base_url,
        )

    print(f"Completed controlled incident {incident_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
