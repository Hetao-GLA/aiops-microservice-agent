"""Run a balanced, reproducibly shuffled batch of all supported faults."""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import random
import time
from typing import Callable, Sequence

from experiment.batch_runner import wait_until_healthy
from experiment.fault_injector import (
    inject_database_disconnect,
    inject_http_500,
    inject_service_stop,
)
from experiment.ground_truth import GroundTruthRecorder
from experiment.local_telemetry import (
    JsonlAppendCapture,
    LocalTelemetrySession,
    new_telemetry_run_id,
    resolve_probe_timeout,
)


SUPPORTED_FAULTS = (
    "database_disconnect",
    "service_stop",
    "http_500",
)


def build_balanced_schedule(
    *,
    repetitions_per_fault: int,
    seed: int,
) -> list[str]:
    if repetitions_per_fault < 1:
        raise ValueError("repetitions_per_fault must be at least 1")
    schedule = [
        fault
        for fault in SUPPORTED_FAULTS
        for _ in range(repetitions_per_fault)
    ]
    random.Random(seed).shuffle(schedule)
    return schedule


def run_mixed_batch(
    *,
    repetitions_per_fault: int,
    seed: int,
    duration_seconds: int,
    cooldown_seconds: float,
    base_url: str,
    ground_truth_path: str | Path,
    result_path: str | Path,
    injectors: dict[str, Callable[..., str]] | None = None,
    telemetry_root: str | Path | None = None,
    telemetry_pre_seconds: float = 10.0,
    telemetry_post_seconds: float = 10.0,
    telemetry_interval_seconds: float = 1.0,
    telemetry_request_timeout_seconds: float | None = None,
    detection_source_path: str | Path | None = None,
    detection_archive_dir: str | Path = "data/detections",
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    if cooldown_seconds < 0:
        raise ValueError("cooldown_seconds cannot be negative")

    selected_injectors = injectors or {
        "database_disconnect": inject_database_disconnect,
        "service_stop": inject_service_stop,
        "http_500": inject_http_500,
    }
    schedule = build_balanced_schedule(
        repetitions_per_fault=repetitions_per_fault,
        seed=seed,
    )
    recorder = GroundTruthRecorder(ground_truth_path)
    trials: list[dict[str, object]] = []
    telemetry_directories: list[str] = []
    started_at = datetime.now(UTC)
    run_id = (
        new_telemetry_run_id("mixed")
        if telemetry_root is not None or detection_source_path is not None
        else None
    )
    resolved_telemetry_timeout = (
        resolve_probe_timeout(
            telemetry_interval_seconds, telemetry_request_timeout_seconds
        )
        if telemetry_root is not None
        else None
    )

    if not wait_until_healthy(
        base_url=base_url,
        endpoint="/health/database",
        timeout_seconds=30,
    ):
        raise RuntimeError("system was not healthy before the mixed batch")

    detection_capture = (
        JsonlAppendCapture(detection_source_path)
        if detection_source_path is not None
        else None
    )
    if detection_capture is not None:
        detection_capture.start()

    for trial_number, fault in enumerate(schedule, start=1):
        telemetry_session = None
        if telemetry_root is not None and run_id is not None:
            telemetry_session = LocalTelemetrySession(
                root=telemetry_root,
                run_id=run_id,
                trial_number=trial_number,
                expected_fault={
                    "database_disconnect": "database_connection_failure",
                    "service_stop": "service_stopped",
                    "http_500": "http_500_failure",
                }[fault],
                base_url=base_url,
                pre_seconds=telemetry_pre_seconds,
                post_seconds=telemetry_post_seconds,
                interval_seconds=telemetry_interval_seconds,
                request_timeout_seconds=resolved_telemetry_timeout,
            )
        arguments: dict[str, object] = {
            "duration_seconds": duration_seconds,
            "recorder": recorder,
        }
        if fault == "http_500":
            arguments["base_url"] = base_url
        try:
            if telemetry_session is not None:
                telemetry_session.start()
            incident_id = selected_injectors[fault](**arguments)
            trials.append(
                {
                    "trial_number": trial_number,
                    "injected_fault": fault,
                    "incident_id": incident_id,
                }
            )

            if not wait_until_healthy(
                base_url=base_url,
                endpoint="/health/database",
                timeout_seconds=30,
            ):
                raise RuntimeError(
                    f"system did not recover after mixed trial {trial_number}"
                )
            if telemetry_session is not None:
                telemetry_directories.append(
                    str(telemetry_session.finish(incident_id))
                )
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": trial_number,
                        "total_trials": len(schedule),
                        "injected_fault": fault,
                        "incident_id": incident_id,
                    }
                )
        except BaseException as exc:
            if telemetry_session is not None:
                telemetry_session.abort(exc)
            raise

        if trial_number < len(schedule):
            time.sleep(cooldown_seconds)

    detection_archive_path: Path | None = None
    detection_record_count = 0
    if detection_capture is not None and run_id is not None:
        detection_archive_path = Path(detection_archive_dir) / f"{run_id}.jsonl"
        detection_record_count = detection_capture.write_archive(
            detection_archive_path
        )

    result: dict[str, object] = {
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(UTC).isoformat(),
        "random_seed": seed,
        "repetitions_per_fault": repetitions_per_fault,
        "fault_duration_seconds": duration_seconds,
        "cooldown_seconds": cooldown_seconds,
        "schedule": schedule,
        "trials": trials,
        "telemetry": {
            "enabled": telemetry_root is not None,
            "run_id": run_id,
            "root": str(telemetry_root) if telemetry_root is not None else None,
            "pre_seconds": telemetry_pre_seconds,
            "post_seconds": telemetry_post_seconds,
            "interval_seconds": telemetry_interval_seconds,
            "request_timeout_seconds": resolved_telemetry_timeout,
            "incident_directories": telemetry_directories,
        },
        "detection_archive": {
            "enabled": detection_capture is not None,
            "source": (
                str(detection_source_path)
                if detection_source_path is not None
                else None
            ),
            "path": (
                str(detection_archive_path)
                if detection_archive_path is not None
                else None
            ),
            "records": detection_record_count,
        },
    }
    output_path = Path(result_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a balanced mixed batch.")
    parser.add_argument("--repetitions-per-fault", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--cooldown", type=float, default=8)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--ground-truth",
        default="data/ground_truth/mixed-batch.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/results/mixed-batch.json",
    )
    parser.add_argument(
        "--telemetry-root",
        default="data/raw/local",
        help="Root directory for per-incident probes and Docker logs.",
    )
    parser.add_argument("--telemetry-pre-seconds", type=float, default=10.0)
    parser.add_argument("--telemetry-post-seconds", type=float, default=10.0)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    parser.add_argument("--telemetry-request-timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable local telemetry capture for a pipeline-only run.",
    )
    parser.add_argument(
        "--detection-source",
        default="data/detections/detections.jsonl",
        help="Append-only detector JSONL file to slice for this run.",
    )
    parser.add_argument(
        "--detection-archive-dir",
        default="data/detections",
    )
    parser.add_argument(
        "--no-detection-archive",
        action="store_true",
        help="Do not archive detector records appended during this run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_mixed_batch(
        repetitions_per_fault=args.repetitions_per_fault,
        seed=args.seed,
        duration_seconds=args.duration,
        cooldown_seconds=args.cooldown,
        base_url=args.base_url,
        ground_truth_path=args.ground_truth,
        result_path=args.output,
        telemetry_root=None if args.no_telemetry else args.telemetry_root,
        telemetry_pre_seconds=args.telemetry_pre_seconds,
        telemetry_post_seconds=args.telemetry_post_seconds,
        telemetry_interval_seconds=args.telemetry_interval_seconds,
        telemetry_request_timeout_seconds=args.telemetry_request_timeout_seconds,
        detection_source_path=(
            None if args.no_detection_archive else args.detection_source
        ),
        detection_archive_dir=args.detection_archive_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
