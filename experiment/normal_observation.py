"""Observe normal workload and report any detector false positives."""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Sequence

import httpx


def count_detection_starts(path: str | Path) -> int:
    detection_path = Path(path)
    if not detection_path.exists():
        return 0
    count = 0
    with detection_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip() and json.loads(line).get("event") == "detection_started":
                count += 1
    return count


def observe_normal_operation(
    *,
    duration_seconds: int,
    base_url: str,
    detection_path: str | Path,
    result_path: str | Path,
    interval_seconds: float = 1.0,
) -> dict[str, object]:
    if duration_seconds < 1:
        raise ValueError("duration_seconds must be positive")

    started_at = datetime.now(UTC)
    healthy_samples = 0
    unhealthy_samples = 0
    starts_before = count_detection_starts(detection_path)
    deadline = time.monotonic() + duration_seconds

    with httpx.Client(base_url=base_url, timeout=4.0) as client:
        client.post("/internal/faults/http-500", json={"enabled": False}).raise_for_status()
        while time.monotonic() < deadline:
            try:
                live = client.get("/health/live")
                database = client.get("/health/database")
                if live.status_code == 200 and database.status_code == 200:
                    healthy_samples += 1
                else:
                    unhealthy_samples += 1
            except httpx.HTTPError:
                unhealthy_samples += 1
            time.sleep(interval_seconds)

    false_positive_detections = (
        count_detection_starts(detection_path) - starts_before
    )
    result: dict[str, object] = {
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(UTC).isoformat(),
        "duration_seconds": duration_seconds,
        "healthy_samples": healthy_samples,
        "unhealthy_samples": unhealthy_samples,
        "false_positive_detections": false_positive_detections,
        "passed": unhealthy_samples == 0 and false_positive_detections == 0,
    }
    output_path = Path(result_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Observe a no-fault period.")
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--detections",
        default="data/detections/no-fault-observation.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/results/no-fault-observation.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = observe_normal_operation(
        duration_seconds=args.duration,
        base_url=args.base_url,
        detection_path=args.detections,
        result_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

