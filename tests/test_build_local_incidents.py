from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment.data_engineering.build_local_incidents import (
    build_local_records,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw_root = tmp_path / "raw"
    run_id = "batch-test"
    incident_id = "INC-TEST"
    incident_dir = raw_root / run_id / incident_id
    incident_dir.mkdir(parents=True)
    probes = [
        {
            "timestamp": "2026-08-24T12:00:00Z",
            "probe": "database_health",
            "status_code": 200,
            "available": True,
            "payload": {"healthy": True},
            "latency_ms": 10.0,
        },
        {
            "timestamp": "2026-08-24T12:00:02Z",
            "probe": "database_health",
            "status_code": 503,
            "available": True,
            "payload": {"healthy": False},
            "latency_ms": 30.0,
        },
    ]
    logs = [
        {
            "timestamp": "2026-08-24T12:00:01Z",
            "service": "order-service",
            "container": "api",
            "stream": "stderr",
            "message": "database connection failed",
        }
    ]
    _write_jsonl(incident_dir / "probes.jsonl", probes)
    _write_jsonl(incident_dir / "docker-logs.jsonl", logs)
    (incident_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "incident_id": incident_id,
                "expected_fault": "database_connection_failure",
                "collector_started_at": "2026-08-24T12:00:00Z",
                "collector_ended_at": "2026-08-24T12:00:04Z",
                "pre_seconds": 1,
                "post_seconds": 1,
                "interval_seconds": 1,
                "request_timeout_seconds": 0.8,
                "probe_sample_count": len(probes),
                "docker_log_record_count": len(logs),
                "collector_error": None,
                "docker_capture_errors": [],
            }
        ),
        encoding="utf-8",
    )
    truth_path = tmp_path / "ground-truth.jsonl"
    _write_jsonl(
        truth_path,
        [
            {
                "timestamp": "2026-08-24T12:00:01Z",
                "incident_id": incident_id,
                "event": "fault_started",
                "service": "order-service",
                "fault_type": "database_connection_failure",
                "root_cause": "postgres_container_stopped",
            },
            {
                "timestamp": "2026-08-24T12:00:03Z",
                "incident_id": incident_id,
                "event": "fault_ended",
                "service": "order-service",
                "fault_type": "database_connection_failure",
                "root_cause": "postgres_container_stopped",
                "recovery_succeeded": True,
            },
        ],
    )
    detection_dir = tmp_path / "detections"
    _write_jsonl(
        detection_dir / f"{run_id}.jsonl",
        [
            {"detection_id": "DET-1", "event": "detection_started"},
            {"detection_id": "DET-1", "event": "detection_ended"},
        ],
    )
    return raw_root, truth_path, detection_dir


def test_local_builder_creates_one_grouped_incident(tmp_path: Path) -> None:
    raw_root, truth_path, detection_dir = _fixture(tmp_path)

    records = build_local_records(
        raw_root=raw_root,
        ground_truth_paths=[truth_path],
        detection_archive_dir=detection_dir,
    )

    assert len(records) == 1
    record = records[0]
    assert record.incident_id == "local:INC-TEST"
    assert record.local_fault_label == "database_connection_failure"
    assert record.root_cause_service == "database"
    assert record.split_group == "INC-TEST"
    assert record.log_text == "[order-service] database connection failed"
    assert record.metric_pre_rows == 1
    assert record.metric_post_rows == 1
    assert record.metric_features[
        "database_health__status_code__mean_delta"
    ] == 303.0
    assert record.provenance["detection_pairs"] == 1


def test_local_builder_rejects_telemetry_count_mismatch(tmp_path: Path) -> None:
    raw_root, truth_path, detection_dir = _fixture(tmp_path)
    metadata_path = raw_root / "batch-test" / "INC-TEST" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["probe_sample_count"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="Probe row count"):
        build_local_records(
            raw_root=raw_root,
            ground_truth_paths=[truth_path],
            detection_archive_dir=detection_dir,
        )


def test_local_builder_requires_complete_detection_archive(tmp_path: Path) -> None:
    raw_root, truth_path, detection_dir = _fixture(tmp_path)
    (detection_dir / "batch-test.jsonl").unlink()

    with pytest.raises(FileNotFoundError):
        build_local_records(
            raw_root=raw_root,
            ground_truth_paths=[truth_path],
            detection_archive_dir=detection_dir,
        )
