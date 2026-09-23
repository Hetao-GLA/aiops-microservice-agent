from types import SimpleNamespace
from pathlib import Path
import json
import pytest
from experiment.campaign_runner import (
    load_config,
    validate_detection_pairs,
    validate_fault_durations,
    validate_metric_latencies,
)


def make_config(tmp_path, duration_limit=24, latency_limit=1000):
    payload = json.loads(Path(
        "experiment/configs/local-campaign-v1.json"
    ).read_text(encoding="utf-8"))
    payload["paths"]["ground_truth"] = str(tmp_path / "truth.jsonl")
    payload["acceptance"]["maximum_fault_duration_seconds"] = duration_limit
    payload["acceptance"]["maximum_metric_latency_ms"] = latency_limit
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return load_config(path)


def write_incident(path, seconds):
    records = [
        {"incident_id": "one", "event": "fault_started",
         "fault_type": "http_500_failure", "timestamp": "2026-01-01T00:00:00Z"},
        {"incident_id": "one", "event": "fault_ended",
         "fault_type": "http_500_failure",
         "timestamp": f"2026-01-01T00:00:{seconds:02d}Z"},
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in records))


def test_duration_acceptance_rejects_suspended_incident(tmp_path):
    config = make_config(tmp_path)
    write_incident(config.paths["ground_truth"], 25)
    with pytest.raises(RuntimeError, match="Fault duration acceptance failed"):
        validate_fault_durations(config)


def test_duration_acceptance_reports_valid_range(tmp_path):
    config = make_config(tmp_path)
    write_incident(config.paths["ground_truth"], 12)
    result = validate_fault_durations(config)
    assert result["maximum_seconds"] == 12
    assert result["limit_seconds"] == 24


def test_metric_latency_acceptance_rejects_scheduler_stall(tmp_path):
    config = make_config(tmp_path)
    records = [SimpleNamespace(metric_features={
        "liveness__latency_ms__post_mean": 1200.0,
        "liveness__available__post_mean": 1.0,
    })]
    with pytest.raises(RuntimeError, match="Metric latency acceptance failed"):
        validate_metric_latencies(config, records)


def test_recovery_failure_is_rejected(tmp_path):
    config = make_config(tmp_path)
    records = [
        {"incident_id": "one", "event": "fault_started",
         "fault_type": "http_500_failure", "timestamp": "2026-01-01T00:00:00Z"},
        {"incident_id": "one", "event": "recovery_failed",
         "fault_type": "http_500_failure", "timestamp": "2026-01-01T00:00:12Z"},
    ]
    config.paths["ground_truth"].write_text(
        "".join(json.dumps(item) + "\n" for item in records)
    )
    with pytest.raises(RuntimeError, match="Recovery acceptance failed"):
        validate_fault_durations(config)


def test_incomplete_detection_pair_is_rejected(tmp_path):
    config = make_config(tmp_path)
    evaluation = {
        "matches": [{
            "incident_id": "one",
            "detection_id": "det-one",
            "recovery_detection_latency_seconds": None,
        }],
        "summary": {"false_positive_detections": 0},
    }
    with pytest.raises(RuntimeError, match="Detection-pair acceptance failed"):
        validate_detection_pairs(config, evaluation)
