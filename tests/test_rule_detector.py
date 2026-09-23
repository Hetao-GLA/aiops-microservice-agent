import json

from experiment.rule_detector import DetectionRecorder, OperationsHealthRule


def test_rule_emits_one_start_and_one_end_per_incident(tmp_path) -> None:
    output = tmp_path / "detections.jsonl"
    rule = OperationsHealthRule(DetectionRecorder(output))

    rule.process_sample(observed_fault=None, evidence={"status_code": 200})
    rule.process_sample(
        observed_fault="database_connection_failure",
        evidence={"status_code": 503},
    )
    rule.process_sample(
        observed_fault="database_connection_failure",
        evidence={"status_code": 503},
    )
    rule.process_sample(observed_fault=None, evidence={"status_code": 200})
    rule.process_sample(observed_fault=None, evidence={"status_code": 200})

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "detection_started",
        "detection_ended",
    ]
    assert records[0]["detection_id"] == records[1]["detection_id"]
    assert records[0]["predicted_fault"] == "database_connection_failure"


def test_rule_distinguishes_service_stop(tmp_path) -> None:
    output = tmp_path / "detections.jsonl"
    rule = OperationsHealthRule(DetectionRecorder(output))

    rule.process_sample(
        observed_fault="service_stopped",
        evidence={"probe": "/health/live", "error_type": "ConnectError"},
    )
    rule.process_sample(observed_fault=None, evidence={"status_code": 200})

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[0]["predicted_fault"] == "service_stopped"
    assert records[0]["predicted_component"] == "service"


def test_rule_distinguishes_http_500_failure(tmp_path) -> None:
    output = tmp_path / "detections.jsonl"
    rule = OperationsHealthRule(DetectionRecorder(output))

    rule.process_sample(
        observed_fault="http_500_failure",
        evidence={"probe": "/metrics/orders", "http_500_rate": 1.0},
    )
    rule.process_sample(observed_fault=None, evidence={"status_code": 200})

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[0]["predicted_fault"] == "http_500_failure"
    assert records[0]["predicted_component"] == "application"
