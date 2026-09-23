import json

from experiment.ground_truth import GroundTruthRecorder


def test_ground_truth_is_append_only_jsonl(tmp_path) -> None:
    output = tmp_path / "incidents.jsonl"
    recorder = GroundTruthRecorder(output)
    incident_id = recorder.new_incident_id()

    recorder.record(
        incident_id=incident_id,
        event="fault_started",
        service="order-service",
        fault_type="database_connection_failure",
        root_cause="postgres_container_stopped",
    )
    recorder.record(
        incident_id=incident_id,
        event="fault_ended",
        service="order-service",
        fault_type="database_connection_failure",
        root_cause="postgres_container_stopped",
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "fault_started",
        "fault_ended",
    ]
    assert all(record["incident_id"] == incident_id for record in records)

