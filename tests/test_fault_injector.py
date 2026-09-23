import json
from types import SimpleNamespace

from experiment.fault_injector import (
    inject_database_disconnect,
    inject_http_500,
    inject_service_stop,
)
from experiment.ground_truth import GroundTruthRecorder


def test_database_fault_stops_waits_and_restores(tmp_path) -> None:
    commands: list[list[str]] = []
    waits: list[float] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[:2] == ["docker", "inspect"]:
            return SimpleNamespace(stdout="2026-08-10T12:00:00.000000Z\n")
        return SimpleNamespace(stdout="")

    recorder = GroundTruthRecorder(tmp_path / "incidents.jsonl")
    incident_id = inject_database_disconnect(
        duration_seconds=5,
        recorder=recorder,
        run_command=fake_run,
        sleep=waits.append,
    )

    assert commands == [
        ["docker", "compose", "stop", "database"],
        [
            "docker",
            "inspect",
            "--format",
            "{{.State.FinishedAt}}",
            "aiops-database",
        ],
        ["docker", "compose", "start", "database"],
    ]
    assert waits == [5]

    records = [
        json.loads(line)
        for line in (tmp_path / "incidents.jsonl").read_text().splitlines()
    ]
    assert records[0]["incident_id"] == incident_id
    assert records[0]["event"] == "fault_started"
    assert records[0]["timestamp"] == "2026-08-10T12:00:00.000000Z"
    assert records[0]["timing_source"] == "docker.State.FinishedAt"
    assert records[1]["event"] == "fault_ended"
    assert records[1]["recovery_succeeded"] is True


def test_service_stop_targets_api_container(tmp_path) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[:2] == ["docker", "inspect"]:
            return SimpleNamespace(stdout="2026-08-10T12:00:00Z\n")
        return SimpleNamespace(stdout="")

    output = tmp_path / "service-stop.jsonl"
    incident_id = inject_service_stop(
        duration_seconds=3,
        recorder=GroundTruthRecorder(output),
        run_command=fake_run,
        sleep=lambda _seconds: None,
    )

    assert commands[0] == ["docker", "compose", "stop", "api"]
    assert commands[1][-1] == "aiops-order-service"
    assert commands[2] == ["docker", "compose", "start", "api"]
    first = json.loads(output.read_text().splitlines()[0])
    assert first["incident_id"] == incident_id
    assert first["fault_type"] == "service_stopped"
    assert first["root_cause"] == "api_container_stopped"


def test_http_500_fault_uses_application_timestamps(tmp_path) -> None:
    toggles: list[bool] = []

    def fake_toggle(enabled: bool) -> str:
        toggles.append(enabled)
        return (
            "2026-08-10T12:00:00+00:00"
            if enabled
            else "2026-08-10T12:00:08+00:00"
        )

    output = tmp_path / "http-500.jsonl"
    incident_id = inject_http_500(
        duration_seconds=8,
        recorder=GroundTruthRecorder(output),
        toggle=fake_toggle,
        sleep=lambda _seconds: None,
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert toggles == [True, False]
    assert records[0]["incident_id"] == incident_id
    assert records[0]["fault_type"] == "http_500_failure"
    assert records[0]["timestamp"] == "2026-08-10T12:00:00+00:00"
    assert records[1]["timestamp"] == "2026-08-10T12:00:08+00:00"
