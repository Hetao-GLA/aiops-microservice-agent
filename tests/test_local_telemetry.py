from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from experiment.local_telemetry import (
    JsonlAppendCapture,
    capture_docker_logs,
    parse_docker_log_lines,
    resolve_probe_timeout,
)


def test_parse_docker_log_lines_preserves_service_and_stream() -> None:
    records = parse_docker_log_lines(
        "2026-08-24T12:00:00.123456789Z first message\nplain fallback\n",
        service="order-service",
        container="aiops-order-service",
        stream="stderr",
    )

    assert records[0] == {
        "timestamp": "2026-08-24T12:00:00.123456789Z",
        "service": "order-service",
        "container": "aiops-order-service",
        "stream": "stderr",
        "message": "first message",
    }
    assert records[1]["timestamp"] is None
    assert records[1]["message"] == "plain fallback"


def test_capture_docker_logs_uses_bounded_time_range() -> None:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        assert kwargs == {"check": False, "text": True, "capture_output": True}
        return SimpleNamespace(
            returncode=0,
            stdout="2026-08-24T12:00:01Z api ready\n",
            stderr="",
        )

    records, errors = capture_docker_logs(
        started_at=datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
        ended_at=datetime(2026, 8, 24, 12, 1, tzinfo=UTC),
        containers={"order-service": "aiops-order-service"},
        run_command=fake_run,
    )

    assert errors == []
    assert records[0]["message"] == "api ready"
    assert commands == [
        [
            "docker",
            "logs",
            "--timestamps",
            "--since",
            "2026-08-24T12:00:00Z",
            "--until",
            "2026-08-24T12:01:00Z",
            "aiops-order-service",
        ]
    ]


def test_capture_docker_logs_reports_failed_container() -> None:
    def fake_run(_command, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="not found")

    records, errors = capture_docker_logs(
        started_at=datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
        ended_at=datetime(2026, 8, 24, 12, 1, tzinfo=UTC),
        containers={"database": "missing-database"},
        run_command=fake_run,
    )

    assert records == []
    assert errors == [
        {
            "service": "database",
            "container": "missing-database",
            "error": "not found",
        }
    ]


def test_jsonl_append_capture_excludes_historical_records(tmp_path) -> None:
    source = tmp_path / "detections.jsonl"
    source.write_text('{"id":"old"}\n', encoding="utf-8")
    capture = JsonlAppendCapture(source)
    capture.start()
    with source.open("a", encoding="utf-8") as handle:
        handle.write('{"id":"new-1"}\n')
        handle.write('{"id":"new-2"}\n')

    archive = tmp_path / "run.jsonl"
    assert capture.write_archive(archive) == 2
    assert archive.read_text(encoding="utf-8").splitlines() == [
        '{"id": "new-1"}',
        '{"id": "new-2"}',
    ]


def test_default_probe_timeout_stays_below_sampling_interval() -> None:
    assert resolve_probe_timeout(1.0, None) == 0.8
    assert resolve_probe_timeout(0.5, None) == 0.4
    assert resolve_probe_timeout(10.0, None) == 2.0
