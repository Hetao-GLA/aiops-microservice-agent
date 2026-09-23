from __future__ import annotations

import json
from pathlib import Path

from experiment import batch_runner


def test_batch_runner_archives_only_current_detection_records(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "detections.jsonl"
    source.write_text('{"detection_id":"old"}\n', encoding="utf-8")
    session_events: list[str] = []

    class FakeTelemetrySession:
        def __init__(self, **kwargs) -> None:
            self.root = Path(kwargs["root"])
            self.run_id = kwargs["run_id"]

        def start(self) -> None:
            session_events.append("start")

        def finish(self, incident_id: str) -> Path:
            session_events.append(f"finish:{incident_id}")
            path = self.root / self.run_id / incident_id
            path.mkdir(parents=True)
            return path

        def abort(self, _error: BaseException) -> None:
            session_events.append("abort")

    def fake_injector(**_kwargs) -> str:
        with source.open("a", encoding="utf-8") as handle:
            handle.write('{"detection_id":"new","event":"detection_started"}\n')
            handle.write('{"detection_id":"new","event":"detection_ended"}\n')
        return "INC-TEST"

    monkeypatch.setattr(batch_runner, "wait_until_healthy", lambda **_kwargs: True)
    monkeypatch.setattr(batch_runner, "LocalTelemetrySession", FakeTelemetrySession)
    monkeypatch.setattr(
        batch_runner, "new_telemetry_run_id", lambda _prefix: "batch-test"
    )

    result = batch_runner.run_batch(
        trials=1,
        duration_seconds=1,
        cooldown_seconds=0,
        base_url="http://unused",
        ground_truth_path=tmp_path / "ground-truth.jsonl",
        result_path=tmp_path / "result.json",
        fault="database_disconnect",
        injector=fake_injector,
        telemetry_root=tmp_path / "raw",
        telemetry_pre_seconds=0,
        telemetry_post_seconds=0,
        detection_source_path=source,
        detection_archive_dir=tmp_path / "archives",
    )

    assert session_events == ["start", "finish:INC-TEST"]
    assert result["incident_ids"] == ["INC-TEST"]
    assert result["detection_archive"]["records"] == 2
    archive = Path(result["detection_archive"]["path"])
    archived = [json.loads(line) for line in archive.read_text().splitlines()]
    assert [record["event"] for record in archived] == [
        "detection_started",
        "detection_ended",
    ]
    assert all(record["detection_id"] != "old" for record in archived)
