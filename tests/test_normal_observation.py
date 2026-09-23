import json

from experiment.normal_observation import count_detection_starts


def test_count_detection_starts_handles_absent_and_existing_files(tmp_path) -> None:
    path = tmp_path / "detections.jsonl"
    assert count_detection_starts(path) == 0

    path.write_text(
        "\n".join(
            [
                json.dumps({"event": "detection_started"}),
                json.dumps({"event": "detection_ended"}),
                json.dumps({"event": "detection_started"}),
            ]
        ),
        encoding="utf-8",
    )
    assert count_detection_starts(path) == 2

