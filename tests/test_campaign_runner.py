from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment.campaign_runner import campaign_plan, load_config, preflight


def test_frozen_campaign_has_balanced_sixty_incidents() -> None:
    config = load_config(Path("experiment/configs/local-campaign-v1.json"))
    plan = campaign_plan(config)

    assert plan["expected_incidents"] == 60
    assert plan["expected_incidents_per_fault"] == 20
    assert plan["random_seed"] == 20260825
    assert plan["estimated_minimum_runtime_minutes"] >= 35
    assert plan["telemetry"]["request_timeout_seconds"] <= plan["telemetry"][
        "interval_seconds"
    ]


def test_campaign_validation_rejects_inconsistent_expected_count(
    tmp_path: Path,
) -> None:
    source = Path("experiment/configs/local-campaign-v1.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["acceptance"]["expected_incidents"] = 59
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="balanced schedule size"):
        load_config(config_path)


def test_preflight_refuses_to_overwrite_campaign_output(tmp_path: Path) -> None:
    source = Path("experiment/configs/local-campaign-v1.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    for name in payload["paths"]:
        payload["paths"][name] = str(tmp_path / f"{name}.json")
    detection_source = Path(payload["paths"]["detection_source"])
    detection_source.write_text("", encoding="utf-8")
    existing_output = Path(payload["paths"]["ground_truth"])
    existing_output.write_text("existing", encoding="utf-8")
    payload["acceptance"]["minimum_free_disk_bytes"] = 1
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_config(config_path)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        preflight(config)
