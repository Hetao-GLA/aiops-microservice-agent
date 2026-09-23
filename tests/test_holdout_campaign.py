from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment import holdout_campaign
from experiment.campaign_runner import campaign_plan, load_config


def test_v2_config_is_balanced_and_separate_from_v1() -> None:
    v1 = load_config("experiment/configs/local-campaign-v1.json")
    v2 = load_config("experiment/configs/local-campaign-v2.json")
    plan = campaign_plan(v2)
    assert plan["expected_incidents"] == 30
    assert plan["expected_incidents_per_fault"] == 10
    assert plan["fault_duration_seconds"] == 12
    for key in ("ground_truth", "batch_result", "evaluation", "processed_incidents", "processed_manifest"):
        assert v1.paths[key] != v2.paths[key]


@pytest.mark.parametrize("failure", [None, "campaign", "override"])
def test_workload_is_restored_on_success_and_failure(tmp_path: Path, monkeypatch, failure) -> None:
    config = load_config("experiment/configs/local-campaign-v2.json")
    output = tmp_path / "protocol.json"
    config.payload["holdout"]["protocol_output"] = str(output)
    config.payload["holdout"]["warmup_seconds"] = 0
    monkeypatch.setattr(holdout_campaign, "holdout_plan", lambda config: {"status": "planned"})
    monkeypatch.setattr(holdout_campaign, "_healthy_and_no_fault", lambda url: None)
    state = {"interval_seconds": 1.0, "image_id": "unchanged", "container_id": "test"}
    changes = []
    def set_workload(override):
        changes.append(override)
        state["interval_seconds"] = 0.25 if override else 1.0
        if failure == "override" and override:
            raise RuntimeError("override startup failed")
    monkeypatch.setattr(holdout_campaign, "set_workload", set_workload)
    monkeypatch.setattr(holdout_campaign, "workload_state", lambda: dict(state))
    def campaign(config):
        assert json.loads(output.read_text())["status"] == "collecting"
        if failure == "campaign":
            raise RuntimeError("campaign failed")
        return {"incidents_completed": 30}
    monkeypatch.setattr(holdout_campaign, "run_campaign", campaign)
    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            holdout_campaign.run_holdout_campaign(config)
    else:
        holdout_campaign.run_holdout_campaign(config)
    result = json.loads(output.read_text())
    assert changes[-1] is None
    assert state["interval_seconds"] == 1.0
    assert result["workload_restored"] is True
    assert result["status"] == ("failed" if failure else "collected")


def test_nonbaseline_workload_is_not_changed(tmp_path: Path, monkeypatch) -> None:
    config = load_config("experiment/configs/local-campaign-v2.json")
    config.payload["holdout"]["protocol_output"] = str(tmp_path / "protocol.json")
    monkeypatch.setattr(holdout_campaign, "holdout_plan", lambda config: {})
    monkeypatch.setattr(holdout_campaign, "workload_state", lambda: {"interval_seconds": 2.0})
    with pytest.raises(RuntimeError, match="refusing to change"):
        holdout_campaign.run_holdout_campaign(config)
    assert not (tmp_path / "protocol.json").exists()
