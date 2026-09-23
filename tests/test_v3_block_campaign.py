from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment.campaign_runner import load_config, sha256_file
from experiment import v3_block_campaign as v3


MASTER = Path("experiment/configs/local-campaign-v3-blocks.json")


def use_temp_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        v3,
        "block_config_path",
        lambda block: tmp_path / f"block-{block:02d}.json",
    )
    monkeypatch.setattr(v3, "read_protocol", lambda config: None)


def test_v3_is_workload_only_and_has_unique_balanced_blocks():
    master = v3.load_master(MASTER)
    payloads = [
        v3.block_payload(MASTER, master, block)
        for block in range(1, master["blocks"] + 1)
    ]
    assert master["fault_duration_seconds"] == 8
    assert master["holdout"]["experimental_interval_seconds"] == 0.25
    assert all(item["repetitions_per_fault"] == 1 for item in payloads)
    assert all(item["acceptance"]["expected_incidents"] == 3 for item in payloads)
    assert len({item["random_seed"] for item in payloads}) == 10
    assert len({item["paths"]["ground_truth"] for item in payloads}) == 10
    assert all(
        item["block_metadata"]["independent_variable"]
        == "workload_interval_seconds"
        for item in payloads
    )


def test_prepared_config_is_immutable(tmp_path, monkeypatch):
    use_temp_configs(tmp_path, monkeypatch)
    master = v3.load_master(MASTER)
    path = v3.prepare_block(MASTER, master, 1)
    assert load_config(path).expected_incidents == 3
    changed = json.loads(path.read_text())
    changed["fault_duration_seconds"] = 9
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="config changed"):
        v3.prepare_block(MASTER, master, 1)


def test_status_reports_prepared_blocks_as_ready(tmp_path, monkeypatch):
    use_temp_configs(tmp_path, monkeypatch)
    master = v3.load_master(MASTER)
    v3.prepare_all(MASTER, master)
    status = v3.campaign_status(MASTER, master)
    assert status["accepted_blocks"] == 0
    assert {row["status"] for row in status["blocks"]} == {"ready"}


def test_completed_block_requires_all_quality_gates(tmp_path, monkeypatch):
    use_temp_configs(tmp_path, monkeypatch)
    master = v3.load_master(MASTER)
    config = load_config(v3.prepare_block(MASTER, master, 1))
    protocol = {
        "status": "collected",
        "workload_restored": True,
        "post_run_health_check_passed": True,
        "campaign": {"config_sha256": sha256_file(config.source_path)},
        "code_and_config_sha256": {
            str(v3.ORCHESTRATOR_PATH): sha256_file(v3.ORCHESTRATOR_PATH)
        },
        "campaign_summary": {
            "incidents_completed": 3,
            "fault_counts": {label: 1 for label in v3.FAULT_LABELS},
            "fault_duration_quality": {"maximum_seconds": 9.5},
            "metric_latency_quality": {"maximum_ms": 800},
            "detection_pair_quality": {
                "incomplete_pairs": 0,
                "false_positive_detections": 0,
            },
            "telemetry_run_id": "run-1",
        },
    }
    assert v3.validate_completed_block(master, config, protocol)[
        "telemetry_run_id"
    ] == "run-1"
    protocol["campaign_summary"]["metric_latency_quality"]["maximum_ms"] = 1001
    with pytest.raises(RuntimeError, match="latency acceptance"):
        v3.validate_completed_block(master, config, protocol)


def test_finalise_refuses_incomplete_campaign_without_outputs(
    tmp_path, monkeypatch
):
    use_temp_configs(tmp_path, monkeypatch)
    master = v3.load_master(MASTER)
    master["aggregate_paths"] = {
        name: str(tmp_path / f"aggregate-{name}")
        for name in master["aggregate_paths"]
    }
    with pytest.raises(RuntimeError, match="has not run"):
        v3.finalise(MASTER, master)
    assert not any(tmp_path.glob("aggregate-*"))
