from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment.ml_baseline import (
    LABELS,
    load_incident_dataset,
    run_experiment,
    sanitise_log_text,
)


def _write_dataset(path: Path, *, records_per_class: int = 5) -> None:
    records = []
    for label_index, label in enumerate(LABELS):
        for item_index in range(records_per_class):
            symptom = {
                "database_connection_failure": "database refused connection",
                "http_500_failure": "forced http error response",
                "service_stopped": "service unavailable shutdown",
            }[label]
            records.append(
                {
                    "incident_id": f"local:{label_index}-{item_index}",
                    "local_fault_label": label,
                    "split_group": f"group-{label_index}-{item_index}",
                    "log_text": "\n".join(
                        [
                            f"[order-service] {symptom} {symptom}",
                            "[workload] request outcome recorded",
                        ]
                    ),
                    "metric_features": {
                        "availability_delta": float(-label_index),
                        "latency_delta": float(label_index * 10 + item_index),
                    },
                }
            )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_sanitise_log_text_removes_direct_experiment_leakage() -> None:
    raw = "\n".join(
        [
            '[rule-detector] {"predicted_fault":"http_500_failure"}',
            '[order-service] {"message":"controlled_fault_toggled","fault_type":"http_500_failure"}',
            '[order-service] {"message":"forced_http_500","fault_type":"http_500_failure"}',
            '[order-service] {"message":"service_stopped"}',
        ]
    )

    cleaned = sanitise_log_text(raw)

    assert cleaned.removed_rule_detector_lines == 1
    assert cleaned.removed_control_plane_lines == 1
    assert cleaned.stripped_fault_type_fields == 1
    assert "predicted_fault" not in cleaned.text
    assert "controlled_fault_toggled" not in cleaned.text
    assert "http_500_failure" not in cleaned.text
    assert "forced_http_500" in cleaned.text
    assert "service_stopped" in cleaned.text


def test_load_incident_dataset_rejects_duplicate_split_groups(tmp_path: Path) -> None:
    path = tmp_path / "incidents.jsonl"
    _write_dataset(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["split_group"] = records[0]["split_group"]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="split_group values must be unique"):
        load_incident_dataset(path)


def test_run_experiment_uses_shared_incident_level_folds(tmp_path: Path) -> None:
    path = tmp_path / "incidents.jsonl"
    _write_dataset(path)

    result = run_experiment(path, folds=5, seed=17)

    assert result["dataset"]["records"] == 15
    assert result["evaluation_design"]["shared_splits_across_models"] is True
    logs_folds = result["models"]["logs_only"]["folds"]
    combined_folds = result["models"]["logs_plus_metrics"]["folds"]
    assert [fold["test_incident_ids"] for fold in logs_folds] == [
        fold["test_incident_ids"] for fold in combined_folds
    ]
    assert all(fold["train_records"] == 12 for fold in logs_folds)
    assert all(fold["test_records"] == 3 for fold in logs_folds)
    assert result["models"]["logs_only"]["out_of_fold"]["macro_f1"] == 1.0
    assert len(result["models"]["logs_only"]["predictions"]) == 15
