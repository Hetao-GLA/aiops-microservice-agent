from __future__ import annotations

import json
from pathlib import Path

from experiment.rcaeval_public_baseline import (
    evaluate_model,
    load_dataset,
    make_splits,
)


SERVICES = ["a", "b", "c", "d", "e"]
FAULTS = ["cpu", "delay", "disk", "loss", "mem", "socket"]


def _spec() -> dict[str, object]:
    return {
        "selection": {
            "dataset": "RE2-OB",
            "expected_cases": 90,
            "expected_root_cause_services": SERVICES,
            "expected_faults": FAULTS,
            "expected_repetitions_per_service_fault_pair": 3,
        },
        "task": {"name": "public_root_cause_localisation"},
        "evaluation": {
            "classifier": {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 2000,
                "random_seed": 7,
            },
            "text_features": {
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_df": 0.98,
                "max_features": 10000,
                "sublinear_tf": True,
            },
        },
    }


def _write_dataset(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for service_index, service in enumerate(SERVICES):
            for fault_index, fault in enumerate(FAULTS):
                for repetition in (1, 2, 3):
                    record = {
                        "incident_id": f"rcaeval:{service}_{fault}_{repetition}",
                        "source": "rcaeval",
                        "dataset": "RE2-OB",
                        "task": "public_root_cause_localisation",
                        "root_cause_service": service,
                        "original_fault_label": fault,
                        "local_fault_label": None,
                        "log_text": f"[{service}] unique_{service} unique_{service} fault_{fault}",
                        "log_record_count": 1,
                        "metric_features": {
                            f"metric_{service}__mean_delta": float(service_index + 1),
                            f"fault_{fault}__mean_delta": float(fault_index + 1),
                            "nullable": None,
                        },
                        "provenance": {"repetition": repetition},
                    }
                    handle.write(json.dumps(record) + "\n")


def test_locked_dataset_and_splits(tmp_path: Path) -> None:
    path = tmp_path / "incidents.jsonl"
    _write_dataset(path)
    dataset = load_dataset(path, _spec())

    repetition_splits = make_splits(dataset, "repetition-held-out")
    fault_splits = make_splits(dataset, "fault-type-held-out")

    assert dataset.class_counts == {service: 18 for service in SERVICES}
    assert len(repetition_splits) == 3
    assert all(len(train) == 60 and len(test) == 30 for _, train, test in repetition_splits)
    assert len(fault_splits) == 6
    assert all(len(train) == 75 and len(test) == 15 for _, train, test in fault_splits)


def test_logs_model_produces_one_oof_prediction_per_case(tmp_path: Path) -> None:
    path = tmp_path / "incidents.jsonl"
    _write_dataset(path)
    spec = _spec()
    dataset = load_dataset(path, spec)
    result = evaluate_model(
        "logs_only", dataset, make_splits(dataset, "repetition-held-out"), spec
    )

    assert result["out_of_fold"]["top_1_accuracy"] == 1.0
    assert len(result["predictions"]) == 90

