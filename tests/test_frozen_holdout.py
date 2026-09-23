from __future__ import annotations

import json
from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline

from experiment.frozen_holdout import evaluate_holdout, freeze_models, verify_bundle
from experiment.frozen_contribution_audit import run_audit
from experiment.ml_baseline import LABELS, MODEL_NAMES, sha256_file


def _dataset(path: Path, run: str, *, year: int, count: int = 5) -> None:
    rows = []
    for label_index, label in enumerate(LABELS):
        symptom = ("database refused connection", "forced http error response", "service unavailable shutdown")[label_index]
        for index in range(count):
            rows.append({
                "schema_version": 1, "source": "local", "task": "local_fault_classification",
                "system_name": "local-order-platform", "incident_id": f"{run}-{label_index}-{index}",
                "split_group": f"group-{run}-{label_index}-{index}", "local_fault_label": label,
                "window_start": f"{year}-01-01T00:00:00Z", "window_end": f"{year}-01-01T00:01:00Z",
                "log_text": f"[order-service] {symptom} {symptom}\n[workload] request outcome recorded",
                "metric_features": {"latency": float(label_index * 10 + index), "availability": float(-label_index)},
                "provenance": {"run_id": run, "docker_logs_sha256": f"logs-{run}-{label_index}-{index}",
                               "probes_sha256": f"probes-{run}-{label_index}-{index}"},
            })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def frozen(tmp_path: Path) -> tuple[Path, Path, Path]:
    train, test, bundle = tmp_path / "train.jsonl", tmp_path / "test.jsonl", tmp_path / "bundle"
    _dataset(train, "v1", year=2020)
    _dataset(test, "v2", year=2099, count=3)
    freeze_models(train, bundle)
    return train, test, bundle


def test_frozen_prediction_never_fits_and_preserves_artifact(frozen, monkeypatch) -> None:
    train, test, bundle = frozen
    before = sha256_file(bundle / "models.joblib")
    def forbidden(*args, **kwargs):
        raise AssertionError("fit must not be used on the holdout")
    monkeypatch.setattr(Pipeline, "fit", forbidden)
    result = evaluate_holdout(bundle, test)
    assert result["training"]["records"] == 15
    assert result["test"]["records"] == 9
    assert result["evaluation_design"]["test_fitting_or_tuning"] is False
    assert all(len(result["models"][name]["predictions"]) == 9 for name in MODEL_NAMES)
    assert result["models"]["logs_only"]["metrics"]["macro_f1"] == 1.0
    assert sha256_file(bundle / "models.joblib") == before


def test_frozen_bundle_refuses_overwrite(frozen) -> None:
    train, _, bundle = frozen
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_models(train, bundle)


def test_holdout_rejects_training_file(frozen) -> None:
    train, _, bundle = frozen
    with pytest.raises(ValueError, match="training dataset"):
        evaluate_holdout(bundle, train)


@pytest.mark.parametrize("change,error", [
    ("incident_id", "incident IDs"), ("split_group", "split groups"),
    ("run_id", "campaign run IDs"), ("raw_hash", "raw telemetry"),
    ("time", "after the model was frozen"), ("source", "local fault-classification"),
    ("metric", "Metric feature schema"), ("system", "same local system"),
])
def test_holdout_rejects_incompatible_or_overlapping_records(frozen, change, error) -> None:
    train, test, bundle = frozen
    original = json.loads(train.read_text().splitlines()[0])
    records = [json.loads(line) for line in test.read_text().splitlines()]
    if change in ("incident_id", "split_group"):
        records[0][change] = original[change]
    elif change == "run_id":
        records[0]["provenance"]["run_id"] = "v1"
    elif change == "raw_hash":
        records[0]["provenance"]["docker_logs_sha256"] = original["provenance"]["docker_logs_sha256"]
    elif change == "time":
        records[0]["window_start"] = "2020-01-01T00:00:00Z"
    elif change == "source":
        records[0]["source"] = "RCAEval"
    elif change == "metric":
        records[0]["metric_features"].pop("latency")
    elif change == "system":
        records[0]["system_name"] = "other-platform"
    test.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match=error):
        evaluate_holdout(bundle, test)


def test_tampered_model_is_rejected_before_deserialisation(frozen, monkeypatch) -> None:
    _, test, bundle = frozen
    (bundle / "models.joblib").write_bytes(b"not a trusted artifact")
    def forbidden(*args, **kwargs):
        raise AssertionError("Do not deserialize a model whose hash changed")
    monkeypatch.setattr("experiment.frozen_holdout.joblib.load", forbidden)
    with pytest.raises(ValueError, match="model hash mismatch"):
        evaluate_holdout(bundle, test)


def test_environment_or_source_change_invalidates_bundle(frozen) -> None:
    _, _, bundle = frozen
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_sha256"]["experiment/ml_baseline.py"] = "changed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source changed"):
        verify_bundle(bundle)


def test_dependency_version_change_invalidates_bundle(frozen, monkeypatch) -> None:
    from experiment import frozen_holdout
    _, _, bundle = frozen
    versions = frozen_holdout.runtime_versions()
    versions["scikit_learn"] = "different-version"
    monkeypatch.setattr(frozen_holdout, "runtime_versions", lambda: versions)
    with pytest.raises(ValueError, match="Runtime versions differ"):
        verify_bundle(bundle)


def test_bundle_requires_complete_source_manifest(frozen) -> None:
    _, _, bundle = frozen
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_sha256"].pop("experiment/ml_baseline.py")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source list is incomplete"):
        verify_bundle(bundle)


def test_contribution_audit_uses_frozen_transforms_without_fitting(
    frozen, monkeypatch
) -> None:
    train, test, bundle = frozen
    before = sha256_file(bundle / "models.joblib")

    def forbidden(*args, **kwargs):
        raise AssertionError("Contribution audit must not fit")

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    result = run_audit(bundle, train, test)
    decomposition = result["database_decision_decomposition"]
    assert result["method"]["fitting_or_tuning"] is False
    assert len(decomposition["incidents"]) == 3
    assert decomposition["maximum_reconstruction_error"] <= 1e-8
    assert len(result["metric_distribution_shift"]["features"]) == 2
    assert sha256_file(bundle / "models.joblib") == before
