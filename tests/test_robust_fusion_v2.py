from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from experiment.ml_baseline import LABELS, load_incident_dataset, sha256_file
from experiment.robust_fusion_v2 import (
    DenseTrainingRangeScaler,
    GatedRobustFusionClassifier,
    MetricEngineer,
    cross_validate_candidate,
    evaluate_candidate,
    freeze_candidate,
)


def _metrics(label_index: int, index: int) -> dict[str, float]:
    total_pre = 10.0 + index
    total_post = 12.0 + index
    successful_pre = total_pre - label_index
    successful_post = total_post - label_index - 1
    failure_pre = total_pre - successful_pre
    failure_post = total_post - successful_post
    values: dict[str, float] = {
        "order_metrics__latency_ms__post_mean": float(label_index * 10 + index),
    }
    phases = {
        "pre_mean": {
            "successful_requests": successful_pre,
            "total_requests": total_pre,
            "failure_count": failure_pre,
            "http_500_count": float(label_index),
            "window_seconds": 5.0,
        },
        "post_mean": {
            "successful_requests": successful_post,
            "total_requests": total_post,
            "failure_count": failure_post,
            "http_500_count": float(label_index + 1),
            "window_seconds": 5.0,
        },
    }
    for family in (
        "successful_requests",
        "total_requests",
        "failure_count",
        "http_500_count",
        "window_seconds",
    ):
        pre = phases["pre_mean"][family]
        post = phases["post_mean"][family]
        for phase, value in (
            ("pre_mean", pre),
            ("post_mean", post),
            ("mean_delta", post - pre),
        ):
            values[f"order_metrics__payload__{family}__{phase}"] = value
    return values


def _rows(count: int = 4) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    labels: list[str] = []
    symptoms = (
        "database refused connection",
        "forced http error response",
        "service unavailable shutdown",
    )
    for label_index, label in enumerate(LABELS):
        for index in range(count):
            rows.append(
                {
                    "log_text": f"{symptoms[label_index]} {symptoms[label_index]} request outcome",
                    "metric_features": _metrics(label_index, index),
                }
            )
            labels.append(label)
    return rows, labels


def _dataset(path: Path, run: str, *, year: int, count: int = 5) -> None:
    rows, labels = _rows(count)
    records = []
    for index, (row, label) in enumerate(zip(rows, labels)):
        records.append(
            {
                "schema_version": 1,
                "source": "local",
                "task": "local_fault_classification",
                "system_name": "local-order-platform",
                "incident_id": f"{run}-{index}",
                "split_group": f"group-{run}-{index}",
                "local_fault_label": label,
                "window_start": f"{year}-01-01T00:00:00Z",
                "window_end": f"{year}-01-01T00:01:00Z",
                "log_text": row["log_text"],
                "metric_features": row["metric_features"],
                "provenance": {
                    "run_id": run,
                    "docker_logs_sha256": f"logs-{run}-{index}",
                    "probes_sha256": f"probes-{run}-{index}",
                },
            }
        )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _spec(path: Path, training: Path) -> None:
    source = Path("experiment/configs/robust-fusion-v2-spec.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    value["training"] = {
        "path": str(training),
        "sha256": sha256_file(training),
        "records": len(training.read_text(encoding="utf-8").splitlines()),
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def test_metric_engineer_drops_absolute_families_and_adds_rates() -> None:
    metrics = _metrics(label_index=1, index=0)
    engineer = MetricEngineer().fit([{"metric_features": metrics}])
    result = engineer.transform([{"metric_features": metrics}])[0]
    for family in (
        "successful_requests",
        "total_requests",
        "failure_count",
        "http_500_count",
        "window_seconds",
    ):
        assert not any(f"__{family}__" in name for name in result)
    assert result["order_metrics__payload__successful_rate__pre_mean"] == pytest.approx(0.9)
    assert result["order_metrics__payload__successful_rate__post_mean"] == pytest.approx(10 / 12)
    assert result["order_metrics__payload__failure_rate__pre_mean"] == pytest.approx(0.1)
    assert result["order_metrics__payload__failure_rate__mean_delta"] == pytest.approx(2 / 12 - 0.1)
    assert len(result) == 7


def test_scaler_uses_training_range_for_ood_and_clipping() -> None:
    scaler = DenseTrainingRangeScaler().fit(
        [{"a": 0.0, "b": 10.0}, {"a": 2.0, "b": 20.0}]
    )
    values = [{"a": 4.0, "b": 15.0}, {"a": -1.0, "b": 30.0}]
    assert scaler.ood_fraction(values).tolist() == [0.5, 1.0]
    transformed = scaler.transform(values)
    assert transformed[0, 0] == pytest.approx(1.0)
    assert transformed[1, 0] == pytest.approx(-1.0)
    assert transformed[1, 1] == pytest.approx(1.0)


def test_gate_falls_back_when_more_than_ten_percent_are_outside() -> None:
    rows, labels = _rows()
    model = GatedRobustFusionClassifier().fit(rows, labels)
    in_range = {"log_text": rows[0]["log_text"], "metric_features": dict(rows[0]["metric_features"])}
    shifted = {"log_text": rows[0]["log_text"], "metric_features": dict(rows[0]["metric_features"])}
    shifted["metric_features"]["order_metrics__latency_ms__post_mean"] = 10_000.0
    routing = model.routing([in_range, shifted])
    assert routing["ood_fractions"][0] == 0.0
    assert bool(routing["fallback"][0]) is False
    assert routing["ood_fractions"][1] == pytest.approx(1 / 7)
    assert bool(routing["fallback"][1]) is True
    assert np.allclose(
        model.predict_proba([shifted]), model.logs_model_.predict_proba([shifted])
    )


def test_cross_validation_fits_metric_ranges_on_training_fold_only(
    tmp_path: Path, monkeypatch
) -> None:
    rows, _ = _rows(count=5)
    dataset_path = tmp_path / "cv.jsonl"
    records = []
    for index, row in enumerate(rows):
        label = LABELS[index // 5]
        records.append(
            {
                "schema_version": 1,
                "source": "local",
                "task": "local_fault_classification",
                "system_name": "local-order-platform",
                "incident_id": f"cv-{index}",
                "split_group": f"cv-group-{index}",
                "local_fault_label": label,
                "window_start": "2020-01-01T00:00:00Z",
                "window_end": "2020-01-01T00:01:00Z",
                "log_text": row["log_text"],
                "metric_features": row["metric_features"],
            }
        )
    dataset_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    dataset = load_incident_dataset(dataset_path)
    observed: list[int] = []
    original_fit = DenseTrainingRangeScaler.fit

    def recording_fit(self, values, y=None):
        materialised = list(values)
        observed.append(len(materialised))
        return original_fit(self, materialised, y)

    monkeypatch.setattr(DenseTrainingRangeScaler, "fit", recording_fit)
    result = cross_validate_candidate(dataset, folds=5, seed=20260830, threshold=0.1)
    assert observed == [12, 12, 12, 12, 12]
    assert result["preprocessing_fit_scope"] == "training_fold_only"


def test_frozen_candidate_evaluation_never_fits_and_preserves_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    training = tmp_path / "train.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    spec = tmp_path / "spec.json"
    bundle = tmp_path / "bundle"
    _dataset(training, "v1", year=2020)
    _dataset(holdout, "v4", year=2099, count=3)
    _spec(spec, training)
    freeze_candidate(training, bundle, spec_path=spec)
    before = sha256_file(bundle / "model.joblib")

    def forbidden(*args, **kwargs):
        raise AssertionError("fit must not be used on a frozen holdout")

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(GatedRobustFusionClassifier, "fit", forbidden)
    result = evaluate_candidate(bundle, holdout)
    assert result["evaluation_design"]["fitting_or_tuning"] is False
    assert result["evaluation_design"]["captured_after_freeze"] is True
    assert result["test"]["records"] == 9
    assert sha256_file(bundle / "model.joblib") == before


def test_frozen_candidate_refuses_overwrite_and_spec_hash_mismatch(tmp_path: Path) -> None:
    training = tmp_path / "train.jsonl"
    spec = tmp_path / "spec.json"
    bundle = tmp_path / "bundle"
    _dataset(training, "v1", year=2020)
    _spec(spec, training)
    freeze_candidate(training, bundle, spec_path=spec)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_candidate(training, bundle, spec_path=spec)
    other_bundle = tmp_path / "other-bundle"
    value = json.loads(spec.read_text())
    value["training"]["sha256"] = "changed"
    spec.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hash differs"):
        freeze_candidate(training, other_bundle, spec_path=spec)


def test_python_module_cli_creates_canonically_loadable_pickle(tmp_path: Path) -> None:
    training = tmp_path / "train.jsonl"
    diagnostic = tmp_path / "diagnostic.jsonl"
    spec = tmp_path / "spec.json"
    bundle = tmp_path / "bundle"
    output = tmp_path / "development.json"
    figure = tmp_path / "development.png"
    _dataset(training, "v1", year=2020)
    _dataset(diagnostic, "v3", year=2021, count=3)
    _spec(spec, training)
    value = json.loads(spec.read_text())
    value["development_diagnostic"] = {
        "path": str(diagnostic),
        "sha256": sha256_file(diagnostic),
        "confirmatory": False,
    }
    spec.write_text(json.dumps(value))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiment.robust_fusion_v2",
            "develop-freeze",
            "--spec",
            str(spec),
            "--bundle-dir",
            str(bundle),
            "--output",
            str(output),
            "--figure",
            str(figure),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    model = joblib.load(bundle / "model.joblib")
    assert isinstance(model, GatedRobustFusionClassifier)
