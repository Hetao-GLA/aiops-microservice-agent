"""Develop, freeze, and evaluate the gated robust-fusion v2 candidate.

The candidate is trained only on the local v1 campaign.  The already-observed
v3 campaign is a post-hoc development diagnostic, never confirmatory evidence.
Future confirmation must use a later, disjoint campaign captured after freeze.

Joblib is pickle-based.  Load only bundles produced locally by this project.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils.validation import check_is_fitted

from experiment.frozen_holdout import (
    _provenance,
    _timestamp,
    _validate_metric_schema,
    runtime_versions,
    write_json,
)
from experiment.ml_baseline import (
    LABELS,
    IncidentDataset,
    _classification_metrics,
    _metric_summary,
    _subset,
    _text_pipeline,
    build_model,
    load_incident_dataset,
    make_splits,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "gated-robust-fusion-v2"
MODEL_FILENAME = "model.joblib"
DEFAULT_SPEC = Path("experiment/configs/robust-fusion-v2-spec.json")
DROP_ABSOLUTE_FAMILIES = (
    "successful_requests",
    "total_requests",
    "failure_count",
    "http_500_count",
    "window_seconds",
)
RATE_FAMILIES = ("successful_rate", "failure_rate")
RATE_PHASES = ("pre_mean", "post_mean", "mean_delta")
FROZEN_SOURCES = (
    "experiment/robust_fusion_v2.py",
    "experiment/ml_baseline.py",
    "experiment/frozen_holdout.py",
    "experiment/data_engineering/build_local_incidents.py",
    "experiment/data_engineering/incident_schema.py",
)


def _stabilise_pickle_module_names() -> None:
    """Make ``python -m`` artifacts loadable through the canonical module."""

    if __name__ != "__main__":
        return
    canonical = "experiment.robust_fusion_v2"
    sys.modules[canonical] = sys.modules[__name__]
    for class_type in (
        MetricEngineer,
        DenseTrainingRangeScaler,
        GatedRobustFusionClassifier,
    ):
        class_type.__module__ = canonical


def _as_rows(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(values)
    if not rows:
        raise ValueError("At least one incident row is required")
    return rows


class MetricEngineer(BaseEstimator, TransformerMixin):
    """Drop workload-volume features and derive bounded request-rate features."""

    def __init__(
        self,
        drop_absolute_families: tuple[str, ...] = DROP_ABSOLUTE_FAMILIES,
    ) -> None:
        self.drop_absolute_families = drop_absolute_families

    @staticmethod
    def _metrics(row: dict[str, Any]) -> dict[str, float]:
        raw = row.get("metric_features")
        if not isinstance(raw, dict) or not raw:
            raise ValueError("Each row must contain non-empty metric_features")
        result: dict[str, float] = {}
        for name, value in raw.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Metric {name!r} is not numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"Metric {name!r} is not finite")
            result[str(name)] = numeric
        return result

    def fit(
        self,
        values: Iterable[dict[str, Any]],
        y: Sequence[str] | None = None,
    ) -> "MetricEngineer":
        rows = _as_rows(values)
        schemas = [sorted(self._metrics(row)) for row in rows]
        if any(schema != schemas[0] for schema in schemas[1:]):
            raise ValueError("Raw metric feature schema differs between rows")
        self.raw_feature_names_in_ = np.asarray(schemas[0], dtype=object)
        engineered = self._engineer(self._metrics(rows[0]))
        self.feature_names_out_ = np.asarray(sorted(engineered), dtype=object)
        return self

    def _engineer(self, metrics: dict[str, float]) -> dict[str, float]:
        dropped_tokens = tuple(
            f"order_metrics__payload__{family}__"
            for family in self.drop_absolute_families
        )
        result = {
            name: value
            for name, value in metrics.items()
            if not name.startswith(dropped_tokens)
        }
        total_prefix = "order_metrics__payload__total_requests__"
        numerator_names = {
            "successful_rate": "successful_requests",
            "failure_rate": "failure_count",
        }
        for rate_name, numerator_name in numerator_names.items():
            phase_values: dict[str, float] = {}
            for phase in ("pre_mean", "post_mean"):
                numerator_key = (
                    f"order_metrics__payload__{numerator_name}__{phase}"
                )
                denominator_key = total_prefix + phase
                if numerator_key not in metrics or denominator_key not in metrics:
                    raise ValueError(
                        f"Rate engineering requires {numerator_key!r} and "
                        f"{denominator_key!r}"
                    )
                denominator = metrics[denominator_key]
                phase_values[phase] = (
                    metrics[numerator_key] / denominator if denominator > 0 else 0.0
                )
                result[
                    f"order_metrics__payload__{rate_name}__{phase}"
                ] = phase_values[phase]
            result[
                f"order_metrics__payload__{rate_name}__mean_delta"
            ] = phase_values["post_mean"] - phase_values["pre_mean"]
        return result

    def transform(
        self, values: Iterable[dict[str, Any]]
    ) -> list[dict[str, float]]:
        check_is_fitted(self, ("raw_feature_names_in_", "feature_names_out_"))
        rows = _as_rows(values)
        expected_raw = [str(value) for value in self.raw_feature_names_in_]
        transformed: list[dict[str, float]] = []
        for row in rows:
            metrics = self._metrics(row)
            if sorted(metrics) != expected_raw:
                raise ValueError("Raw metric feature schema differs from training")
            engineered = self._engineer(metrics)
            if sorted(engineered) != [str(value) for value in self.feature_names_out_]:
                raise RuntimeError("Engineered metric schema changed unexpectedly")
            transformed.append(engineered)
        return transformed

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


class DenseTrainingRangeScaler(BaseEstimator, TransformerMixin):
    """Clip by training min/max and scale by training median/IQR."""

    def fit(
        self,
        values: Iterable[dict[str, float]],
        y: Sequence[str] | None = None,
    ) -> "DenseTrainingRangeScaler":
        rows = list(values)
        if not rows:
            raise ValueError("At least one engineered metric row is required")
        names = sorted(rows[0])
        matrix = self._matrix(rows, names)
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)
        self.data_min_ = np.min(matrix, axis=0)
        self.data_max_ = np.max(matrix, axis=0)
        self.center_ = np.median(matrix, axis=0)
        q25, q75 = np.percentile(matrix, [25, 75], axis=0)
        iqr = q75 - q25
        self.scale_ = np.where(iqr > 0, iqr, 1.0)
        return self

    @staticmethod
    def _matrix(
        rows: Sequence[dict[str, float]], names: Sequence[str]
    ) -> np.ndarray:
        expected = list(names)
        matrix: list[list[float]] = []
        for row in rows:
            if sorted(row) != expected:
                raise ValueError("Engineered metric feature schema differs from training")
            converted = [float(row[name]) for name in expected]
            if not all(math.isfinite(value) for value in converted):
                raise ValueError("Engineered metrics must be finite")
            matrix.append(converted)
        return np.asarray(matrix, dtype=float)

    def _raw_matrix(self, values: Iterable[dict[str, float]]) -> np.ndarray:
        check_is_fitted(
            self,
            ("feature_names_in_", "data_min_", "data_max_", "center_", "scale_"),
        )
        rows = list(values)
        if not rows:
            return np.empty((0, self.n_features_in_), dtype=float)
        return self._matrix(rows, [str(value) for value in self.feature_names_in_])

    def transform(self, values: Iterable[dict[str, float]]) -> np.ndarray:
        matrix = self._raw_matrix(values)
        clipped = np.clip(matrix, self.data_min_, self.data_max_)
        return (clipped - self.center_) / self.scale_

    def ood_fraction(self, values: Iterable[dict[str, float]]) -> np.ndarray:
        matrix = self._raw_matrix(values)
        if matrix.shape[0] == 0:
            return np.empty(0, dtype=float)
        outside = (matrix < self.data_min_) | (matrix > self.data_max_)
        return np.mean(outside, axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_.copy()


def _metric_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("engineer", MetricEngineer()),
            ("scale", DenseTrainingRangeScaler()),
        ]
    )


class GatedRobustFusionClassifier(ClassifierMixin, BaseEstimator):
    """Use robust fusion in-range and a frozen-configuration log model OOD."""

    def __init__(
        self,
        *,
        seed: int = 20260830,
        ood_fraction_threshold: float = 0.1,
    ) -> None:
        self.seed = seed
        self.ood_fraction_threshold = ood_fraction_threshold

    def fit(
        self, values: Iterable[dict[str, Any]], y: Sequence[str]
    ) -> "GatedRobustFusionClassifier":
        rows = _as_rows(values)
        labels = list(y)
        if len(rows) != len(labels):
            raise ValueError("Rows and labels must have the same length")
        if not 0 <= self.ood_fraction_threshold <= 1:
            raise ValueError("OOD fraction threshold must be between zero and one")
        self.logs_model_ = build_model("logs_only", seed=self.seed)
        self.logs_model_.fit(rows, labels)
        features = FeatureUnion(
            [("logs", _text_pipeline()), ("metrics", _metric_pipeline())]
        )
        classifier = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2_000,
            random_state=self.seed,
        )
        self.fusion_model_ = Pipeline(
            [("features", features), ("classifier", classifier)]
        )
        self.fusion_model_.fit(rows, labels)
        fusion_classes = np.asarray(
            self.fusion_model_.named_steps["classifier"].classes_, dtype=object
        )
        log_classes = np.asarray(
            self.logs_model_.named_steps["classifier"].classes_, dtype=object
        )
        if not np.array_equal(fusion_classes, log_classes):
            raise RuntimeError("Fusion and fallback class order differs")
        self.classes_ = fusion_classes
        return self

    def _metric_steps(self) -> tuple[MetricEngineer, DenseTrainingRangeScaler]:
        check_is_fitted(self, ("fusion_model_", "logs_model_", "classes_"))
        features = self.fusion_model_.named_steps["features"]
        metric_branch = dict(features.transformer_list)["metrics"]
        return (
            metric_branch.named_steps["engineer"],
            metric_branch.named_steps["scale"],
        )

    def routing(self, values: Iterable[dict[str, Any]]) -> dict[str, Any]:
        rows = _as_rows(values)
        engineer, scaler = self._metric_steps()
        engineered = engineer.transform(rows)
        fractions = scaler.ood_fraction(engineered)
        fallback = fractions > self.ood_fraction_threshold
        return {
            "ood_fractions": fractions,
            "fallback": fallback,
            "fusion": ~fallback,
        }

    def predict_proba(self, values: Iterable[dict[str, Any]]) -> np.ndarray:
        rows = _as_rows(values)
        routing = self.routing(rows)
        fusion_probabilities = self.fusion_model_.predict_proba(rows)
        log_probabilities = self.logs_model_.predict_proba(rows)
        result = np.asarray(fusion_probabilities, dtype=float).copy()
        result[routing["fallback"]] = log_probabilities[routing["fallback"]]
        return result

    def predict(self, values: Iterable[dict[str, Any]]) -> np.ndarray:
        probabilities = self.predict_proba(values)
        return self.classes_[np.argmax(probabilities, axis=1)]


def _load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or spec.get("model_id") != MODEL_ID:
        raise ValueError("Unsupported robust-fusion specification")
    expected = {
        "drop_absolute_families": list(DROP_ABSOLUTE_FAMILIES),
        "add_rates": list(RATE_FAMILIES),
        "rate_phases": list(RATE_PHASES),
    }
    engineering = spec.get("metric_engineering", {})
    for key, value in expected.items():
        if engineering.get(key) != value:
            raise ValueError(f"Specification differs from implementation: {key}")
    fusion = spec.get("fusion", {})
    if fusion != {
        "classifier": "LogisticRegression",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 2000,
    }:
        raise ValueError("Specification differs from fusion implementation")
    threshold = spec.get("routing", {}).get(
        "fallback_when_fraction_strictly_greater_than"
    )
    if threshold != 0.1:
        raise ValueError("Specification differs from routing implementation")
    return spec


def _resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _validate_spec_datasets(spec: dict[str, Any]) -> tuple[Path, Path]:
    training = _resolve_recorded_path(spec["training"]["path"])
    diagnostic = _resolve_recorded_path(spec["development_diagnostic"]["path"])
    for name, path, details in (
        ("training", training, spec["training"]),
        ("development diagnostic", diagnostic, spec["development_diagnostic"]),
    ):
        if sha256_file(path) != details["sha256"]:
            raise ValueError(f"{name.capitalize()} dataset hash differs from specification")
    return training, diagnostic


def _routing_summary(routing: dict[str, Any]) -> dict[str, Any]:
    fractions = np.asarray(routing["ood_fractions"], dtype=float)
    fallback = np.asarray(routing["fallback"], dtype=bool)
    return {
        "records": int(len(fractions)),
        "fusion_records": int(np.sum(~fallback)),
        "fallback_records": int(np.sum(fallback)),
        "coverage": round(float(np.mean(~fallback)), 6),
        "fallback_rate": round(float(np.mean(fallback)), 6),
        "ood_fraction": {
            "mean": round(float(np.mean(fractions)), 6),
            "median": round(float(np.median(fractions)), 6),
            "min": round(float(np.min(fractions)), 6),
            "max": round(float(np.max(fractions)), 6),
        },
    }


def _confusion(truth: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
    matrix = confusion_matrix(truth, predicted, labels=LABELS)
    return {
        actual: {
            label: int(matrix[row_index, column_index])
            for column_index, label in enumerate(LABELS)
        }
        for row_index, actual in enumerate(LABELS)
    }


def cross_validate_candidate(
    dataset: IncidentDataset,
    *,
    folds: int,
    seed: int,
    threshold: float,
) -> dict[str, Any]:
    splits = make_splits(dataset, folds=folds, seed=seed)
    candidate_oof: list[str | None] = [None] * len(dataset.rows)
    logs_oof: list[str | None] = [None] * len(dataset.rows)
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    all_fractions: list[float] = []
    all_fallback: list[bool] = []
    template = GatedRobustFusionClassifier(
        seed=seed, ood_fraction_threshold=threshold
    )
    for fold_number, (train_indexes, test_indexes) in enumerate(splits, start=1):
        model = clone(template)
        train_rows = _subset(dataset.rows, train_indexes)
        train_labels = _subset(dataset.labels, train_indexes)
        test_rows = _subset(dataset.rows, test_indexes)
        test_labels = _subset(dataset.labels, test_indexes)
        model.fit(train_rows, train_labels)
        candidate_predictions = [str(value) for value in model.predict(test_rows)]
        logs_predictions = [str(value) for value in model.logs_model_.predict(test_rows)]
        probabilities = model.predict_proba(test_rows)
        routing = model.routing(test_rows)
        routing_summary = _routing_summary(routing)
        all_fractions.extend(float(value) for value in routing["ood_fractions"])
        all_fallback.extend(bool(value) for value in routing["fallback"])
        fold_rows.append(
            {
                "fold": fold_number,
                "train_records": len(train_indexes),
                "test_records": len(test_indexes),
                "test_incident_ids": _subset(dataset.incident_ids, test_indexes),
                "candidate_metrics": _classification_metrics(
                    test_labels, candidate_predictions
                ),
                "logs_only_metrics": _classification_metrics(
                    test_labels, logs_predictions
                ),
                "routing": routing_summary,
            }
        )
        for local_index, dataset_index in enumerate(test_indexes):
            candidate = candidate_predictions[local_index]
            logs = logs_predictions[local_index]
            candidate_oof[dataset_index] = candidate
            logs_oof[dataset_index] = logs
            prediction_rows.append(
                {
                    "incident_id": dataset.incident_ids[dataset_index],
                    "fold": fold_number,
                    "actual": dataset.labels[dataset_index],
                    "candidate_prediction": candidate,
                    "logs_only_prediction": logs,
                    "candidate_confidence": round(
                        float(
                            probabilities[
                                local_index, list(model.classes_).index(candidate)
                            ]
                        ),
                        6,
                    ),
                    "ood_fraction": round(
                        float(routing["ood_fractions"][local_index]), 6
                    ),
                    "fallback_to_logs": bool(routing["fallback"][local_index]),
                    "correct": candidate == dataset.labels[dataset_index],
                }
            )
    if any(value is None for value in candidate_oof + logs_oof):
        raise RuntimeError("Not every incident received an out-of-fold prediction")
    candidate_predictions = [str(value) for value in candidate_oof]
    logs_predictions = [str(value) for value in logs_oof]
    routing = {
        "ood_fractions": np.asarray(all_fractions, dtype=float),
        "fallback": np.asarray(all_fallback, dtype=bool),
    }
    metric_names = ("accuracy", "macro_f1", "weighted_f1")
    return {
        "method": "shared_5_fold_StratifiedGroupKFold",
        "preprocessing_fit_scope": "training_fold_only",
        "folds": fold_rows,
        "candidate": {
            "out_of_fold_metrics": _classification_metrics(
                dataset.labels, candidate_predictions
            ),
            "confusion_matrix": _confusion(dataset.labels, candidate_predictions),
            "cross_validation": {
                name: _metric_summary(
                    [float(fold["candidate_metrics"][name]) for fold in fold_rows]
                )
                for name in metric_names
            },
            "routing": _routing_summary(routing),
        },
        "logs_only": {
            "out_of_fold_metrics": _classification_metrics(
                dataset.labels, logs_predictions
            ),
            "confusion_matrix": _confusion(dataset.labels, logs_predictions),
            "cross_validation": {
                name: _metric_summary(
                    [float(fold["logs_only_metrics"][name]) for fold in fold_rows]
                )
                for name in metric_names
            },
        },
        "predictions": sorted(prediction_rows, key=lambda row: row["incident_id"]),
    }


def freeze_candidate(
    training_path: Path,
    output_dir: Path,
    *,
    spec_path: Path = DEFAULT_SPEC,
) -> dict[str, Any]:
    _stabilise_pickle_module_names()
    if output_dir.exists():
        raise FileExistsError(
            f"Frozen bundle already exists; refusing to overwrite: {output_dir}"
        )
    spec = _load_spec(spec_path)
    if sha256_file(training_path) != spec["training"]["sha256"]:
        raise ValueError("Training dataset hash differs from specification")
    dataset = load_incident_dataset(training_path)
    if len(dataset.rows) != int(spec["training"]["records"]):
        raise ValueError("Training record count differs from specification")
    _validate_metric_schema(dataset, dataset.metric_feature_names)
    provenance = _provenance(training_path)
    threshold = float(
        spec["routing"]["fallback_when_fraction_strictly_greater_than"]
    )
    seed = int(spec["random_seed"])
    model = GatedRobustFusionClassifier(
        seed=seed, ood_fraction_threshold=threshold
    ).fit(dataset.rows, dataset.labels)
    engineer, scaler = model._metric_steps()
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact = output_dir / MODEL_FILENAME
    joblib.dump(model, artifact, compress=3)
    recorded_spec_path = (
        str(spec_path.relative_to(ROOT))
        if spec_path.is_absolute() and spec_path.is_relative_to(ROOT)
        else str(spec_path)
    )
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "Frozen v1 candidate for later holdout-only prediction",
        "random_seed": seed,
        "ood_fraction_threshold": threshold,
        "model_sha256": sha256_file(artifact),
        "versions": runtime_versions(),
        "source_sha256": {
            name: sha256_file(ROOT / name) for name in FROZEN_SOURCES
        },
        "specification": {
            "path": recorded_spec_path,
            "sha256": sha256_file(spec_path),
        },
        "training": {
            "path": str(training_path),
            "sha256": sha256_file(training_path),
            "records": len(dataset.rows),
            "class_counts": dataset.class_counts,
            "incident_ids": dataset.incident_ids,
            "split_groups": dataset.groups,
            "metric_feature_names": dataset.metric_feature_names,
            "provenance": provenance,
            "leakage_controls": dataset.leakage_audit,
        },
        "engineered_metrics": {
            "feature_names": [str(value) for value in engineer.feature_names_out_],
            "feature_count": int(scaler.n_features_in_),
            "training_min": {
                str(name): float(value)
                for name, value in zip(scaler.feature_names_in_, scaler.data_min_)
            },
            "training_max": {
                str(name): float(value)
                for name, value in zip(scaler.feature_names_in_, scaler.data_max_)
            },
            "training_median": {
                str(name): float(value)
                for name, value in zip(scaler.feature_names_in_, scaler.center_)
            },
            "training_iqr_scale": {
                str(name): float(value)
                for name, value in zip(scaler.feature_names_in_, scaler.scale_)
            },
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def verify_candidate_bundle(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("model_id") != MODEL_ID:
        raise ValueError("Unsupported robust-fusion bundle")
    if runtime_versions() != manifest["versions"]:
        raise ValueError("Runtime versions differ from the frozen environment")
    if set(manifest["source_sha256"]) != set(FROZEN_SOURCES):
        raise ValueError("Frozen source list is incomplete")
    for name, expected_hash in manifest["source_sha256"].items():
        if sha256_file(ROOT / name) != expected_hash:
            raise ValueError(f"Frozen preprocessing/evaluation source changed: {name}")
    spec_path = _resolve_recorded_path(manifest["specification"]["path"])
    if sha256_file(spec_path) != manifest["specification"]["sha256"]:
        raise ValueError("Frozen specification hash mismatch")
    if sha256_file(bundle_dir / MODEL_FILENAME) != manifest["model_sha256"]:
        raise ValueError("Frozen model hash mismatch")
    return manifest


def _validate_evaluation_dataset(
    manifest: dict[str, Any],
    input_path: Path,
    *,
    require_after_freeze: bool,
) -> tuple[IncidentDataset, dict[str, Any]]:
    dataset = load_incident_dataset(input_path)
    provenance = _provenance(input_path)
    training = manifest["training"]
    if sha256_file(input_path) == training["sha256"]:
        raise ValueError("Holdout cannot be the training dataset")
    for name, test_values, train_values in (
        ("incident IDs", dataset.incident_ids, training["incident_ids"]),
        ("split groups", dataset.groups, training["split_groups"]),
        ("campaign run IDs", provenance["run_ids"], training["provenance"]["run_ids"]),
        (
            "raw telemetry",
            provenance["raw_telemetry_hashes"],
            training["provenance"]["raw_telemetry_hashes"],
        ),
    ):
        if set(test_values).intersection(train_values):
            raise ValueError(f"Training/holdout overlap in {name}")
    if provenance["systems"] != training["provenance"]["systems"]:
        raise ValueError("This protocol evaluates the same local system only")
    if require_after_freeze:
        lower_bound = max(
            _timestamp(manifest["created_at"]),
            _timestamp(training["provenance"]["latest_window_end"]),
        )
        if _timestamp(provenance["earliest_window_start"]) <= lower_bound:
            raise ValueError("Holdout telemetry must start after the model was frozen")
    _validate_metric_schema(dataset, training["metric_feature_names"])
    return dataset, provenance


def _predict_dataset(
    model: GatedRobustFusionClassifier,
    dataset: IncidentDataset,
) -> dict[str, Any]:
    candidate_predictions = [str(value) for value in model.predict(dataset.rows)]
    logs_predictions = [str(value) for value in model.logs_model_.predict(dataset.rows)]
    probabilities = model.predict_proba(dataset.rows)
    logs_probabilities = model.logs_model_.predict_proba(dataset.rows)
    routing = model.routing(dataset.rows)
    classes = [str(value) for value in model.classes_]

    def model_result(
        predicted: list[str], probability_matrix: np.ndarray
    ) -> dict[str, Any]:
        return {
            "metrics": _classification_metrics(dataset.labels, predicted),
            "confusion_matrix": _confusion(dataset.labels, predicted),
            "predictions": [
                {
                    "incident_id": incident_id,
                    "actual": actual,
                    "predicted": guess,
                    "correct": actual == guess,
                    "confidence": round(
                        float(probability_matrix[index, classes.index(guess)]), 6
                    ),
                }
                for index, (incident_id, actual, guess) in enumerate(
                    zip(dataset.incident_ids, dataset.labels, predicted)
                )
            ],
        }

    candidate = model_result(candidate_predictions, probabilities)
    candidate["routing"] = _routing_summary(routing)
    for index, prediction in enumerate(candidate["predictions"]):
        prediction["ood_fraction"] = round(
            float(routing["ood_fractions"][index]), 6
        )
        prediction["fallback_to_logs"] = bool(routing["fallback"][index])
    return {
        "gated_robust_fusion_v2": candidate,
        "logs_only": model_result(logs_predictions, logs_probabilities),
    }


def evaluate_candidate(
    bundle_dir: Path,
    input_path: Path,
    *,
    require_after_freeze: bool = True,
) -> dict[str, Any]:
    manifest = verify_candidate_bundle(bundle_dir)
    dataset, provenance = _validate_evaluation_dataset(
        manifest, input_path, require_after_freeze=require_after_freeze
    )
    # Never call fit or fit_transform here; deserialize only after hash checks.
    model = joblib.load(bundle_dir / MODEL_FILENAME)
    results = _predict_dataset(model, dataset)
    candidate = results["gated_robust_fusion_v2"]
    logs = results["logs_only"]
    return {
        "schema_version": 1,
        "experiment_id": MODEL_ID + ("-confirmation" if require_after_freeze else "-v3-diagnostic"),
        "created_at": datetime.now(UTC).isoformat(),
        "bundle": {
            "path": str(bundle_dir),
            "manifest_sha256": sha256_file(bundle_dir / "manifest.json"),
            "model_sha256": manifest["model_sha256"],
            "frozen_at": manifest["created_at"],
        },
        "training": {
            key: manifest["training"][key]
            for key in ("path", "sha256", "records", "class_counts")
        },
        "test": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "records": len(dataset.rows),
            "class_counts": dataset.class_counts,
            "provenance": provenance,
        },
        "evaluation_design": {
            "fitting_or_tuning": False,
            "training_test_disjoint": True,
            "captured_after_freeze": require_after_freeze,
            "confirmatory": require_after_freeze,
            "metric_schema_unchanged": True,
            "source_runtime_model_and_spec_verified": True,
        },
        "models": results,
        "comparison": {
            "candidate_minus_logs_only_macro_f1": round(
                candidate["metrics"]["macro_f1"] - logs["metrics"]["macro_f1"],
                6,
            )
        },
        "limitations": [
            "Same system and known fault classes; not cross-system or unknown-fault generalisation.",
            "Complete incident windows make this retrospective classification, not online early diagnosis.",
            "OOD routing measures metric-range shift only; it does not detect every distribution shift.",
            "Classifier probabilities are not calibrated confidence guarantees.",
        ],
    }


def develop_and_freeze(
    spec_path: Path,
    bundle_dir: Path,
) -> dict[str, Any]:
    spec = _load_spec(spec_path)
    training_path, diagnostic_path = _validate_spec_datasets(spec)
    training = load_incident_dataset(training_path)
    if len(training.rows) != int(spec["training"]["records"]):
        raise ValueError("Training record count differs from specification")
    threshold = float(
        spec["routing"]["fallback_when_fraction_strictly_greater_than"]
    )
    seed = int(spec["random_seed"])
    development_cv = cross_validate_candidate(
        training, folds=5, seed=seed, threshold=threshold
    )
    manifest = freeze_candidate(
        training_path, bundle_dir, spec_path=spec_path
    )
    before = sha256_file(bundle_dir / MODEL_FILENAME)
    diagnostic = evaluate_candidate(
        bundle_dir,
        diagnostic_path,
        require_after_freeze=False,
    )
    after = sha256_file(bundle_dir / MODEL_FILENAME)
    if before != after or after != manifest["model_sha256"]:
        raise RuntimeError("Frozen model artifact changed during v3 diagnostic")
    return {
        "schema_version": 1,
        "experiment_id": MODEL_ID + "-development",
        "created_at": datetime.now(UTC).isoformat(),
        "design": {
            "specification": str(spec_path),
            "specification_sha256": sha256_file(spec_path),
            "candidate_rules_locked_before_implementation_and_fitting": True,
            "training_data": "local campaign v1 only",
            "v3_role": "post_hoc_development_diagnostic_only",
            "v3_fitting_or_threshold_selection": False,
            "confirmation_required": "new local campaign v4 captured after freeze",
        },
        "training": {
            "path": str(training_path),
            "sha256": sha256_file(training_path),
            "records": len(training.rows),
            "class_counts": training.class_counts,
        },
        "v1_cross_validation": development_cv,
        "frozen_bundle": {
            "path": str(bundle_dir),
            "frozen_at": manifest["created_at"],
            "model_sha256": manifest["model_sha256"],
            "engineered_metric_count": manifest["engineered_metrics"]["feature_count"],
        },
        "v3_post_hoc_diagnostic": diagnostic,
        "integrity": {
            "model_sha256_before_v3_diagnostic": before,
            "model_sha256_after_v3_diagnostic": after,
            "artifact_unchanged": before == after,
        },
    }


def _matrix_array(matrix: dict[str, dict[str, int]]) -> np.ndarray:
    return np.asarray(
        [[matrix[actual][predicted] for predicted in LABELS] for actual in LABELS],
        dtype=int,
    )


def render_development_figure(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((path.parent / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Database", "HTTP 500", "Service stopped"]
    cv = result["v1_cross_validation"]["candidate"]
    diagnostic = result["v3_post_hoc_diagnostic"]["models"][
        "gated_robust_fusion_v2"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, model_result, title in (
        (axes[0], cv, "v1 grouped OOF"),
        (axes[1], diagnostic, "v3 post-hoc diagnostic"),
    ):
        matrix = _matrix_array(model_result["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        axis.set_xticks(range(3), labels, rotation=20, ha="right")
        axis.set_yticks(range(3), labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        metrics = model_result.get("out_of_fold_metrics", model_result.get("metrics"))
        axis.set_title(f"{title}\nMacro F1 = {metrics['macro_f1']:.3f}")
        for row in range(3):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                )
    routing = diagnostic["routing"]
    values = [routing["coverage"], routing["fallback_rate"]]
    axes[2].bar(["Fusion coverage", "Logs fallback"], values, color=["#2a9d8f", "#e76f51"])
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Fraction of v3 incidents")
    axes[2].set_title("OOD routing on v3")
    for index, value in enumerate(values):
        axes[2].text(index, value + 0.02, f"{value:.3f}", ha="center")
    fig.suptitle("Gated robust-fusion v2 development evidence")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    develop = commands.add_parser("develop-freeze")
    develop.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    develop.add_argument(
        "--bundle-dir", type=Path, default=Path("data/models/gated-robust-fusion-v2")
    )
    develop.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/gated-robust-fusion-v2-development.json"),
    )
    develop.add_argument(
        "--figure",
        type=Path,
        default=Path("data/results/figures/gated-robust-fusion-v2-development.png"),
    )
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument(
        "--bundle", type=Path, default=Path("data/models/gated-robust-fusion-v2")
    )
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "develop-freeze":
        if args.output.exists() or args.figure.exists():
            raise FileExistsError("Development outputs already exist; refusing to overwrite")
        result = develop_and_freeze(args.spec, args.bundle_dir)
        render_development_figure(result, args.figure)
        result["artifacts"] = {
            "figure": str(args.figure),
            "figure_sha256": sha256_file(args.figure),
        }
        write_json(args.output, result)
        summary = {
            "output": str(args.output),
            "bundle": str(args.bundle_dir),
            "v1_oof": result["v1_cross_validation"]["candidate"]["out_of_fold_metrics"],
            "v3_diagnostic": result["v3_post_hoc_diagnostic"]["models"][
                "gated_robust_fusion_v2"
            ]["metrics"],
            "v3_routing": result["v3_post_hoc_diagnostic"]["models"][
                "gated_robust_fusion_v2"
            ]["routing"],
            "model_sha256": result["frozen_bundle"]["model_sha256"],
        }
    else:
        if args.output.exists():
            raise FileExistsError("Evaluation output already exists; refusing to overwrite")
        result = evaluate_candidate(args.bundle, args.input, require_after_freeze=True)
        write_json(args.output, result)
        summary = {
            "output": str(args.output),
            "metrics": result["models"]["gated_robust_fusion_v2"]["metrics"],
            "routing": result["models"]["gated_robust_fusion_v2"]["routing"],
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
