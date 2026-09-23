"""Explain a frozen combined model without fitting or changing its predictions."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from experiment.frozen_holdout import evaluate_holdout, verify_bundle, write_json
from experiment.ml_baseline import (
    LABELS,
    _feature_names,
    load_incident_dataset,
    sha256_file,
)


ACTUAL_LABEL = "database_connection_failure"
COMPETING_LABEL = "service_stopped"


def _scores(matrix: Any, coefficients: np.ndarray) -> np.ndarray:
    return np.asarray(matrix @ coefficients.T, dtype=float)


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(values)), 6),
        "minimum": round(float(np.min(values)), 6),
        "maximum": round(float(np.max(values)), 6),
    }


def _preference_counts(scores: np.ndarray, classes: list[str]) -> dict[str, int]:
    labels = [classes[int(index)] for index in np.argmax(scores, axis=1)]
    return dict(sorted(Counter(labels).items()))


def _rank_contributions(
    names: list[str],
    values: np.ndarray,
    *,
    descending: bool,
    limit: int = 15,
) -> list[dict[str, Any]]:
    indexes = np.argsort(values)
    if descending:
        indexes = indexes[::-1]
    return [
        {
            "feature": names[int(index)],
            "mean_margin_contribution": round(float(values[int(index)]), 6),
        }
        for index in indexes[:limit]
    ]


def _metric_matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    return np.asarray(
        [
            [float(row["metric_features"][name]) for name in names]
            for row in rows
        ],
        dtype=float,
    )


def metric_shift_audit(
    train_rows: list[dict[str, Any]],
    train_labels: list[str],
    test_rows: list[dict[str, Any]],
    test_labels: list[str],
    *,
    names: list[str],
    scales: np.ndarray,
    margin_coefficients: np.ndarray,
) -> list[dict[str, Any]]:
    train = _metric_matrix(train_rows, names)
    test = _metric_matrix(test_rows, names)
    train_mask = np.asarray(train_labels) == ACTUAL_LABEL
    test_mask = np.asarray(test_labels) == ACTUAL_LABEL
    train_actual = train[train_mask]
    test_actual = test[test_mask]
    result = []
    for index, name in enumerate(names):
        train_column = train[:, index]
        test_column = test[:, index]
        train_db = train_actual[:, index]
        test_db = test_actual[:, index]
        train_std = float(np.std(train_column))
        db_std = float(np.std(train_db))
        coefficient = float(margin_coefficients[index])
        train_db_contribution = float(np.mean(train_db / scales[index]) * coefficient)
        test_db_contribution = float(np.mean(test_db / scales[index]) * coefficient)
        result.append(
            {
                "feature": name,
                "train": {
                    "mean": round(float(np.mean(train_column)), 6),
                    "minimum": round(float(np.min(train_column)), 6),
                    "maximum": round(float(np.max(train_column)), 6),
                    "standard_deviation": round(train_std, 6),
                },
                "test": {
                    "mean": round(float(np.mean(test_column)), 6),
                    "minimum": round(float(np.min(test_column)), 6),
                    "maximum": round(float(np.max(test_column)), 6),
                    "outside_global_training_range_records": int(
                        np.sum(
                            (test_column < np.min(train_column))
                            | (test_column > np.max(train_column))
                        )
                    ),
                },
                "database_class": {
                    "train_mean": round(float(np.mean(train_db)), 6),
                    "test_mean": round(float(np.mean(test_db)), 6),
                    "standardised_mean_shift": (
                        round(float((np.mean(test_db) - np.mean(train_db)) / db_std), 6)
                        if db_std > 0
                        else None
                    ),
                    "outside_class_training_range_records": int(
                        np.sum(
                            (test_db < np.min(train_db))
                            | (test_db > np.max(train_db))
                        )
                    ),
                },
                "frozen_transform_and_margin": {
                    "scaler_scale": round(float(scales[index]), 6),
                    "service_minus_database_coefficient": round(coefficient, 6),
                    "train_database_mean_contribution": round(
                        train_db_contribution, 6
                    ),
                    "test_database_mean_contribution": round(
                        test_db_contribution, 6
                    ),
                    "contribution_shift": round(
                        test_db_contribution - train_db_contribution, 6
                    ),
                },
            }
        )
    return result


def run_audit(
    bundle_dir: Path,
    training_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    manifest = verify_bundle(bundle_dir)
    if sha256_file(training_path) != manifest["training"]["sha256"]:
        raise ValueError("Training data differs from the frozen manifest")
    holdout_result = evaluate_holdout(bundle_dir, test_path)
    training = load_incident_dataset(training_path)
    test = load_incident_dataset(test_path)
    models = joblib.load(bundle_dir / "models.joblib")
    model = models["logs_plus_metrics"]
    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    branches = dict(features.transformer_list)
    log_matrix = branches["logs"].transform(test.rows)
    metric_matrix = branches["metrics"].transform(test.rows)
    full_matrix = features.transform(test.rows)
    coefficients = np.asarray(classifier.coef_, dtype=float)
    intercept = np.asarray(classifier.intercept_, dtype=float)
    classes = [str(value) for value in classifier.classes_]
    actual_index = classes.index(ACTUAL_LABEL)
    competing_index = classes.index(COMPETING_LABEL)
    mask = np.asarray(test.labels) == ACTUAL_LABEL
    selected_indexes = np.flatnonzero(mask)

    log_scores = _scores(log_matrix, coefficients[:, : log_matrix.shape[1]])
    metric_scores = _scores(
        metric_matrix, coefficients[:, log_matrix.shape[1] :]
    )
    full_scores = _scores(full_matrix, coefficients) + intercept
    log_margin = (
        log_scores[mask, competing_index] - log_scores[mask, actual_index]
    )
    metric_margin = (
        metric_scores[mask, competing_index] - metric_scores[mask, actual_index]
    )
    intercept_margin = float(intercept[competing_index] - intercept[actual_index])
    full_margin = (
        full_scores[mask, competing_index] - full_scores[mask, actual_index]
    )
    residual = full_margin - (log_margin + metric_margin + intercept_margin)
    if float(np.max(np.abs(residual))) > 1e-8:
        raise RuntimeError("Branch contributions do not reconstruct the decision margin")

    all_names = _feature_names(model, "logs_plus_metrics")
    margin_weights = coefficients[competing_index] - coefficients[actual_index]
    selected_matrix = full_matrix[mask]
    mean_feature_values = np.asarray(selected_matrix.mean(axis=0)).ravel()
    mean_contributions = mean_feature_values * margin_weights
    predictions = [str(value) for value in model.predict(test.rows)]
    incidents = []
    for local_index, dataset_index in enumerate(selected_indexes):
        incidents.append(
            {
                "incident_id": test.incident_ids[int(dataset_index)],
                "full_prediction": predictions[int(dataset_index)],
                "service_minus_database_margin": round(
                    float(full_margin[local_index]), 6
                ),
                "intercept_contribution": round(intercept_margin, 6),
                "log_contribution": round(float(log_margin[local_index]), 6),
                "metric_contribution": round(float(metric_margin[local_index]), 6),
                "reconstruction_residual": round(float(residual[local_index]), 12),
            }
        )

    metric_transform = branches["metrics"]
    vectoriser = metric_transform.named_steps["vectorise"]
    scaler = metric_transform.named_steps["scale"]
    metric_names = [str(value) for value in vectoriser.get_feature_names_out()]
    metric_coefficients = margin_weights[log_matrix.shape[1] :]
    shifts = metric_shift_audit(
        training.rows,
        training.labels,
        test.rows,
        test.labels,
        names=metric_names,
        scales=np.asarray(scaler.scale_, dtype=float),
        margin_coefficients=metric_coefficients,
    )
    by_standardised_shift = sorted(
        shifts,
        key=lambda item: abs(
            item["database_class"]["standardised_mean_shift"]
            if item["database_class"]["standardised_mean_shift"] is not None
            else 0.0
        ),
        reverse=True,
    )
    by_margin_shift = sorted(
        shifts,
        key=lambda item: abs(
            item["frozen_transform_and_margin"]["contribution_shift"]
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "audit_id": "local-frozen-contribution-audit-v1-to-v3",
        "created_at": datetime.now(UTC).isoformat(),
        "method": {
            "model": "frozen logs_plus_metrics pipeline",
            "fitting_or_tuning": False,
            "target_margin": f"{COMPETING_LABEL} minus {ACTUAL_LABEL}",
            "positive_margin_meaning": "evidence toward the observed wrong service-stop class",
            "decomposition": "intercept + log contribution + metric contribution",
            "post_hoc": True,
        },
        "bundle": {
            "path": str(bundle_dir),
            "model_sha256": manifest["model_sha256"],
            "manifest_sha256": sha256_file(bundle_dir / "manifest.json"),
        },
        "datasets": {
            "training": {
                "path": str(training_path),
                "sha256": sha256_file(training_path),
                "records": len(training.rows),
            },
            "test": {
                "path": str(test_path),
                "sha256": sha256_file(test_path),
                "records": len(test.rows),
            },
        },
        "reference_holdout_metrics": {
            name: holdout_result["models"][name]["metrics"]
            for name in ("logs_only", "logs_plus_metrics")
        },
        "database_decision_decomposition": {
            "incidents": incidents,
            "full_predictions": dict(
                sorted(Counter(item["full_prediction"] for item in incidents).items())
            ),
            "log_branch_preference_without_intercept": _preference_counts(
                log_scores[mask], classes
            ),
            "metric_branch_preference_without_intercept": _preference_counts(
                metric_scores[mask], classes
            ),
            "intercept_margin": round(intercept_margin, 6),
            "log_margin": _summary(log_margin),
            "metric_margin": _summary(metric_margin),
            "full_margin": _summary(full_margin),
            "maximum_reconstruction_error": round(
                float(np.max(np.abs(residual))), 12
            ),
            "top_features_toward_service_stop": _rank_contributions(
                all_names, mean_contributions, descending=True
            ),
            "top_features_toward_database": _rank_contributions(
                all_names, mean_contributions, descending=False
            ),
        },
        "metric_distribution_shift": {
            "features": shifts,
            "features_outside_global_training_range": sum(
                item["test"]["outside_global_training_range_records"] > 0
                for item in shifts
            ),
            "top_standardised_database_shifts": by_standardised_shift[:15],
            "top_service_minus_database_margin_shifts": by_margin_shift[:15],
        },
        "limitations": [
            "Post-hoc mechanism audit after v3 labels and predictions were observed.",
            "Branch scores are an exact linear decomposition of this frozen model, not causal effects under intervention.",
            "TF-IDF and scaled metric magnitudes are model-space contributions and are not directly comparable as raw units.",
            "Any redesigned model informed by this audit requires a new confirmation dataset.",
        ],
    }


def render_figure(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "MPLCONFIGDIR", str((output_path.parent / ".matplotlib").resolve())
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decomposition = result["database_decision_decomposition"]
    incidents = decomposition["incidents"]
    indexes = np.arange(1, len(incidents) + 1)
    logs = np.asarray([item["log_contribution"] for item in incidents])
    metrics = np.asarray([item["metric_contribution"] for item in incidents])
    full = np.asarray([item["service_minus_database_margin"] for item in incidents])
    shifts = result["metric_distribution_shift"][
        "top_service_minus_database_margin_shifts"
    ][:10]
    labels = [item["feature"].replace("order_metrics__payload__", "") for item in shifts]
    values = [
        item["frozen_transform_and_margin"]["contribution_shift"]
        for item in shifts
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(indexes - 0.18, logs, width=0.36, label="Log contribution")
    axes[0].bar(indexes + 0.18, metrics, width=0.36, label="Metric contribution")
    axes[0].plot(indexes, full, "ko", label="Full margin")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Database incident")
    axes[0].set_ylabel("Service-stop minus database decision margin")
    axes[0].set_title("Exact frozen-model branch decomposition")
    axes[0].legend(fontsize=8)

    positions = np.arange(len(labels))
    colors = ["#d95f02" if value > 0 else "#1b9e77" for value in values]
    axes[1].barh(positions, values, color=colors)
    axes[1].set_yticks(positions, labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("v3 minus v1 database-class margin contribution")
    axes[1].set_title("Largest metric contribution shifts")
    fig.suptitle("Frozen v1 combined model under the v3 workload shift")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, default=Path("data/models/local-v1-frozen")
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=Path("data/processed/incidents/local-campaign-v1.jsonl"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/processed/incidents/local-campaign-v3.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/local-frozen-contribution-audit-v1-to-v3.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(
            "data/results/figures/local-frozen-contribution-audit-v1-to-v3.png"
        ),
    )
    args = parser.parse_args(argv)
    if args.output.exists() or args.figure.exists():
        raise FileExistsError("Contribution-audit outputs already exist")
    result = run_audit(args.bundle, args.training, args.test)
    render_figure(result, args.figure)
    result["artifacts"] = {
        "figure": str(args.figure),
        "figure_sha256": sha256_file(args.figure),
    }
    write_json(args.output, result)
    summary = {
        "output": str(args.output),
        "full_predictions": result["database_decision_decomposition"][
            "full_predictions"
        ],
        "log_margin": result["database_decision_decomposition"]["log_margin"],
        "metric_margin": result["database_decision_decomposition"]["metric_margin"],
        "full_margin": result["database_decision_decomposition"]["full_margin"],
        "features_outside_global_training_range": result[
            "metric_distribution_shift"
        ]["features_outside_global_training_range"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
