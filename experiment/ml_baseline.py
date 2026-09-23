from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


LABELS = (
    "database_connection_failure",
    "http_500_failure",
    "service_stopped",
)
MODEL_NAMES = ("logs_only", "logs_plus_metrics")
RULE_DETECTOR_PREFIX = "[rule-detector]"
CONTROL_MESSAGE = "controlled_fault_toggled"


@dataclass(frozen=True)
class SanitisedLog:
    text: str
    original_lines: int
    retained_lines: int
    removed_rule_detector_lines: int
    removed_control_plane_lines: int
    stripped_fault_type_fields: int


@dataclass(frozen=True)
class IncidentDataset:
    rows: list[dict[str, Any]]
    labels: list[str]
    groups: list[str]
    incident_ids: list[str]
    class_counts: dict[str, int]
    metric_feature_names: list[str]
    leakage_audit: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prefixed_json(line: str) -> tuple[str, dict[str, Any]] | None:
    closing = line.find("] ")
    if closing < 0:
        return None
    prefix = line[: closing + 2]
    payload = line[closing + 2 :]
    if not payload.startswith("{"):
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return prefix, parsed


def sanitise_log_text(log_text: str) -> SanitisedLog:
    """Remove experiment-control leakage while retaining operational evidence."""

    original = log_text.splitlines()
    retained: list[str] = []
    removed_rule = 0
    removed_control = 0
    stripped_fault_type = 0

    for line in original:
        if line.lstrip().startswith(RULE_DETECTOR_PREFIX):
            removed_rule += 1
            continue

        parsed = _prefixed_json(line)
        if parsed is None:
            retained.append(line)
            continue

        prefix, payload = parsed
        if payload.get("message") == CONTROL_MESSAGE:
            removed_control += 1
            continue
        if "fault_type" in payload:
            payload.pop("fault_type")
            stripped_fault_type += 1
        retained.append(
            prefix
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )

    return SanitisedLog(
        text="\n".join(retained),
        original_lines=len(original),
        retained_lines=len(retained),
        removed_rule_detector_lines=removed_rule,
        removed_control_plane_lines=removed_control,
        stripped_fault_type_fields=stripped_fault_type,
    )


def _numeric_metrics(raw: Any, *, line_number: int) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Line {line_number}: metric_features must be a non-empty object")
    converted: dict[str, float] = {}
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Line {line_number}: metric {name!r} is not numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Line {line_number}: metric {name!r} is not finite")
        converted[str(name)] = numeric
    return converted


def load_incident_dataset(path: str | Path) -> IncidentDataset:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    labels: list[str] = []
    groups: list[str] = []
    incident_ids: list[str] = []
    metric_names: set[str] = set()
    totals: Counter[str] = Counter()

    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            incident_id = str(record.get("incident_id", ""))
            label = str(record.get("local_fault_label", ""))
            group = str(record.get("split_group", ""))
            if not incident_id:
                raise ValueError(f"Line {line_number}: incident_id is required")
            if label not in LABELS:
                raise ValueError(f"Line {line_number}: unsupported local label {label!r}")
            if not group:
                raise ValueError(f"Line {line_number}: split_group is required")
            if not isinstance(record.get("log_text"), str):
                raise ValueError(f"Line {line_number}: log_text must be a string")

            sanitised = sanitise_log_text(record["log_text"])
            if not sanitised.text.strip():
                raise ValueError(
                    f"Line {line_number}: no operational log text remains after sanitisation"
                )
            metrics = _numeric_metrics(
                record.get("metric_features"), line_number=line_number
            )
            metric_names.update(metrics)
            rows.append({"log_text": sanitised.text, "metric_features": metrics})
            labels.append(label)
            groups.append(group)
            incident_ids.append(incident_id)
            totals["original_lines"] += sanitised.original_lines
            totals["retained_lines"] += sanitised.retained_lines
            totals["removed_rule_detector_lines"] += (
                sanitised.removed_rule_detector_lines
            )
            totals["removed_control_plane_lines"] += (
                sanitised.removed_control_plane_lines
            )
            totals["stripped_fault_type_fields"] += (
                sanitised.stripped_fault_type_fields
            )

    if not rows:
        raise ValueError("Incident dataset is empty")
    if len(set(incident_ids)) != len(incident_ids):
        raise ValueError("incident_id values must be unique")
    if len(set(groups)) != len(groups):
        raise ValueError("split_group values must be unique for this local campaign")

    class_counts = dict(sorted(Counter(labels).items()))
    if set(class_counts) != set(LABELS):
        raise ValueError(f"All local fault classes are required: {class_counts}")

    remaining_hits = {
        label: sum(label in row["log_text"] for row in rows) for label in LABELS
    }
    return IncidentDataset(
        rows=rows,
        labels=labels,
        groups=groups,
        incident_ids=incident_ids,
        class_counts=class_counts,
        metric_feature_names=sorted(metric_names),
        leakage_audit={
            **dict(totals),
            "excluded_log_prefixes": [RULE_DETECTOR_PREFIX],
            "excluded_control_messages": [CONTROL_MESSAGE],
            "remaining_exact_label_record_counts": remaining_hits,
            "all_records_nonempty_after_sanitisation": all(
                bool(row["log_text"].strip()) for row in rows
            ),
        },
    )


def _select_log_text(rows: Iterable[dict[str, Any]]) -> list[str]:
    return [row["log_text"] for row in rows]


def _select_metrics(rows: Iterable[dict[str, Any]]) -> list[dict[str, float]]:
    return [row["metric_features"] for row in rows]


def _text_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "select",
                FunctionTransformer(_select_log_text, validate=False),
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=10_000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b",
                ),
            ),
        ]
    )


def _metric_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "select",
                FunctionTransformer(_select_metrics, validate=False),
            ),
            ("vectorise", DictVectorizer(sparse=True, sort=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )


def build_model(model_name: str, *, seed: int) -> Pipeline:
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2_000,
        random_state=seed,
    )
    if model_name == "logs_only":
        features: Pipeline | FeatureUnion = _text_pipeline()
    elif model_name == "logs_plus_metrics":
        features = FeatureUnion(
            [("logs", _text_pipeline()), ("metrics", _metric_pipeline())]
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("features", features), ("classifier", classifier)])


def make_splits(
    dataset: IncidentDataset, *, folds: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if min(dataset.class_counts.values()) < folds:
        raise ValueError("Each class must contain at least one record per fold")
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    indexes = np.arange(len(dataset.rows))
    splits: list[tuple[list[int], list[int]]] = []
    for train, test in splitter.split(indexes, dataset.labels, dataset.groups):
        train_list = [int(value) for value in train]
        test_list = [int(value) for value in test]
        train_groups = {dataset.groups[index] for index in train_list}
        test_groups = {dataset.groups[index] for index in test_list}
        if train_groups.intersection(test_groups):
            raise RuntimeError("A split_group crossed a train/test boundary")
        splits.append((train_list, test_list))
    return splits


def _subset(values: Sequence[Any], indexes: Sequence[int]) -> list[Any]:
    return [values[index] for index in indexes]


def _metric_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 6),
        "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _classification_metrics(
    truth: Sequence[str], predicted: Sequence[str]
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=LABELS, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        truth, predicted, average="macro", zero_division=0
    )
    weighted_f1 = precision_recall_fscore_support(
        truth, predicted, average="weighted", zero_division=0
    )[2]
    return {
        "accuracy": round(float(accuracy_score(truth, predicted)), 6),
        "macro_precision": round(float(macro_precision), 6),
        "macro_recall": round(float(macro_recall), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "per_class": {
            label: {
                "support": int(support[index]),
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
            }
            for index, label in enumerate(LABELS)
        },
    }


def _feature_names(fitted: Pipeline, model_name: str) -> list[str]:
    features = fitted.named_steps["features"]
    if model_name == "logs_only":
        return [
            "log:" + name
            for name in features.named_steps["tfidf"].get_feature_names_out()
        ]
    transformers = dict(features.transformer_list)
    log_names = [
        "log:" + name
        for name in transformers["logs"]
        .named_steps["tfidf"]
        .get_feature_names_out()
    ]
    metric_names = [
        "metric:" + name
        for name in transformers["metrics"]
        .named_steps["vectorise"]
        .get_feature_names_out()
    ]
    return log_names + metric_names


def _top_features(fitted: Pipeline, model_name: str, limit: int = 12) -> dict[str, Any]:
    names = _feature_names(fitted, model_name)
    classifier = fitted.named_steps["classifier"]
    result: dict[str, Any] = {}
    for class_index, label in enumerate(classifier.classes_):
        coefficients = classifier.coef_[class_index]
        ranked = np.argsort(coefficients)[::-1][:limit]
        result[str(label)] = [
            {
                "feature": names[int(index)],
                "coefficient": round(float(coefficients[int(index)]), 6),
            }
            for index in ranked
        ]
    return result


def evaluate_model(
    model_name: str,
    dataset: IncidentDataset,
    splits: Sequence[tuple[list[int], list[int]]],
    *,
    seed: int,
) -> dict[str, Any]:
    template = build_model(model_name, seed=seed)
    oof_predictions: list[str | None] = [None] * len(dataset.rows)
    prediction_rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []

    for fold_number, (train_indexes, test_indexes) in enumerate(splits, start=1):
        model = clone(template)
        train_rows = _subset(dataset.rows, train_indexes)
        test_rows = _subset(dataset.rows, test_indexes)
        train_labels = _subset(dataset.labels, train_indexes)
        test_labels = _subset(dataset.labels, test_indexes)
        model.fit(train_rows, train_labels)
        predicted = [str(value) for value in model.predict(test_rows)]
        probabilities = model.predict_proba(test_rows)
        classes = [str(value) for value in model.named_steps["classifier"].classes_]

        fold_metrics = _classification_metrics(test_labels, predicted)
        folds.append(
            {
                "fold": fold_number,
                "train_records": len(train_indexes),
                "test_records": len(test_indexes),
                "train_class_counts": dict(sorted(Counter(train_labels).items())),
                "test_class_counts": dict(sorted(Counter(test_labels).items())),
                "test_incident_ids": _subset(dataset.incident_ids, test_indexes),
                "metrics": {
                    key: value
                    for key, value in fold_metrics.items()
                    if key != "per_class"
                },
            }
        )

        for local_index, dataset_index in enumerate(test_indexes):
            predicted_label = predicted[local_index]
            oof_predictions[dataset_index] = predicted_label
            confidence = float(probabilities[local_index, classes.index(predicted_label)])
            prediction_rows.append(
                {
                    "incident_id": dataset.incident_ids[dataset_index],
                    "fold": fold_number,
                    "actual": dataset.labels[dataset_index],
                    "predicted": predicted_label,
                    "confidence": round(confidence, 6),
                    "correct": predicted_label == dataset.labels[dataset_index],
                }
            )

    if any(value is None for value in oof_predictions):
        raise RuntimeError("Not every incident received an out-of-fold prediction")
    final_predictions = [str(value) for value in oof_predictions]
    oof_metrics = _classification_metrics(dataset.labels, final_predictions)
    matrix = confusion_matrix(dataset.labels, final_predictions, labels=LABELS)
    metric_names = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    )
    cross_validation = {
        name: _metric_summary([float(fold["metrics"][name]) for fold in folds])
        for name in metric_names
    }

    interpretation_model = clone(template)
    interpretation_model.fit(dataset.rows, dataset.labels)
    feature_names = _feature_names(interpretation_model, model_name)
    return {
        "configuration": {
            "classifier": "LogisticRegression",
            "classifier_c": 1.0,
            "classifier_solver": "lbfgs",
            "tfidf_ngram_range": [1, 2],
            "tfidf_min_df": 2,
            "tfidf_max_df": 0.98,
            "tfidf_max_features": 10_000,
            "metrics_enabled": model_name == "logs_plus_metrics",
            "metric_scaler": (
                "StandardScaler(with_mean=False)"
                if model_name == "logs_plus_metrics"
                else None
            ),
        },
        "cross_validation": cross_validation,
        "out_of_fold": oof_metrics,
        "confusion_matrix": {
            actual: {
                predicted: int(matrix[row_index, column_index])
                for column_index, predicted in enumerate(LABELS)
            }
            for row_index, actual in enumerate(LABELS)
        },
        "folds": folds,
        "predictions": sorted(prediction_rows, key=lambda item: item["incident_id"]),
        "interpretation_fit": {
            "scope": "all_records_for_feature_interpretation_only",
            "feature_count": len(feature_names),
            "top_positive_features": _top_features(
                interpretation_model, model_name
            ),
        },
    }


def run_experiment(
    input_path: str | Path,
    *,
    folds: int = 5,
    seed: int = 20260825,
) -> dict[str, Any]:
    source = Path(input_path)
    dataset = load_incident_dataset(source)
    splits = make_splits(dataset, folds=folds, seed=seed)
    models = {
        name: evaluate_model(name, dataset, splits, seed=seed)
        for name in MODEL_NAMES
    }
    logs_f1 = models["logs_only"]["out_of_fold"]["macro_f1"]
    combined_f1 = models["logs_plus_metrics"]["out_of_fold"]["macro_f1"]
    logs_accuracy = models["logs_only"]["out_of_fold"]["accuracy"]
    combined_accuracy = models["logs_plus_metrics"]["out_of_fold"]["accuracy"]
    return {
        "schema_version": 1,
        "experiment_id": "local-ml-baseline-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "path": str(source),
            "sha256": sha256_file(source),
            "records": len(dataset.rows),
            "class_counts": dataset.class_counts,
            "unique_incident_ids": len(set(dataset.incident_ids)),
            "unique_split_groups": len(set(dataset.groups)),
            "metric_feature_union_count": len(dataset.metric_feature_names),
        },
        "leakage_controls": dataset.leakage_audit,
        "evaluation_design": {
            "method": "StratifiedGroupKFold",
            "folds": folds,
            "shuffle": True,
            "random_seed": seed,
            "unit_of_split": "complete incident",
            "shared_splits_across_models": True,
            "all_transformers_fitted_within_training_fold": True,
        },
        "models": models,
        "comparison": {
            "logs_plus_metrics_minus_logs_only_macro_f1": round(
                float(combined_f1 - logs_f1), 6
            ),
            "logs_plus_metrics_minus_logs_only_accuracy": round(
                float(combined_accuracy - logs_accuracy), 6
            ),
        },
        "limitations": [
            "The dataset contains only 60 incidents from one controlled system and one campaign.",
            "Operational logs contain strong known-fault signatures, so high closed-set scores do not establish generalisation.",
            "The rule detector and fault-control messages were excluded to prevent direct answer leakage.",
            "The model is evaluated for incident-level classification, not online detection latency.",
        ],
    }


def _matrix_array(model_result: dict[str, Any]) -> np.ndarray:
    matrix = model_result["confusion_matrix"]
    return np.array(
        [[matrix[actual][predicted] for predicted in LABELS] for actual in LABELS],
        dtype=int,
    )


def render_figure(result: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    matplotlib_config = output.parent / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config.resolve()))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    display_labels = ["Database", "HTTP 500", "Service stopped"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    counts = [result["dataset"]["class_counts"][label] for label in LABELS]
    axes[0, 0].bar(display_labels, counts, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0, 0].set_title("Class balance")
    axes[0, 0].set_ylabel("Incidents")
    axes[0, 0].set_ylim(0, max(counts) * 1.2)
    for index, count in enumerate(counts):
        axes[0, 0].text(index, count + 0.4, str(count), ha="center")

    for fold_index in range(result["evaluation_design"]["folds"]):
        values = [
            result["models"][name]["folds"][fold_index]["metrics"]["macro_f1"]
            for name in MODEL_NAMES
        ]
        axes[0, 1].plot(
            [0, 1], values, marker="o", alpha=0.65, label=f"Fold {fold_index + 1}"
        )
    axes[0, 1].set_xticks([0, 1], ["Logs only", "Logs + metrics"])
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_ylabel("Macro F1")
    axes[0, 1].set_title("Paired fold comparison")
    axes[0, 1].legend(fontsize=8, loc="lower right")

    for axis, model_name, title in (
        (axes[1, 0], "logs_only", "Logs-only confusion matrix"),
        (axes[1, 1], "logs_plus_metrics", "Logs + metrics confusion matrix"),
    ):
        matrix = _matrix_array(result["models"][model_name])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        axis.set_xticks(range(len(LABELS)), display_labels, rotation=20, ha="right")
        axis.set_yticks(range(len(LABELS)), display_labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title(title)
        for row_index in range(len(LABELS)):
            for column_index in range(len(LABELS)):
                axis.text(
                    column_index,
                    row_index,
                    str(int(matrix[row_index, column_index])),
                    ha="center",
                    va="center",
                    color=("white" if matrix[row_index, column_index] > matrix.max() / 2 else "black"),
                )

    fig.suptitle("Local ML baseline v1 - incident-level 5-fold evaluation", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate incident-level TF-IDF logistic-regression baselines."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/incidents/local-campaign-v1.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/local-ml-baseline-v1.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("data/results/figures/local-ml-baseline-v1.png"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = [args.output]
    if not args.no_figure:
        outputs.append(args.figure)
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Experiment outputs already exist; refusing to overwrite: "
            + ", ".join(existing)
        )

    result = run_experiment(args.input, folds=args.folds, seed=args.seed)
    if not args.no_figure:
        render_figure(result, args.figure)
        result["artifacts"] = {
            "figure": str(args.figure),
            "figure_sha256": sha256_file(args.figure),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    console_summary = {
        "experiment_id": result["experiment_id"],
        "dataset": result["dataset"],
        "leakage_controls": result["leakage_controls"],
        "out_of_fold": {
            name: result["models"][name]["out_of_fold"] for name in MODEL_NAMES
        },
        "comparison": result["comparison"],
        "artifacts": result.get("artifacts", {}),
        "result": str(args.output),
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
