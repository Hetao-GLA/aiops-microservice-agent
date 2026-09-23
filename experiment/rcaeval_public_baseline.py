from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


MODEL_NAMES = ("logs_only", "metrics_only", "logs_plus_metrics")
DESIGN_NAMES = ("repetition-held-out", "fault-type-held-out")


@dataclass(frozen=True)
class PublicDataset:
    rows: list[dict[str, Any]]
    labels: list[str]
    faults: list[str]
    repetitions: list[int]
    incident_ids: list[str]
    metric_feature_names: list[str]
    class_counts: dict[str, int]
    fault_counts: dict[str, int]
    log_record_count: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_spec(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported public-baseline spec version")
    return payload


def _numeric_metrics(raw: Any, *, line_number: int) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Line {line_number}: metric_features must be non-empty")
    converted: dict[str, float] = {}
    for name, value in raw.items():
        if value is None:
            converted[str(name)] = 0.0
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Line {line_number}: metric {name!r} is not numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Line {line_number}: metric {name!r} is not finite")
        converted[str(name)] = numeric
    return converted


def load_dataset(path: str | Path, spec: dict[str, Any]) -> PublicDataset:
    expected_services = sorted(spec["selection"]["expected_root_cause_services"])
    expected_faults = sorted(spec["selection"]["expected_faults"])
    expected_cases = int(spec["selection"]["expected_cases"])
    expected_repetitions = int(
        spec["selection"]["expected_repetitions_per_service_fault_pair"]
    )
    rows: list[dict[str, Any]] = []
    labels: list[str] = []
    faults: list[str] = []
    repetitions: list[int] = []
    incident_ids: list[str] = []
    metric_names: set[str] = set()
    log_record_count = 0

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if record.get("source") != "rcaeval":
                raise ValueError(f"Line {line_number}: source must be rcaeval")
            if record.get("dataset") != spec["selection"]["dataset"]:
                raise ValueError(f"Line {line_number}: unexpected public dataset")
            if record.get("task") != spec["task"]["name"]:
                raise ValueError(f"Line {line_number}: unexpected task")
            if record.get("local_fault_label") is not None:
                raise ValueError(f"Line {line_number}: local label must remain unset")

            incident_id = str(record.get("incident_id", ""))
            label = str(record.get("root_cause_service", ""))
            fault = str(record.get("original_fault_label", ""))
            provenance = record.get("provenance")
            if not incident_id:
                raise ValueError(f"Line {line_number}: incident_id is required")
            if label not in expected_services:
                raise ValueError(f"Line {line_number}: unexpected service label {label!r}")
            if fault not in expected_faults:
                raise ValueError(f"Line {line_number}: unexpected fault {fault!r}")
            if not isinstance(provenance, dict) or "repetition" not in provenance:
                raise ValueError(f"Line {line_number}: repetition provenance is required")
            repetition = int(provenance["repetition"])
            if repetition not in range(1, expected_repetitions + 1):
                raise ValueError(f"Line {line_number}: invalid repetition")
            log_text = record.get("log_text")
            if not isinstance(log_text, str) or not log_text.strip():
                raise ValueError(f"Line {line_number}: non-empty log_text is required")
            metrics = _numeric_metrics(
                record.get("metric_features"), line_number=line_number
            )
            metric_names.update(metrics)
            rows.append({"log_text": log_text, "metric_features": metrics})
            labels.append(label)
            faults.append(fault)
            repetitions.append(repetition)
            incident_ids.append(incident_id)
            log_record_count += int(record.get("log_record_count", 0))

    if len(rows) != expected_cases:
        raise ValueError(f"Expected {expected_cases} records, found {len(rows)}")
    if len(set(incident_ids)) != len(incident_ids):
        raise ValueError("incident_id values must be unique")
    if sorted(set(labels)) != expected_services:
        raise ValueError("The locked root-cause service set is incomplete")
    if sorted(set(faults)) != expected_faults:
        raise ValueError("The locked fault set is incomplete")
    combinations = Counter(zip(labels, faults, repetitions))
    expected_combinations = {
        (service, fault, repetition)
        for service in expected_services
        for fault in expected_faults
        for repetition in range(1, expected_repetitions + 1)
    }
    if set(combinations) != expected_combinations or set(combinations.values()) != {1}:
        raise ValueError("Each service/fault/repetition combination must occur once")

    return PublicDataset(
        rows=rows,
        labels=labels,
        faults=faults,
        repetitions=repetitions,
        incident_ids=incident_ids,
        metric_feature_names=sorted(metric_names),
        class_counts=dict(sorted(Counter(labels).items())),
        fault_counts=dict(sorted(Counter(faults).items())),
        log_record_count=log_record_count,
    )


def make_splits(
    dataset: PublicDataset, design_name: str
) -> list[tuple[str, list[int], list[int]]]:
    if design_name == "repetition-held-out":
        held_out_values: Sequence[str | int] = sorted(set(dataset.repetitions))
        values: Sequence[str | int] = dataset.repetitions
    elif design_name == "fault-type-held-out":
        held_out_values = sorted(set(dataset.faults))
        values = dataset.faults
    else:
        raise ValueError(f"Unsupported evaluation design: {design_name}")

    splits: list[tuple[str, list[int], list[int]]] = []
    all_indexes = set(range(len(dataset.rows)))
    for held_out in held_out_values:
        test = [index for index, value in enumerate(values) if value == held_out]
        train = sorted(all_indexes.difference(test))
        if not test or not train or set(test).intersection(train):
            raise RuntimeError("Invalid public benchmark split")
        if set(dataset.labels[index] for index in test) != set(dataset.labels):
            raise RuntimeError("Every test fold must contain every service class")
        if set(dataset.labels[index] for index in train) != set(dataset.labels):
            raise RuntimeError("Every train fold must contain every service class")
        splits.append((str(held_out), train, test))
    test_coverage = Counter(index for _, _, test in splits for index in test)
    if set(test_coverage) != all_indexes or set(test_coverage.values()) != {1}:
        raise RuntimeError("Every incident must be tested exactly once")
    return splits


def _select_log_text(rows: Iterable[dict[str, Any]]) -> list[str]:
    return [row["log_text"] for row in rows]


def _select_metrics(rows: Iterable[dict[str, Any]]) -> list[dict[str, float]]:
    return [row["metric_features"] for row in rows]


def _text_pipeline(spec: dict[str, Any]) -> Pipeline:
    settings = spec["evaluation"]["text_features"]
    return Pipeline(
        [
            ("select", FunctionTransformer(_select_log_text, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=tuple(settings["ngram_range"]),
                    min_df=int(settings["min_df"]),
                    max_df=float(settings["max_df"]),
                    max_features=int(settings["max_features"]),
                    sublinear_tf=bool(settings["sublinear_tf"]),
                    token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b",
                ),
            ),
        ]
    )


def _metric_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("select", FunctionTransformer(_select_metrics, validate=False)),
            ("vectorise", DictVectorizer(sparse=True, sort=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )


def build_model(model_name: str, spec: dict[str, Any]) -> Pipeline:
    settings = spec["evaluation"]["classifier"]
    classifier = LogisticRegression(
        C=float(settings["C"]),
        solver=str(settings["solver"]),
        max_iter=int(settings["max_iter"]),
        random_state=int(settings["random_seed"]),
    )
    if model_name == "logs_only":
        features: Pipeline | FeatureUnion = _text_pipeline(spec)
    elif model_name == "metrics_only":
        features = _metric_pipeline()
    elif model_name == "logs_plus_metrics":
        features = FeatureUnion(
            [("logs", _text_pipeline(spec)), ("metrics", _metric_pipeline())]
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("features", features), ("classifier", classifier)])


def _subset(values: Sequence[Any], indexes: Sequence[int]) -> list[Any]:
    return [values[index] for index in indexes]


def _top_k_accuracy(
    truth: Sequence[str], probabilities: np.ndarray, classes: Sequence[str], k: int
) -> float:
    class_array = np.asarray(classes)
    correct = 0
    for index, actual in enumerate(truth):
        top = np.argsort(probabilities[index])[::-1][:k]
        correct += actual in class_array[top]
    return float(correct / len(truth))


def classification_metrics(
    truth: Sequence[str],
    predicted: Sequence[str],
    probabilities: np.ndarray,
    classes: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    macro_f1 = precision_recall_fscore_support(
        truth, predicted, average="macro", zero_division=0
    )[2]
    return {
        "top_1_accuracy": round(float(accuracy_score(truth, predicted)), 6),
        "top_3_accuracy": round(
            _top_k_accuracy(truth, probabilities, classes, k=3), 6
        ),
        "macro_f1": round(float(macro_f1), 6),
        "per_class": {
            label: {
                "support": int(support[index]),
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
            }
            for index, label in enumerate(labels)
        },
    }


def evaluate_model(
    model_name: str,
    dataset: PublicDataset,
    splits: Sequence[tuple[str, list[int], list[int]]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    labels = sorted(dataset.class_counts)
    template = build_model(model_name, spec)
    oof_predictions: list[str | None] = [None] * len(dataset.rows)
    oof_probabilities = np.zeros((len(dataset.rows), len(labels)), dtype=float)
    prediction_rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []

    for fold_number, (held_out, train_indexes, test_indexes) in enumerate(
        splits, start=1
    ):
        model = clone(template)
        train_rows = _subset(dataset.rows, train_indexes)
        test_rows = _subset(dataset.rows, test_indexes)
        train_labels = _subset(dataset.labels, train_indexes)
        test_labels = _subset(dataset.labels, test_indexes)
        model.fit(train_rows, train_labels)
        predicted = [str(value) for value in model.predict(test_rows)]
        probabilities = model.predict_proba(test_rows)
        classes = [str(value) for value in model.named_steps["classifier"].classes_]
        aligned = np.column_stack(
            [probabilities[:, classes.index(label)] for label in labels]
        )
        fold_metrics = classification_metrics(
            test_labels, predicted, aligned, labels, labels
        )
        folds.append(
            {
                "fold": fold_number,
                "held_out": held_out,
                "train_records": len(train_indexes),
                "test_records": len(test_indexes),
                "metrics": {
                    key: value for key, value in fold_metrics.items() if key != "per_class"
                },
            }
        )
        for local_index, dataset_index in enumerate(test_indexes):
            predicted_label = predicted[local_index]
            oof_predictions[dataset_index] = predicted_label
            oof_probabilities[dataset_index] = aligned[local_index]
            prediction_rows.append(
                {
                    "incident_id": dataset.incident_ids[dataset_index],
                    "held_out": held_out,
                    "actual": dataset.labels[dataset_index],
                    "predicted": predicted_label,
                    "confidence": round(float(max(aligned[local_index])), 6),
                    "correct": predicted_label == dataset.labels[dataset_index],
                }
            )

    if any(value is None for value in oof_predictions):
        raise RuntimeError("Not every public incident received an OOF prediction")
    final_predictions = [str(value) for value in oof_predictions]
    metrics = classification_metrics(
        dataset.labels, final_predictions, oof_probabilities, labels, labels
    )
    matrix = confusion_matrix(dataset.labels, final_predictions, labels=labels)
    return {
        "out_of_fold": metrics,
        "confusion_matrix": {
            actual: {
                predicted: int(matrix[row_index, column_index])
                for column_index, predicted in enumerate(labels)
            }
            for row_index, actual in enumerate(labels)
        },
        "folds": folds,
        "predictions": sorted(prediction_rows, key=lambda row: row["incident_id"]),
    }


def run_experiment(
    input_path: str | Path,
    spec_path: str | Path,
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    dataset = load_dataset(input_path, spec)
    designs: dict[str, Any] = {}
    for design_name in DESIGN_NAMES:
        splits = make_splits(dataset, design_name)
        models = {
            model_name: evaluate_model(model_name, dataset, splits, spec)
            for model_name in MODEL_NAMES
        }
        designs[design_name] = {
            "folds": len(splits),
            "shared_splits_across_models": True,
            "all_transformers_fitted_within_training_fold": True,
            "models": models,
            "comparison": {
                "logs_plus_metrics_minus_logs_only_macro_f1": round(
                    models["logs_plus_metrics"]["out_of_fold"]["macro_f1"]
                    - models["logs_only"]["out_of_fold"]["macro_f1"],
                    6,
                ),
                "logs_plus_metrics_minus_metrics_only_macro_f1": round(
                    models["logs_plus_metrics"]["out_of_fold"]["macro_f1"]
                    - models["metrics_only"]["out_of_fold"]["macro_f1"],
                    6,
                ),
            },
        }

    return {
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "spec": str(spec_path),
            "spec_sha256": sha256_file(spec_path),
            "hyperparameter_tuning": False,
        },
        "dataset": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "records": len(dataset.rows),
            "class_counts": dataset.class_counts,
            "fault_counts": dataset.fault_counts,
            "repetitions": sorted(set(dataset.repetitions)),
            "metric_feature_union_count": len(dataset.metric_feature_names),
            "log_record_count": dataset.log_record_count,
            "public_labels_kept_separate": True,
        },
        "model_configuration": spec["evaluation"],
        "designs": designs,
        "limitations": spec["interpretation_limits"],
    }


def _matrix_array(result: dict[str, Any], design: str) -> np.ndarray:
    labels = sorted(result["dataset"]["class_counts"])
    matrix = result["designs"][design]["models"]["logs_plus_metrics"][
        "confusion_matrix"
    ]
    return np.array(
        [[matrix[actual][predicted] for predicted in labels] for actual in labels],
        dtype=int,
    )


def render_figure(result: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    config = output.parent / ".matplotlib"
    config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config.resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted(result["dataset"]["class_counts"])
    display = [label.replace("service", "") for label in labels]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    positions = np.arange(len(MODEL_NAMES))

    for row, design in enumerate(DESIGN_NAMES):
        model_results = result["designs"][design]["models"]
        accuracy = [
            model_results[name]["out_of_fold"]["top_1_accuracy"]
            for name in MODEL_NAMES
        ]
        macro_f1 = [
            model_results[name]["out_of_fold"]["macro_f1"] for name in MODEL_NAMES
        ]
        axes[row, 0].bar(positions - 0.18, accuracy, width=0.36, label="Top-1 accuracy", color=colors[0])
        axes[row, 0].bar(positions + 0.18, macro_f1, width=0.36, label="Macro F1", color=colors[1])
        axes[row, 0].set_xticks(positions, ["Logs", "Metrics", "Logs + metrics"])
        axes[row, 0].set_ylim(0, 1.05)
        axes[row, 0].set_title(design)
        axes[row, 0].legend(loc="lower right")

        matrix = _matrix_array(result, design)
        axes[row, 1].imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        axes[row, 1].set_xticks(range(len(labels)), display, rotation=25, ha="right")
        axes[row, 1].set_yticks(range(len(labels)), display)
        axes[row, 1].set_xlabel("Predicted")
        axes[row, 1].set_ylabel("Actual")
        axes[row, 1].set_title(f"Logs + metrics: {design}")
        for i in range(len(labels)):
            for j in range(len(labels)):
                axes[row, 1].text(j, i, str(int(matrix[i, j])), ha="center", va="center")

    fig.suptitle("RCAEval RE2-OB public root-cause service baseline", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the locked RCAEval RE2-OB public service baseline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/incidents/rcaeval-re2-ob.jsonl"),
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("experiment/configs/rcaeval-re2-ob-public-baseline-spec.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/rcaeval-re2-ob-public-baseline-v1.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("data/results/figures/rcaeval-re2-ob-public-baseline-v1.png"),
    )
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    outputs = [args.output] + ([] if args.no_figure else [args.figure])
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Experiment outputs already exist; refusing to overwrite: "
            + ", ".join(existing)
        )
    result = run_experiment(args.input, args.spec)
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
    summary = {
        "experiment_id": result["experiment_id"],
        "dataset": result["dataset"],
        "scores": {
            design: {
                name: result["designs"][design]["models"][name]["out_of_fold"]
                for name in MODEL_NAMES
            }
            for design in DESIGN_NAMES
        },
        "result": str(args.output),
        "artifacts": result.get("artifacts", {}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
