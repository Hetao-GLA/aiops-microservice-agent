from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from experiment.rcaeval_public_baseline import (
    PublicDataset,
    classification_metrics,
    load_dataset,
    make_splits,
    sha256_file,
)


REFERENCE = "frozen_metrics_only"
CANDIDATES = (
    "delta_only_lr",
    "logs_plus_delta_lr",
    "robust_service_delta_score",
)
ALL_MODELS = (REFERENCE, *CANDIDATES)
DESIGNS = ("repetition-held-out", "fault-type-held-out")


def _select_log_text(rows: Iterable[dict[str, Any]]) -> list[str]:
    return [row["log_text"] for row in rows]


def _select_delta_metrics(rows: Iterable[dict[str, Any]]) -> list[dict[str, float]]:
    return [
        {
            name: float(value)
            for name, value in row["metric_features"].items()
            if name.endswith("__mean_delta")
        }
        for row in rows
    ]


def _text_pipeline(spec: dict[str, Any]) -> Pipeline:
    settings = spec["text_features"]
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


def _delta_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("select", FunctionTransformer(_select_delta_metrics, validate=False)),
            ("vectorise", DictVectorizer(sparse=True, sort=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )


def build_lr_candidate(name: str, spec: dict[str, Any]) -> Pipeline:
    settings = spec["logistic_regression"]
    classifier = LogisticRegression(
        C=float(settings["C"]),
        solver=str(settings["solver"]),
        max_iter=int(settings["max_iter"]),
        random_state=int(settings["random_seed"]),
    )
    if name == "delta_only_lr":
        features: Pipeline | FeatureUnion = _delta_pipeline()
    elif name == "logs_plus_delta_lr":
        features = FeatureUnion(
            [("logs", _text_pipeline(spec)), ("delta", _delta_pipeline())]
        )
    else:
        raise ValueError(f"Unsupported LR candidate: {name}")
    return Pipeline([("features", features), ("classifier", classifier)])


def _subset(values: Sequence[Any], indexes: Sequence[int]) -> list[Any]:
    return [values[index] for index in indexes]


def _confusion(
    truth: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    matrix = confusion_matrix(truth, predicted, labels=labels)
    return {
        actual: {
            predicted_label: int(matrix[row_index, column_index])
            for column_index, predicted_label in enumerate(labels)
        }
        for row_index, actual in enumerate(labels)
    }


def _rank_probabilities(rankings: Sequence[Sequence[str]], labels: Sequence[str]) -> np.ndarray:
    probabilities = np.zeros((len(rankings), len(labels)), dtype=float)
    for row_index, ranking in enumerate(rankings):
        rank_values = {
            service: float(len(labels) - rank) for rank, service in enumerate(ranking)
        }
        raw = np.asarray([rank_values[label] for label in labels])
        raw = np.exp(raw - raw.max())
        probabilities[row_index] = raw / raw.sum()
    return probabilities


def robust_service_rankings(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    services: Sequence[str],
    *,
    epsilon: float,
    score_cap: float,
) -> tuple[list[list[str]], list[dict[str, float]]]:
    train_delta = _select_delta_metrics(train_rows)
    test_delta = _select_delta_metrics(test_rows)
    features = sorted({name for row in train_delta for name in row})
    if not features:
        raise ValueError("No mean-delta features are available")
    train = np.asarray(
        [[row.get(feature, 0.0) for feature in features] for row in train_delta],
        dtype=float,
    )
    test = np.asarray(
        [[row.get(feature, 0.0) for feature in features] for row in test_delta],
        dtype=float,
    )
    median = np.median(train, axis=0)
    iqr = np.quantile(train, 0.75, axis=0) - np.quantile(train, 0.25, axis=0)
    robust = np.zeros_like(test)
    np.divide(np.abs(test - median), iqr, out=robust, where=iqr > epsilon)
    constant_shift = (iqr <= epsilon) & (np.abs(test - median) > epsilon)
    robust[constant_shift] = score_cap
    robust = np.minimum(robust, score_cap)
    masks = {
        service: np.asarray(
            [feature.startswith(service + "_") for feature in features], dtype=bool
        )
        for service in services
    }
    if any(not mask.any() for mask in masks.values()):
        missing = [service for service, mask in masks.items() if not mask.any()]
        raise ValueError(f"No delta features for candidate services: {missing}")

    rankings: list[list[str]] = []
    score_rows: list[dict[str, float]] = []
    for row_index in range(len(test_rows)):
        scores = {
            service: float(robust[row_index, mask].max())
            for service, mask in masks.items()
        }
        ranking = sorted(services, key=lambda service: (-scores[service], service))
        rankings.append(ranking)
        score_rows.append(scores)
    return rankings, score_rows


def evaluate_lr_candidate(
    name: str,
    dataset: PublicDataset,
    splits: Sequence[tuple[str, list[int], list[int]]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    labels = sorted(dataset.class_counts)
    template = build_lr_candidate(name, spec)
    oof_predictions: list[str | None] = [None] * len(dataset.rows)
    oof_probabilities = np.zeros((len(dataset.rows), len(labels)), dtype=float)
    folds: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    for fold_number, (held_out, train_indexes, test_indexes) in enumerate(splits, 1):
        model = clone(template)
        train_rows = _subset(dataset.rows, train_indexes)
        test_rows = _subset(dataset.rows, test_indexes)
        train_labels = _subset(dataset.labels, train_indexes)
        test_labels = _subset(dataset.labels, test_indexes)
        model.fit(train_rows, train_labels)
        predicted = [str(value) for value in model.predict(test_rows)]
        raw_probabilities = model.predict_proba(test_rows)
        classes = [str(value) for value in model.named_steps["classifier"].classes_]
        aligned = np.column_stack(
            [raw_probabilities[:, classes.index(label)] for label in labels]
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
                "metrics": fold_metrics,
            }
        )
        for local_index, dataset_index in enumerate(test_indexes):
            oof_predictions[dataset_index] = predicted[local_index]
            oof_probabilities[dataset_index] = aligned[local_index]
            predictions.append(
                {
                    "incident_id": dataset.incident_ids[dataset_index],
                    "held_out": held_out,
                    "actual": dataset.labels[dataset_index],
                    "predicted": predicted[local_index],
                    "confidence": round(float(aligned[local_index].max()), 6),
                    "correct": predicted[local_index] == dataset.labels[dataset_index],
                }
            )

    if any(value is None for value in oof_predictions):
        raise RuntimeError("Candidate did not predict every incident")
    final_predictions = [str(value) for value in oof_predictions]
    return {
        "out_of_fold": classification_metrics(
            dataset.labels, final_predictions, oof_probabilities, labels, labels
        ),
        "confusion_matrix": _confusion(dataset.labels, final_predictions, labels),
        "folds": folds,
        "predictions": sorted(predictions, key=lambda row: row["incident_id"]),
    }


def evaluate_robust_scorer(
    dataset: PublicDataset,
    splits: Sequence[tuple[str, list[int], list[int]]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    labels = sorted(dataset.class_counts)
    settings = spec["robust_service_delta_score"]
    oof_predictions: list[str | None] = [None] * len(dataset.rows)
    oof_probabilities = np.zeros((len(dataset.rows), len(labels)), dtype=float)
    folds: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    for fold_number, (held_out, train_indexes, test_indexes) in enumerate(splits, 1):
        train_rows = _subset(dataset.rows, train_indexes)
        test_rows = _subset(dataset.rows, test_indexes)
        test_labels = _subset(dataset.labels, test_indexes)
        train_labels = sorted(set(_subset(dataset.labels, train_indexes)))
        if train_labels != labels:
            raise RuntimeError("Every robust-score training fold must contain all classes")
        rankings, score_rows = robust_service_rankings(
            train_rows,
            test_rows,
            labels,
            epsilon=float(settings["epsilon"]),
            score_cap=float(settings["absolute_score_cap"]),
        )
        predicted = [ranking[0] for ranking in rankings]
        probabilities = _rank_probabilities(rankings, labels)
        fold_metrics = classification_metrics(
            test_labels, predicted, probabilities, labels, labels
        )
        folds.append(
            {
                "fold": fold_number,
                "held_out": held_out,
                "train_records": len(train_indexes),
                "test_records": len(test_indexes),
                "metrics": fold_metrics,
            }
        )
        for local_index, dataset_index in enumerate(test_indexes):
            ranking = rankings[local_index]
            scores = score_rows[local_index]
            oof_predictions[dataset_index] = predicted[local_index]
            oof_probabilities[dataset_index] = probabilities[local_index]
            predictions.append(
                {
                    "incident_id": dataset.incident_ids[dataset_index],
                    "held_out": held_out,
                    "actual": dataset.labels[dataset_index],
                    "predicted": predicted[local_index],
                    "ranking": ranking,
                    "top_score": round(scores[ranking[0]], 6),
                    "score_margin": round(scores[ranking[0]] - scores[ranking[1]], 6),
                    "correct": predicted[local_index] == dataset.labels[dataset_index],
                }
            )

    if any(value is None for value in oof_predictions):
        raise RuntimeError("Robust scorer did not rank every incident")
    final_predictions = [str(value) for value in oof_predictions]
    return {
        "out_of_fold": classification_metrics(
            dataset.labels, final_predictions, oof_probabilities, labels, labels
        ),
        "confusion_matrix": _confusion(dataset.labels, final_predictions, labels),
        "folds": folds,
        "predictions": sorted(predictions, key=lambda row: row["incident_id"]),
    }


def _reference_result(baseline: dict[str, Any], design: str) -> dict[str, Any]:
    reference = baseline["designs"][design]["models"]["metrics_only"]
    return {
        "source": "reused_frozen_public_baseline_without_refit",
        "out_of_fold": reference["out_of_fold"],
        "confusion_matrix": reference["confusion_matrix"],
        "folds": reference["folds"],
        "predictions": reference["predictions"],
    }


def _fold_macro_f1(model: dict[str, Any], held_out: str) -> float:
    for fold in model["folds"]:
        if str(fold["held_out"]) == held_out:
            return float(fold["metrics"]["macro_f1"])
    raise ValueError(f"Held-out fold not found: {held_out}")


def evaluate_promotion(
    designs: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    requirements = spec["promotion_gate"]["requirements"]
    evaluations: dict[str, Any] = {}
    for candidate in CANDIDATES:
        observed = {
            "repetition_held_out_macro_f1": float(
                designs["repetition-held-out"]["models"][candidate]["out_of_fold"]["macro_f1"]
            ),
            "repetition_1_macro_f1": _fold_macro_f1(
                designs["repetition-held-out"]["models"][candidate], "1"
            ),
            "fault_type_held_out_macro_f1": float(
                designs["fault-type-held-out"]["models"][candidate]["out_of_fold"]["macro_f1"]
            ),
            "delay_fold_macro_f1": _fold_macro_f1(
                designs["fault-type-held-out"]["models"][candidate], "delay"
            ),
        }
        checks = {
            name: observed[name] >= float(minimum)
            for name, minimum in requirements.items()
        }
        evaluations[candidate] = {
            "observed": observed,
            "minimum": requirements,
            "checks": checks,
            "passes_all": all(checks.values()),
        }
    selected = next(
        (
            candidate
            for candidate in spec["promotion_gate"]["selection_order"]
            if evaluations[candidate]["passes_all"]
        ),
        None,
    )
    return {
        "selection_order": spec["promotion_gate"]["selection_order"],
        "evaluations": evaluations,
        "selected_for_external_confirmation": selected,
        "development_only": True,
    }


def run_experiment(spec_path: str | Path) -> dict[str, Any]:
    spec_file = Path(spec_path)
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError("Unsupported delta-candidate specification")
    for path_key, hash_key in (
        ("incidents", "incidents_sha256"),
        ("public_baseline_result", "public_baseline_result_sha256"),
        ("public_baseline_spec", "public_baseline_spec_sha256"),
        ("shift_audit_result", "shift_audit_result_sha256"),
        ("shift_audit_spec", "shift_audit_spec_sha256"),
    ):
        actual = sha256_file(spec["inputs"][path_key])
        expected = spec["inputs"][hash_key]
        if actual != expected:
            raise ValueError(
                f"Frozen input hash mismatch for {path_key}: expected={expected}, actual={actual}"
            )

    baseline_spec = json.loads(
        Path(spec["inputs"]["public_baseline_spec"]).read_text(encoding="utf-8")
    )
    dataset = load_dataset(spec["inputs"]["incidents"], baseline_spec)
    baseline = json.loads(
        Path(spec["inputs"]["public_baseline_result"]).read_text(encoding="utf-8")
    )
    delta_features = sorted(
        {
            name
            for row in dataset.rows
            for name in row["metric_features"]
            if name.endswith(spec["delta_features"]["suffix"])
        }
    )
    designs: dict[str, Any] = {}
    for design in DESIGNS:
        splits = make_splits(dataset, design)
        models = {
            REFERENCE: _reference_result(baseline, design),
            "delta_only_lr": evaluate_lr_candidate(
                "delta_only_lr", dataset, splits, spec
            ),
            "logs_plus_delta_lr": evaluate_lr_candidate(
                "logs_plus_delta_lr", dataset, splits, spec
            ),
            "robust_service_delta_score": evaluate_robust_scorer(
                dataset, splits, spec
            ),
        }
        reference_f1 = models[REFERENCE]["out_of_fold"]["macro_f1"]
        designs[design] = {
            "folds": len(splits),
            "models": models,
            "macro_f1_minus_reference": {
                candidate: round(
                    float(models[candidate]["out_of_fold"]["macro_f1"] - reference_f1),
                    6,
                )
                for candidate in CANDIDATES
            },
        }

    promotion = evaluate_promotion(designs, spec)
    return {
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "status": spec["status"],
        "spec": {"path": str(spec_file), "sha256": sha256_file(spec_file)},
        "inputs": spec["inputs"],
        "dataset": {
            "records": len(dataset.rows),
            "classes": dataset.class_counts,
            "faults": dataset.fault_counts,
            "delta_feature_count": len(delta_features),
            "excluded_absolute_metric_features": True,
        },
        "rules": spec["rules"],
        "designs": designs,
        "promotion_gate": promotion,
        "limitations": [
            "The candidates were designed after inspecting RE2-OB baseline errors and are development results.",
            "Promotion requires confirmation on a different public system or suite.",
            "The robust scorer uses service-qualified metric names and is not directly comparable with official metric-level RCAEval rankings.",
            "No traces are used.",
        ],
    }


def render_figure(result: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    config = output.parent / ".matplotlib"
    config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config.resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    display = ["Frozen metrics", "Delta LR", "Logs + delta", "Robust service delta"]
    colors = ["#9D9D9D", "#4C78A8", "#F58518", "#54A24B"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = (
        ("repetition-held-out", None, "Repetition-held-out Macro F1"),
        ("repetition-held-out", "1", "Held-out repetition 1 Macro F1"),
        ("fault-type-held-out", None, "Fault-type-held-out Macro F1"),
        ("fault-type-held-out", "delay", "Held-out delay Macro F1"),
    )
    for axis, (design, held_out, title) in zip(axes.flat, panels):
        values = []
        for model_name in ALL_MODELS:
            model = result["designs"][design]["models"][model_name]
            values.append(
                model["out_of_fold"]["macro_f1"]
                if held_out is None
                else _fold_macro_f1(model, held_out)
            )
        axis.bar(range(len(values)), values, color=colors)
        axis.set_xticks(range(len(values)), display, rotation=18, ha="right")
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
        for index, value in enumerate(values):
            axis.text(index, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
    selected = result["promotion_gate"]["selected_for_external_confirmation"]
    fig.suptitle(
        "RCAEval RE2-OB delta candidates"
        + (f" — selected: {selected}" if selected else " — no candidate passed"),
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate locked RE2-OB delta-only development candidates."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("experiment/configs/rcaeval-re2-ob-delta-candidates-spec.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/rcaeval-re2-ob-delta-candidates-v1.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("data/results/figures/rcaeval-re2-ob-delta-candidates-v1.png"),
    )
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    outputs = [args.output] + ([] if args.no_figure else [args.figure])
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Candidate outputs already exist; refusing to overwrite: " + ", ".join(existing)
        )
    result = run_experiment(args.spec)
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
        "scores": {
            design: {
                model: result["designs"][design]["models"][model]["out_of_fold"]
                for model in ALL_MODELS
            }
            for design in DESIGNS
        },
        "promotion_gate": result["promotion_gate"],
        "result": str(args.output),
        "artifacts": result.get("artifacts", {}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
