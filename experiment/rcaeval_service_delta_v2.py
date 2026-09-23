from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from experiment.rcaeval_public_baseline import (
    PublicDataset,
    build_model,
    classification_metrics,
    load_dataset,
    make_splits,
    sha256_file,
)


REFERENCE = "frozen_full_metrics_lr"
TAIL = "empirical_tail_top2_service_score"
SOFT_VOTE = "full_delta_soft_vote"
CANDIDATES = (TAIL, SOFT_VOTE)
ALL_MODELS = (REFERENCE, *CANDIDATES)
DESIGNS = ("repetition-held-out", "fault-type-held-out")


def _subset(values: Sequence[Any], indexes: Sequence[int]) -> list[Any]:
    return [values[index] for index in indexes]


def _select_delta_metrics(rows: Iterable[dict[str, Any]]) -> list[dict[str, float]]:
    return [
        {
            name: float(value)
            for name, value in row["metric_features"].items()
            if name.endswith("__mean_delta")
        }
        for row in rows
    ]


def _delta_model(spec: dict[str, Any]) -> Pipeline:
    settings = spec["full_delta_soft_vote"]["classifier"]
    return Pipeline(
        [
            (
                "features",
                Pipeline(
                    [
                        (
                            "select",
                            FunctionTransformer(_select_delta_metrics, validate=False),
                        ),
                        ("vectorise", DictVectorizer(sparse=True, sort=True)),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=float(settings["C"]),
                    solver=str(settings["solver"]),
                    max_iter=int(settings["max_iter"]),
                    random_state=int(settings["random_seed"]),
                ),
            ),
        ]
    )


def _rank_probabilities(rankings: Sequence[Sequence[str]], labels: Sequence[str]) -> np.ndarray:
    result = np.zeros((len(rankings), len(labels)), dtype=float)
    for row_index, ranking in enumerate(rankings):
        values = {service: len(labels) - rank for rank, service in enumerate(ranking)}
        raw = np.asarray([float(values[label]) for label in labels])
        raw = np.exp(raw - raw.max())
        result[row_index] = raw / raw.sum()
    return result


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


def empirical_tail_rankings(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    services: Sequence[str],
    *,
    epsilon: float,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    train_delta = _select_delta_metrics(train_rows)
    test_delta = _select_delta_metrics(test_rows)
    features = sorted({name for row in train_delta for name in row})
    train = np.asarray(
        [[row.get(feature, 0.0) for feature in features] for row in train_delta],
        dtype=float,
    )
    test = np.asarray(
        [[row.get(feature, 0.0) for feature in features] for row in test_delta],
        dtype=float,
    )
    median = np.median(train, axis=0)
    train_deviation = np.abs(train - median)
    test_deviation = np.abs(test - median)
    surprise = np.empty_like(test)
    for row_index in range(len(test_rows)):
        counts = np.sum(train_deviation >= test_deviation[row_index], axis=0)
        surprise[row_index] = -np.log10((1.0 + counts) / (len(train_rows) + 1.0))
    minimum = train.min(axis=0)
    maximum = train.max(axis=0)
    outside = (test < minimum) | (test > maximum)
    std = train.std(axis=0)
    continuous = np.log1p(test_deviation / np.maximum(std, epsilon))
    masks = {
        service: np.asarray(
            [feature.startswith(service + "_") for feature in features], dtype=bool
        )
        for service in services
    }
    missing = [service for service, mask in masks.items() if mask.sum() < 2]
    if missing:
        raise ValueError(f"Fewer than two delta features for services: {missing}")

    rankings: list[list[str]] = []
    details: list[dict[str, Any]] = []
    for row_index in range(len(test_rows)):
        keys: dict[str, tuple[float, float, float, float]] = {}
        for service, mask in masks.items():
            values = surprise[row_index, mask]
            top_two = np.sort(values)[-2:]
            keys[service] = (
                float(top_two.mean()),
                float(values.mean()),
                float(outside[row_index, mask].mean()),
                float(continuous[row_index, mask].mean()),
            )
        ranking = sorted(
            services,
            key=lambda service: tuple(-value for value in keys[service]) + (service,),
        )
        exact_tie = keys[ranking[0]] == keys[ranking[1]]
        rankings.append(ranking)
        details.append(
            {
                "numeric_keys": {service: list(keys[service]) for service in services},
                "primary_margin": round(keys[ranking[0]][0] - keys[ranking[1]][0], 9),
                "exact_numeric_tie": exact_tie,
            }
        )
    return rankings, details


def _finish_evaluation(
    dataset: PublicDataset,
    labels: Sequence[str],
    oof_predictions: Sequence[str | None],
    oof_probabilities: np.ndarray,
    folds: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    if any(value is None for value in oof_predictions):
        raise RuntimeError("Not every incident received a v2 prediction")
    final_predictions = [str(value) for value in oof_predictions]
    predicted_counts = Counter(final_predictions)
    exact_ties = sum(bool(row.get("exact_numeric_tie", False)) for row in predictions)
    return {
        "out_of_fold": classification_metrics(
            dataset.labels, final_predictions, oof_probabilities, labels, labels
        ),
        "confusion_matrix": _confusion(dataset.labels, final_predictions, labels),
        "folds": folds,
        "predictions": sorted(predictions, key=lambda row: row["incident_id"]),
        "balance_and_ties": {
            "predicted_class_counts": dict(sorted(predicted_counts.items())),
            "maximum_predicted_class_fraction": round(
                max(predicted_counts.values()) / len(dataset.rows), 6
            ),
            "exact_numeric_ties": exact_ties,
            "exact_numeric_tie_fraction": round(exact_ties / len(dataset.rows), 6),
        },
    }


def evaluate_tail_candidate(
    dataset: PublicDataset,
    splits: Sequence[tuple[str, list[int], list[int]]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    labels = sorted(dataset.class_counts)
    epsilon = float(spec[TAIL]["epsilon"])
    oof_predictions: list[str | None] = [None] * len(dataset.rows)
    oof_probabilities = np.zeros((len(dataset.rows), len(labels)), dtype=float)
    folds: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for fold_number, (held_out, train_indexes, test_indexes) in enumerate(splits, 1):
        train_rows = _subset(dataset.rows, train_indexes)
        test_rows = _subset(dataset.rows, test_indexes)
        test_labels = _subset(dataset.labels, test_indexes)
        rankings, details = empirical_tail_rankings(
            train_rows, test_rows, labels, epsilon=epsilon
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
            oof_predictions[dataset_index] = predicted[local_index]
            oof_probabilities[dataset_index] = probabilities[local_index]
            predictions.append(
                {
                    "incident_id": dataset.incident_ids[dataset_index],
                    "held_out": held_out,
                    "actual": dataset.labels[dataset_index],
                    "predicted": predicted[local_index],
                    "ranking": rankings[local_index],
                    "primary_margin": details[local_index]["primary_margin"],
                    "exact_numeric_tie": details[local_index]["exact_numeric_tie"],
                    "correct": predicted[local_index] == dataset.labels[dataset_index],
                }
            )
    return _finish_evaluation(
        dataset,
        labels,
        oof_predictions,
        oof_probabilities,
        folds,
        predictions,
    )


def evaluate_soft_vote(
    dataset: PublicDataset,
    splits: Sequence[tuple[str, list[int], list[int]]],
    dataset_spec: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    labels = sorted(dataset.class_counts)
    weights = spec[SOFT_VOTE]["probability_weights"]
    full_weight = float(weights["full_metric_branch"])
    delta_weight = float(weights["delta_branch"])
    if not math.isclose(full_weight + delta_weight, 1.0):
        raise ValueError("Soft-vote weights must sum to one")
    full_template = build_model("metrics_only", dataset_spec)
    delta_template = _delta_model(spec)
    oof_predictions: list[str | None] = [None] * len(dataset.rows)
    oof_probabilities = np.zeros((len(dataset.rows), len(labels)), dtype=float)
    folds: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for fold_number, (held_out, train_indexes, test_indexes) in enumerate(splits, 1):
        train_rows = _subset(dataset.rows, train_indexes)
        test_rows = _subset(dataset.rows, test_indexes)
        train_labels = _subset(dataset.labels, train_indexes)
        test_labels = _subset(dataset.labels, test_indexes)
        full = clone(full_template).fit(train_rows, train_labels)
        delta = clone(delta_template).fit(train_rows, train_labels)
        full_classes = [str(value) for value in full.named_steps["classifier"].classes_]
        delta_classes = [str(value) for value in delta.named_steps["classifier"].classes_]
        full_raw = full.predict_proba(test_rows)
        delta_raw = delta.predict_proba(test_rows)
        full_aligned = np.column_stack(
            [full_raw[:, full_classes.index(label)] for label in labels]
        )
        delta_aligned = np.column_stack(
            [delta_raw[:, delta_classes.index(label)] for label in labels]
        )
        probabilities = full_weight * full_aligned + delta_weight * delta_aligned
        predicted = [labels[int(index)] for index in np.argmax(probabilities, axis=1)]
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
            order = np.argsort(probabilities[local_index])[::-1]
            exact_tie = bool(
                probabilities[local_index, order[0]]
                == probabilities[local_index, order[1]]
            )
            oof_predictions[dataset_index] = predicted[local_index]
            oof_probabilities[dataset_index] = probabilities[local_index]
            predictions.append(
                {
                    "incident_id": dataset.incident_ids[dataset_index],
                    "held_out": held_out,
                    "actual": dataset.labels[dataset_index],
                    "predicted": predicted[local_index],
                    "confidence": round(float(probabilities[local_index, order[0]]), 6),
                    "probability_margin": round(
                        float(
                            probabilities[local_index, order[0]]
                            - probabilities[local_index, order[1]]
                        ),
                        9,
                    ),
                    "exact_numeric_tie": exact_tie,
                    "correct": predicted[local_index] == dataset.labels[dataset_index],
                }
            )
    return _finish_evaluation(
        dataset,
        labels,
        oof_predictions,
        oof_probabilities,
        folds,
        predictions,
    )


def _reference_result(raw: dict[str, Any], design: str, model_key: str) -> dict[str, Any]:
    model = raw["designs"][design]["models"][model_key]
    return {
        "source": "reused_frozen_reference_predictions",
        "out_of_fold": model["out_of_fold"],
        "confusion_matrix": model["confusion_matrix"],
        "folds": model["folds"],
        "predictions": model["predictions"],
    }


def _fold_macro(model: dict[str, Any], held_out: str) -> float:
    return next(
        float(fold["metrics"]["macro_f1"])
        for fold in model["folds"]
        if str(fold["held_out"]) == held_out
    )


def evaluate_gate(systems: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    evaluations: dict[str, Any] = {}
    for candidate in CANDIDATES:
        differences = []
        candidate_scores = []
        repetition_folds = []
        delay_differences = []
        top3_scores = []
        class_fractions = []
        tie_fractions = []
        for system in systems.values():
            for design in DESIGNS:
                models = system["designs"][design]["models"]
                candidate_model = models[candidate]
                reference = models[REFERENCE]
                candidate_f1 = float(candidate_model["out_of_fold"]["macro_f1"])
                reference_f1 = float(reference["out_of_fold"]["macro_f1"])
                differences.append(candidate_f1 - reference_f1)
                candidate_scores.append(candidate_f1)
                top3_scores.append(
                    float(candidate_model["out_of_fold"]["top_3_accuracy"])
                )
                class_fractions.append(
                    float(
                        candidate_model["balance_and_ties"][
                            "maximum_predicted_class_fraction"
                        ]
                    )
                )
                tie_fractions.append(
                    float(
                        candidate_model["balance_and_ties"][
                            "exact_numeric_tie_fraction"
                        ]
                    )
                )
            repetition_model = system["designs"]["repetition-held-out"]["models"][
                candidate
            ]
            repetition_folds.extend(
                float(fold["metrics"]["macro_f1"])
                for fold in repetition_model["folds"]
            )
            fault_models = system["designs"]["fault-type-held-out"]["models"]
            delay_differences.append(
                _fold_macro(fault_models[candidate], "delay")
                - _fold_macro(fault_models[REFERENCE], "delay")
            )
        observed = {
            "mean_macro_f1_minus_reference": round(float(np.mean(differences)), 6),
            "worst_system_design_macro_f1_minus_reference": round(
                float(min(differences)), 6
            ),
            "minimum_system_design_macro_f1": round(float(min(candidate_scores)), 6),
            "minimum_repetition_fold_macro_f1": round(float(min(repetition_folds)), 6),
            "mean_delay_macro_f1_minus_reference": round(
                float(np.mean(delay_differences)), 6
            ),
            "minimum_top3_accuracy": round(float(min(top3_scores)), 6),
            "maximum_predicted_class_fraction": round(float(max(class_fractions)), 6),
            "maximum_exact_numeric_tie_fraction": round(float(max(tie_fractions)), 6),
        }
        minimums = spec["promotion_gate"]["minimums"]
        maximums = spec["promotion_gate"]["maximums"]
        checks = {
            **{
                name: observed[name] >= float(limit)
                for name, limit in minimums.items()
            },
            **{
                name: observed[name] <= float(limit)
                for name, limit in maximums.items()
            },
        }
        evaluations[candidate] = {
            "observed": observed,
            "minimums": minimums,
            "maximums": maximums,
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
        "selected_for_third_system_confirmation": selected,
        "development_only": True,
    }


def run_experiment(spec_path: str | Path) -> dict[str, Any]:
    spec_file = Path(spec_path)
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError("Unsupported service-delta v2 specification")
    systems: dict[str, Any] = {}
    verified_hashes: dict[str, str] = {}
    for system_name, settings in spec["systems"].items():
        for path_key, hash_key in (
            ("incidents", "incidents_sha256"),
            ("reference_result", "reference_result_sha256"),
        ):
            actual = sha256_file(settings[path_key])
            if actual != settings[hash_key]:
                raise ValueError(f"Frozen v2 input hash mismatch: {settings[path_key]}")
            verified_hashes[settings[path_key]] = actual
        dataset_spec = json.loads(
            Path(settings["dataset_spec"]).read_text(encoding="utf-8")
        )
        dataset = load_dataset(settings["incidents"], dataset_spec)
        reference_raw = json.loads(
            Path(settings["reference_result"]).read_text(encoding="utf-8")
        )
        designs: dict[str, Any] = {}
        for design in DESIGNS:
            splits = make_splits(dataset, design)
            models = {
                REFERENCE: _reference_result(
                    reference_raw, design, settings["reference_model_key"]
                ),
                TAIL: evaluate_tail_candidate(dataset, splits, spec),
                SOFT_VOTE: evaluate_soft_vote(
                    dataset, splits, dataset_spec, spec
                ),
            }
            reference_f1 = float(models[REFERENCE]["out_of_fold"]["macro_f1"])
            designs[design] = {
                "folds": len(splits),
                "models": models,
                "macro_f1_minus_reference": {
                    candidate: round(
                        float(
                            models[candidate]["out_of_fold"]["macro_f1"]
                            - reference_f1
                        ),
                        6,
                    )
                    for candidate in CANDIDATES
                },
            }
        systems[system_name] = {
            "dataset": {
                "records": len(dataset.rows),
                "classes": dataset.class_counts,
                "faults": dataset.fault_counts,
                "metric_feature_union_count": len(dataset.metric_feature_names),
            },
            "designs": designs,
        }
    gate = evaluate_gate(systems, spec)
    return {
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "status": spec["status"],
        "spec": {"path": str(spec_file), "sha256": sha256_file(spec_file)},
        "verified_input_hashes": verified_hashes,
        "rules": spec["rules"],
        "systems": systems,
        "promotion_gate": gate,
        "limitations": [
            "Both RE2-OB and RE2-SS informed v2 and are development systems.",
            "Only an untouched third system can confirm a promoted candidate.",
            "The service score assumes service-qualified metric names and a known candidate set.",
            "No traces are used and scores are not official RCAEval metric-level Avg@k results.",
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = ["#9D9D9D", "#54A24B", "#4C78A8"]
    display = ["Full metrics LR", "Tail Top-2", "Full/delta vote"]
    for axis, (system_name, design) in zip(
        axes.flat,
        [
            ("RE2-OB", "repetition-held-out"),
            ("RE2-OB", "fault-type-held-out"),
            ("RE2-SS", "repetition-held-out"),
            ("RE2-SS", "fault-type-held-out"),
        ],
    ):
        models = result["systems"][system_name]["designs"][design]["models"]
        values = [models[name]["out_of_fold"]["macro_f1"] for name in ALL_MODELS]
        axis.bar(range(3), values, color=colors)
        axis.set_xticks(range(3), display, rotation=15, ha="right")
        axis.set_ylim(0, 1.05)
        axis.set_title(f"{system_name}: {design}")
        for index, value in enumerate(values):
            axis.text(index, value + 0.02, f"{value:.3f}", ha="center")
    selected = result["promotion_gate"]["selected_for_third_system_confirmation"]
    fig.suptitle(
        "RCAEval service-delta v2 development"
        + (f" — selected: {selected}" if selected else " — no candidate passed"),
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Develop service-delta v2 on RE2-OB and RE2-SS."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("experiment/configs/rcaeval-service-delta-v2-development-spec.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/rcaeval-service-delta-v2-development.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("data/results/figures/rcaeval-service-delta-v2-development.png"),
    )
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    outputs = [args.output] + ([] if args.no_figure else [args.figure])
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "v2 outputs already exist; refusing to overwrite: " + ", ".join(existing)
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
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "scores": {
                    system_name: {
                        design: {
                            model: system["designs"][design]["models"][model][
                                "out_of_fold"
                            ]
                            for model in ALL_MODELS
                        }
                        for design in DESIGNS
                    }
                    for system_name, system in result["systems"].items()
                },
                "promotion_gate": result["promotion_gate"],
                "result": str(args.output),
                "artifacts": result.get("artifacts", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
