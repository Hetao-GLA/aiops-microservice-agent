"""Run no-fit early-window stress tests on the frozen v4 incidents."""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import joblib
import numpy as np

from experiment.data_engineering.build_local_incidents import (
    _flatten_probe_values,
    _parse_timestamp,
    read_jsonl,
)
from experiment.frozen_holdout import write_json
from experiment.ml_baseline import (
    LABELS,
    _classification_metrics,
    load_incident_dataset,
    sha256_file,
)
from experiment.robust_fusion_v2 import (
    MODEL_FILENAME,
    _confusion,
    evaluate_candidate,
    verify_candidate_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ID = "early-diagnosis-v4-stress-test"
DEFAULT_SPEC = Path("experiment/configs/early-diagnosis-v4-stress-spec.json")
PHASE_SUFFIXES = ("__pre_mean", "__post_mean", "__mean_delta")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or spec.get("analysis_id") != ANALYSIS_ID:
        raise ValueError("Unsupported early-diagnosis specification")
    cutoffs = spec.get("time_windows", {}).get(
        "post_injection_cutoffs_seconds"
    )
    if cutoffs != [1, 3, 5, 8]:
        raise ValueError("Early-diagnosis cutoffs differ from the implementation")
    if spec.get("telemetry_rules", {}).get("missing_post_numeric_value") != 0.0:
        raise ValueError("Early missing-value sentinel differs from implementation")
    restrictions = spec.get("restrictions", {})
    if not all(
        restrictions.get(key) is True
        for key in (
            "no_fit",
            "no_threshold_selection",
            "no_feature_selection",
            "no_cutoff_selection_for_model_claims",
            "must_not_be_called_independent_confirmation",
        )
    ):
        raise ValueError("Early-diagnosis restrictions are incomplete")
    return spec


def _metric_bases(expected_feature_names: Sequence[str]) -> list[str]:
    bases: set[str] = set()
    phases: dict[str, set[str]] = defaultdict(set)
    for feature in expected_feature_names:
        suffix = next(
            (value for value in PHASE_SUFFIXES if feature.endswith(value)), None
        )
        if suffix is None:
            raise ValueError(f"Unexpected metric feature name: {feature}")
        base = feature[: -len(suffix)]
        bases.add(base)
        phases[base].add(suffix)
    if any(phases[base] != set(PHASE_SUFFIXES) for base in bases):
        raise ValueError("Every raw metric must have pre, post, and delta features")
    return sorted(bases)


def summarise_early_probes(
    probes: Sequence[dict[str, Any]],
    injected_at: datetime,
    expected_feature_names: Sequence[str],
    *,
    missing_post_value: float = 0.0,
) -> tuple[int, int, dict[str, float]]:
    pre_values: dict[str, list[float]] = defaultdict(list)
    post_values: dict[str, list[float]] = defaultdict(list)
    pre_rows = 0
    post_rows = 0
    for sample in probes:
        is_pre = _parse_timestamp(str(sample["timestamp"])) < injected_at
        target = pre_values if is_pre else post_values
        if is_pre:
            pre_rows += 1
        else:
            post_rows += 1
        for key, value in _flatten_probe_values(sample).items():
            target[key].append(float(value))
    if pre_rows == 0 or post_rows == 0:
        raise ValueError("Early variants require both pre- and post-injection probes")

    features: dict[str, float] = {}
    for base in _metric_bases(expected_feature_names):
        if not pre_values.get(base):
            raise ValueError(f"Pre-fault metric is absent: {base}")
        pre_mean = float(mean(pre_values[base]))
        post_mean = (
            float(mean(post_values[base]))
            if post_values.get(base)
            else float(missing_post_value)
        )
        features[base + "__pre_mean"] = pre_mean
        features[base + "__post_mean"] = post_mean
        features[base + "__mean_delta"] = post_mean - pre_mean
    if sorted(features) != sorted(expected_feature_names):
        raise RuntimeError("Early metric schema differs from frozen training")
    return pre_rows, post_rows, features


def _verify_raw_file(path: Path, expected_hash: str, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected_hash:
        raise ValueError(f"Raw {name} hash differs from v4 provenance: {path}")


def build_early_record(
    source_record: dict[str, Any],
    *,
    cutoff_seconds: int,
    expected_metric_names: Sequence[str],
    missing_post_value: float = 0.0,
) -> dict[str, Any]:
    if cutoff_seconds <= 0:
        raise ValueError("Early cutoff must be positive")
    provenance = source_record["provenance"]
    raw_dir = _resolve(str(provenance["raw_incident_directory"]))
    metadata_path = raw_dir / "metadata.json"
    probes_path = raw_dir / "probes.jsonl"
    logs_path = raw_dir / "docker-logs.jsonl"
    for path, key, label in (
        (metadata_path, "metadata_sha256", "metadata"),
        (probes_path, "probes_sha256", "probes"),
        (logs_path, "docker_logs_sha256", "Docker logs"),
    ):
        _verify_raw_file(path, str(provenance[key]), label)

    injected_at = _parse_timestamp(str(source_record["injected_at"]))
    cutoff_at = injected_at + timedelta(seconds=cutoff_seconds)
    fault_ended_at = _parse_timestamp(str(provenance["fault_ended_at"]))
    if cutoff_at >= fault_ended_at:
        raise ValueError(
            f"Cutoff reaches recovery evidence for {source_record['incident_id']}"
        )
    logs = [
        record
        for record in read_jsonl(logs_path)
        if _parse_timestamp(str(record["timestamp"])) <= cutoff_at
    ]
    probes = [
        record
        for record in read_jsonl(probes_path)
        if _parse_timestamp(str(record["timestamp"])) <= cutoff_at
    ]
    if not logs or not probes:
        raise ValueError("Early variant has empty telemetry")
    pre_rows, post_rows, metric_features = summarise_early_probes(
        probes,
        injected_at,
        expected_metric_names,
        missing_post_value=missing_post_value,
    )
    result = copy.deepcopy(source_record)
    result["dataset"] = f"{source_record['dataset']}:early+{cutoff_seconds}s"
    result["window_end"] = cutoff_at.isoformat()
    result["log_text"] = "\n".join(
        f"[{record['service']}] {record['message']}" for record in logs
    )
    result["log_record_count"] = len(logs)
    result["log_services"] = sorted({str(record["service"]) for record in logs})
    result["metric_row_count"] = len(probes)
    result["metric_pre_rows"] = pre_rows
    result["metric_post_rows"] = post_rows
    result["metric_features"] = metric_features
    result["provenance"]["analysis_variant"] = {
        "analysis_id": ANALYSIS_ID,
        "cutoff_seconds_after_injection": cutoff_seconds,
        "cutoff_at": cutoff_at.isoformat(),
        "cutoff_inclusive": True,
        "missing_post_numeric_value": float(missing_post_value),
        "recovery_evidence_excluded": True,
    }
    return result


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_variants(
    source_path: Path,
    output_dir: Path,
    *,
    cutoffs: Sequence[int],
    expected_metric_names: Sequence[str],
    missing_post_value: float,
) -> list[dict[str, Any]]:
    source_records = read_jsonl(source_path)
    variants: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        output_path = output_dir / f"early-diagnosis-v4-at-{cutoff}s.jsonl"
        if output_path.exists():
            raise FileExistsError(
                f"Early-window dataset already exists; refusing to overwrite: {output_path}"
            )
        records = [
            build_early_record(
                record,
                cutoff_seconds=int(cutoff),
                expected_metric_names=expected_metric_names,
                missing_post_value=missing_post_value,
            )
            for record in source_records
        ]
        _write_jsonl(output_path, records)
        variants.append(
            {
                "cutoff_seconds": int(cutoff),
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "records": len(records),
                "total_log_records": sum(record["log_record_count"] for record in records),
                "total_probe_records": sum(record["metric_row_count"] for record in records),
                "total_post_probe_records": sum(record["metric_post_rows"] for record in records),
            }
        )
    return variants


def _ungated_result(model: Any, dataset_path: Path) -> dict[str, Any]:
    dataset = load_incident_dataset(dataset_path)
    predicted = [str(value) for value in model.fusion_model_.predict(dataset.rows)]
    probabilities = model.fusion_model_.predict_proba(dataset.rows)
    classes = [
        str(value)
        for value in model.fusion_model_.named_steps["classifier"].classes_
    ]
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
                    float(probabilities[index, classes.index(guess)]), 6
                ),
            }
            for index, (incident_id, actual, guess) in enumerate(
                zip(dataset.incident_ids, dataset.labels, predicted)
            )
        ],
    }


def _evaluate_window(
    bundle_dir: Path,
    dataset_path: Path,
    model: Any,
    *,
    label: str,
    cutoff_seconds: int | None,
) -> dict[str, Any]:
    result = evaluate_candidate(
        bundle_dir, dataset_path, require_after_freeze=True
    )
    result["experiment_id"] = ANALYSIS_ID + "-" + label
    result["evaluation_design"].update(
        {
            "confirmatory": False,
            "post_hoc_derived_from_observed_v4": True,
            "recovery_evidence_excluded": cutoff_seconds is not None,
            "cutoff_seconds_after_injection": cutoff_seconds,
            "no_fit_or_tuning": True,
        }
    )
    result["models"]["robust_fusion_branch_without_routing"] = _ungated_result(
        model, dataset_path
    )
    return result


def _curve_row(
    evaluation: dict[str, Any],
    *,
    label: str,
    cutoff_seconds: int | None,
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    models = evaluation["models"]
    return {
        "label": label,
        "cutoff_seconds": cutoff_seconds,
        "telemetry": telemetry,
        "models": {
            name: {
                "metrics": models[name]["metrics"],
                "confusion_matrix": models[name]["confusion_matrix"],
            }
            for name in (
                "gated_robust_fusion_v2",
                "logs_only",
                "robust_fusion_branch_without_routing",
            )
        },
        "routing": models["gated_robust_fusion_v2"]["routing"],
    }


def _milestones(curve: Sequence[dict[str, Any]], thresholds: Sequence[float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    early_rows = [row for row in curve if row["cutoff_seconds"] is not None]
    for model_name in (
        "gated_robust_fusion_v2",
        "logs_only",
        "robust_fusion_branch_without_routing",
    ):
        result[model_name] = {}
        for threshold in thresholds:
            matches = [
                row["cutoff_seconds"]
                for row in early_rows
                if row["models"][model_name]["metrics"]["macro_f1"] >= threshold
            ]
            result[model_name][str(threshold)] = min(matches) if matches else None
    return result


def run_stress_test(
    spec_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    source_path = _resolve(spec["source_dataset"]["path"])
    if sha256_file(source_path) != spec["source_dataset"]["sha256"]:
        raise ValueError("v4 source dataset hash differs from the locked specification")
    if len(read_jsonl(source_path)) != int(spec["source_dataset"]["records"]):
        raise ValueError("v4 source record count differs from the locked specification")
    bundle_dir = _resolve(spec["frozen_candidate"]["bundle"])
    manifest = verify_candidate_bundle(bundle_dir)
    if manifest["model_sha256"] != spec["frozen_candidate"]["model_sha256"]:
        raise ValueError("Frozen candidate hash differs from the locked specification")
    model_hash_before = sha256_file(bundle_dir / MODEL_FILENAME)
    model = joblib.load(bundle_dir / MODEL_FILENAME)
    cutoffs = [int(value) for value in spec["time_windows"]["post_injection_cutoffs_seconds"]]
    variants = build_variants(
        source_path,
        output_dir,
        cutoffs=cutoffs,
        expected_metric_names=manifest["training"]["metric_feature_names"],
        missing_post_value=float(
            spec["telemetry_rules"]["missing_post_numeric_value"]
        ),
    )
    detailed: dict[str, Any] = {}
    curve: list[dict[str, Any]] = []
    for variant in variants:
        label = f"+{variant['cutoff_seconds']}s"
        evaluation = _evaluate_window(
            bundle_dir,
            Path(variant["path"]),
            model,
            label=label,
            cutoff_seconds=int(variant["cutoff_seconds"]),
        )
        detailed[label] = evaluation
        curve.append(
            _curve_row(
                evaluation,
                label=label,
                cutoff_seconds=int(variant["cutoff_seconds"]),
                telemetry=variant,
            )
        )

    full_dataset = load_incident_dataset(source_path)
    full_evaluation = _evaluate_window(
        bundle_dir,
        source_path,
        model,
        label="complete",
        cutoff_seconds=None,
    )
    detailed["complete"] = full_evaluation
    curve.append(
        _curve_row(
            full_evaluation,
            label="Complete",
            cutoff_seconds=None,
            telemetry={
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "records": len(full_dataset.rows),
                "reference_only": True,
            },
        )
    )
    model_hash_after = sha256_file(bundle_dir / MODEL_FILENAME)
    if model_hash_before != model_hash_after:
        raise RuntimeError("Frozen candidate changed during the stress test")
    return {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "created_at": datetime.now().astimezone().isoformat(),
        "design": {
            "specification": str(spec_path),
            "specification_sha256": sha256_file(spec_path),
            "post_hoc": True,
            "independent_confirmation": False,
            "fitting_or_tuning": False,
            "recovery_evidence_excluded_for_early_windows": True,
            "cutoffs_selected_before_prediction": cutoffs,
            "repeated_windows_are_not_independent_samples": True,
        },
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "records": len(full_dataset.rows),
        },
        "bundle": {
            "path": str(bundle_dir),
            "model_sha256": manifest["model_sha256"],
            "model_unchanged": model_hash_before == model_hash_after,
        },
        "variant_datasets": variants,
        "curve": curve,
        "descriptive_milestones_seconds": _milestones(
            curve, [float(value) for value in spec["reporting"]["descriptive_milestones"]]
        ),
        "detailed_evaluations": detailed,
        "limitations": [
            "Post-hoc analysis of v4 after its labels and full-window results were observed.",
            "Repeated cutoffs reuse the same incidents and are not independent samples.",
            "The frozen models were trained on complete windows, not optimised for online diagnosis.",
            "A zero sentinel represents early unavailable numeric probe fields and may itself cause OOD routing.",
            "Same system and three strong known faults; no cross-system or unknown-fault claim.",
        ],
    }


def render_figure(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault(
        "MPLCONFIGDIR", str((output_path.parent / ".matplotlib").resolve())
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = result["curve"]
    x = np.arange(len(curve))
    labels = [row["label"] for row in curve]
    model_styles = (
        ("gated_robust_fusion_v2", "Gated robust fusion", "#2a9d8f", "o"),
        ("logs_only", "Logs only", "#264653", "s"),
        (
            "robust_fusion_branch_without_routing",
            "Ungated robust fusion",
            "#e76f51",
            "^",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for name, title, color, marker in model_styles:
        values = [row["models"][name]["metrics"]["macro_f1"] for row in curve]
        axes[0].plot(x, values, marker=marker, color=color, label=title)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Frozen-model performance by cutoff")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    class_titles = {
        LABELS[0]: "Database",
        LABELS[1]: "HTTP 500",
        LABELS[2]: "Service stopped",
    }
    for label, color, marker in zip(LABELS, ("#457b9d", "#f4a261", "#8d5a97"), ("o", "s", "^")):
        values = [
            row["models"]["gated_robust_fusion_v2"]["metrics"]["per_class"][label]["f1"]
            for row in curve
        ]
        axes[1].plot(x, values, marker=marker, color=color, label=class_titles[label])
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Per-class F1")
    axes[1].set_title("Gated candidate by fault class")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    coverage = [row["routing"]["coverage"] for row in curve]
    fallback = [row["routing"]["fallback_rate"] for row in curve]
    axes[2].plot(x, coverage, marker="o", label="Fusion coverage", color="#2a9d8f")
    axes[2].plot(x, fallback, marker="s", label="Logs fallback", color="#e76f51")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Fraction of incidents")
    axes[2].set_title("Frozen OOD routing")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("Post-hoc early-diagnosis stress test on v4 incidents")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--variant-dir", type=Path, default=Path("data/processed/incidents")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/early-diagnosis-v4-stress-test.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("data/results/figures/early-diagnosis-v4-stress-test.png"),
    )
    args = parser.parse_args(argv)
    if args.output.exists() or args.figure.exists():
        raise FileExistsError("Early stress-test outputs already exist")
    result = run_stress_test(args.spec, args.variant_dir)
    render_figure(result, args.figure)
    result["artifacts"] = {
        "figure": str(args.figure),
        "figure_sha256": sha256_file(args.figure),
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "curve": [
                    {
                        "label": row["label"],
                        "candidate_macro_f1": row["models"]["gated_robust_fusion_v2"]["metrics"]["macro_f1"],
                        "logs_macro_f1": row["models"]["logs_only"]["metrics"]["macro_f1"],
                        "ungated_macro_f1": row["models"]["robust_fusion_branch_without_routing"]["metrics"]["macro_f1"],
                        "fusion_coverage": row["routing"]["coverage"],
                    }
                    for row in result["curve"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
