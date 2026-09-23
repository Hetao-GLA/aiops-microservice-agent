from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from experiment.rcaeval_public_baseline import (
    PublicDataset,
    _numeric_metrics,
    evaluate_model,
    make_splits,
    sha256_file,
)
from experiment.rcaeval_service_delta_v2 import (
    TAIL,
    _fold_macro,
    evaluate_tail_candidate,
)


REFERENCE = "full_metrics_logistic_regression"
CANDIDATE = TAIL
DESIGNS = ("repetition-held-out", "fault-type-held-out")


def validate_frozen_inputs(spec: dict[str, Any]) -> dict[str, str]:
    checks = {
        spec["source"]["index_path"]: spec["source"]["index_sha256"],
        spec["frozen_development_evidence"]["result"]: spec[
            "frozen_development_evidence"
        ]["result_sha256"],
        spec["frozen_development_evidence"]["spec"]: spec[
            "frozen_development_evidence"
        ]["spec_sha256"],
        spec["frozen_implementations"]["candidate_module"]: spec[
            "frozen_implementations"
        ]["candidate_module_sha256"],
        spec["frozen_implementations"]["reference_module"]: spec[
            "frozen_implementations"
        ]["reference_module_sha256"],
    }
    verified: dict[str, str] = {}
    for path, expected in checks.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Frozen confirmation input hash mismatch for {path}: "
                f"expected={expected}, actual={actual}"
            )
        verified[path] = actual

    development = json.loads(
        Path(spec["frozen_development_evidence"]["result"]).read_text(
            encoding="utf-8"
        )
    )
    selected = development["promotion_gate"][
        "selected_for_third_system_confirmation"
    ]
    expected_candidate = spec["frozen_development_evidence"]["selected_candidate"]
    passed = development["promotion_gate"]["evaluations"][expected_candidate][
        "passes_all"
    ]
    if selected != expected_candidate or expected_candidate != CANDIDATE or not passed:
        raise ValueError("Frozen development result did not promote the locked candidate")
    return verified


def validate_acquisition(
    spec_path: str | Path,
    download_manifest_path: str | Path,
    processed_manifest_path: str | Path,
    incident_path: str | Path,
) -> dict[str, Any]:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    spec_hash = sha256_file(spec_path)
    download_path = Path(download_manifest_path)
    download = json.loads(download_path.read_text(encoding="utf-8"))
    if download.get("spec_sha256") != spec_hash:
        raise ValueError("Download manifest does not reference the locked TT spec")
    totals = download.get("totals", {})
    expected_cases = int(spec["selection"]["expected_cases"])
    expected_names = sorted(str(name) for name in spec["telemetry"]["files"])
    expected_files = expected_cases * len(expected_names)
    if totals.get("cases") != expected_cases or totals.get("files") != expected_files:
        raise ValueError(f"Incomplete RE2-TT download totals: {totals}")
    if len(download.get("cases", [])) != expected_cases:
        raise ValueError("RE2-TT download manifest has the wrong case count")

    root = download_path.parent
    verified_files = 0
    for case_entry in download["cases"]:
        case = str(case_entry["case"])
        files = case_entry.get("files", [])
        if sorted(str(entry["name"]) for entry in files) != expected_names:
            raise ValueError(f"Case does not contain the locked metric files: {case}")
        for file_entry in files:
            path = root / case / str(file_entry["name"])
            if sha256_file(path) != str(file_entry["sha256"]):
                raise ValueError(f"Downloaded file hash mismatch: {path}")
            verified_files += 1

    processed_path = Path(processed_manifest_path)
    processed = json.loads(processed_path.read_text(encoding="utf-8"))
    incident_hash = sha256_file(incident_path)
    if (
        processed.get("records") != expected_cases
        or processed.get("jsonl_sha256") != incident_hash
    ):
        raise ValueError("Processed RE2-TT manifest does not match the incident JSONL")
    if processed.get("datasets") != [spec["selection"]["dataset"]]:
        raise ValueError("Processed manifest is not the locked RE2-TT subset")
    if processed.get("local_fault_labels") != []:
        raise ValueError("Public confirmation labels must remain separate")
    return {
        "download_manifest": str(download_path),
        "download_manifest_sha256": sha256_file(download_path),
        "downloaded_cases": int(totals["cases"]),
        "downloaded_files": int(totals["files"]),
        "downloaded_bytes": int(totals["bytes"]),
        "all_downloaded_file_hashes_verified": verified_files == expected_files,
        "processed_manifest": str(processed_path),
        "processed_manifest_sha256": sha256_file(processed_path),
        "incident_path": str(incident_path),
        "incident_sha256": incident_hash,
        "incident_records": int(processed["records"]),
        "modalities": ["metrics"],
        "public_labels_kept_separate": True,
    }


def load_metrics_dataset(path: str | Path, spec: dict[str, Any]) -> PublicDataset:
    selection = spec["selection"]
    expected_services = sorted(selection["expected_root_cause_services"])
    expected_faults = sorted(selection["expected_faults"])
    expected_cases = int(selection["expected_cases"])
    expected_repetitions = int(
        selection["expected_repetitions_per_service_fault_pair"]
    )
    rows: list[dict[str, Any]] = []
    labels: list[str] = []
    faults: list[str] = []
    repetitions: list[int] = []
    incident_ids: list[str] = []
    metric_names: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if record.get("source") != "rcaeval":
                raise ValueError(f"Line {line_number}: source must be rcaeval")
            if record.get("dataset") != selection["dataset"]:
                raise ValueError(f"Line {line_number}: unexpected dataset")
            if record.get("task") != spec["task"]["name"]:
                raise ValueError(f"Line {line_number}: unexpected task")
            if record.get("local_fault_label") is not None:
                raise ValueError(f"Line {line_number}: local label must remain unset")
            if record.get("modalities") != ["metrics"]:
                raise ValueError(f"Line {line_number}: only metrics may be present")
            if record.get("log_text") or int(record.get("log_record_count", 0)) != 0:
                raise ValueError(f"Line {line_number}: logs must not be fabricated")
            incident_id = str(record.get("incident_id", ""))
            label = str(record.get("root_cause_service", ""))
            fault = str(record.get("original_fault_label", ""))
            provenance = record.get("provenance")
            if not incident_id or label not in expected_services or fault not in expected_faults:
                raise ValueError(f"Line {line_number}: invalid identity or public label")
            if not isinstance(provenance, dict) or "repetition" not in provenance:
                raise ValueError(f"Line {line_number}: repetition provenance is required")
            repetition = int(provenance["repetition"])
            if repetition not in range(1, expected_repetitions + 1):
                raise ValueError(f"Line {line_number}: invalid repetition")
            metrics = _numeric_metrics(
                record.get("metric_features"), line_number=line_number
            )
            metric_names.update(metrics)
            rows.append({"log_text": "", "metric_features": metrics})
            labels.append(label)
            faults.append(fault)
            repetitions.append(repetition)
            incident_ids.append(incident_id)

    if len(rows) != expected_cases or len(set(incident_ids)) != expected_cases:
        raise ValueError("RE2-TT incident count or uniqueness is invalid")
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
        log_record_count=0,
    )


def evaluate_confirmation_gate(
    designs: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    differences: list[float] = []
    candidate_scores: list[float] = []
    top3_scores: list[float] = []
    class_fractions: list[float] = []
    tie_fractions: list[float] = []
    for design in DESIGNS:
        models = designs[design]["models"]
        candidate = models[CANDIDATE]
        reference = models[REFERENCE]
        candidate_f1 = float(candidate["out_of_fold"]["macro_f1"])
        differences.append(candidate_f1 - float(reference["out_of_fold"]["macro_f1"]))
        candidate_scores.append(candidate_f1)
        top3_scores.append(float(candidate["out_of_fold"]["top_3_accuracy"]))
        class_fractions.append(
            float(candidate["balance_and_ties"]["maximum_predicted_class_fraction"])
        )
        tie_fractions.append(
            float(candidate["balance_and_ties"]["exact_numeric_tie_fraction"])
        )
    repetition_folds = [
        float(fold["metrics"]["macro_f1"])
        for fold in designs["repetition-held-out"]["models"][CANDIDATE]["folds"]
    ]
    fault_models = designs["fault-type-held-out"]["models"]
    observed = {
        "mean_macro_f1_minus_reference": round(mean(differences), 6),
        "worst_design_macro_f1_minus_reference": round(min(differences), 6),
        "minimum_design_macro_f1": round(min(candidate_scores), 6),
        "minimum_repetition_fold_macro_f1": round(min(repetition_folds), 6),
        "delay_macro_f1_minus_reference": round(
            _fold_macro(fault_models[CANDIDATE], "delay")
            - _fold_macro(fault_models[REFERENCE], "delay"),
            6,
        ),
        "minimum_top3_accuracy": round(min(top3_scores), 6),
        "maximum_predicted_class_fraction": round(max(class_fractions), 6),
        "maximum_exact_numeric_tie_fraction": round(max(tie_fractions), 6),
    }
    minimums = spec["confirmation_gate"]["minimums"]
    maximums = spec["confirmation_gate"]["maximums"]
    checks = {
        **{name: observed[name] >= float(limit) for name, limit in minimums.items()},
        **{name: observed[name] <= float(limit) for name, limit in maximums.items()},
    }
    passed = all(checks.values())
    return {
        "observed": observed,
        "minimums": minimums,
        "maximums": maximums,
        "checks": checks,
        "confirmation_passed": passed,
        "decision": (
            "three_system_confirmation_passed"
            if passed
            else "third_system_confirmation_failed"
        ),
    }


def run_confirmation(
    spec_path: str | Path,
    incident_path: str | Path,
    download_manifest_path: str | Path,
    processed_manifest_path: str | Path,
) -> dict[str, Any]:
    spec_file = Path(spec_path)
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError("Unsupported RE2-TT confirmation specification")
    frozen_hashes = validate_frozen_inputs(spec)
    acquisition = validate_acquisition(
        spec_file, download_manifest_path, processed_manifest_path, incident_path
    )
    dataset = load_metrics_dataset(incident_path, spec)
    designs: dict[str, Any] = {}
    for design in DESIGNS:
        splits = make_splits(dataset, design)
        reference = evaluate_model("metrics_only", dataset, splits, spec)
        candidate = evaluate_tail_candidate(dataset, splits, spec)
        designs[design] = {
            "folds": len(splits),
            "models": {REFERENCE: reference, CANDIDATE: candidate},
            "candidate_minus_reference_macro_f1": round(
                float(candidate["out_of_fold"]["macro_f1"])
                - float(reference["out_of_fold"]["macro_f1"]),
                6,
            ),
        }
    gate = evaluate_confirmation_gate(designs, spec)
    return {
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "status": spec["status"],
        "spec": {"path": str(spec_file), "sha256": sha256_file(spec_file)},
        "frozen_input_hashes": frozen_hashes,
        "acquisition": acquisition,
        "dataset": {
            "records": len(dataset.rows),
            "classes": dataset.class_counts,
            "faults": dataset.fault_counts,
            "repetitions": sorted(set(dataset.repetitions)),
            "metric_feature_union_count": len(dataset.metric_feature_names),
            "log_record_count": 0,
        },
        "rules": spec["rules"],
        "designs": designs,
        "confirmation_gate": gate,
        "limitations": spec["interpretation_limits"],
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
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = ["#9D9D9D", "#54A24B"]
    labels = ["Full metrics LR", "Empirical-tail Top-2"]
    panels = (
        ("repetition-held-out", None, "Repetition-held-out Macro F1"),
        ("repetition-held-out", "folds", "Macro F1 by held-out repetition"),
        ("fault-type-held-out", None, "Fault-type-held-out Macro F1"),
        ("fault-type-held-out", "delay", "Held-out delay Macro F1"),
    )
    for axis, (design, focus, title) in zip(axes.flat, panels):
        models = result["designs"][design]["models"]
        if focus == "folds":
            held_out = [str(fold["held_out"]) for fold in models[REFERENCE]["folds"]]
            positions = list(range(len(held_out)))
            width = 0.36
            for offset, model_name, color, label in (
                (-width / 2, REFERENCE, colors[0], labels[0]),
                (width / 2, CANDIDATE, colors[1], labels[1]),
            ):
                values = [_fold_macro(models[model_name], value) for value in held_out]
                axis.bar(
                    [position + offset for position in positions],
                    values,
                    width,
                    color=color,
                    label=label,
                )
            axis.set_xticks(positions, held_out)
            axis.legend(fontsize=8)
        else:
            values = [
                (
                    float(models[name]["out_of_fold"]["macro_f1"])
                    if focus is None
                    else _fold_macro(models[name], focus)
                )
                for name in (REFERENCE, CANDIDATE)
            ]
            axis.bar(range(2), values, color=colors)
            axis.set_xticks(range(2), labels, rotation=15, ha="right")
            for index, value in enumerate(values):
                axis.text(index, value + 0.02, f"{value:.3f}", ha="center")
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
    decision = result["confirmation_gate"]["decision"]
    fig.suptitle(f"RCAEval RE2-TT service-delta v2 — {decision}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Confirm frozen service-delta v2 on untouched RCAEval RE2-TT."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(
            "experiment/configs/rcaeval-re2-tt-service-delta-v2-confirmation-spec.json"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/incidents/rcaeval-re2-tt-metrics.jsonl"),
    )
    parser.add_argument(
        "--download-manifest",
        type=Path,
        default=Path("data/external/rcaeval/re2-tt/subset-manifest.json"),
    )
    parser.add_argument(
        "--processed-manifest",
        type=Path,
        default=Path(
            "data/processed/incidents/rcaeval-re2-tt-metrics-manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/results/rcaeval-re2-tt-service-delta-v2-confirmation.json"
        ),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(
            "data/results/figures/rcaeval-re2-tt-service-delta-v2-confirmation.png"
        ),
    )
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    outputs = [args.output] + ([] if args.no_figure else [args.figure])
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Confirmation outputs already exist; refusing to overwrite: "
            + ", ".join(existing)
        )
    result = run_confirmation(
        args.spec, args.input, args.download_manifest, args.processed_manifest
    )
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
                for model in (REFERENCE, CANDIDATE)
            }
            for design in DESIGNS
        },
        "confirmation_gate": result["confirmation_gate"],
        "result": str(args.output),
        "artifacts": result.get("artifacts", {}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
