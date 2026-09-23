from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from experiment.rcaeval_delta_candidates import (
    _fold_macro_f1,
    evaluate_robust_scorer,
)
from experiment.rcaeval_public_baseline import (
    evaluate_model,
    load_dataset,
    make_splits,
    sha256_file,
)


DESIGNS = ("repetition-held-out", "fault-type-held-out")
REFERENCE = "full_metrics_logistic_regression"
CANDIDATE = "robust_service_delta_score"


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
                f"Frozen confirmation input hash mismatch for {path}: expected={expected}, actual={actual}"
            )
        verified[path] = actual
    return verified


def validate_acquisition(
    spec_path: str | Path,
    download_manifest_path: str | Path,
    processed_manifest_path: str | Path,
    incident_path: str | Path,
) -> dict[str, Any]:
    spec_hash = sha256_file(spec_path)
    download_path = Path(download_manifest_path)
    download = json.loads(download_path.read_text(encoding="utf-8"))
    if download.get("spec_sha256") != spec_hash:
        raise ValueError("Download manifest does not reference the locked confirmation spec")
    totals = download.get("totals", {})
    if totals.get("cases") != 90 or totals.get("files") != 270:
        raise ValueError(f"Incomplete RE2-SS download totals: {totals}")
    if len(download.get("cases", [])) != 90:
        raise ValueError("RE2-SS download manifest must contain 90 cases")

    root = download_path.parent
    file_hashes: dict[tuple[str, str], str] = {}
    for case_entry in download["cases"]:
        case = str(case_entry["case"])
        files = case_entry.get("files", [])
        if len(files) != 3:
            raise ValueError(f"Case does not contain three locked files: {case}")
        for file_entry in files:
            name = str(file_entry["name"])
            path = root / case / name
            actual = sha256_file(path)
            expected = str(file_entry["sha256"])
            if actual != expected:
                raise ValueError(f"Downloaded file hash mismatch: {path}")
            file_hashes[(case, name)] = actual

    processed_path = Path(processed_manifest_path)
    processed = json.loads(processed_path.read_text(encoding="utf-8"))
    incident_hash = sha256_file(incident_path)
    if processed.get("records") != 90 or processed.get("jsonl_sha256") != incident_hash:
        raise ValueError("Processed RE2-SS manifest does not match the incident JSONL")
    if processed.get("datasets") != ["RE2-SS"]:
        raise ValueError("Processed manifest is not the locked RE2-SS subset")
    if processed.get("local_fault_labels") != []:
        raise ValueError("Public confirmation labels must remain separate")

    return {
        "download_manifest": str(download_path),
        "download_manifest_sha256": sha256_file(download_path),
        "downloaded_cases": int(totals["cases"]),
        "downloaded_files": int(totals["files"]),
        "downloaded_bytes": int(totals["bytes"]),
        "all_downloaded_file_hashes_verified": len(file_hashes) == 270,
        "processed_manifest": str(processed_path),
        "processed_manifest_sha256": sha256_file(processed_path),
        "incident_path": str(incident_path),
        "incident_sha256": incident_hash,
        "incident_records": int(processed["records"]),
        "public_labels_kept_separate": True,
    }


def evaluate_confirmation_gate(
    designs: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    repetition = designs["repetition-held-out"]["models"]
    fault = designs["fault-type-held-out"]["models"]
    candidate_rep = repetition[CANDIDATE]
    reference_rep = repetition[REFERENCE]
    candidate_fault = fault[CANDIDATE]
    reference_fault = fault[REFERENCE]
    observed = {
        "candidate_repetition_macro_f1": float(
            candidate_rep["out_of_fold"]["macro_f1"]
        ),
        "candidate_fault_macro_f1": float(
            candidate_fault["out_of_fold"]["macro_f1"]
        ),
        "candidate_min_repetition_fold_macro_f1": min(
            float(fold["metrics"]["macro_f1"]) for fold in candidate_rep["folds"]
        ),
        "candidate_delay_macro_f1": _fold_macro_f1(candidate_fault, "delay"),
        "candidate_repetition_top3_accuracy": float(
            candidate_rep["out_of_fold"]["top_3_accuracy"]
        ),
        "candidate_fault_top3_accuracy": float(
            candidate_fault["out_of_fold"]["top_3_accuracy"]
        ),
        "candidate_minus_reference_repetition_macro_f1": round(
            float(
                candidate_rep["out_of_fold"]["macro_f1"]
                - reference_rep["out_of_fold"]["macro_f1"]
            ),
            6,
        ),
        "candidate_minus_reference_fault_macro_f1": round(
            float(
                candidate_fault["out_of_fold"]["macro_f1"]
                - reference_fault["out_of_fold"]["macro_f1"]
            ),
            6,
        ),
        "candidate_minus_reference_delay_macro_f1": round(
            _fold_macro_f1(candidate_fault, "delay")
            - _fold_macro_f1(reference_fault, "delay"),
            6,
        ),
    }
    requirements = spec["confirmation_gate"]["requirements"]
    checks = {
        name: observed[name] >= float(minimum)
        for name, minimum in requirements.items()
    }
    return {
        "observed": observed,
        "minimum": requirements,
        "checks": checks,
        "confirmation_passed": all(checks.values()),
        "decision": (
            "cross_system_confirmation_passed"
            if all(checks.values())
            else "cross_system_confirmation_failed"
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
        raise ValueError("Unsupported RE2-SS confirmation specification")
    frozen_hashes = validate_frozen_inputs(spec)
    acquisition = validate_acquisition(
        spec_file,
        download_manifest_path,
        processed_manifest_path,
        incident_path,
    )
    dataset = load_dataset(incident_path, spec)
    designs: dict[str, Any] = {}
    for design in DESIGNS:
        splits = make_splits(dataset, design)
        reference = evaluate_model("metrics_only", dataset, splits, spec)
        candidate = evaluate_robust_scorer(dataset, splits, spec)
        designs[design] = {
            "folds": len(splits),
            "models": {REFERENCE: reference, CANDIDATE: candidate},
            "candidate_minus_reference_macro_f1": round(
                float(
                    candidate["out_of_fold"]["macro_f1"]
                    - reference["out_of_fold"]["macro_f1"]
                ),
                6,
            ),
        }
    gate = evaluate_confirmation_gate(designs, spec)
    return {
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
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
            "log_record_count": dataset.log_record_count,
        },
        "rules": spec["rules"],
        "designs": designs,
        "confirmation_gate": gate,
        "limitations": [
            "This confirms an algorithm across two RCAEval systems, not a pre-trained model transfer.",
            "Both algorithms fit training-fold statistics or coefficients within RE2-SS.",
            "The candidate assumes service-qualified metric names and a known five-service class set.",
            "No traces are used and the result is not an official metric-level Avg@k score.",
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
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = ["#9D9D9D", "#54A24B"]
    labels = ["Full metrics LR", "Robust service delta"]
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
                values = [
                    _fold_macro_f1(models[model_name], value) for value in held_out
                ]
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
                    models[model_name]["out_of_fold"]["macro_f1"]
                    if focus is None
                    else _fold_macro_f1(models[model_name], focus)
                )
                for model_name in (REFERENCE, CANDIDATE)
            ]
            axis.bar(range(2), values, color=colors)
            axis.set_xticks(range(2), labels, rotation=15, ha="right")
            for index, value in enumerate(values):
                axis.text(index, value + 0.02, f"{value:.3f}", ha="center")
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
    decision = result["confirmation_gate"]["decision"]
    fig.suptitle(f"RCAEval RE2-SS robust-delta confirmation — {decision}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Confirm the frozen robust-delta algorithm on RCAEval RE2-SS."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("experiment/configs/rcaeval-re2-ss-confirmation-spec.json"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/incidents/rcaeval-re2-ss.jsonl"),
    )
    parser.add_argument(
        "--download-manifest",
        type=Path,
        default=Path("data/external/rcaeval/re2-ss/subset-manifest.json"),
    )
    parser.add_argument(
        "--processed-manifest",
        type=Path,
        default=Path("data/processed/incidents/rcaeval-re2-ss-manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/rcaeval-re2-ss-robust-delta-confirmation-v1.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("data/results/figures/rcaeval-re2-ss-robust-delta-confirmation-v1.png"),
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
        args.spec,
        args.input,
        args.download_manifest,
        args.processed_manifest,
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
