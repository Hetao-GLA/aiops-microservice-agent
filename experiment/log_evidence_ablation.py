"""Run source and lexical-signature ablations on the +1-second v4 logs."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from experiment.early_diagnosis_stress import _ungated_result
from experiment.frozen_holdout import write_json
from experiment.ml_baseline import LABELS, sha256_file
from experiment.robust_fusion_v2 import (
    MODEL_FILENAME,
    evaluate_candidate,
    verify_candidate_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ID = "log-evidence-ablation-v4-1s"
DEFAULT_SPEC = Path("experiment/configs/log-evidence-ablation-v4-1s-spec.json")
VARIANTS = (
    "all_logs",
    "order_service_only",
    "workload_only",
    "masked_all_logs",
    "masked_order_service_only",
    "masked_workload_only",
)
SERVICE_PATTERN = re.compile(r"^\[([^]]+)\]\s")
STATUS_PATTERN = re.compile(r"\b[1-5]\d{2}\b")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or spec.get("analysis_id") != ANALYSIS_ID:
        raise ValueError("Unsupported log-evidence ablation specification")
    if spec.get("variants") != list(VARIANTS):
        raise ValueError("Ablation variants differ from the implementation")
    if spec.get("uniform_mask", {}).get("replacement_token") != "maskedtoken":
        raise ValueError("Ablation replacement token differs from implementation")
    if not spec.get("uniform_mask", {}).get(
        "mask_all_three_digit_status_codes_100_to_599"
    ):
        raise ValueError("Status-code masking must remain enabled")
    restrictions = spec.get("restrictions", {})
    if not all(
        restrictions.get(key) is True
        for key in (
            "no_fit",
            "no_threshold_selection",
            "no_mask_selection_after_prediction",
            "no_variant_selection_for_model_claims",
            "must_not_be_called_independent_confirmation",
        )
    ):
        raise ValueError("Ablation restrictions are incomplete")
    return spec


def service_of_line(line: str) -> str | None:
    match = SERVICE_PATTERN.match(line)
    return match.group(1) if match else None


def filter_log_sources(log_text: str, allowed_services: set[str] | None) -> str:
    lines = log_text.splitlines()
    if allowed_services is not None:
        lines = [line for line in lines if service_of_line(line) in allowed_services]
    if not lines:
        raise ValueError("Log-source ablation removed every line")
    return "\n".join(lines)


def mask_log_text(
    log_text: str,
    *,
    terms: Sequence[str],
    replacement: str = "maskedtoken",
) -> str:
    ordered = sorted({str(term) for term in terms if str(term)}, key=len, reverse=True)
    if not ordered:
        raise ValueError("At least one lexical signature is required")
    pattern = re.compile("|".join(re.escape(term) for term in ordered), re.IGNORECASE)
    result = STATUS_PATTERN.sub(replacement, log_text)
    result = pattern.sub(replacement, result)
    return result


def _variant_rules(spec: dict[str, Any], variant: str) -> tuple[set[str] | None, bool]:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported ablation variant: {variant}")
    masked = variant.startswith("masked_")
    source_name = variant.removeprefix("masked_")
    if source_name == "all_logs":
        services = None
    elif source_name == "order_service_only":
        services = set(spec["sources"]["order_service_only"])
    elif source_name == "workload_only":
        services = set(spec["sources"]["workload_only"])
    else:
        raise ValueError(f"Unsupported ablation source: {source_name}")
    return services, masked


def transform_record(
    record: dict[str, Any], spec: dict[str, Any], variant: str
) -> dict[str, Any]:
    services, masked = _variant_rules(spec, variant)
    text = filter_log_sources(str(record["log_text"]), services)
    if masked:
        text = mask_log_text(
            text,
            terms=spec["uniform_mask"]["case_insensitive_substrings"],
            replacement=str(spec["uniform_mask"]["replacement_token"]),
        )
    result = copy.deepcopy(record)
    result["dataset"] = f"{record['dataset']}:ablation:{variant}"
    result["log_text"] = text
    result["log_record_count"] = len(text.splitlines())
    result["log_services"] = sorted(
        {service for line in text.splitlines() if (service := service_of_line(line))}
    )
    result["provenance"]["log_evidence_ablation"] = {
        "analysis_id": ANALYSIS_ID,
        "variant": variant,
        "allowed_services": sorted(services) if services is not None else None,
        "uniform_mask_applied": masked,
        "metric_features_unchanged": True,
    }
    return result


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_variants(
    source_path: Path, output_dir: Path, spec: dict[str, Any]
) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results: list[dict[str, Any]] = []
    for variant in VARIANTS:
        path = output_dir / f"log-evidence-ablation-v4-1s-{variant}.jsonl"
        if path.exists():
            raise FileExistsError(
                f"Ablation dataset already exists; refusing to overwrite: {path}"
            )
        transformed = [transform_record(record, spec, variant) for record in records]
        _write_jsonl(path, transformed)
        results.append(
            {
                "variant": variant,
                "path": str(path),
                "sha256": sha256_file(path),
                "records": len(transformed),
                "total_log_records": sum(row["log_record_count"] for row in transformed),
                "mean_log_records": round(
                    float(np.mean([row["log_record_count"] for row in transformed])), 3
                ),
            }
        )
    return results


def _evaluate_variant(
    bundle_dir: Path, path: Path, model: Any, variant: str
) -> dict[str, Any]:
    result = evaluate_candidate(bundle_dir, path, require_after_freeze=True)
    result["experiment_id"] = f"{ANALYSIS_ID}-{variant}"
    result["evaluation_design"].update(
        {
            "confirmatory": False,
            "post_hoc_log_evidence_ablation": True,
            "variant": variant,
            "no_fit_or_tuning": True,
        }
    )
    result["models"]["robust_fusion_branch_without_routing"] = _ungated_result(
        model, path
    )
    return result


def _routing_signature(evaluation: dict[str, Any]) -> list[tuple[str, bool, float]]:
    predictions = evaluation["models"]["gated_robust_fusion_v2"]["predictions"]
    return sorted(
        (
            str(row["incident_id"]),
            bool(row["fallback_to_logs"]),
            float(row["ood_fraction"]),
        )
        for row in predictions
    )


def run_ablation(spec_path: Path, output_dir: Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    source_path = _resolve(spec["source_dataset"]["path"])
    if sha256_file(source_path) != spec["source_dataset"]["sha256"]:
        raise ValueError("+1-second source hash differs from the locked specification")
    source_records = [line for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(source_records) != int(spec["source_dataset"]["records"]):
        raise ValueError("+1-second source record count differs from specification")
    bundle_dir = _resolve(spec["frozen_candidate"]["bundle"])
    manifest = verify_candidate_bundle(bundle_dir)
    if manifest["model_sha256"] != spec["frozen_candidate"]["model_sha256"]:
        raise ValueError("Frozen model differs from the locked specification")
    before = sha256_file(bundle_dir / MODEL_FILENAME)
    model = joblib.load(bundle_dir / MODEL_FILENAME)
    variants = build_variants(source_path, output_dir, spec)
    evaluations: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    routing_reference: list[tuple[str, bool, float]] | None = None
    for artifact in variants:
        variant = str(artifact["variant"])
        evaluation = _evaluate_variant(
            bundle_dir, Path(artifact["path"]), model, variant
        )
        signature = _routing_signature(evaluation)
        if routing_reference is None:
            routing_reference = signature
        elif signature != routing_reference:
            raise RuntimeError("Metric-only routing changed across log ablations")
        evaluations[variant] = evaluation
        models = evaluation["models"]
        summaries.append(
            {
                **artifact,
                "gated": models["gated_robust_fusion_v2"]["metrics"],
                "logs_only": models["logs_only"]["metrics"],
                "ungated": models["robust_fusion_branch_without_routing"]["metrics"],
                "routing": models["gated_robust_fusion_v2"]["routing"],
            }
        )
    after = sha256_file(bundle_dir / MODEL_FILENAME)
    if before != after:
        raise RuntimeError("Frozen candidate changed during evidence ablation")
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
            "uniform_mask_fixed_before_prediction": True,
            "all_variants_reported": True,
        },
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "records": len(source_records),
        },
        "bundle": {
            "path": str(bundle_dir),
            "model_sha256": manifest["model_sha256"],
            "model_unchanged": before == after,
        },
        "summaries": summaries,
        "detailed_evaluations": evaluations,
        "routing_identical_across_log_variants": True,
        "limitations": [
            "Post-hoc dependency test after the +1-second result was observed.",
            "Inference-time token masking creates an artificial lexical distribution shift.",
            "Source-only variants remove context and do not reproduce a naturally instrumented system.",
            "The model is given a known incident boundary; this is classification, not detection.",
            "Same 30 incidents are reused across all variants and are not independent samples.",
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

    rows = result["summaries"]
    labels = [
        "All",
        "Service",
        "Workload",
        "Masked all",
        "Masked service",
        "Masked workload",
    ]
    x = np.arange(len(rows))
    width = 0.24
    model_fields = (
        ("gated", "Gated", "#2a9d8f"),
        ("logs_only", "Logs only", "#264653"),
        ("ungated", "Ungated fusion", "#e76f51"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for offset, (field, title, color) in zip((-width, 0.0, width), model_fields):
        values = [row[field]["macro_f1"] for row in rows]
        axes[0].bar(x + offset, values, width, label=title, color=color)
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Frozen models under log ablation")
    axes[0].legend(fontsize=8)

    class_titles = {
        LABELS[0]: "Database",
        LABELS[1]: "HTTP 500",
        LABELS[2]: "Service stopped",
    }
    for label, color, marker in zip(LABELS, ("#457b9d", "#f4a261", "#8d5a97"), ("o", "s", "^")):
        values = [row["logs_only"]["per_class"][label]["f1"] for row in rows]
        axes[1].plot(x, values, marker=marker, color=color, label=class_titles[label])
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Logs-only per-class F1")
    axes[1].set_title("Which class signatures survive?")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    line_counts = [row["mean_log_records"] for row in rows]
    axes[2].bar(x, line_counts, color="#6c757d")
    axes[2].set_xticks(x, labels, rotation=25, ha="right")
    axes[2].set_ylabel("Mean retained log rows per incident")
    axes[2].set_title("Evidence volume after source filtering")
    fig.suptitle("Post-hoc +1-second log-evidence ablation")
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
        default=Path("data/results/log-evidence-ablation-v4-1s.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("data/results/figures/log-evidence-ablation-v4-1s.png"),
    )
    args = parser.parse_args(argv)
    if args.output.exists() or args.figure.exists():
        raise FileExistsError("Log-evidence ablation outputs already exist")
    result = run_ablation(args.spec, args.variant_dir)
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
                "variants": [
                    {
                        "variant": row["variant"],
                        "gated_macro_f1": row["gated"]["macro_f1"],
                        "logs_macro_f1": row["logs_only"]["macro_f1"],
                        "ungated_macro_f1": row["ungated"]["macro_f1"],
                        "fusion_coverage": row["routing"]["coverage"],
                    }
                    for row in result["summaries"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
