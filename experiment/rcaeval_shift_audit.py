from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DESIGNS = ("repetition-held-out", "fault-type-held-out")
MODELS = ("logs_only", "metrics_only", "logs_plus_metrics")
DIAGNOSTIC_FIELDS = (
    "metric_range_breach_fraction",
    "metric_mean_absolute_z",
    "metric_mean_absolute_robust_z",
    "log_unseen_line_fraction",
    "log_unseen_unique_fraction",
    "log_service_proportion_l1",
)

UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HEX_RE = re.compile(r"\b(?:0x)?[0-9a-f]{8,}\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![a-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
SERVICE_PREFIX_RE = re.compile(r"^\[([^\]]+)\]\s*")


@dataclass(frozen=True)
class AuditRecord:
    incident_id: str
    label: str
    fault: str
    repetition: int
    metrics: dict[str, float]
    templates: Counter[str]
    log_services: Counter[str]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_log_line(line: str) -> str:
    value = line.lower()
    value = UUID_RE.sub("<uuid>", value)
    value = IPV4_RE.sub("<ip>", value)
    value = HEX_RE.sub("<hex>", value)
    value = NUMBER_RE.sub("<num>", value)
    return SPACE_RE.sub(" ", value).strip()


def _numeric_metrics(raw: Any, *, line_number: int) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Line {line_number}: metric_features must be non-empty")
    result: dict[str, float] = {}
    for name, value in raw.items():
        if value is None:
            result[str(name)] = 0.0
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Line {line_number}: non-numeric metric {name!r}")
        elif not math.isfinite(float(value)):
            raise ValueError(f"Line {line_number}: non-finite metric {name!r}")
        else:
            result[str(name)] = float(value)
    return result


def load_records(path: str | Path) -> list[AuditRecord]:
    records: list[AuditRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            raw = json.loads(raw_line)
            if raw.get("source") != "rcaeval" or raw.get("dataset") != "RE2-OB":
                raise ValueError(f"Line {line_number}: unexpected data source")
            provenance = raw.get("provenance")
            if not isinstance(provenance, dict) or "repetition" not in provenance:
                raise ValueError(f"Line {line_number}: repetition is required")
            templates: Counter[str] = Counter()
            services: Counter[str] = Counter()
            log_text = raw.get("log_text")
            if not isinstance(log_text, str) or not log_text:
                raise ValueError(f"Line {line_number}: log_text is required")
            for line in log_text.splitlines():
                normalised = normalise_log_line(line)
                if normalised:
                    templates[normalised] += 1
                match = SERVICE_PREFIX_RE.match(line)
                services[(match.group(1) if match else "<unknown>").lower()] += 1
            records.append(
                AuditRecord(
                    incident_id=str(raw["incident_id"]),
                    label=str(raw["root_cause_service"]),
                    fault=str(raw["original_fault_label"]),
                    repetition=int(provenance["repetition"]),
                    metrics=_numeric_metrics(
                        raw.get("metric_features"), line_number=line_number
                    ),
                    templates=templates,
                    log_services=services,
                )
            )
    if len(records) != 90:
        raise ValueError(f"Expected 90 audit records, found {len(records)}")
    ids = [record.incident_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Audit incident IDs must be unique")
    return records


def make_splits(
    records: Sequence[AuditRecord], design: str
) -> list[tuple[str, list[int], list[int]]]:
    if design == "repetition-held-out":
        values: Sequence[str | int] = [record.repetition for record in records]
    elif design == "fault-type-held-out":
        values = [record.fault for record in records]
    else:
        raise ValueError(f"Unsupported audit design: {design}")
    splits = []
    all_indexes = set(range(len(records)))
    for value in sorted(set(values)):
        test = [index for index, item in enumerate(values) if item == value]
        train = sorted(all_indexes.difference(test))
        splits.append((str(value), train, test))
    coverage = Counter(index for _, _, test in splits for index in test)
    if set(coverage) != all_indexes or set(coverage.values()) != {1}:
        raise RuntimeError("Audit splits do not cover every incident exactly once")
    return splits


def _matrix(
    records: Sequence[AuditRecord], indexes: Sequence[int], features: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [[records[index].metrics.get(feature, 0.0) for feature in features] for index in indexes],
        dtype=float,
    )


def metric_fold_diagnostics(
    records: Sequence[AuditRecord],
    train_indexes: Sequence[int],
    test_indexes: Sequence[int],
    features: Sequence[str],
    *,
    epsilon: float,
    z_cap: float,
    services: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = _matrix(records, train_indexes, features)
    test = _matrix(records, test_indexes, features)
    minimum = train.min(axis=0)
    maximum = train.max(axis=0)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    median = np.median(train, axis=0)
    q1 = np.quantile(train, 0.25, axis=0)
    q3 = np.quantile(train, 0.75, axis=0)
    iqr = q3 - q1

    breach = (test < minimum - epsilon) | (test > maximum + epsilon)
    z = np.zeros_like(test)
    np.divide(np.abs(test - mean), std, out=z, where=std > epsilon)
    constant_shift = (std <= epsilon) & (np.abs(test - mean) > epsilon)
    z[constant_shift] = z_cap
    z = np.minimum(z, z_cap)
    robust_z = np.zeros_like(test)
    np.divide(np.abs(test - median), iqr, out=robust_z, where=iqr > epsilon)
    robust_constant_shift = (iqr <= epsilon) & (np.abs(test - median) > epsilon)
    robust_z[robust_constant_shift] = z_cap
    robust_z = np.minimum(robust_z, z_cap)

    block_masks = {
        suffix: np.asarray([feature.endswith("__" + suffix) for feature in features])
        for suffix in ("pre_mean", "post_mean", "mean_delta")
    }
    delta_mask = block_masks["mean_delta"]
    service_masks = {
        service: np.asarray(
            [feature.startswith(service + "_") for feature in features]
        )
        & delta_mask
        for service in services
    }

    rows: list[dict[str, Any]] = []
    for local_index, record_index in enumerate(test_indexes):
        row: dict[str, Any] = {
            "incident_id": records[record_index].incident_id,
            "metric_range_breach_fraction": round(float(breach[local_index].mean()), 6),
            "metric_mean_absolute_z": round(float(z[local_index].mean()), 6),
            "metric_mean_absolute_robust_z": round(
                float(robust_z[local_index].mean()), 6
            ),
            "metric_blocks": {},
        }
        for block, mask in block_masks.items():
            row["metric_blocks"][block] = {
                "range_breach_fraction": round(float(breach[local_index, mask].mean()), 6),
                "mean_absolute_z": round(float(z[local_index, mask].mean()), 6),
                "mean_absolute_robust_z": round(
                    float(robust_z[local_index, mask].mean()), 6
                ),
            }
        service_scores = {
            service: float(z[local_index, mask].max()) if mask.any() else 0.0
            for service, mask in service_masks.items()
        }
        ranking = sorted(services, key=lambda service: (-service_scores[service], service))
        true_rank = ranking.index(records[record_index].label) + 1
        row["metric_signal"] = {
            "ranking": ranking,
            "true_service_rank": true_rank,
            "hit_at_1": true_rank <= 1,
            "hit_at_3": true_rank <= 3,
        }
        rows.append(row)

    feature_shift = np.mean(z, axis=0)
    ranked = np.argsort(feature_shift)[::-1]
    top_features = [
        {
            "feature": features[int(index)],
            "mean_absolute_z": round(float(feature_shift[int(index)]), 6),
            "test_range_breach_fraction": round(float(breach[:, int(index)].mean()), 6),
        }
        for index in ranked
    ]
    return rows, top_features


def log_fold_diagnostics(
    records: Sequence[AuditRecord],
    train_indexes: Sequence[int],
    test_indexes: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    training_templates: set[str] = set()
    service_names: set[str] = set()
    for index in train_indexes:
        training_templates.update(records[index].templates)
        service_names.update(records[index].log_services)
    for index in test_indexes:
        service_names.update(records[index].log_services)
    ordered_services = sorted(service_names)

    training_proportions = []
    for index in train_indexes:
        counter = records[index].log_services
        total = sum(counter.values()) or 1
        training_proportions.append(
            np.asarray([counter[service] / total for service in ordered_services])
        )
    mean_training_proportion = np.mean(training_proportions, axis=0)

    combined_unseen: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for index in test_indexes:
        record = records[index]
        total_lines = sum(record.templates.values()) or 1
        unseen = Counter(
            {template: count for template, count in record.templates.items() if template not in training_templates}
        )
        combined_unseen.update(unseen)
        unique_total = len(record.templates) or 1
        service_total = sum(record.log_services.values()) or 1
        service_proportion = np.asarray(
            [record.log_services[service] / service_total for service in ordered_services]
        )
        rows.append(
            {
                "incident_id": record.incident_id,
                "log_unseen_line_fraction": round(sum(unseen.values()) / total_lines, 6),
                "log_unseen_unique_fraction": round(len(unseen) / unique_total, 6),
                "log_service_proportion_l1": round(
                    float(np.abs(service_proportion - mean_training_proportion).sum()),
                    6,
                ),
            }
        )
    top_unseen = [
        {"template": template, "test_lines": int(count)}
        for template, count in combined_unseen.most_common()
    ]
    return rows, top_unseen


def _mean(rows: Sequence[dict[str, Any]], field: str) -> float:
    return round(float(np.mean([float(row[field]) for row in rows])), 6)


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {field: _mean(rows, field) for field in DIAGNOSTIC_FIELDS}


def _prediction_maps(baseline: dict[str, Any]) -> dict[str, dict[str, dict[str, bool]]]:
    result: dict[str, dict[str, dict[str, bool]]] = {}
    for design in DESIGNS:
        result[design] = {}
        for model in MODELS:
            predictions = baseline["designs"][design]["models"][model]["predictions"]
            mapping = {str(row["incident_id"]): bool(row["correct"]) for row in predictions}
            if len(mapping) != 90:
                raise ValueError(f"Frozen prediction table is incomplete: {design}/{model}")
            result[design][model] = mapping
    return result


def run_audit(spec_path: str | Path) -> dict[str, Any]:
    spec_file = Path(spec_path)
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError("Unsupported shift-audit spec")
    inputs = spec["inputs"]
    for path_key, hash_key in (
        ("incidents", "incidents_sha256"),
        ("baseline_result", "baseline_result_sha256"),
        ("baseline_spec", "baseline_spec_sha256"),
    ):
        actual = sha256_file(inputs[path_key])
        if actual != inputs[hash_key]:
            raise ValueError(
                f"Frozen input hash mismatch for {path_key}: expected={inputs[hash_key]}, actual={actual}"
            )

    records = load_records(inputs["incidents"])
    baseline = json.loads(Path(inputs["baseline_result"]).read_text(encoding="utf-8"))
    predictions = _prediction_maps(baseline)
    features = sorted({name for record in records for name in record.metrics})
    services = sorted({record.label for record in records})
    metric_settings = spec["metric_drift"]
    log_settings = spec["log_drift"]
    designs: dict[str, Any] = {}

    for design in DESIGNS:
        diagnostic_rows: list[dict[str, Any]] = []
        folds: list[dict[str, Any]] = []
        for held_out, train_indexes, test_indexes in make_splits(records, design):
            metric_rows, shifted = metric_fold_diagnostics(
                records,
                train_indexes,
                test_indexes,
                features,
                epsilon=float(metric_settings["epsilon"]),
                z_cap=float(metric_settings["absolute_z_cap"]),
                services=services,
            )
            log_rows, unseen = log_fold_diagnostics(records, train_indexes, test_indexes)
            metric_by_id = {row["incident_id"]: row for row in metric_rows}
            log_by_id = {row["incident_id"]: row for row in log_rows}
            fold_rows = []
            for index in test_indexes:
                record = records[index]
                row = {
                    "incident_id": record.incident_id,
                    "held_out": held_out,
                    "root_cause_service": record.label,
                    "fault": record.fault,
                    "repetition": record.repetition,
                    **metric_by_id[record.incident_id],
                    **log_by_id[record.incident_id],
                    "frozen_prediction_correct": {
                        model: predictions[design][model][record.incident_id]
                        for model in MODELS
                    },
                }
                fold_rows.append(row)
                diagnostic_rows.append(row)
            folds.append(
                {
                    "held_out": held_out,
                    "records": len(fold_rows),
                    "frozen_errors": {
                        model: sum(
                            not row["frozen_prediction_correct"][model]
                            for row in fold_rows
                        )
                        for model in MODELS
                    },
                    "diagnostics": _summary(fold_rows),
                    "metric_signal_localisation": {
                        "hit_at_1": round(
                            sum(row["metric_signal"]["hit_at_1"] for row in fold_rows)
                            / len(fold_rows),
                            6,
                        ),
                        "hit_at_3": round(
                            sum(row["metric_signal"]["hit_at_3"] for row in fold_rows)
                            / len(fold_rows),
                            6,
                        ),
                    },
                    "top_shifted_metric_features": shifted[
                        : int(metric_settings["top_shifted_features"])
                    ],
                    "top_unseen_log_templates": unseen[
                        : int(log_settings["top_unseen_templates"])
                    ],
                }
            )

        error_association: dict[str, Any] = {}
        for model in MODELS:
            correct = [
                row for row in diagnostic_rows if row["frozen_prediction_correct"][model]
            ]
            wrong = [
                row for row in diagnostic_rows if not row["frozen_prediction_correct"][model]
            ]
            error_association[model] = {
                "correct": {"records": len(correct), **_summary(correct)},
                "wrong": {"records": len(wrong), **_summary(wrong)},
            }

        focus = str(spec["scope"]["focus_folds"][design])
        focus_rows = [row for row in diagnostic_rows if row["held_out"] == focus]
        other_rows = [row for row in diagnostic_rows if row["held_out"] != focus]
        designs[design] = {
            "focus_fold": focus,
            "folds": folds,
            "overall_diagnostics": _summary(diagnostic_rows),
            "focus_comparison": {
                "focus": {"records": len(focus_rows), **_summary(focus_rows)},
                "other_folds": {"records": len(other_rows), **_summary(other_rows)},
                "focus_minus_other": {
                    field: round(_mean(focus_rows, field) - _mean(other_rows, field), 6)
                    for field in DIAGNOSTIC_FIELDS
                },
            },
            "error_association": error_association,
            "diagnostic_rows": sorted(
                diagnostic_rows, key=lambda row: row["incident_id"]
            ),
        }

    return {
        "schema_version": 1,
        "audit_id": spec["audit_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "spec": {"path": str(spec_file), "sha256": sha256_file(spec_file)},
        "inputs": inputs,
        "dataset": {
            "records": len(records),
            "metric_feature_union_count": len(features),
            "root_cause_services": services,
            "faults": sorted({record.fault for record in records}),
            "repetitions": sorted({record.repetition for record in records}),
        },
        "analysis_rules": spec["analysis_rules"],
        "designs": designs,
        "limitations": [
            "The audit is descriptive and reuses one frozen benchmark result.",
            "Associations between drift measures and errors are not causal estimates.",
            "Template normalisation is intentionally simple and may merge distinct messages.",
            "The signal-localisation check uses service-qualified metric names and is not an official RCAEval score.",
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

    baseline = json.loads(
        Path(result["inputs"]["baseline_result"]).read_text(encoding="utf-8")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for row_index, design in enumerate(DESIGNS):
        folds = result["designs"][design]["folds"]
        labels = [fold["held_out"] for fold in folds]
        accuracy = [
            next(
                fold["metrics"]["top_1_accuracy"]
                for fold in baseline["designs"][design]["models"]["logs_plus_metrics"]["folds"]
                if str(fold["held_out"]) == label
            )
            for label in labels
        ]
        axes[row_index, 0].bar(labels, accuracy, color="#4C78A8")
        axes[row_index, 0].set_ylim(0, 1.05)
        axes[row_index, 0].set_ylabel("Frozen fusion accuracy")
        axes[row_index, 0].set_title(design)

        range_breach = [
            fold["diagnostics"]["metric_range_breach_fraction"] for fold in folds
        ]
        unseen = [fold["diagnostics"]["log_unseen_line_fraction"] for fold in folds]
        signal = [fold["metric_signal_localisation"]["hit_at_1"] for fold in folds]
        positions = np.arange(len(labels))
        width = 0.26
        axes[row_index, 1].bar(positions - width, range_breach, width, label="Metric range breach", color="#F58518")
        axes[row_index, 1].bar(positions, unseen, width, label="Unseen log lines", color="#54A24B")
        axes[row_index, 1].bar(positions + width, signal, width, label="Metric signal hit@1", color="#B279A2")
        axes[row_index, 1].set_xticks(positions, labels, rotation=20)
        axes[row_index, 1].set_ylim(0, 1.05)
        axes[row_index, 1].set_title(f"Diagnostics: {design}")
        axes[row_index, 1].legend(fontsize=8)

    fig.suptitle("RCAEval RE2-OB frozen-prediction shift audit", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit shift around the frozen RCAEval RE2-OB predictions."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("experiment/configs/rcaeval-re2-ob-shift-audit-spec.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/rcaeval-re2-ob-shift-audit-v1.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("data/results/figures/rcaeval-re2-ob-shift-audit-v1.png"),
    )
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    outputs = [args.output] + ([] if args.no_figure else [args.figure])
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Audit outputs already exist; refusing to overwrite: " + ", ".join(existing)
        )

    result = run_audit(args.spec)
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
        "audit_id": result["audit_id"],
        "focus": {
            design: result["designs"][design]["focus_comparison"]
            for design in DESIGNS
        },
        "error_association": {
            design: result["designs"][design]["error_association"]
            for design in DESIGNS
        },
        "result": str(args.output),
        "artifacts": result.get("artifacts", {}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
