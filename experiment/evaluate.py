"""Evaluate detector incidents against independent ground-truth incidents."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence


@dataclass
class IncidentInterval:
    identifier: str
    label: str
    started_at: datetime
    ended_at: datetime | None


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
    return records


def build_ground_truth_intervals(
    records: Iterable[dict[str, Any]],
) -> list[IncidentInterval]:
    incidents: dict[str, IncidentInterval] = {}
    for record in records:
        incident_id = record["incident_id"]
        if record["event"] == "fault_started":
            incidents[incident_id] = IncidentInterval(
                identifier=incident_id,
                label=record["fault_type"],
                started_at=_parse_timestamp(record["timestamp"]),
                ended_at=None,
            )
        elif record["event"] in {"fault_ended", "recovery_failed"}:
            if incident_id not in incidents:
                raise ValueError(f"ground-truth end without start: {incident_id}")
            incidents[incident_id].ended_at = _parse_timestamp(record["timestamp"])
    return sorted(incidents.values(), key=lambda incident: incident.started_at)


def build_detection_intervals(
    records: Iterable[dict[str, Any]],
) -> list[IncidentInterval]:
    detections: dict[str, IncidentInterval] = {}
    for record in records:
        detection_id = record["detection_id"]
        if record["event"] == "detection_started":
            detections[detection_id] = IncidentInterval(
                identifier=detection_id,
                label=record["predicted_fault"],
                started_at=_parse_timestamp(record["timestamp"]),
                ended_at=None,
            )
        elif record["event"] == "detection_ended":
            if detection_id not in detections:
                raise ValueError(f"detection end without start: {detection_id}")
            detections[detection_id].ended_at = _parse_timestamp(record["timestamp"])
    return sorted(detections.values(), key=lambda detection: detection.started_at)


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def evaluate_intervals(
    ground_truth: Sequence[IncidentInterval],
    detections: Sequence[IncidentInterval],
    *,
    early_tolerance_seconds: float = 5.0,
    late_tolerance_seconds: float = 15.0,
    timing_boundary_tolerance_seconds: float = 0.5,
) -> dict[str, Any]:
    """Match each truth incident to the nearest temporally plausible detection."""

    unmatched_detection_ids = {detection.identifier for detection in detections}
    pairs: list[dict[str, Any]] = []

    for truth in ground_truth:
        truth_end = truth.ended_at or truth.started_at
        candidates = [
            detection
            for detection in detections
            if detection.identifier in unmatched_detection_ids
            and (
                truth.started_at.timestamp() - early_tolerance_seconds
                <= detection.started_at.timestamp()
                <= truth_end.timestamp() + late_tolerance_seconds
            )
        ]

        if not candidates:
            pairs.append(
                {
                    "incident_id": truth.identifier,
                    "expected_fault": truth.label,
                    "detection_id": None,
                    "predicted_fault": None,
                    "classification_correct": False,
                    "detection_latency_seconds": None,
                    "raw_detection_latency_seconds": None,
                    "timing_boundary_adjusted": False,
                    "recovery_detection_latency_seconds": None,
                }
            )
            continue

        detection = min(
            candidates,
            key=lambda item: abs(
                (item.started_at - truth.started_at).total_seconds()
            ),
        )
        unmatched_detection_ids.remove(detection.identifier)
        raw_detection_latency = (
            detection.started_at - truth.started_at
        ).total_seconds()
        timing_boundary_adjusted = (
            -timing_boundary_tolerance_seconds <= raw_detection_latency < 0
        )
        detection_latency = (
            0.0 if timing_boundary_adjusted else raw_detection_latency
        )
        recovery_latency = None
        if truth.ended_at is not None and detection.ended_at is not None:
            recovery_latency = (detection.ended_at - truth.ended_at).total_seconds()

        pairs.append(
            {
                "incident_id": truth.identifier,
                "expected_fault": truth.label,
                "detection_id": detection.identifier,
                "predicted_fault": detection.label,
                "classification_correct": detection.label == truth.label,
                "detection_latency_seconds": _round_optional(detection_latency),
                "raw_detection_latency_seconds": _round_optional(
                    raw_detection_latency
                ),
                "timing_boundary_adjusted": timing_boundary_adjusted,
                "recovery_detection_latency_seconds": _round_optional(
                    recovery_latency
                ),
            }
        )

    matched = [pair for pair in pairs if pair["detection_id"] is not None]
    correct = [pair for pair in matched if pair["classification_correct"]]
    detection_latencies = [
        pair["detection_latency_seconds"]
        for pair in matched
        if pair["detection_latency_seconds"] is not None
    ]
    recovery_latencies = [
        pair["recovery_detection_latency_seconds"]
        for pair in matched
        if pair["recovery_detection_latency_seconds"] is not None
    ]

    true_positives = len(matched)
    false_positives = len(unmatched_detection_ids)
    false_negatives = len(ground_truth) - true_positives
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = true_positives / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    detection_by_id = {
        detection.identifier: detection for detection in detections
    }
    labels = sorted(
        {incident.label for incident in ground_truth}
        | {detection.label for detection in detections}
    )
    confusion_matrix: dict[str, dict[str, int]] = {
        label: {prediction: 0 for prediction in [*labels, "MISSED"]}
        for label in labels
    }
    for pair in pairs:
        predicted = pair["predicted_fault"] or "MISSED"
        confusion_matrix[pair["expected_fault"]][predicted] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        support = sum(pair["expected_fault"] == label for pair in pairs)
        class_true_positives = sum(
            pair["expected_fault"] == label
            and pair["predicted_fault"] == label
            for pair in pairs
        )
        class_false_negatives = support - class_true_positives
        class_false_positives = sum(
            pair["expected_fault"] != label
            and pair["predicted_fault"] == label
            for pair in pairs
        ) + sum(
            detection_by_id[detection_id].label == label
            for detection_id in unmatched_detection_ids
        )
        class_precision = (
            class_true_positives
            / (class_true_positives + class_false_positives)
            if class_true_positives + class_false_positives
            else 0.0
        )
        class_recall = (
            class_true_positives
            / (class_true_positives + class_false_negatives)
            if class_true_positives + class_false_negatives
            else 0.0
        )
        class_f1 = (
            2
            * class_precision
            * class_recall
            / (class_precision + class_recall)
            if class_precision + class_recall
            else 0.0
        )
        per_class[label] = {
            "support": support,
            "true_positives": class_true_positives,
            "false_positives": class_false_positives,
            "false_negatives": class_false_negatives,
            "precision": round(class_precision, 4),
            "recall": round(class_recall, 4),
            "f1": round(class_f1, 4),
        }

    return {
        "summary": {
            "ground_truth_incidents": len(ground_truth),
            "detection_incidents": len(detections),
            "matched_incidents": true_positives,
            "correctly_classified_incidents": len(correct),
            "missed_incidents": false_negatives,
            "false_positive_detections": false_positives,
            "event_precision": round(precision, 4),
            "event_recall": round(recall, 4),
            "event_f1": round(f1, 4),
            "classification_accuracy": round(
                len(correct) / true_positives, 4
            )
            if true_positives
            else 0.0,
            "mean_detection_latency_seconds": _round_optional(
                mean(detection_latencies) if detection_latencies else None
            ),
            "median_detection_latency_seconds": _round_optional(
                median(detection_latencies) if detection_latencies else None
            ),
            "mean_recovery_detection_latency_seconds": _round_optional(
                mean(recovery_latencies) if recovery_latencies else None
            ),
            "timing_boundary_adjustments": sum(
                pair["timing_boundary_adjusted"] for pair in pairs
            ),
        },
        "matches": pairs,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
        "unmatched_detection_ids": sorted(unmatched_detection_ids),
        "configuration": {
            "early_tolerance_seconds": early_tolerance_seconds,
            "late_tolerance_seconds": late_tolerance_seconds,
            "timing_boundary_tolerance_seconds": timing_boundary_tolerance_seconds,
        },
    }


def evaluate_files(
    ground_truth_path: str | Path,
    detections_path: str | Path,
) -> dict[str, Any]:
    truth = build_ground_truth_intervals(read_jsonl(ground_truth_path))
    detections = build_detection_intervals(read_jsonl(detections_path))
    return evaluate_intervals(truth, detections)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate detector output.")
    parser.add_argument(
        "--ground-truth",
        default="data/ground_truth/incidents.jsonl",
    )
    parser.add_argument(
        "--detections",
        default="data/detections/detections.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/results/evaluation.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = evaluate_files(args.ground_truth, args.detections)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
