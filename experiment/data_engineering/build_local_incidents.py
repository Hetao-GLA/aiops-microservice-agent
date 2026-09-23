from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from pathlib import Path
from statistics import mean
from typing import Any

from experiment.data_engineering.incident_schema import IncidentRecord, SCHEMA_VERSION


ROOT_CAUSE_SERVICES = {
    "postgres_container_stopped": "database",
    "api_container_stopped": "order-service",
    "forced_application_error_mode": "order-service",
}


def _parse_timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError(f"Timestamp must include a timezone: {value!r}")
    return timestamp


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


@dataclass(frozen=True)
class GroundTruthPair:
    started: dict[str, Any]
    ended: dict[str, Any]
    source_path: Path


def load_ground_truth(paths: Iterable[Path]) -> dict[str, GroundTruthPair]:
    grouped: dict[str, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    materialised = list(paths)
    if not materialised:
        raise ValueError("At least one Ground Truth path is required")
    for path in materialised:
        for record in read_jsonl(path):
            grouped[str(record["incident_id"])].append((record, path))

    pairs: dict[str, GroundTruthPair] = {}
    for incident_id, entries in grouped.items():
        starts = [(record, path) for record, path in entries if record["event"] == "fault_started"]
        ends = [
            (record, path)
            for record, path in entries
            if record["event"] in {"fault_ended", "recovery_failed"}
        ]
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError(
                f"Ground Truth {incident_id} must have exactly one start and one end"
            )
        if starts[0][1] != ends[0][1]:
            raise ValueError(f"Ground Truth pair spans multiple files: {incident_id}")
        pairs[incident_id] = GroundTruthPair(
            started=starts[0][0], ended=ends[0][0], source_path=starts[0][1]
        )
    return pairs


def _flatten_probe_values(sample: dict[str, Any]) -> dict[str, float]:
    prefix = str(sample["probe"])
    values: dict[str, float] = {
        f"{prefix}__available": 1.0 if sample.get("available") else 0.0,
    }
    for name in ("status_code", "latency_ms"):
        value = sample.get(name)
        if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)):
            values[f"{prefix}__{name}"] = float(value)
    payload = sample.get("payload")
    if isinstance(payload, dict):
        for name, value in payload.items():
            if isinstance(value, bool):
                values[f"{prefix}__payload__{name}"] = float(value)
            elif isinstance(value, Real) and math.isfinite(float(value)):
                values[f"{prefix}__payload__{name}"] = float(value)
    return values


def summarise_probes(
    probes: list[dict[str, Any]], injected_at: datetime
) -> tuple[int, int, dict[str, float | None]]:
    pre_values: dict[str, list[float]] = defaultdict(list)
    post_values: dict[str, list[float]] = defaultdict(list)
    pre_rows = 0
    post_rows = 0
    for sample in probes:
        target = pre_values if _parse_timestamp(str(sample["timestamp"])) < injected_at else post_values
        if target is pre_values:
            pre_rows += 1
        else:
            post_rows += 1
        for key, value in _flatten_probe_values(sample).items():
            target[key].append(value)

    features: dict[str, float | None] = {}
    for key in sorted(set(pre_values) | set(post_values)):
        pre_mean = mean(pre_values[key]) if pre_values.get(key) else None
        post_mean = mean(post_values[key]) if post_values.get(key) else None
        features[f"{key}__pre_mean"] = pre_mean
        features[f"{key}__post_mean"] = post_mean
        features[f"{key}__mean_delta"] = (
            post_mean - pre_mean
            if pre_mean is not None and post_mean is not None
            else None
        )
    return pre_rows, post_rows, features


def validate_detection_archive(path: Path) -> tuple[int, int]:
    records = read_jsonl(path)
    starts: Counter[str] = Counter()
    ends: Counter[str] = Counter()
    for record in records:
        detection_id = str(record["detection_id"])
        if record["event"] == "detection_started":
            starts[detection_id] += 1
        elif record["event"] == "detection_ended":
            ends[detection_id] += 1
    if starts != ends or any(count != 1 for count in starts.values()):
        raise ValueError(f"Detection archive has incomplete or duplicate pairs: {path}")
    return len(records), len(starts)


def build_local_incident(
    incident_dir: Path,
    truth: GroundTruthPair,
    *,
    detection_archive_dir: Path,
    require_detection_archive: bool = True,
) -> IncidentRecord:
    metadata_path = incident_dir / "metadata.json"
    probes_path = incident_dir / "probes.jsonl"
    logs_path = incident_dir / "docker-logs.jsonl"
    for path in (metadata_path, probes_path, logs_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    incident_id = incident_dir.name
    if metadata.get("incident_id") != incident_id:
        raise ValueError(f"Incident directory and metadata ID differ: {incident_dir}")
    run_id = str(metadata["run_id"])
    if incident_dir.parent.name != run_id:
        raise ValueError(f"Run directory and metadata run ID differ: {incident_dir}")
    if metadata.get("collector_error") is not None:
        raise ValueError(f"Collector error recorded for {incident_id}")
    if metadata.get("docker_capture_errors"):
        raise ValueError(f"Docker capture errors recorded for {incident_id}")

    started = truth.started
    ended = truth.ended
    if ended["event"] != "fault_ended" or not ended.get("recovery_succeeded", False):
        raise ValueError(f"Incident did not recover successfully: {incident_id}")
    fault_type = str(started["fault_type"])
    if metadata.get("expected_fault") != fault_type:
        raise ValueError(f"Telemetry and Ground Truth labels differ: {incident_id}")
    if ended.get("fault_type") != fault_type:
        raise ValueError(f"Ground Truth start/end labels differ: {incident_id}")

    injected_at = _parse_timestamp(str(started["timestamp"]))
    fault_ended_at = _parse_timestamp(str(ended["timestamp"]))
    window_start = _parse_timestamp(str(metadata["collector_started_at"]))
    window_end = _parse_timestamp(str(metadata["collector_ended_at"]))
    if not window_start <= injected_at <= fault_ended_at <= window_end:
        raise ValueError(f"Ground Truth falls outside telemetry window: {incident_id}")

    probes = read_jsonl(probes_path)
    logs = read_jsonl(logs_path)
    if len(probes) != int(metadata["probe_sample_count"]):
        raise ValueError(f"Probe row count does not match metadata: {incident_id}")
    if len(logs) != int(metadata["docker_log_record_count"]):
        raise ValueError(f"Docker log row count does not match metadata: {incident_id}")
    if not probes or not logs:
        raise ValueError(f"Local incident has empty telemetry: {incident_id}")

    pre_rows, post_rows, metric_features = summarise_probes(probes, injected_at)
    log_text = "\n".join(
        f"[{record['service']}] {record['message']}" for record in logs
    )
    root_cause = str(started["root_cause"])
    detection_path = detection_archive_dir / f"{run_id}.jsonl"
    detection_records = 0
    detection_pairs = 0
    if detection_path.is_file():
        detection_records, detection_pairs = validate_detection_archive(detection_path)
    elif require_detection_archive:
        raise FileNotFoundError(detection_path)

    record = IncidentRecord(
        incident_id=f"local:{incident_id}",
        source="local",
        dataset=run_id,
        system_name="local-order-platform",
        task="local_fault_classification",
        injected_at=str(started["timestamp"]),
        window_start=str(metadata["collector_started_at"]),
        window_end=str(metadata["collector_ended_at"]),
        root_cause_service=ROOT_CAUSE_SERVICES.get(
            root_cause, str(started["service"])
        ),
        original_fault_label=fault_type,
        local_fault_label=fault_type,
        split_group=incident_id,
        modalities=["logs", "metrics"],
        log_text=log_text,
        log_record_count=len(logs),
        log_services=sorted({str(record["service"]) for record in logs}),
        metric_row_count=len(probes),
        metric_pre_rows=pre_rows,
        metric_post_rows=post_rows,
        metric_features=metric_features,
        provenance={
            "run_id": run_id,
            "raw_incident_directory": str(incident_dir),
            "metadata_sha256": _sha256(metadata_path),
            "probes_sha256": _sha256(probes_path),
            "docker_logs_sha256": _sha256(logs_path),
            "ground_truth_path": str(truth.source_path),
            "ground_truth_sha256": _sha256(truth.source_path),
            "original_root_cause": root_cause,
            "fault_ended_at": str(ended["timestamp"]),
            "detection_archive_path": str(detection_path) if detection_path.is_file() else None,
            "detection_archive_sha256": _sha256(detection_path) if detection_path.is_file() else None,
            "detection_records": detection_records,
            "detection_pairs": detection_pairs,
            "telemetry_pre_seconds": metadata.get("pre_seconds"),
            "telemetry_post_seconds": metadata.get("post_seconds"),
            "telemetry_interval_seconds": metadata.get("interval_seconds"),
            "telemetry_request_timeout_seconds": metadata.get(
                "request_timeout_seconds"
            ),
        },
    )
    record.validate()
    return record


def discover_incident_directories(
    raw_root: Path, selected_run_ids: set[str] | None = None
) -> list[Path]:
    directories: list[Path] = []
    for run_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        if selected_run_ids is not None and run_dir.name not in selected_run_ids:
            continue
        directories.extend(
            sorted(
                path
                for path in run_dir.iterdir()
                if path.is_dir() and path.name.startswith("INC-")
            )
        )
    if not directories:
        raise ValueError("No local incident directories matched the selection")
    return directories


def build_local_records(
    *,
    raw_root: Path,
    ground_truth_paths: Iterable[Path],
    detection_archive_dir: Path,
    selected_run_ids: set[str] | None = None,
    require_detection_archive: bool = True,
) -> list[IncidentRecord]:
    truth = load_ground_truth(ground_truth_paths)
    records: list[IncidentRecord] = []
    for incident_dir in discover_incident_directories(raw_root, selected_run_ids):
        if incident_dir.name not in truth:
            raise ValueError(f"No Ground Truth found for {incident_dir.name}")
        records.append(
            build_local_incident(
                incident_dir,
                truth[incident_dir.name],
                detection_archive_dir=detection_archive_dir,
                require_detection_archive=require_detection_archive,
            )
        )
    incident_ids = [record.incident_id for record in records]
    split_groups = [record.split_group for record in records]
    if len(incident_ids) != len(set(incident_ids)):
        raise ValueError("Duplicate local incident IDs would cause data leakage")
    if len(split_groups) != len(set(split_groups)):
        raise ValueError("Duplicate local split groups would cause data leakage")
    return records


def write_jsonl(records: list[IncidentRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def write_manifest(
    records: list[IncidentRecord], output_path: Path, jsonl_path: Path
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": "local",
        "task": "local_fault_classification",
        "records": len(records),
        "incident_ids": [record.incident_id for record in records],
        "run_ids": sorted({record.dataset for record in records}),
        "fault_label_counts": dict(
            sorted(Counter(record.local_fault_label for record in records).items())
        ),
        "total_log_records": sum(record.log_record_count for record in records),
        "total_probe_records": sum(record.metric_row_count for record in records),
        "metric_feature_union_count": len(
            {key for record in records for key in record.metric_features}
        ),
        "jsonl": str(jsonl_path),
        "jsonl_sha256": _sha256(jsonl_path),
        "quality_checks": {
            "unique_incident_ids": True,
            "unique_split_groups": True,
            "ground_truth_pairs_complete": True,
            "telemetry_counts_match_metadata": True,
            "collector_errors_absent": True,
            "detection_archives_complete": all(
                int(record.provenance["detection_pairs"]) >= 1 for record in records
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert local raw telemetry into incident schema v1."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/local"))
    parser.add_argument(
        "--ground-truth",
        action="append",
        type=Path,
        default=None,
        help="Ground Truth JSONL path; repeat for multiple files.",
    )
    parser.add_argument(
        "--ground-truth-glob", default="data/ground_truth/*.jsonl"
    )
    parser.add_argument(
        "--detection-archive-dir", type=Path, default=Path("data/detections")
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=None,
        help="Include only this telemetry run ID; repeat for multiple runs.",
    )
    parser.add_argument(
        "--allow-missing-detection-archive", action="store_true"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/incidents/local-incidents.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/incidents/local-incidents-manifest.json"),
    )
    args = parser.parse_args()

    ground_truth_paths = (
        args.ground_truth
        if args.ground_truth is not None
        else sorted(Path().glob(args.ground_truth_glob))
    )
    records = build_local_records(
        raw_root=args.raw_root,
        ground_truth_paths=ground_truth_paths,
        detection_archive_dir=args.detection_archive_dir,
        selected_run_ids=set(args.run_id) if args.run_id else None,
        require_detection_archive=not args.allow_missing_detection_archive,
    )
    write_jsonl(records, args.output)
    write_manifest(records, args.manifest, args.output)
    print(f"Wrote {len(records)} local incident records -> {args.output}")
    print(f"Wrote conversion manifest -> {args.manifest}")


if __name__ == "__main__":
    main()

