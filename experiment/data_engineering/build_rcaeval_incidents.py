from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiment.data_engineering.incident_schema import IncidentRecord, SCHEMA_VERSION


RCA_REQUIRED_COLUMNS = {
    "case",
    "dataset",
    "system_name",
    "root_cause_service",
    "fault",
    "inject_time",
    "has_logs",
}


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional data environment
        raise RuntimeError(
            "RCAEval incident building requires requirements-data.txt"
        ) from exc
    return pd


def _iso_utc(unix_seconds: int | float) -> str:
    return datetime.fromtimestamp(float(unix_seconds), tz=UTC).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_mean(series) -> float | None:
    value = series.mean(skipna=True)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def summarise_metrics(metrics, inject_time: int, window_start: int, window_end: int):
    if "time" not in metrics.columns:
        raise ValueError("RCAEval metrics are missing the 'time' column")
    window = metrics[(metrics["time"] >= window_start) & (metrics["time"] <= window_end)]
    pre = window[window["time"] < inject_time]
    post = window[window["time"] >= inject_time]

    features: dict[str, float | None] = {}
    for column in sorted(column for column in window.columns if column != "time"):
        pre_mean = _finite_mean(pre[column])
        post_mean = _finite_mean(post[column])
        features[f"{column}__pre_mean"] = pre_mean
        features[f"{column}__post_mean"] = post_mean
        features[f"{column}__mean_delta"] = (
            post_mean - pre_mean
            if pre_mean is not None and post_mean is not None
            else None
        )
    return window, pre, post, features


def build_case_incident(
    case_dir: Path,
    metadata: dict[str, Any],
    *,
    pre_seconds: int = 60,
    post_seconds: int = 120,
) -> IncidentRecord:
    if pre_seconds <= 0 or post_seconds <= 0:
        raise ValueError("pre_seconds and post_seconds must both be positive")

    pd = _pandas()
    logs_path = case_dir / "logs.parquet"
    metrics_path = case_dir / "metrics.parquet"
    inject_path = case_dir / "inject_time.txt"
    for path in (logs_path, metrics_path, inject_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    inject_time = int(inject_path.read_text(encoding="utf-8").strip())
    indexed_inject_time = int(metadata["inject_time"])
    if inject_time != indexed_inject_time:
        raise ValueError(
            f"Injection time mismatch for {case_dir.name}: "
            f"file={inject_time}, index={indexed_inject_time}"
        )
    window_start = inject_time - pre_seconds
    window_end = inject_time + post_seconds

    logs = pd.read_parquet(logs_path)
    required_log_columns = {"timestamp", "container_name", "message"}
    missing_log_columns = required_log_columns.difference(logs.columns)
    if missing_log_columns:
        raise ValueError(
            f"RCAEval logs are missing columns: {sorted(missing_log_columns)}"
        )
    log_window = logs[
        (logs["timestamp"] >= window_start) & (logs["timestamp"] <= window_end)
    ].sort_values("timestamp", kind="stable")
    log_lines = [
        f"[{container}] {message}"
        for container, message in zip(
            log_window["container_name"].astype(str),
            log_window["message"].astype(str),
        )
    ]

    metrics = pd.read_parquet(metrics_path)
    metric_window, metric_pre, metric_post, metric_features = summarise_metrics(
        metrics, inject_time, window_start, window_end
    )

    record = IncidentRecord(
        incident_id=f"rcaeval:{case_dir.name}",
        source="rcaeval",
        dataset=str(metadata["dataset"]),
        system_name=str(metadata["system_name"]),
        task="public_root_cause_localisation",
        injected_at=_iso_utc(inject_time),
        window_start=_iso_utc(window_start),
        window_end=_iso_utc(window_end),
        root_cause_service=str(metadata["root_cause_service"]),
        original_fault_label=str(metadata["fault"]),
        local_fault_label=None,
        split_group=case_dir.name,
        modalities=["logs", "metrics"],
        log_text="\n".join(log_lines),
        log_record_count=int(len(log_window)),
        log_services=sorted(log_window["container_name"].astype(str).unique().tolist()),
        metric_row_count=int(len(metric_window)),
        metric_pre_rows=int(len(metric_pre)),
        metric_post_rows=int(len(metric_post)),
        metric_features=metric_features,
        provenance={
            "case": case_dir.name,
            "dataset": str(metadata["dataset"]),
            "root_cause_service": str(metadata["root_cause_service"]),
            "fault": str(metadata["fault"]),
            "repetition": int(metadata["repetition"]),
            "logs_sha256": _sha256(logs_path),
            "metrics_sha256": _sha256(metrics_path),
            "inject_time_sha256": _sha256(inject_path),
            "pre_seconds": pre_seconds,
            "post_seconds": post_seconds,
        },
    )
    record.validate()
    return record


def load_index(index_path: Path) -> dict[str, dict[str, Any]]:
    pd = _pandas()
    frame = pd.read_parquet(index_path)
    missing = RCA_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"RCAEval index is missing columns: {sorted(missing)}")
    if frame["case"].duplicated().any():
        duplicates = sorted(frame.loc[frame["case"].duplicated(), "case"].tolist())
        raise ValueError(f"RCAEval index contains duplicate cases: {duplicates}")
    return {
        str(row["case"]): row.to_dict()
        for _, row in frame.iterrows()
    }


def build_records(
    root: Path,
    index_path: Path,
    *,
    pre_seconds: int = 60,
    post_seconds: int = 120,
) -> list[IncidentRecord]:
    index = load_index(index_path)
    case_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not case_dirs:
        raise ValueError(f"No RCAEval case directories found under {root}")

    records: list[IncidentRecord] = []
    for case_dir in case_dirs:
        if case_dir.name not in index:
            raise ValueError(
                f"Case is not present in the RCAEval index: {case_dir.name}"
            )
        records.append(
            build_case_incident(
                case_dir,
                index[case_dir.name],
                pre_seconds=pre_seconds,
                post_seconds=post_seconds,
            )
        )

    incident_ids = [record.incident_id for record in records]
    split_groups = [record.split_group for record in records]
    if len(incident_ids) != len(set(incident_ids)):
        raise ValueError("Duplicate incident IDs would cause data leakage")
    if len(split_groups) != len(set(split_groups)):
        raise ValueError("Duplicate split groups would cause data leakage")
    return records


def build_pilot_records(
    pilot_root: Path,
    index_path: Path,
    *,
    pre_seconds: int = 60,
    post_seconds: int = 120,
) -> list[IncidentRecord]:
    """Backward-compatible wrapper for the original six-case pilot."""

    return build_records(
        pilot_root,
        index_path,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
    )


def write_jsonl(records: Iterable[IncidentRecord], output_path: Path) -> int:
    materialised = list(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in materialised:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return len(materialised)


def write_manifest(
    records: list[IncidentRecord], output_path: Path, jsonl_path: Path
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": "rcaeval",
        "task": "public_root_cause_localisation",
        "records": len(records),
        "incident_ids": [record.incident_id for record in records],
        "datasets": sorted({record.dataset for record in records}),
        "root_cause_services": sorted(
            {record.root_cause_service for record in records}
        ),
        "root_cause_service_counts": dict(
            sorted(
                {
                    service: sum(
                        record.root_cause_service == service for record in records
                    )
                    for service in {record.root_cause_service for record in records}
                }.items()
            )
        ),
        "original_fault_labels": sorted(
            {record.original_fault_label for record in records}
        ),
        "original_fault_label_counts": dict(
            sorted(
                {
                    fault: sum(
                        record.original_fault_label == fault for record in records
                    )
                    for fault in {record.original_fault_label for record in records}
                }.items()
            )
        ),
        "repetitions": sorted(
            {int(record.provenance["repetition"]) for record in records}
        ),
        "local_fault_labels": sorted(
            {record.local_fault_label for record in records if record.local_fault_label}
        ),
        "jsonl": str(jsonl_path),
        "jsonl_sha256": _sha256(jsonl_path),
        "leakage_checks": {
            "unique_incident_ids": True,
            "unique_split_groups": True,
            "public_labels_not_mapped_to_local_classes": all(
                record.local_fault_label is None for record in records
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an RCAEval case subset into one JSONL row per incident."
    )
    parser.add_argument(
        "--root",
        "--pilot-root",
        dest="root",
        type=Path,
        default=Path("data/external/rcaeval/pilot"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/external/rcaeval/cases.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/incidents/rcaeval-pilot.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/incidents/rcaeval-pilot-manifest.json"),
    )
    parser.add_argument("--pre-seconds", type=int, default=60)
    parser.add_argument("--post-seconds", type=int, default=120)
    args = parser.parse_args()

    records = build_records(
        args.root,
        args.index,
        pre_seconds=args.pre_seconds,
        post_seconds=args.post_seconds,
    )
    count = write_jsonl(records, args.output)
    write_manifest(records, args.manifest, args.output)
    print(f"Wrote {count} incident records -> {args.output}")
    print(f"Wrote conversion manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
