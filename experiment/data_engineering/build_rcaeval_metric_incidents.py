from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiment.data_engineering.build_rcaeval_incidents import (
    _iso_utc,
    _sha256,
    load_index,
    summarise_metrics,
    write_jsonl,
    write_manifest,
)
from experiment.data_engineering.incident_schema import IncidentRecord


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional data environment
        raise RuntimeError(
            "RCAEval metric-only incident building requires requirements-data.txt"
        ) from exc
    return pd


def build_metric_case_incident(
    case_dir: Path,
    metadata: dict[str, Any],
    *,
    pre_seconds: int = 60,
    post_seconds: int = 120,
) -> IncidentRecord:
    if pre_seconds <= 0 or post_seconds <= 0:
        raise ValueError("pre_seconds and post_seconds must both be positive")

    metrics_path = case_dir / "metrics.parquet"
    inject_path = case_dir / "inject_time.txt"
    for path in (metrics_path, inject_path):
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
    metrics = _pandas().read_parquet(metrics_path)
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
        modalities=["metrics"],
        log_text="",
        log_record_count=0,
        log_services=[],
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
            "metrics_sha256": _sha256(metrics_path),
            "inject_time_sha256": _sha256(inject_path),
            "pre_seconds": pre_seconds,
            "post_seconds": post_seconds,
            "logs_intentionally_not_acquired": True,
        },
    )
    record.validate()
    return record


def build_metric_records(
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
            raise ValueError(f"Case is not present in the RCAEval index: {case_dir.name}")
        records.append(
            build_metric_case_incident(
                case_dir,
                index[case_dir.name],
                pre_seconds=pre_seconds,
                post_seconds=post_seconds,
            )
        )
    incident_ids = [record.incident_id for record in records]
    if len(incident_ids) != len(set(incident_ids)):
        raise ValueError("Duplicate incident IDs would cause data leakage")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a metric-only RCAEval subset into incident JSONL."
    )
    parser.add_argument(
        "--root", type=Path, default=Path("data/external/rcaeval/re2-tt")
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/external/rcaeval/cases.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/incidents/rcaeval-re2-tt-metrics.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/incidents/rcaeval-re2-tt-metrics-manifest.json"
        ),
    )
    parser.add_argument("--pre-seconds", type=int, default=60)
    parser.add_argument("--post-seconds", type=int, default=120)
    args = parser.parse_args()

    records = build_metric_records(
        args.root,
        args.index,
        pre_seconds=args.pre_seconds,
        post_seconds=args.post_seconds,
    )
    count = write_jsonl(records, args.output)
    write_manifest(records, args.manifest, args.output)
    print(f"Wrote {count} metric-only incident records -> {args.output}")
    print(f"Wrote conversion manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
