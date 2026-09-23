from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from experiment.data_engineering.build_local_incidents import (
    build_local_records,
    write_jsonl as write_incident_jsonl,
    write_manifest as write_incident_manifest,
)
from experiment.evaluate import (
    build_ground_truth_intervals,
    evaluate_files,
    read_jsonl,
)
from experiment.mixed_runner import SUPPORTED_FAULTS, run_mixed_batch


@dataclass(frozen=True)
class CampaignConfig:
    source_path: Path
    payload: dict[str, Any]

    @property
    def campaign_id(self) -> str:
        return str(self.payload["campaign_id"])

    @property
    def expected_incidents(self) -> int:
        return int(self.payload["acceptance"]["expected_incidents"])

    @property
    def repetitions_per_fault(self) -> int:
        return int(self.payload["repetitions_per_fault"])

    @property
    def paths(self) -> dict[str, Path]:
        return {name: Path(value) for name, value in self.payload["paths"].items()}

    @property
    def telemetry(self) -> dict[str, Any]:
        return dict(self.payload["telemetry"])

    def validate(self) -> None:
        if int(self.payload.get("schema_version", 0)) != 1:
            raise ValueError("Campaign schema_version must be 1")
        if not self.campaign_id:
            raise ValueError("campaign_id cannot be empty")
        if self.repetitions_per_fault < 1:
            raise ValueError("repetitions_per_fault must be at least 1")
        calculated = self.repetitions_per_fault * len(SUPPORTED_FAULTS)
        if self.expected_incidents != calculated:
            raise ValueError(
                f"Expected incident count {self.expected_incidents} does not match "
                f"the balanced schedule size {calculated}"
            )
        expected_per_fault = int(
            self.payload["acceptance"]["expected_incidents_per_fault"]
        )
        if expected_per_fault != self.repetitions_per_fault:
            raise ValueError("Per-fault acceptance count differs from repetitions")
        if int(self.payload["fault_duration_seconds"]) < 1:
            raise ValueError("fault_duration_seconds must be positive")
        if float(self.payload["cooldown_seconds"]) < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        telemetry = self.telemetry
        for name in ("pre_seconds", "post_seconds"):
            if float(telemetry[name]) < 0:
                raise ValueError(f"telemetry {name} cannot be negative")
        if float(telemetry["interval_seconds"]) <= 0:
            raise ValueError("telemetry interval_seconds must be positive")
        if float(telemetry["request_timeout_seconds"]) <= 0:
            raise ValueError("telemetry request_timeout_seconds must be positive")
        if float(telemetry["request_timeout_seconds"]) > float(
            telemetry["interval_seconds"]
        ):
            raise ValueError("Probe timeout cannot exceed the sampling interval")
        acceptance = self.payload["acceptance"]
        duration_limit = acceptance.get("maximum_fault_duration_seconds")
        if duration_limit is not None and float(duration_limit) < int(
            self.payload["fault_duration_seconds"]
        ):
            raise ValueError("Maximum accepted fault duration is below the target")
        latency_limit = acceptance.get("maximum_metric_latency_ms")
        if latency_limit is not None and float(latency_limit) <= 0:
            raise ValueError("Maximum metric latency must be positive")
        required_paths = {
            "ground_truth",
            "batch_result",
            "evaluation",
            "campaign_summary",
            "detection_source",
            "detection_archive_dir",
            "processed_incidents",
            "processed_manifest",
        }
        missing = required_paths.difference(self.paths)
        if missing:
            raise ValueError(f"Campaign paths are missing: {sorted(missing)}")


def load_config(path: str | Path) -> CampaignConfig:
    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    config = CampaignConfig(source_path=source_path, payload=payload)
    config.validate()
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def campaign_plan(config: CampaignConfig) -> dict[str, Any]:
    telemetry = config.telemetry
    per_incident_seconds = (
        float(telemetry["pre_seconds"])
        + float(config.payload["fault_duration_seconds"])
        + float(telemetry["post_seconds"])
        + float(config.payload["cooldown_seconds"])
    )
    estimated_seconds = (
        per_incident_seconds * config.expected_incidents
        - float(config.payload["cooldown_seconds"])
    )
    return {
        "campaign_id": config.campaign_id,
        "config_sha256": sha256_file(config.source_path),
        "faults": list(SUPPORTED_FAULTS),
        "random_seed": int(config.payload["random_seed"]),
        "expected_incidents": config.expected_incidents,
        "expected_incidents_per_fault": config.repetitions_per_fault,
        "estimated_minimum_runtime_minutes": round(estimated_seconds / 60, 1),
        "fault_duration_seconds": int(config.payload["fault_duration_seconds"]),
        "cooldown_seconds": float(config.payload["cooldown_seconds"]),
        "telemetry": telemetry,
        "paths": {name: str(path) for name, path in config.paths.items()},
    }


def preflight(config: CampaignConfig) -> dict[str, Any]:
    paths = config.paths
    outputs = [
        paths["ground_truth"],
        paths["batch_result"],
        paths["evaluation"],
        paths["campaign_summary"],
        paths["processed_incidents"],
        paths["processed_manifest"],
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Campaign outputs already exist; refusing to overwrite: "
            + ", ".join(existing)
        )
    detection_source = paths["detection_source"]
    if not detection_source.is_file():
        raise FileNotFoundError(detection_source)
    free_bytes = shutil.disk_usage(Path.cwd()).free
    minimum_free = int(config.payload["acceptance"]["minimum_free_disk_bytes"])
    if free_bytes < minimum_free:
        raise RuntimeError(
            f"Insufficient free disk space: {free_bytes} < {minimum_free}"
        )
    return {
        "outputs_absent": True,
        "detection_source_exists": True,
        "free_disk_bytes": free_bytes,
        "minimum_free_disk_bytes": minimum_free,
    }


def _progress(event: dict[str, object]) -> None:
    print(
        "PROGRESS "
        + json.dumps(event, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


def validate_fault_durations(config: CampaignConfig) -> dict[str, Any]:
    limit = config.payload["acceptance"].get("maximum_fault_duration_seconds")
    ground_truth_records = read_jsonl(config.paths["ground_truth"])
    recovery_failures = [
        str(record["incident_id"])
        for record in ground_truth_records
        if record.get("event") == "recovery_failed"
    ]
    if (
        config.payload["acceptance"].get("require_complete_recovery")
        and recovery_failures
    ):
        raise RuntimeError(f"Recovery acceptance failed: {recovery_failures}")
    intervals = build_ground_truth_intervals(ground_truth_records)
    durations = {}
    for incident in intervals:
        if incident.ended_at is None:
            raise RuntimeError(f"Ground truth has no end event: {incident.identifier}")
        durations[incident.identifier] = (
            incident.ended_at - incident.started_at
        ).total_seconds()
    result = {
        "incidents": len(durations),
        "minimum_seconds": round(min(durations.values()), 3),
        "maximum_seconds": round(max(durations.values()), 3),
        "limit_seconds": float(limit) if limit is not None else None,
        "recovery_failures": len(recovery_failures),
    }
    if limit is not None:
        invalid = {
            name: round(value, 3)
            for name, value in durations.items()
            if value > float(limit)
        }
        if invalid:
            raise RuntimeError(f"Fault duration acceptance failed: {invalid}")
    return result


def validate_metric_latencies(
    config: CampaignConfig,
    records: Sequence[Any],
) -> dict[str, Any]:
    limit = config.payload["acceptance"].get("maximum_metric_latency_ms")
    observed = [
        float(value)
        for record in records
        for name, value in record.metric_features.items()
        if "latency_ms" in name and value is not None
    ]
    result = {
        "features_checked": len(observed),
        "maximum_ms": round(max(observed), 3),
        "limit_ms": float(limit) if limit is not None else None,
    }
    if limit is not None and max(observed) > float(limit):
        raise RuntimeError(
            f"Metric latency acceptance failed: {max(observed):.3f} ms "
            f"> {float(limit):.3f} ms"
        )
    return result


def validate_detection_pairs(
    config: CampaignConfig,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    incomplete = [
        str(match["incident_id"])
        for match in evaluation["matches"]
        if match["detection_id"] is None
        or match["recovery_detection_latency_seconds"] is None
    ]
    false_positives = int(
        evaluation["summary"]["false_positive_detections"]
    )
    if (
        config.payload["acceptance"].get("require_complete_detection_pairs")
        and (incomplete or false_positives)
    ):
        raise RuntimeError(
            "Detection-pair acceptance failed: "
            f"incomplete={incomplete}, false_positives={false_positives}"
        )
    return {
        "incomplete_pairs": len(incomplete),
        "false_positive_detections": false_positives,
    }


def run_campaign(config: CampaignConfig) -> dict[str, Any]:
    checks = preflight(config)
    paths = config.paths
    telemetry = config.telemetry
    plan = campaign_plan(config)
    print("CAMPAIGN " + json.dumps(plan, ensure_ascii=False), flush=True)

    batch_result = run_mixed_batch(
        repetitions_per_fault=config.repetitions_per_fault,
        seed=int(config.payload["random_seed"]),
        duration_seconds=int(config.payload["fault_duration_seconds"]),
        cooldown_seconds=float(config.payload["cooldown_seconds"]),
        base_url=str(config.payload["base_url"]),
        ground_truth_path=paths["ground_truth"],
        result_path=paths["batch_result"],
        telemetry_root=Path(str(telemetry["root"])),
        telemetry_pre_seconds=float(telemetry["pre_seconds"]),
        telemetry_post_seconds=float(telemetry["post_seconds"]),
        telemetry_interval_seconds=float(telemetry["interval_seconds"]),
        telemetry_request_timeout_seconds=float(
            telemetry["request_timeout_seconds"]
        ),
        detection_source_path=paths["detection_source"],
        detection_archive_dir=paths["detection_archive_dir"],
        progress_callback=_progress,
    )

    completed = len(batch_result["trials"])
    if completed != config.expected_incidents:
        raise RuntimeError(
            f"Campaign completed {completed} incidents; expected {config.expected_incidents}"
        )
    schedule_counts = Counter(batch_result["schedule"])
    expected_schedule = {
        fault: config.repetitions_per_fault for fault in SUPPORTED_FAULTS
    }
    if dict(schedule_counts) != expected_schedule:
        raise RuntimeError(f"Campaign schedule is not balanced: {schedule_counts}")
    duration_quality = validate_fault_durations(config)

    detection_archive = Path(batch_result["detection_archive"]["path"])
    evaluation = evaluate_files(paths["ground_truth"], detection_archive)
    detection_pair_quality = validate_detection_pairs(config, evaluation)
    paths["evaluation"].parent.mkdir(parents=True, exist_ok=True)
    paths["evaluation"].write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    run_id = str(batch_result["telemetry"]["run_id"])
    records = build_local_records(
        raw_root=Path(str(telemetry["root"])),
        ground_truth_paths=[paths["ground_truth"]],
        detection_archive_dir=paths["detection_archive_dir"],
        selected_run_ids={run_id},
        require_detection_archive=True,
    )
    label_counts = Counter(record.local_fault_label for record in records)
    expected_labels = {
        "database_connection_failure": config.repetitions_per_fault,
        "service_stopped": config.repetitions_per_fault,
        "http_500_failure": config.repetitions_per_fault,
    }
    if dict(label_counts) != expected_labels:
        raise RuntimeError(f"Converted label counts are not balanced: {label_counts}")
    metric_latency_quality = validate_metric_latencies(config, records)
    write_incident_jsonl(records, paths["processed_incidents"])
    write_incident_manifest(
        records, paths["processed_manifest"], paths["processed_incidents"]
    )

    summary = {
        "schema_version": 1,
        "campaign_id": config.campaign_id,
        "config_path": str(config.source_path),
        "config_sha256": sha256_file(config.source_path),
        "preflight": checks,
        "telemetry_run_id": run_id,
        "incidents_completed": completed,
        "fault_counts": dict(sorted(label_counts.items())),
        "fault_duration_quality": duration_quality,
        "metric_latency_quality": metric_latency_quality,
        "detection_pair_quality": detection_pair_quality,
        "detection_archive": str(detection_archive),
        "evaluation_summary": evaluation["summary"],
        "processed_incidents": str(paths["processed_incidents"]),
        "processed_manifest": str(paths["processed_manifest"]),
    }
    paths["campaign_summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["campaign_summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a frozen local fault-data campaign end to end."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiment/configs/local-campaign-v1.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        output = {"plan": campaign_plan(config), "preflight": preflight(config)}
    else:
        output = run_campaign(config)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
