"""Collect and evaluate the post-freeze v4 robust-fusion confirmation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiment.campaign_runner import (
    CampaignConfig,
    load_config,
    sha256_file,
    validate_metric_latencies,
)
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
from experiment.frozen_holdout import verify_bundle, write_json
from experiment.holdout_campaign import holdout_plan, run_holdout_campaign
from experiment.ml_baseline import LABELS
from experiment.robust_fusion_v2 import (
    MODEL_FILENAME,
    evaluate_candidate,
    verify_candidate_bundle,
)


DEFAULT_MASTER = Path("experiment/configs/local-campaign-v4-blocks.json")
ORCHESTRATOR_PATH = Path("experiment/v4_confirmation_campaign.py")
FAULT_LABELS = set(LABELS)
MODEL_KEYS = ("gated_robust_fusion_v2", "logs_only")


def load_master(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported v4 master schema")
    if int(payload.get("blocks", 0)) < 1:
        raise ValueError("v4 blocks must be positive")
    if int(payload.get("repetitions_per_fault_per_block", 0)) != 1:
        raise ValueError("Each v4 block must contain one incident per fault")
    if int(payload["fault_duration_seconds"]) != 8:
        raise ValueError("v4 changes workload only; fault duration must remain 8 seconds")
    holdout = payload.get("holdout", {})
    if not holdout.get("candidate_bundle"):
        raise ValueError("v4 candidate bundle is required")
    if float(holdout["experimental_interval_seconds"]) == float(
        holdout["baseline_interval_seconds"]
    ):
        raise ValueError("v4 must specify a workload shift")
    if float(payload["acceptance"]["maximum_fault_duration_seconds"]) < 8:
        raise ValueError("Fault-duration acceptance limit is below the target")
    if float(payload["acceptance"]["maximum_metric_latency_ms"]) <= 0:
        raise ValueError("Metric-latency acceptance limit must be positive")
    required_outputs = {
        "ground_truth",
        "detections",
        "evaluation",
        "summary",
        "processed_incidents",
        "processed_manifest",
        "model_result",
        "model_figure",
    }
    if set(payload["aggregate_paths"]) != required_outputs:
        raise ValueError("v4 aggregate paths are incomplete")
    return payload


def block_config_path(block: int) -> Path:
    return Path(
        f"data/results/local-campaign-v4-blocks/config-block-{block:02d}.json"
    )


def block_payload(
    master_path: Path, master: dict[str, Any], block: int
) -> dict[str, Any]:
    total = int(master["blocks"])
    if block < 1 or block > total:
        raise ValueError(f"Block must be in 1..{total}")
    name = f"{master['campaign_id']}-block-{block:02d}"
    holdout = dict(master["holdout"])
    holdout["protocol_output"] = f"data/results/{name}-protocol.json"
    holdout["additional_code_paths"] = [str(ORCHESTRATOR_PATH)]
    return {
        "schema_version": 1,
        "campaign_id": name,
        "description": (
            f"v4 robust-fusion confirmation block {block}/{total}; "
            "one incident per known fault and no model inspection."
        ),
        "random_seed": int(master["base_seed"]) + block - 1,
        "repetitions_per_fault": 1,
        "fault_duration_seconds": int(master["fault_duration_seconds"]),
        "cooldown_seconds": float(master["cooldown_seconds"]),
        "base_url": master["base_url"],
        "telemetry": dict(master["telemetry"]),
        "paths": {
            "ground_truth": f"data/ground_truth/{name}.jsonl",
            "batch_result": f"data/results/{name}.json",
            "evaluation": f"data/results/evaluation-{name}.json",
            "campaign_summary": f"data/results/{name}-summary.json",
            "detection_source": "data/detections/detections.jsonl",
            "detection_archive_dir": "data/detections",
            "processed_incidents": f"data/processed/incidents/{name}.jsonl",
            "processed_manifest": f"data/processed/incidents/{name}-manifest.json",
        },
        "acceptance": {
            "expected_incidents": 3,
            "expected_incidents_per_fault": 1,
            "require_complete_recovery": True,
            "require_complete_detection_pairs": True,
            "minimum_free_disk_bytes": int(
                master["acceptance"]["minimum_free_disk_bytes"]
            ),
            "maximum_fault_duration_seconds": float(
                master["acceptance"]["maximum_fault_duration_seconds"]
            ),
            "maximum_metric_latency_ms": float(
                master["acceptance"]["maximum_metric_latency_ms"]
            ),
        },
        "holdout": holdout,
        "block_metadata": {
            "block": block,
            "total_blocks": total,
            "master_path": str(master_path),
            "master_sha256": sha256_file(master_path),
            "independent_variable": "workload_interval_seconds",
            "controlled_fault_duration_seconds": 8,
            "candidate_model_locked_before_collection": True,
        },
    }


def prepare_block(master_path: Path, master: dict[str, Any], block: int) -> Path:
    path = block_config_path(block)
    expected = block_payload(master_path, master, block)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError(f"Prepared block config changed: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, expected)
    return path


def prepare_all(master_path: Path, master: dict[str, Any]) -> list[Path]:
    return [
        prepare_block(master_path, master, block)
        for block in range(1, int(master["blocks"]) + 1)
    ]


def read_protocol(config: CampaignConfig) -> dict[str, Any] | None:
    path = Path(config.payload["holdout"]["protocol_output"])
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def confirmation_plan(config: CampaignConfig) -> dict[str, Any]:
    candidate_bundle = Path(config.payload["holdout"]["candidate_bundle"])
    candidate = verify_candidate_bundle(candidate_bundle)
    plan = holdout_plan(config)
    plan["candidate_freeze"] = {
        "bundle": str(candidate_bundle),
        "model_sha256": candidate["model_sha256"],
        "manifest_sha256": sha256_file(candidate_bundle / "manifest.json"),
        "frozen_at": candidate["created_at"],
        "verified_before_collection": True,
    }
    return plan


def run_confirmation_block(config: CampaignConfig) -> dict[str, Any]:
    candidate_bundle = Path(config.payload["holdout"]["candidate_bundle"])
    candidate = verify_candidate_bundle(candidate_bundle)
    before_hash = sha256_file(candidate_bundle / MODEL_FILENAME)
    verified_at = datetime.now(UTC).isoformat()
    protocol_path = Path(config.payload["holdout"]["protocol_output"])
    try:
        run_holdout_campaign(config)
    finally:
        if protocol_path.exists():
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            after = verify_candidate_bundle(candidate_bundle)
            after_hash = sha256_file(candidate_bundle / MODEL_FILENAME)
            protocol["candidate_freeze"] = {
                "bundle": str(candidate_bundle),
                "model_sha256": candidate["model_sha256"],
                "manifest_sha256": sha256_file(candidate_bundle / "manifest.json"),
                "frozen_at": candidate["created_at"],
                "verified_before_collection_at": verified_at,
                "verified_after_collection_at": datetime.now(UTC).isoformat(),
                "model_sha256_after_collection": after["model_sha256"],
                "model_artifact_unchanged": (
                    before_hash == after_hash == candidate["model_sha256"]
                ),
            }
            write_json(protocol_path, protocol)
    protocol = read_protocol(config)
    if protocol is None:
        raise RuntimeError("Confirmation protocol was not created")
    return protocol


def validate_completed_block(
    master: dict[str, Any],
    config: CampaignConfig,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    if protocol.get("status") != "collected":
        raise RuntimeError(f"{config.campaign_id} is not collected")
    if not protocol.get("workload_restored"):
        raise RuntimeError(f"{config.campaign_id} did not restore workload")
    if not protocol.get("post_run_health_check_passed"):
        raise RuntimeError(f"{config.campaign_id} failed its post-run health check")
    if protocol["campaign"]["config_sha256"] != sha256_file(config.source_path):
        raise RuntimeError(f"{config.campaign_id} config hash changed")
    expected_code_hash = sha256_file(ORCHESTRATOR_PATH)
    if (
        protocol["code_and_config_sha256"].get(str(ORCHESTRATOR_PATH))
        != expected_code_hash
    ):
        raise RuntimeError(f"{config.campaign_id} orchestrator hash changed")
    candidate = verify_candidate_bundle(
        Path(master["holdout"]["candidate_bundle"])
    )
    candidate_record = protocol.get("candidate_freeze", {})
    if candidate_record.get("model_sha256") != candidate["model_sha256"]:
        raise RuntimeError(f"{config.campaign_id} candidate model changed")
    if not candidate_record.get("model_artifact_unchanged"):
        raise RuntimeError(
            f"{config.campaign_id} candidate was not stable through collection"
        )
    summary = protocol["campaign_summary"]
    if summary["incidents_completed"] != 3:
        raise RuntimeError(f"{config.campaign_id} did not collect three incidents")
    if set(summary["fault_counts"]) != FAULT_LABELS:
        raise RuntimeError(f"{config.campaign_id} fault classes are incomplete")
    if any(value != 1 for value in summary["fault_counts"].values()):
        raise RuntimeError(f"{config.campaign_id} is not balanced")
    duration = summary["fault_duration_quality"]
    latency = summary["metric_latency_quality"]
    detection_pairs = summary["detection_pair_quality"]
    if duration["maximum_seconds"] > float(
        master["acceptance"]["maximum_fault_duration_seconds"]
    ):
        raise RuntimeError(f"{config.campaign_id} exceeded duration acceptance")
    if latency["maximum_ms"] > float(
        master["acceptance"]["maximum_metric_latency_ms"]
    ):
        raise RuntimeError(f"{config.campaign_id} exceeded latency acceptance")
    if detection_pairs["incomplete_pairs"] or detection_pairs[
        "false_positive_detections"
    ]:
        raise RuntimeError(f"{config.campaign_id} has incomplete detection pairs")
    return summary


def campaign_status(master_path: Path, master: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for block in range(1, int(master["blocks"]) + 1):
        path = block_config_path(block)
        if not path.exists():
            rows.append({"block": block, "status": "unprepared"})
            continue
        expected = block_payload(master_path, master, block)
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            rows.append({"block": block, "status": "config_changed"})
            continue
        config = load_config(path)
        protocol = read_protocol(config)
        if protocol is None:
            rows.append({"block": block, "status": "ready"})
            continue
        status = str(protocol.get("status", "unknown"))
        row: dict[str, Any] = {"block": block, "status": status}
        if status == "collected":
            try:
                summary = validate_completed_block(master, config, protocol)
                row.update(
                    {
                        "duration_max_seconds": summary["fault_duration_quality"][
                            "maximum_seconds"
                        ],
                        "metric_latency_max_ms": summary["metric_latency_quality"][
                            "maximum_ms"
                        ],
                        "run_id": summary["telemetry_run_id"],
                    }
                )
            except (KeyError, RuntimeError, ValueError) as exc:
                row["status"] = "invalid"
                row["error"] = str(exc)
        else:
            row["error"] = protocol.get("error")
        rows.append(row)
    return {
        "campaign_id": master["campaign_id"],
        "blocks": rows,
        "accepted_blocks": sum(row["status"] == "collected" for row in rows),
        "expected_blocks": int(master["blocks"]),
    }


def _write_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _ensure_absent(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "v4 aggregate outputs already exist; refusing to overwrite: "
            + ", ".join(existing)
        )


def _matrix_array(matrix: dict[str, dict[str, int]]) -> np.ndarray:
    return np.asarray(
        [[matrix[actual][predicted] for predicted in LABELS] for actual in LABELS],
        dtype=int,
    )


def render_confirmation_figure(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "MPLCONFIGDIR", str((path.parent / ".matplotlib").resolve())
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Database", "HTTP 500", "Service stopped"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, key, title in zip(
        axes[:2], MODEL_KEYS, ("Gated robust fusion v2", "Logs-only comparator")
    ):
        model = result["models"][key]
        matrix = _matrix_array(model["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        axis.set_xticks(range(3), labels, rotation=20, ha="right")
        axis.set_yticks(range(3), labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title(f"{title}\nMacro F1 = {model['metrics']['macro_f1']:.3f}")
        for row in range(3):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                )
    routing = result["models"]["gated_robust_fusion_v2"]["routing"]
    values = [routing["coverage"], routing["fallback_rate"]]
    axes[2].bar(
        ["Fusion coverage", "Logs fallback"],
        values,
        color=["#2a9d8f", "#e76f51"],
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Fraction of v4 incidents")
    axes[2].set_title("Frozen OOD routing")
    for index, value in enumerate(values):
        axes[2].text(index, value + 0.02, f"{value:.3f}", ha="center")
    fig.suptitle("Independent post-freeze v4 confirmation")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def finalise(master_path: Path, master: dict[str, Any]) -> dict[str, Any]:
    configs = [
        load_config(prepare_block(master_path, master, block))
        for block in range(1, int(master["blocks"]) + 1)
    ]
    protocols = []
    summaries = []
    for config in configs:
        protocol = read_protocol(config)
        if protocol is None:
            raise RuntimeError(f"{config.campaign_id} has not run")
        summaries.append(validate_completed_block(master, config, protocol))
        protocols.append(protocol)

    aggregate = {
        name: Path(value) for name, value in master["aggregate_paths"].items()
    }
    _ensure_absent(list(aggregate.values()))
    guard_bundle = Path(master["holdout"]["frozen_bundle"])
    guard = verify_bundle(guard_bundle)
    if any(
        protocol["frozen_model_sha256"] != guard["model_sha256"]
        for protocol in protocols
    ):
        raise RuntimeError("v4 collection blocks did not use one guard artifact")
    candidate_bundle = Path(master["holdout"]["candidate_bundle"])
    candidate = verify_candidate_bundle(candidate_bundle)
    if any(
        protocol["candidate_freeze"]["model_sha256"]
        != candidate["model_sha256"]
        for protocol in protocols
    ):
        raise RuntimeError("v4 blocks did not preserve one candidate artifact")

    truth_records = [
        record
        for config in configs
        for record in read_jsonl(config.paths["ground_truth"])
    ]
    detection_sources = [Path(summary["detection_archive"]) for summary in summaries]
    detection_records = [
        record for path in detection_sources for record in read_jsonl(path)
    ]
    _write_records(aggregate["ground_truth"], truth_records)
    _write_records(aggregate["detections"], detection_records)
    rule_result = evaluate_files(aggregate["ground_truth"], aggregate["detections"])
    write_json(aggregate["evaluation"], rule_result)

    run_ids = {str(summary["telemetry_run_id"]) for summary in summaries}
    records = build_local_records(
        raw_root=Path(str(master["telemetry"]["root"])),
        ground_truth_paths=[aggregate["ground_truth"]],
        detection_archive_dir=Path("data/detections"),
        selected_run_ids=run_ids,
        require_detection_archive=True,
    )
    expected_per_class = int(master["blocks"])
    counts = Counter(record.local_fault_label for record in records)
    if len(records) != expected_per_class * 3:
        raise RuntimeError(f"v4 aggregate incident count is {len(records)}")
    if counts != Counter({label: expected_per_class for label in FAULT_LABELS}):
        raise RuntimeError(f"v4 aggregate is not balanced: {counts}")
    metric_quality = validate_metric_latencies(configs[0], records)
    intervals = build_ground_truth_intervals(truth_records)
    durations = [
        (item.ended_at - item.started_at).total_seconds()
        for item in intervals
        if item.ended_at is not None
    ]
    if len(durations) != len(records):
        raise RuntimeError("v4 aggregate has incomplete ground-truth intervals")
    duration_max = max(durations)
    if duration_max > float(master["acceptance"]["maximum_fault_duration_seconds"]):
        raise RuntimeError("v4 aggregate exceeded duration acceptance")
    write_incident_jsonl(records, aggregate["processed_incidents"])
    write_incident_manifest(
        records, aggregate["processed_manifest"], aggregate["processed_incidents"]
    )

    model_result = evaluate_candidate(
        candidate_bundle,
        aggregate["processed_incidents"],
        require_after_freeze=True,
    )
    model_result["experiment_id"] = "gated-robust-fusion-v2-v4-confirmation"
    model_result["v4_protocol"] = {
        "master_path": str(master_path),
        "master_sha256": sha256_file(master_path),
        "blocks": int(master["blocks"]),
        "single_changed_collection_factor": "workload_interval_seconds",
        "test_fitting_or_tuning": False,
        "candidate_locked_before_all_telemetry": True,
    }
    render_confirmation_figure(model_result, aggregate["model_figure"])
    model_result["artifacts"] = {
        "figure": str(aggregate["model_figure"]),
        "figure_sha256": sha256_file(aggregate["model_figure"]),
    }
    write_json(aggregate["model_result"], model_result)
    result = {
        "schema_version": 1,
        "campaign_id": master["campaign_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "master_path": str(master_path),
        "master_sha256": sha256_file(master_path),
        "candidate_model_sha256": candidate["model_sha256"],
        "candidate_frozen_at": candidate["created_at"],
        "accepted_blocks": int(master["blocks"]),
        "incidents": len(records),
        "fault_counts": dict(sorted(counts.items())),
        "run_ids": sorted(run_ids),
        "duration_quality": {
            "maximum_seconds": round(duration_max, 3),
            "limit_seconds": float(
                master["acceptance"]["maximum_fault_duration_seconds"]
            ),
        },
        "metric_latency_quality": metric_quality,
        "rule_evaluation": rule_result["summary"],
        "model_metrics": {
            name: model_result["models"][name]["metrics"] for name in MODEL_KEYS
        },
        "routing": model_result["models"]["gated_robust_fusion_v2"]["routing"],
        "paths": {name: str(path) for name, path in aggregate.items()},
    }
    write_json(aggregate["summary"], result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--dry-run-block", type=int)
    actions.add_argument("--run-block", type=int)
    actions.add_argument("--finalise", action="store_true")
    args = parser.parse_args(argv)
    master = load_master(args.master)
    if args.prepare:
        output = {"configs": [str(path) for path in prepare_all(args.master, master)]}
    elif args.status:
        output = campaign_status(args.master, master)
    elif args.dry_run_block is not None:
        path = prepare_block(args.master, master, args.dry_run_block)
        output = confirmation_plan(load_config(path))
    elif args.run_block is not None:
        for earlier in range(1, args.run_block):
            config = load_config(prepare_block(args.master, master, earlier))
            protocol = read_protocol(config)
            if protocol is None:
                raise RuntimeError(f"Block {earlier} must complete first")
            validate_completed_block(master, config, protocol)
        path = prepare_block(args.master, master, args.run_block)
        output = run_confirmation_block(load_config(path))
    else:
        output = finalise(args.master, master)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
