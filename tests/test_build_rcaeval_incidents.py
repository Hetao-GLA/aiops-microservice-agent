from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiment.data_engineering.build_rcaeval_incidents import (
    build_case_incident,
    write_jsonl,
)


def test_build_case_incident_uses_one_complete_time_window(tmp_path: Path) -> None:
    case_dir = tmp_path / "re2ob_checkoutservice_socket_1"
    case_dir.mkdir()
    (case_dir / "inject_time.txt").write_text("100\n", encoding="utf-8")
    pd.DataFrame(
        {
            "timestamp": [89, 90, 100, 120, 121],
            "container_name": ["frontend", "frontend", "checkoutservice", "checkoutservice", "frontend"],
            "message": ["outside", "before", "failure", "recovery", "outside"],
        }
    ).to_parquet(case_dir / "logs.parquet", index=False)
    pd.DataFrame(
        {
            "time": [89, 90, 99, 100, 120, 121],
            "checkoutservice_socket": [0.0, 1.0, 3.0, 5.0, 7.0, 9.0],
        }
    ).to_parquet(case_dir / "metrics.parquet", index=False)
    metadata = {
        "case": case_dir.name,
        "dataset": "RE2-OB",
        "system_name": "Online Boutique",
        "root_cause_service": "checkoutservice",
        "fault": "socket",
        "repetition": 1,
        "inject_time": 100,
        "has_logs": True,
    }

    record = build_case_incident(
        case_dir, metadata, pre_seconds=10, post_seconds=20
    )

    assert record.incident_id == f"rcaeval:{case_dir.name}"
    assert record.log_record_count == 3
    assert record.log_text.splitlines() == [
        "[frontend] before",
        "[checkoutservice] failure",
        "[checkoutservice] recovery",
    ]
    assert record.metric_row_count == 4
    assert record.metric_pre_rows == 2
    assert record.metric_post_rows == 2
    assert record.metric_features["checkoutservice_socket__pre_mean"] == 2.0
    assert record.metric_features["checkoutservice_socket__post_mean"] == 6.0
    assert record.metric_features["checkoutservice_socket__mean_delta"] == 4.0
    assert record.local_fault_label is None
    assert record.provenance["repetition"] == 1


def test_jsonl_output_keeps_one_row_per_incident(tmp_path: Path) -> None:
    case_dir = tmp_path / "re2ob_checkoutservice_socket_1"
    case_dir.mkdir()
    (case_dir / "inject_time.txt").write_text("100\n", encoding="utf-8")
    pd.DataFrame(
        {
            "timestamp": [100],
            "container_name": ["checkoutservice"],
            "message": ["failure"],
        }
    ).to_parquet(case_dir / "logs.parquet", index=False)
    pd.DataFrame(
        {"time": [99, 100], "checkoutservice_socket": [1.0, 5.0]}
    ).to_parquet(case_dir / "metrics.parquet", index=False)
    metadata = {
        "case": case_dir.name,
        "dataset": "RE2-OB",
        "system_name": "Online Boutique",
        "root_cause_service": "checkoutservice",
        "fault": "socket",
        "repetition": 1,
        "inject_time": 100,
        "has_logs": True,
    }
    record = build_case_incident(case_dir, metadata, pre_seconds=10, post_seconds=20)
    output = tmp_path / "incidents.jsonl"

    assert write_jsonl([record], output) == 1
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["incident_id"] == record.incident_id
