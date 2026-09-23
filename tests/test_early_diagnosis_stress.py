from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment.early_diagnosis_stress import (
    build_early_record,
    build_variants,
    load_spec,
    summarise_early_probes,
)
from experiment.ml_baseline import sha256_file


SPEC = Path("experiment/configs/early-diagnosis-v4-stress-spec.json")
FEATURES = [
    "order_metrics__available__pre_mean",
    "order_metrics__available__post_mean",
    "order_metrics__available__mean_delta",
    "order_metrics__status_code__pre_mean",
    "order_metrics__status_code__post_mean",
    "order_metrics__status_code__mean_delta",
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _source_record(tmp_path: Path) -> dict:
    raw = tmp_path / "raw"
    raw.mkdir()
    metadata = raw / "metadata.json"
    probes = raw / "probes.jsonl"
    logs = raw / "docker-logs.jsonl"
    metadata.write_text(json.dumps({"incident_id": "INC-1"}))
    _write_jsonl(
        probes,
        [
            {
                "timestamp": "2026-01-01T00:00:09Z",
                "probe": "order_metrics",
                "available": True,
                "status_code": 200,
            },
            {
                "timestamp": "2026-01-01T00:00:10.500Z",
                "probe": "order_metrics",
                "available": False,
                "status_code": None,
            },
            {
                "timestamp": "2026-01-01T00:00:12Z",
                "probe": "order_metrics",
                "available": True,
                "status_code": 200,
            },
        ],
    )
    _write_jsonl(
        logs,
        [
            {
                "timestamp": "2026-01-01T00:00:09Z",
                "service": "order-service",
                "message": "healthy",
            },
            {
                "timestamp": "2026-01-01T00:00:10.500Z",
                "service": "workload",
                "message": "request failed",
            },
            {
                "timestamp": "2026-01-01T00:00:12Z",
                "service": "workload",
                "message": "late evidence",
            },
        ],
    )
    return {
        "schema_version": 1,
        "incident_id": "local:INC-1",
        "dataset": "run-1",
        "injected_at": "2026-01-01T00:00:10Z",
        "window_start": "2026-01-01T00:00:00Z",
        "window_end": "2026-01-01T00:00:30Z",
        "log_text": "full",
        "log_record_count": 3,
        "log_services": ["order-service", "workload"],
        "metric_row_count": 3,
        "metric_pre_rows": 1,
        "metric_post_rows": 2,
        "metric_features": {name: 1.0 for name in FEATURES},
        "provenance": {
            "run_id": "run-1",
            "raw_incident_directory": str(raw),
            "metadata_sha256": sha256_file(metadata),
            "probes_sha256": sha256_file(probes),
            "docker_logs_sha256": sha256_file(logs),
            "fault_ended_at": "2026-01-01T00:00:20Z",
        },
    }


def test_locked_spec_has_fixed_cutoffs_and_restrictions() -> None:
    spec = load_spec(SPEC)
    assert spec["time_windows"]["post_injection_cutoffs_seconds"] == [1, 3, 5, 8]
    assert spec["restrictions"]["no_fit"] is True
    assert spec["restrictions"]["must_not_be_called_independent_confirmation"] is True


def test_early_summary_uses_zero_for_unobserved_post_numeric_value() -> None:
    probes = [
        {
            "timestamp": "2026-01-01T00:00:09Z",
            "probe": "order_metrics",
            "available": True,
            "status_code": 200,
        },
        {
            "timestamp": "2026-01-01T00:00:10.5Z",
            "probe": "order_metrics",
            "available": False,
            "status_code": None,
        },
    ]
    pre, post, features = summarise_early_probes(
        probes,
        injected_at=__import__("datetime").datetime.fromisoformat(
            "2026-01-01T00:00:10+00:00"
        ),
        expected_feature_names=FEATURES,
    )
    assert (pre, post) == (1, 1)
    assert features["order_metrics__available__post_mean"] == 0.0
    assert features["order_metrics__status_code__post_mean"] == 0.0
    assert features["order_metrics__status_code__mean_delta"] == -200.0


def test_build_early_record_excludes_later_and_recovery_evidence(tmp_path: Path) -> None:
    source = _source_record(tmp_path)
    result = build_early_record(
        source, cutoff_seconds=1, expected_metric_names=FEATURES
    )
    assert result["window_end"] == "2026-01-01T00:00:11+00:00"
    assert result["log_record_count"] == 2
    assert "request failed" in result["log_text"]
    assert "late evidence" not in result["log_text"]
    assert result["metric_row_count"] == 2
    assert result["metric_post_rows"] == 1
    assert result["provenance"]["analysis_variant"][
        "recovery_evidence_excluded"
    ] is True


def test_raw_hash_change_is_rejected(tmp_path: Path) -> None:
    source = _source_record(tmp_path)
    raw = Path(source["provenance"]["raw_incident_directory"])
    (raw / "docker-logs.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="hash differs"):
        build_early_record(source, cutoff_seconds=1, expected_metric_names=FEATURES)


def test_variant_outputs_refuse_overwrite(tmp_path: Path) -> None:
    source = _source_record(tmp_path)
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(source_path, [source])
    output_dir = tmp_path / "variants"
    build_variants(
        source_path,
        output_dir,
        cutoffs=[1],
        expected_metric_names=FEATURES,
        missing_post_value=0.0,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_variants(
            source_path,
            output_dir,
            cutoffs=[1],
            expected_metric_names=FEATURES,
            missing_post_value=0.0,
        )
