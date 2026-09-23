from __future__ import annotations

from pathlib import Path

import pytest

from experiment.log_evidence_ablation import (
    filter_log_sources,
    load_spec,
    mask_log_text,
    transform_record,
)


SPEC = Path("experiment/configs/log-evidence-ablation-v4-1s-spec.json")
LOG_TEXT = "\n".join(
    [
        '[order-service] database connection refused HTTP/1.1 500 Internal Server Error',
        '[workload] workload_request_failed status_code=500',
        '[rule-detector] database unavailable',
    ]
)


def test_locked_ablation_spec_requires_all_six_variants() -> None:
    spec = load_spec(SPEC)
    assert len(spec["variants"]) == 6
    assert spec["uniform_mask"]["applied_identically_to_every_class"] is True
    assert spec["restrictions"]["no_fit"] is True


def test_source_filter_retains_only_requested_service() -> None:
    result = filter_log_sources(LOG_TEXT, {"order-service"})
    assert result.startswith("[order-service]")
    assert "[workload]" not in result
    with pytest.raises(ValueError, match="removed every line"):
        filter_log_sources(LOG_TEXT, {"missing-service"})


def test_uniform_mask_removes_terms_and_all_http_status_codes() -> None:
    spec = load_spec(SPEC)
    result = mask_log_text(
        LOG_TEXT,
        terms=spec["uniform_mask"]["case_insensitive_substrings"],
        replacement=spec["uniform_mask"]["replacement_token"],
    ).lower()
    for token in ("database", "connection", "refused", "http", "error", "failed", "500"):
        assert token not in result
    assert "maskedtoken" in result


def test_transform_preserves_metrics_and_records_variant() -> None:
    spec = load_spec(SPEC)
    record = {
        "dataset": "run-1:early+1s",
        "log_text": LOG_TEXT,
        "log_record_count": 3,
        "log_services": ["order-service", "rule-detector", "workload"],
        "metric_features": {"metric__pre_mean": 1.0},
        "provenance": {},
    }
    result = transform_record(record, spec, "masked_workload_only")
    assert result["log_record_count"] == 1
    assert result["log_services"] == ["workload"]
    assert result["metric_features"] == record["metric_features"]
    assert result["provenance"]["log_evidence_ablation"]["uniform_mask_applied"] is True
