from __future__ import annotations

from dataclasses import replace

import pytest

from experiment.data_engineering.incident_schema import IncidentRecord


def _record() -> IncidentRecord:
    return IncidentRecord(
        incident_id="rcaeval:case-1",
        source="rcaeval",
        dataset="RE2-OB",
        system_name="Online Boutique",
        task="public_root_cause_localisation",
        injected_at="2026-08-20T12:00:00Z",
        window_start="2026-08-20T11:59:00Z",
        window_end="2026-08-20T12:02:00Z",
        root_cause_service="checkoutservice",
        original_fault_label="socket",
        local_fault_label=None,
        split_group="case-1",
        modalities=["logs", "metrics"],
        log_text="[checkoutservice] socket failure",
        log_record_count=1,
        log_services=["checkoutservice"],
        metric_row_count=181,
        metric_pre_rows=60,
        metric_post_rows=121,
        metric_features={"checkoutservice_socket__mean_delta": 4.0},
        provenance={"case": "case-1"},
    )


def test_valid_public_incident_serialises_with_schema_version() -> None:
    payload = _record().to_dict()

    assert payload["schema_version"] == 1
    assert payload["local_fault_label"] is None
    assert payload["split_group"] == "case-1"


def test_public_incident_cannot_be_forced_into_local_label_space() -> None:
    record = replace(_record(), local_fault_label="database_connection_failure")

    with pytest.raises(ValueError, match="Public incidents"):
        record.validate()


def test_injection_time_must_be_inside_window() -> None:
    record = replace(_record(), injected_at="2026-08-20T12:03:00Z")

    with pytest.raises(ValueError, match="inside the incident window"):
        record.validate()

