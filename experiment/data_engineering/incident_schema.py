from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


SCHEMA_VERSION = 1
LOCAL_FAULT_LABELS = frozenset(
    {
        "database_connection_failure",
        "service_stopped",
        "http_500_failure",
    }
)
TASKS = frozenset({"local_fault_classification", "public_root_cause_localisation"})


def _parse_timestamp(value: str) -> datetime:
    normalised = value[:-1] + "+00:00" if value.endswith("Z") else value
    timestamp = datetime.fromisoformat(normalised)
    if timestamp.tzinfo is None:
        raise ValueError(f"Timestamp must include a timezone: {value!r}")
    return timestamp


@dataclass(frozen=True)
class IncidentRecord:
    """One complete incident, kept intact for grouped train/test splitting."""

    incident_id: str
    source: str
    dataset: str
    system_name: str
    task: str
    injected_at: str
    window_start: str
    window_end: str
    root_cause_service: str
    original_fault_label: str
    local_fault_label: str | None
    split_group: str
    modalities: list[str]
    log_text: str
    log_record_count: int
    log_services: list[str]
    metric_row_count: int
    metric_pre_rows: int
    metric_post_rows: int
    metric_features: dict[str, float | None]
    provenance: dict[str, Any]
    schema_version: int = field(default=SCHEMA_VERSION, init=False)

    def validate(self) -> None:
        required_strings = {
            "incident_id": self.incident_id,
            "source": self.source,
            "dataset": self.dataset,
            "system_name": self.system_name,
            "task": self.task,
            "root_cause_service": self.root_cause_service,
            "original_fault_label": self.original_fault_label,
            "split_group": self.split_group,
        }
        empty = sorted(name for name, value in required_strings.items() if not value)
        if empty:
            raise ValueError(f"Required incident fields are empty: {empty}")
        if self.task not in TASKS:
            raise ValueError(f"Unsupported incident task: {self.task!r}")
        if self.local_fault_label is not None and self.local_fault_label not in LOCAL_FAULT_LABELS:
            raise ValueError(f"Unsupported local fault label: {self.local_fault_label!r}")
        if self.source == "local" and self.task != "local_fault_classification":
            raise ValueError("Local incidents must use the local fault-classification task")
        if self.source == "local" and self.local_fault_label is None:
            raise ValueError("Local incidents must have a local fault label")
        if self.source != "local" and self.local_fault_label is not None:
            raise ValueError(
                "Public incidents must retain their original labels and cannot be "
                "assigned a local fault label without a separate mapping experiment"
            )
        injected_at = _parse_timestamp(self.injected_at)
        window_start = _parse_timestamp(self.window_start)
        window_end = _parse_timestamp(self.window_end)
        if not window_start <= injected_at <= window_end:
            raise ValueError("Injection time must fall inside the incident window")
        counts = {
            "log_record_count": self.log_record_count,
            "metric_row_count": self.metric_row_count,
            "metric_pre_rows": self.metric_pre_rows,
            "metric_post_rows": self.metric_post_rows,
        }
        negative = sorted(name for name, value in counts.items() if value < 0)
        if negative:
            raise ValueError(f"Incident counts cannot be negative: {negative}")
        if self.log_record_count > 0 and not self.log_text:
            raise ValueError("log_text cannot be empty when log records are present")
        if len(self.modalities) != len(set(self.modalities)):
            raise ValueError("modalities must not contain duplicates")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
