"""Append-only ground-truth records for controlled fault experiments."""

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


class GroundTruthRecorder:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)

    def new_incident_id(self) -> str:
        return f"INC-{uuid4().hex[:12].upper()}"

    def record(
        self,
        *,
        incident_id: str,
        event: str,
        service: str,
        fault_type: str,
        root_cause: str,
        timestamp: datetime | str | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        if isinstance(timestamp, datetime):
            recorded_at = timestamp.isoformat()
        elif isinstance(timestamp, str):
            recorded_at = timestamp
        else:
            recorded_at = datetime.now(UTC).isoformat()

        entry: dict[str, Any] = {
            "timestamp": recorded_at,
            "incident_id": incident_id,
            "event": event,
            "service": service,
            "fault_type": fault_type,
            "root_cause": root_cause,
            **details,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        return entry
