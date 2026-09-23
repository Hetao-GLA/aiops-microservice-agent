"""Local incident state plus append-only audit storage."""

from __future__ import annotations

import json
from pathlib import Path
import re
from threading import RLock
from typing import Any

from ops_agent.errors import IncidentNotFound
from ops_agent.models import AgentIncident, AuditEvent, utc_now


SAFE_ID = re.compile(r"^[A-Z0-9-]+$")


class IncidentStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.incident_root = self.root / "incidents"
        self.audit_path = self.root / "audit.jsonl"
        self._lock = RLock()

    def _path(self, incident_id: str) -> Path:
        if not SAFE_ID.fullmatch(incident_id):
            raise ValueError("Invalid incident identifier")
        return self.incident_root / f"{incident_id}.json"

    def _write(self, incident: AgentIncident) -> None:
        path = self._path(incident.incident_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            incident.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    def _append_global_audit(self, incident_id: str, event: AuditEvent) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"incident_id": incident_id, **event.model_dump(mode="json")}
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def save(
        self,
        incident: AgentIncident,
        *,
        event: str,
        actor: str,
        details: dict[str, Any] | None = None,
    ) -> AgentIncident:
        with self._lock:
            timestamp = utc_now()
            audit = AuditEvent(
                timestamp=timestamp,
                event=event,
                actor=actor,
                details=details or {},
            )
            incident.updated_at = timestamp
            incident.audit.append(audit)
            self._write(incident)
            self._append_global_audit(incident.incident_id, audit)
            return incident.model_copy(deep=True)

    def get(self, incident_id: str) -> AgentIncident:
        with self._lock:
            path = self._path(incident_id)
            if not path.is_file():
                raise IncidentNotFound(f"Unknown Agent incident: {incident_id}")
            return AgentIncident.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[AgentIncident]:
        with self._lock:
            if not self.incident_root.is_dir():
                return []
            incidents = [
                AgentIncident.model_validate_json(path.read_text(encoding="utf-8"))
                for path in self.incident_root.glob("*.json")
            ]
            return sorted(incidents, key=lambda item: item.created_at, reverse=True)
