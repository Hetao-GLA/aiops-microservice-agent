"""Validated data contracts for Agent diagnosis, approval and recovery."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


FaultType = Literal[
    "database_connection_failure",
    "http_500_failure",
    "service_stopped",
]
ActionId = Literal["start_database", "disable_http_500", "start_api"]
RiskLevel = Literal["low", "medium", "high"]


class AgentStatus(str, Enum):
    awaiting_approval = "awaiting_approval"
    approved = "approved"
    rejected = "rejected"
    executing = "executing"
    verifying = "verifying"
    resolved = "resolved"
    execution_failed = "execution_failed"
    verification_failed = "verification_failed"


class Runbook(BaseModel):
    fault_type: FaultType
    title: str = Field(min_length=1)
    likely_root_cause: str = Field(min_length=1)
    action_id: ActionId
    risk_level: RiskLevel
    recovery_steps: list[str] = Field(min_length=1)
    verification_steps: list[str] = Field(min_length=1)
    rollback_note: str = Field(min_length=1)


class DiagnosisRequest(BaseModel):
    detection_id: str | None = Field(default=None, max_length=128)
    predicted_fault: FaultType | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    diagnostic_source: str | None = Field(default=None, max_length=128)
    evidence: dict[str, Any] = Field(default_factory=dict)
    collect_live_evidence: bool = True
    requested_by: str = Field(default="operator", min_length=1, max_length=128)


class ApprovalRequest(BaseModel):
    approved: bool
    approver: str = Field(min_length=1, max_length=128)
    note: str = Field(default="", max_length=1000)


class PlannerOutput(BaseModel):
    summary: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    evidence_used: list[str] = Field(default_factory=list)
    recommended_action: ActionId
    provider: str
    model: str | None = None


class Diagnosis(BaseModel):
    fault_type: FaultType
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    detection_id: str | None = None
    evidence: dict[str, Any]
    planner: PlannerOutput
    policy_action_accepted: bool


class ApprovalRecord(BaseModel):
    approved: bool
    approver: str
    note: str
    decided_at: str


class ExecutionRecord(BaseModel):
    action_id: ActionId
    started_at: str
    completed_at: str | None = None
    succeeded: bool | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None


class VerificationRecord(BaseModel):
    passed: bool
    checked_at: str
    attempts: int = Field(ge=1)
    checks: dict[str, Any]


class AuditEvent(BaseModel):
    timestamp: str
    event: str
    actor: str
    details: dict[str, Any] = Field(default_factory=dict)


class AgentIncident(BaseModel):
    incident_id: str
    created_at: str
    updated_at: str
    status: AgentStatus
    requested_by: str
    diagnosis: Diagnosis
    runbook: Runbook
    proposed_action: ActionId
    approval: ApprovalRecord | None = None
    execution: ExecutionRecord | None = None
    verification: VerificationRecord | None = None
    audit: list[AuditEvent] = Field(default_factory=list)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
