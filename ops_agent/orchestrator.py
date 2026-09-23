"""Human-approved Agent state machine."""

from __future__ import annotations

from threading import RLock
from typing import Any
from uuid import uuid4

from ops_agent.errors import InvalidTransition, NoActiveFault
from ops_agent.models import (
    AgentIncident,
    AgentStatus,
    ApprovalRecord,
    ApprovalRequest,
    Diagnosis,
    DiagnosisRequest,
    ExecutionRecord,
    VerificationRecord,
    utc_now,
)
from ops_agent.planner import Planner
from ops_agent.policy import authorise_execution, validate_recommendation, validate_runbook
from ops_agent.recovery import RecoveryExecutor
from ops_agent.runbooks import RunbookRegistry
from ops_agent.store import IncidentStore
from ops_agent.tools import EvidenceToolkit


class OperationsAgent:
    def __init__(
        self,
        *,
        store: IncidentStore,
        toolkit: EvidenceToolkit,
        runbooks: RunbookRegistry,
        planner: Planner,
        recovery: RecoveryExecutor,
    ) -> None:
        self.store = store
        self.toolkit = toolkit
        self.runbooks = runbooks
        self.planner = planner
        self.recovery = recovery
        self._lock = RLock()

    @property
    def planner_provider(self) -> str:
        return self.planner.provider_name

    def diagnose(self, request: DiagnosisRequest) -> AgentIncident:
        with self._lock:
            snapshot = (
                self.toolkit.collect_snapshot()
                if request.collect_live_evidence
                else {}
            )
            observed_fault, observed_confidence, classification_evidence = (
                self.toolkit.classify_snapshot(snapshot)
                if snapshot
                else (None, 0.0, {"decisive_signal": "not_collected"})
            )
            fault_type = request.predicted_fault or observed_fault
            if fault_type is None:
                raise NoActiveFault(
                    "No active supported fault was found; no recovery was proposed"
                )
            confidence = (
                request.confidence
                if request.confidence is not None
                else (observed_confidence if observed_fault == fault_type else 0.5)
            )
            source = request.diagnostic_source or (
                "submitted_detector" if request.predicted_fault else "live_rule_probe"
            )
            runbook = self.runbooks.get(fault_type)
            validate_runbook(runbook)
            evidence: dict[str, Any] = {
                "submitted": request.evidence,
                "live_snapshot": snapshot,
                "observed_fault": observed_fault,
                "classification_evidence": classification_evidence,
            }
            planner_context = {
                "trusted_fault_type": fault_type,
                "trusted_confidence": confidence,
                "diagnostic_source": source,
                "classification_evidence": classification_evidence,
                "evidence": evidence,
                "human_approval_required": True,
            }
            plan = self.planner.plan(planner_context, runbook)
            policy_accepted = validate_recommendation(
                fault_type, plan.recommended_action
            )
            now = utc_now()
            incident = AgentIncident(
                incident_id=f"AGT-{uuid4().hex[:12].upper()}",
                created_at=now,
                updated_at=now,
                status=AgentStatus.awaiting_approval,
                requested_by=request.requested_by,
                diagnosis=Diagnosis(
                    fault_type=fault_type,
                    confidence=float(confidence),
                    source=source,
                    detection_id=request.detection_id,
                    evidence=evidence,
                    planner=plan,
                    policy_action_accepted=policy_accepted,
                ),
                runbook=runbook,
                proposed_action=runbook.action_id,
            )
            return self.store.save(
                incident,
                event="diagnosis_created",
                actor=request.requested_by,
                details={
                    "fault_type": fault_type,
                    "proposed_action": runbook.action_id,
                    "planner_recommendation_accepted": policy_accepted,
                },
            )

    def approve(
        self, incident_id: str, request: ApprovalRequest
    ) -> AgentIncident:
        with self._lock:
            incident = self.store.get(incident_id)
            if incident.status != AgentStatus.awaiting_approval:
                raise InvalidTransition(
                    f"Incident {incident_id} is not awaiting approval"
                )
            incident.approval = ApprovalRecord(
                approved=request.approved,
                approver=request.approver,
                note=request.note,
                decided_at=utc_now(),
            )
            incident.status = (
                AgentStatus.approved if request.approved else AgentStatus.rejected
            )
            return self.store.save(
                incident,
                event="recovery_approved" if request.approved else "recovery_rejected",
                actor=request.approver,
                details={"note": request.note},
            )

    def execute(self, incident_id: str) -> AgentIncident:
        with self._lock:
            incident = self.store.get(incident_id)
            action_id = authorise_execution(incident)
            started_at = utc_now()
            incident.status = AgentStatus.executing
            incident.execution = ExecutionRecord(
                action_id=action_id,
                started_at=started_at,
            )
            incident = self.store.save(
                incident,
                event="recovery_started",
                actor="policy-executor",
                details={"action_id": action_id},
            )
            try:
                detail = self.recovery.execute(action_id)
            except Exception as exc:
                incident.execution.completed_at = utc_now()
                incident.execution.succeeded = False
                incident.execution.error_type = type(exc).__name__
                incident.status = AgentStatus.execution_failed
                return self.store.save(
                    incident,
                    event="recovery_execution_failed",
                    actor="policy-executor",
                    details={"action_id": action_id, "error_type": type(exc).__name__},
                )

            incident.execution.completed_at = utc_now()
            incident.execution.succeeded = True
            incident.execution.detail = detail
            incident.status = AgentStatus.verifying
            incident = self.store.save(
                incident,
                event="recovery_action_completed",
                actor="policy-executor",
                details={"action_id": action_id},
            )
            return self._verify(incident)

    def _verify(self, incident: AgentIncident) -> AgentIncident:
        try:
            raw = self.recovery.verify(incident.proposed_action)
            verification = VerificationRecord.model_validate(raw)
        except Exception as exc:
            verification = VerificationRecord(
                passed=False,
                checked_at=utc_now(),
                attempts=1,
                checks={"error_type": type(exc).__name__},
            )
        incident.verification = verification
        incident.status = (
            AgentStatus.resolved
            if verification.passed
            else AgentStatus.verification_failed
        )
        return self.store.save(
            incident,
            event=(
                "recovery_verified"
                if verification.passed
                else "recovery_verification_failed"
            ),
            actor="verification-tool",
            details={
                "attempts": verification.attempts,
                "passed": verification.passed,
            },
        )

    def retry_verification(self, incident_id: str) -> AgentIncident:
        with self._lock:
            incident = self.store.get(incident_id)
            if incident.status != AgentStatus.verification_failed:
                raise InvalidTransition(
                    "Verification can be retried only after verification failure"
                )
            incident.status = AgentStatus.verifying
            incident = self.store.save(
                incident,
                event="verification_retry_started",
                actor="operator",
            )
            return self._verify(incident)

    def get(self, incident_id: str) -> AgentIncident:
        return self.store.get(incident_id)

    def list(self) -> list[AgentIncident]:
        return self.store.list()
