"""Hard safety policy independent of model output."""

from __future__ import annotations

from ops_agent.errors import InvalidTransition, PolicyViolation
from ops_agent.models import ActionId, AgentIncident, AgentStatus, FaultType, Runbook


EXPECTED_ACTION: dict[FaultType, ActionId] = {
    "database_connection_failure": "start_database",
    "http_500_failure": "disable_http_500",
    "service_stopped": "start_api",
}


def validate_runbook(runbook: Runbook) -> None:
    expected = EXPECTED_ACTION[runbook.fault_type]
    if runbook.action_id != expected:
        raise PolicyViolation(
            f"Runbook action {runbook.action_id} is not allowed for {runbook.fault_type}"
        )


def validate_recommendation(fault_type: FaultType, action_id: ActionId) -> bool:
    return EXPECTED_ACTION[fault_type] == action_id


def authorise_execution(incident: AgentIncident) -> ActionId:
    if incident.status != AgentStatus.approved:
        raise InvalidTransition("Recovery requires an approved incident")
    if incident.approval is None or not incident.approval.approved:
        raise PolicyViolation("A positive human approval record is required")
    expected = EXPECTED_ACTION[incident.diagnosis.fault_type]
    if incident.proposed_action != expected or incident.runbook.action_id != expected:
        raise PolicyViolation("Proposed recovery action does not match the allow-list")
    return expected
