from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from ops_agent.api import create_app
from ops_agent.errors import InvalidTransition
from ops_agent.models import (
    AgentStatus,
    ApprovalRequest,
    DiagnosisRequest,
    PlannerOutput,
)
from ops_agent.orchestrator import OperationsAgent
from ops_agent.planner import (
    DeterministicPlanner,
    OpenAIResponsesPlanner,
    READ_ONLY_TOOL_NAMES,
)
from ops_agent.policy import EXPECTED_ACTION
from ops_agent.recovery import RecoveryExecutor
from ops_agent.runbooks import RunbookRegistry
from ops_agent.store import IncidentStore
from ops_agent.tools import EvidenceToolkit


def _snapshot(fault: str | None) -> dict[str, object]:
    healthy = {"reachable": True, "status_code": 200, "body": {}}
    snapshot: dict[str, object] = {
        "health": {"liveness": dict(healthy), "database": dict(healthy)},
        "metrics": {
            "reachable": True,
            "status_code": 200,
            "body": {
                "total_requests": 5,
                "http_500_count": 0,
                "http_500_rate": 0.0,
            },
        },
        "fault_control": {"reachable": True, "status_code": 200},
        "recent_detections": [],
    }
    if fault == "service_stopped":
        snapshot["health"]["liveness"] = {
            "reachable": False,
            "status_code": None,
        }
    elif fault == "database_connection_failure":
        snapshot["health"]["database"] = {
            "reachable": True,
            "status_code": 503,
        }
    elif fault == "http_500_failure":
        snapshot["metrics"]["body"] = {
            "total_requests": 5,
            "http_500_count": 4,
            "http_500_rate": 0.8,
        }
    return snapshot


class FakeToolkit:
    def __init__(self, fault: str | None) -> None:
        self.snapshot = _snapshot(fault)

    def collect_snapshot(self):
        return self.snapshot

    classify_snapshot = staticmethod(EvidenceToolkit.classify_snapshot)


class FakeRecovery:
    def __init__(self, *, verification_passed: bool = True) -> None:
        self.executed: list[str] = []
        self.verification_passed = verification_passed

    def execute(self, action_id: str):
        self.executed.append(action_id)
        return {"allow_listed_action": action_id}

    def verify(self, action_id: str):
        return {
            "passed": self.verification_passed,
            "checked_at": "2026-08-30T00:00:00+00:00",
            "attempts": 1,
            "checks": {"action_id": action_id},
        }


def _agent(tmp_path: Path, fault: str = "database_connection_failure"):
    recovery = FakeRecovery()
    agent = OperationsAgent(
        store=IncidentStore(tmp_path / "agent-data"),
        toolkit=FakeToolkit(fault),
        runbooks=RunbookRegistry(),
        planner=DeterministicPlanner(),
        recovery=recovery,
    )
    return agent, recovery


@pytest.mark.parametrize(
    ("fault", "action"),
    [
        ("database_connection_failure", "start_database"),
        ("http_500_failure", "disable_http_500"),
        ("service_stopped", "start_api"),
    ],
)
def test_agent_requires_approval_then_executes_allow_listed_action(
    tmp_path: Path, fault: str, action: str
) -> None:
    agent, recovery = _agent(tmp_path, fault)
    incident = agent.diagnose(DiagnosisRequest(requested_by="tester"))

    assert incident.status == AgentStatus.awaiting_approval
    assert incident.proposed_action == action
    with pytest.raises(InvalidTransition, match="approved"):
        agent.execute(incident.incident_id)

    approved = agent.approve(
        incident.incident_id,
        ApprovalRequest(approved=True, approver="human", note="checked"),
    )
    assert approved.status == AgentStatus.approved

    resolved = agent.execute(incident.incident_id)

    assert resolved.status == AgentStatus.resolved
    assert recovery.executed == [action]
    assert resolved.verification is not None and resolved.verification.passed
    assert [event.event for event in resolved.audit] == [
        "diagnosis_created",
        "recovery_approved",
        "recovery_started",
        "recovery_action_completed",
        "recovery_verified",
    ]


def test_rejected_incident_cannot_execute(tmp_path: Path) -> None:
    agent, recovery = _agent(tmp_path)
    incident = agent.diagnose(DiagnosisRequest())
    rejected = agent.approve(
        incident.incident_id,
        ApprovalRequest(approved=False, approver="human", note="unsafe now"),
    )

    assert rejected.status == AgentStatus.rejected
    with pytest.raises(InvalidTransition):
        agent.execute(incident.incident_id)
    assert recovery.executed == []


def test_model_recommendation_cannot_change_policy_action(tmp_path: Path) -> None:
    class WrongPlanner:
        provider_name = "test-model"

        def plan(self, context, runbook):
            return PlannerOutput(
                summary="wrong recommendation",
                reasoning="test",
                evidence_used=[],
                recommended_action="start_api",
                provider=self.provider_name,
            )

    agent = OperationsAgent(
        store=IncidentStore(tmp_path),
        toolkit=FakeToolkit("database_connection_failure"),
        runbooks=RunbookRegistry(),
        planner=WrongPlanner(),
        recovery=FakeRecovery(),
    )

    incident = agent.diagnose(DiagnosisRequest())

    assert incident.diagnosis.planner.recommended_action == "start_api"
    assert incident.diagnosis.policy_action_accepted is False
    assert incident.proposed_action == "start_database"


def test_store_writes_append_only_audit(tmp_path: Path) -> None:
    agent, _ = _agent(tmp_path)
    incident = agent.diagnose(DiagnosisRequest())
    agent.approve(
        incident.incident_id,
        ApprovalRequest(approved=True, approver="human"),
    )

    audit_path = tmp_path / "agent-data" / "audit.jsonl"
    events = [json.loads(line)["event"] for line in audit_path.read_text().splitlines()]
    assert events == ["diagnosis_created", "recovery_approved"]


def test_read_only_llm_tools_exclude_recovery() -> None:
    names = {tool["name"] for tool in OpenAIResponsesPlanner.TOOLS}

    assert names == READ_ONLY_TOOL_NAMES
    assert not names.intersection(EXPECTED_ACTION.values())
    assert all("execute" not in name and "recover" not in name for name in names)


def test_openai_planner_runs_read_only_function_loop() -> None:
    responses = [
        SimpleNamespace(
            output=[
                {
                    "type": "function_call",
                    "name": "get_current_health",
                    "arguments": "{}",
                    "call_id": "call-1",
                }
            ],
            output_text="",
        ),
        SimpleNamespace(
            output=[],
            output_text=json.dumps(
                {
                    "summary": "Database unavailable",
                    "reasoning": "Health evidence and Runbook agree.",
                    "evidence_used": ["database health"],
                    "recommended_action": "start_database",
                }
            ),
        ),
    ]

    class FakeResponses:
        def create(self, **kwargs):
            return responses.pop(0)

    called: list[tuple[str, dict[str, object]]] = []

    def execute(name, arguments):
        called.append((name, arguments))
        return {"database": {"status_code": 503}}

    planner = OpenAIResponsesPlanner(
        model="test-model",
        tool_executor=execute,
        client=SimpleNamespace(responses=FakeResponses()),
    )
    runbook = RunbookRegistry().get("database_connection_failure")

    result = planner.plan({"trusted_fault_type": runbook.fault_type}, runbook)

    assert called == [("get_current_health", {})]
    assert result.provider == "openai_responses"
    assert result.recommended_action == "start_database"


def test_recovery_executor_builds_only_fixed_compose_start(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def run_command(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="started")

    executor = RecoveryExecutor(
        base_url="http://localhost:8000",
        project_root=tmp_path,
        compose_file=tmp_path / "docker-compose.yml",
        run_command=run_command,
    )

    result = executor.execute("start_database")

    assert commands == [
        [
            "docker",
            "compose",
            "-f",
            str(tmp_path / "docker-compose.yml"),
            "start",
            "database",
        ]
    ]
    assert result["service"] == "database"


def test_agent_api_exposes_complete_approval_flow(tmp_path: Path) -> None:
    agent, _ = _agent(tmp_path, "service_stopped")
    with TestClient(create_app(agent)) as client:
        health = client.get("/agent/health")
        diagnosis = client.post(
            "/agent/incidents/diagnose", json={"requested_by": "api-test"}
        )
        incident_id = diagnosis.json()["incident_id"]
        blocked = client.post(f"/agent/incidents/{incident_id}/execute")
        approved = client.post(
            f"/agent/incidents/{incident_id}/approval",
            json={"approved": True, "approver": "reviewer", "note": "ok"},
        )
        executed = client.post(f"/agent/incidents/{incident_id}/execute")
        dashboard = client.get("/")

    assert health.status_code == 200
    assert health.json()["human_approval_required"] is True
    assert diagnosis.status_code == 201
    assert blocked.status_code == 409
    assert approved.json()["status"] == "approved"
    assert executed.json()["status"] == "resolved"
    assert "Human-approved Operations Agent" in dashboard.text
