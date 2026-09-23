"""Build the real host-side Agent from environment configuration."""

from __future__ import annotations

from ops_agent.config import AgentSettings, get_settings
from ops_agent.orchestrator import OperationsAgent
from ops_agent.planner import build_planner
from ops_agent.recovery import RecoveryExecutor
from ops_agent.runbooks import RunbookRegistry
from ops_agent.store import IncidentStore
from ops_agent.tools import EvidenceToolkit, ReadOnlyToolExecutor


def build_operations_agent(
    settings: AgentSettings | None = None,
) -> OperationsAgent:
    selected = settings or get_settings()
    runbooks = RunbookRegistry()
    toolkit = EvidenceToolkit(
        base_url=selected.target_base_url,
        detection_path=selected.detection_path,
        project_root=selected.project_root,
        compose_file=selected.compose_file,
        command_timeout_seconds=selected.command_timeout_seconds,
    )
    read_tools = ReadOnlyToolExecutor(toolkit, runbooks)
    planner = build_planner(
        provider=selected.planner_provider,
        model=selected.openai_model,
        tool_executor=read_tools,
    )
    recovery = RecoveryExecutor(
        base_url=selected.target_base_url,
        project_root=selected.project_root,
        compose_file=selected.compose_file,
        command_timeout_seconds=selected.command_timeout_seconds,
        verification_timeout_seconds=selected.verification_timeout_seconds,
        verification_interval_seconds=selected.verification_interval_seconds,
    )
    return OperationsAgent(
        store=IncidentStore(selected.data_dir),
        toolkit=toolkit,
        runbooks=runbooks,
        planner=planner,
        recovery=recovery,
    )
