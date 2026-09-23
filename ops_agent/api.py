"""FastAPI control plane for the host-side operations Agent."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.responses import HTMLResponse, JSONResponse

from ops_agent.errors import AgentError, IncidentNotFound
from ops_agent.factory import build_operations_agent
from ops_agent.models import AgentIncident, ApprovalRequest, DiagnosisRequest, Runbook
from ops_agent.orchestrator import OperationsAgent
from ops_agent.planner import READ_ONLY_TOOL_NAMES


DASHBOARD_PATH = Path(__file__).with_name("static") / "index.html"


def create_app(operations_agent: OperationsAgent | None = None) -> FastAPI:
    app = FastAPI(
        title="Human-approved Microservice Operations Agent",
        version="0.1.0",
        description=(
            "AI-assisted diagnosis with read-only planning tools, local Runbooks, "
            "human approval and allow-listed recovery."
        ),
    )
    app.state.operations_agent = operations_agent or build_operations_agent()

    @app.exception_handler(AgentError)
    async def handle_agent_error(_: Request, exc: AgentError) -> JSONResponse:
        code = 404 if isinstance(exc, IncidentNotFound) else 409
        return JSONResponse(status_code=code, content={"detail": str(exc)})

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return DASHBOARD_PATH.read_text(encoding="utf-8")

    @app.get("/agent/health")
    def agent_health(request: Request) -> dict[str, object]:
        agent: OperationsAgent = request.app.state.operations_agent
        return {
            "status": "healthy",
            "planner_provider": agent.planner_provider,
            "human_approval_required": True,
            "arbitrary_command_execution": False,
            "read_only_planner_tools": sorted(READ_ONLY_TOOL_NAMES),
        }

    @app.get("/agent/runbooks", response_model=list[Runbook])
    def list_runbooks(request: Request) -> list[Runbook]:
        agent: OperationsAgent = request.app.state.operations_agent
        return agent.runbooks.list()

    @app.get("/agent/incidents", response_model=list[AgentIncident])
    def list_incidents(request: Request) -> list[AgentIncident]:
        agent: OperationsAgent = request.app.state.operations_agent
        return agent.list()

    @app.post(
        "/agent/incidents/diagnose",
        response_model=AgentIncident,
        status_code=status.HTTP_201_CREATED,
    )
    def diagnose(
        payload: DiagnosisRequest, request: Request
    ) -> AgentIncident:
        agent: OperationsAgent = request.app.state.operations_agent
        return agent.diagnose(payload)

    @app.get("/agent/incidents/{incident_id}", response_model=AgentIncident)
    def get_incident(incident_id: str, request: Request) -> AgentIncident:
        agent: OperationsAgent = request.app.state.operations_agent
        return agent.get(incident_id)

    @app.post(
        "/agent/incidents/{incident_id}/approval", response_model=AgentIncident
    )
    def approve(
        incident_id: str, payload: ApprovalRequest, request: Request
    ) -> AgentIncident:
        agent: OperationsAgent = request.app.state.operations_agent
        return agent.approve(incident_id, payload)

    @app.post(
        "/agent/incidents/{incident_id}/execute", response_model=AgentIncident
    )
    def execute(incident_id: str, request: Request) -> AgentIncident:
        agent: OperationsAgent = request.app.state.operations_agent
        return agent.execute(incident_id)

    @app.post(
        "/agent/incidents/{incident_id}/verify", response_model=AgentIncident
    )
    def retry_verification(incident_id: str, request: Request) -> AgentIncident:
        agent: OperationsAgent = request.app.state.operations_agent
        return agent.retry_verification(incident_id)

    return app


app = create_app()
