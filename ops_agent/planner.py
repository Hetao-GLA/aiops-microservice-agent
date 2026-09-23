"""Deterministic and optional OpenAI Responses API planning adapters."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from typing import Any, Protocol

from ops_agent.models import PlannerOutput, Runbook


READ_ONLY_TOOL_NAMES = frozenset(
    {
        "get_current_health",
        "get_current_metrics",
        "get_recent_logs",
        "get_recent_detections",
        "get_runbook",
    }
)


class Planner(Protocol):
    provider_name: str

    def plan(self, context: dict[str, Any], runbook: Runbook) -> PlannerOutput: ...


class DeterministicPlanner:
    provider_name = "deterministic"

    def plan(self, context: dict[str, Any], runbook: Runbook) -> PlannerOutput:
        decisive = context.get("classification_evidence", {}).get(
            "decisive_signal", "submitted diagnosis"
        )
        return PlannerOutput(
            summary=f"Likely {runbook.fault_type}; approval is required before recovery.",
            reasoning=(
                f"The trusted diagnosis points to {runbook.fault_type} using "
                f"{decisive}. The local Runbook permits only {runbook.action_id}."
            ),
            evidence_used=[str(decisive), "validated local Runbook"],
            recommended_action=runbook.action_id,
            provider=self.provider_name,
        )


class OpenAIResponsesPlanner:
    """LLM explanation and read-only tool use; never executes recovery."""

    provider_name = "openai_responses"
    TOOLS: list[dict[str, Any]] = [
        {
            "type": "function",
            "name": "get_current_health",
            "description": "Read current service liveness and database health.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "get_current_metrics",
            "description": "Read the current five-second order metric window.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "get_recent_logs",
            "description": "Read a bounded tail of one allow-listed Compose service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "enum": ["api", "database", "workload", "detector"],
                    },
                    "tail": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                "required": ["service", "tail"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "get_recent_detections",
            "description": "Read recent structured rule-detection records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20}
                },
                "required": ["limit"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "get_runbook",
            "description": "Read one validated local recovery Runbook.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fault_type": {
                        "type": "string",
                        "enum": [
                            "database_connection_failure",
                            "http_500_failure",
                            "service_stopped",
                        ],
                    }
                },
                "required": ["fault_type"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]

    def __init__(
        self,
        *,
        model: str,
        tool_executor: Callable[[str, dict[str, Any]], Any],
        client: Any | None = None,
        max_rounds: int = 5,
    ) -> None:
        if not model:
            raise ValueError("OPENAI_MODEL is required for the OpenAI planner")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "Install requirements-agent.txt to enable the OpenAI planner"
                ) from exc
            client = OpenAI()
        self.client = client
        self.model = model
        self.tool_executor = tool_executor
        self.max_rounds = max_rounds

    @staticmethod
    def _field(item: Any, name: str) -> Any:
        if isinstance(item, dict):
            return item.get(name)
        return getattr(item, name, None)

    @staticmethod
    def _serialise_item(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return item
        if hasattr(item, "model_dump"):
            return item.model_dump(exclude_none=True)
        raise TypeError("Unsupported Responses API output item")

    @staticmethod
    def _parse_output(text: str, runbook: Runbook, model: str) -> PlannerOutput:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].lstrip()
        payload = json.loads(cleaned)
        return PlannerOutput(
            summary=str(payload["summary"]),
            reasoning=str(payload["reasoning"]),
            evidence_used=[str(value) for value in payload.get("evidence_used", [])],
            recommended_action=payload.get(
                "recommended_action", runbook.action_id
            ),
            provider="openai_responses",
            model=model,
        )

    def plan(self, context: dict[str, Any], runbook: Runbook) -> PlannerOutput:
        instructions = (
            "You are a constrained microservice operations diagnosis planner. "
            "Use only the supplied read-only tools. Never claim to execute recovery, "
            "never request arbitrary commands, and never bypass human approval. "
            "The trusted fault label and local Runbook action are policy inputs, not "
            "choices to override. Return only JSON with summary, reasoning, "
            "evidence_used (array), and recommended_action."
        )
        input_items: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "Prepare an explainable recovery recommendation for this context:\n"
                    + json.dumps(
                        {
                            "context": context,
                            "locked_runbook": runbook.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                        default=str,
                    )[:16000]
                ),
            }
        ]
        for _ in range(self.max_rounds):
            response = self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=input_items,
                tools=self.TOOLS,
                tool_choice="auto",
                parallel_tool_calls=False,
                max_output_tokens=900,
                store=False,
            )
            calls = [
                item
                for item in response.output
                if self._field(item, "type") == "function_call"
            ]
            input_items.extend(self._serialise_item(item) for item in response.output)
            if not calls:
                return self._parse_output(response.output_text, runbook, self.model)
            for call in calls:
                name = str(self._field(call, "name"))
                if name not in READ_ONLY_TOOL_NAMES:
                    raise ValueError("The model requested a non-read-only tool")
                arguments = json.loads(self._field(call, "arguments") or "{}")
                output = self.tool_executor(name, arguments)
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(self._field(call, "call_id")),
                        "output": json.dumps(output, ensure_ascii=False, default=str),
                    }
                )
        raise RuntimeError("OpenAI planner exceeded the read-only tool-call limit")


class ResilientPlanner:
    def __init__(self, primary: Planner, fallback: DeterministicPlanner) -> None:
        self.primary = primary
        self.fallback = fallback
        self.provider_name = primary.provider_name

    def plan(self, context: dict[str, Any], runbook: Runbook) -> PlannerOutput:
        try:
            return self.primary.plan(context, runbook)
        except Exception as exc:
            output = self.fallback.plan(context, runbook)
            output.provider = "deterministic_fallback"
            output.reasoning += f" LLM planning was unavailable ({type(exc).__name__})."
            return output


def build_planner(
    *,
    provider: str,
    model: str | None,
    tool_executor: Callable[[str, dict[str, Any]], Any],
) -> Planner:
    fallback = DeterministicPlanner()
    if provider != "openai":
        return fallback
    if not model or not os.getenv("OPENAI_API_KEY"):
        return fallback
    return ResilientPlanner(
        OpenAIResponsesPlanner(
            model=model,
            tool_executor=tool_executor,
        ),
        fallback,
    )
