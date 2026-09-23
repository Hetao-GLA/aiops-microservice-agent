"""Environment-based configuration for the host-side operations Agent."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class AgentSettings:
    target_base_url: str
    data_dir: Path
    detection_path: Path
    project_root: Path
    compose_file: Path | None
    command_timeout_seconds: float
    verification_timeout_seconds: float
    verification_interval_seconds: float
    planner_provider: str
    openai_model: str | None


def get_settings() -> AgentSettings:
    project_root = Path(
        os.getenv("OPS_AGENT_PROJECT_ROOT", Path.cwd())
    ).resolve()
    compose_value = os.getenv("OPS_AGENT_COMPOSE_FILE", "docker-compose.yml").strip()
    provider = os.getenv("OPS_AGENT_PLANNER", "deterministic").strip().lower()
    if provider not in {"deterministic", "openai"}:
        raise ValueError("OPS_AGENT_PLANNER must be deterministic or openai")
    model = os.getenv("OPENAI_MODEL", "").strip() or None
    return AgentSettings(
        target_base_url=os.getenv(
            "OPS_AGENT_TARGET_BASE_URL", "http://localhost:8000"
        ).rstrip("/"),
        data_dir=Path(
            os.getenv("OPS_AGENT_DATA_DIR", "data/agent")
        ),
        detection_path=Path(
            os.getenv(
                "OPS_AGENT_DETECTION_PATH", "data/detections/detections.jsonl"
            )
        ),
        project_root=project_root,
        compose_file=(project_root / compose_value).resolve() if compose_value else None,
        command_timeout_seconds=float(
            os.getenv("OPS_AGENT_COMMAND_TIMEOUT_SECONDS", "30")
        ),
        verification_timeout_seconds=float(
            os.getenv("OPS_AGENT_VERIFICATION_TIMEOUT_SECONDS", "30")
        ),
        verification_interval_seconds=float(
            os.getenv("OPS_AGENT_VERIFICATION_INTERVAL_SECONDS", "1")
        ),
        planner_provider=provider,
        openai_model=model,
    )
