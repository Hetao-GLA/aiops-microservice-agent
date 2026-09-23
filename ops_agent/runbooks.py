"""Validated local Runbook registry."""

from __future__ import annotations

import json
from pathlib import Path

from ops_agent.models import FaultType, Runbook


DEFAULT_RUNBOOK_DIR = Path(__file__).with_name("runbooks")


class RunbookRegistry:
    def __init__(self, root: str | Path = DEFAULT_RUNBOOK_DIR) -> None:
        self.root = Path(root)
        self._runbooks = self._load()

    def _load(self) -> dict[str, Runbook]:
        paths = sorted(self.root.glob("*.json"))
        if not paths:
            raise ValueError(f"No Agent Runbooks found under {self.root}")
        runbooks: dict[str, Runbook] = {}
        for path in paths:
            runbook = Runbook.model_validate_json(path.read_text(encoding="utf-8"))
            if runbook.fault_type in runbooks:
                raise ValueError(f"Duplicate Runbook for {runbook.fault_type}")
            runbooks[runbook.fault_type] = runbook
        return runbooks

    def get(self, fault_type: FaultType) -> Runbook:
        try:
            return self._runbooks[fault_type]
        except KeyError as exc:
            raise KeyError(f"No Runbook for {fault_type}") from exc

    def list(self) -> list[Runbook]:
        return [self._runbooks[name] for name in sorted(self._runbooks)]

    def as_tool_output(self, fault_type: FaultType) -> dict[str, object]:
        return json.loads(self.get(fault_type).model_dump_json())
