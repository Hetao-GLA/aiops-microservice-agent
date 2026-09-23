from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiment.data_import.download_public_indexes import sha256_file
from experiment.data_import.download_rcaeval_subset import (
    _reuse_cached_file,
    remote_url,
    select_cases,
)


def _write_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    rows = []
    for service in ("a", "b"):
        for fault in ("cpu", "mem"):
            for repetition in (1, 2):
                rows.append(
                    {
                        "case": f"re2ob_{service}_{fault}_{repetition}",
                        "dataset": "RE2-OB",
                        "system_name": "Online Boutique",
                        "root_cause_service": service,
                        "fault": fault,
                        "repetition": repetition,
                        "has_logs": True,
                        "n_metrics": 10,
                    }
                )
    index = tmp_path / "cases.parquet"
    pd.DataFrame(rows).to_parquet(index, index=False)
    spec = {
        "source": {"index_sha256": sha256_file(index)},
        "selection": {
            "dataset": "RE2-OB",
            "system_name": "Online Boutique",
            "require_logs": True,
            "require_metrics": True,
            "expected_cases": 8,
            "expected_root_cause_services": ["a", "b"],
            "expected_faults": ["cpu", "mem"],
            "expected_repetitions_per_service_fault_pair": 2,
        },
    }
    return index, spec


def test_selection_requires_complete_locked_balance(tmp_path: Path) -> None:
    index, spec = _write_fixture(tmp_path)

    selected = select_cases(index, spec)

    assert len(selected) == 8
    assert selected[0]["case"] == "re2ob_a_cpu_1"


def test_selection_rejects_index_hash_change(tmp_path: Path) -> None:
    index, spec = _write_fixture(tmp_path)
    spec["source"]["index_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="index hash mismatch"):
        select_cases(index, spec)


def test_selection_can_require_metrics_without_requiring_logs(tmp_path: Path) -> None:
    index, spec = _write_fixture(tmp_path)
    frame = pd.read_parquet(index)
    frame.loc[0, "has_logs"] = False
    frame.to_parquet(index, index=False)
    spec["source"]["index_sha256"] = sha256_file(index)
    spec["selection"]["require_logs"] = False

    selected = select_cases(index, spec)

    assert len(selected) == 8


def test_cached_file_is_copied_without_network(tmp_path: Path) -> None:
    cache = tmp_path / "cache" / "case" / "logs.parquet"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"telemetry")
    destination = tmp_path / "target" / "case" / "logs.parquet"

    assert _reuse_cached_file([cache], destination) is True
    assert destination.read_bytes() == b"telemetry"


def test_remote_url_targets_one_case_file() -> None:
    assert remote_url("re2ob_a_cpu_1", "logs.parquet").endswith(
        "/re2ob_a_cpu_1/logs.parquet?download=true"
    )
