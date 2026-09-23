from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment.data_engineering.build_rcaeval_metric_incidents import (
    build_metric_case_incident,
)


def test_builds_metric_only_incident_without_fabricated_logs(tmp_path: Path) -> None:
    case = tmp_path / "re2tt_service_cpu_1"
    case.mkdir()
    (case / "inject_time.txt").write_text("100\n", encoding="utf-8")
    pd.DataFrame(
        {
            "time": [50, 99, 100, 150],
            "service_cpu": [1.0, 1.0, 3.0, 3.0],
        }
    ).to_parquet(case / "metrics.parquet", index=False)
    metadata = {
        "dataset": "RE2-TT",
        "system_name": "Train Ticket",
        "root_cause_service": "service",
        "fault": "cpu",
        "repetition": 1,
        "inject_time": 100,
    }

    record = build_metric_case_incident(case, metadata, pre_seconds=60, post_seconds=120)

    assert record.modalities == ["metrics"]
    assert record.log_text == ""
    assert record.log_record_count == 0
    assert record.metric_features["service_cpu__mean_delta"] == 2.0
    assert record.provenance["logs_intentionally_not_acquired"] is True
