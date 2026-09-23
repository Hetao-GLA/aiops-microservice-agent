import json
from pathlib import Path

from experiment.data_import.public_index_audit import (
    _status_code,
    render_markdown,
    summarise_lo2v2,
)


def test_status_code_is_extracted_from_lo2_test_name() -> None:
    assert _status_code("get_client_404_no_client") == "404"
    assert _status_code("access_token_auth_header_error_401") == "401"
    assert _status_code("correct") is None


def test_summarise_lo2v2_distinguishes_expected_errors(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    path.write_text(
        json.dumps(
            {
                "LO2v2": [
                    {
                        "tests": [
                            {"test": "correct", "logs": [], "traces": []},
                            {
                                "test": "get_client_404_no_client",
                                "logs": [{"line_count": 7}],
                                "traces": [{"line_count": 3}],
                            },
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = summarise_lo2v2(path)

    assert summary["runs"] == 1
    assert summary["correct_tests"] == 1
    assert summary["designed_error_tests"] == 1
    assert summary["response_status_counts"] == {"404": 1}
    assert summary["contains_500_test"] is False
    assert summary["total_log_lines"] == 7
    assert summary["total_trace_lines"] == 3


def test_markdown_records_separate_dataset_roles() -> None:
    audit = {
        "generated_at": "2026-08-20T00:00:00+00:00",
        "rcaeval": {
            "cases": 735,
            "cases_with_logs": 360,
            "cases_with_traces": 240,
            "candidate_subsets": {
                "RE2-OB": {"cases": 90, "socket_cases": 15},
                "RE3-OB": {"cases": 30},
            },
        },
        "lo2v2": {
            "runs": 115,
            "tests": 6210,
            "correct_tests": 115,
            "designed_error_tests": 6095,
            "response_status_counts": {"400": 40, "401": 10, "404": 3},
            "contains_500_test": False,
            "total_log_lines": 100,
            "total_trace_lines": 20,
        },
    }

    report = render_markdown(audit)

    assert "separate root-cause localisation benchmark" in report
    assert "do not use LO2v2 to train" in report
    assert "Keep locally generated incidents" in report
