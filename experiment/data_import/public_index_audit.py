from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


RCA_REQUIRED_COLUMNS = {
    "case",
    "dataset",
    "system_name",
    "root_cause_service",
    "fault",
    "inject_time",
    "has_logs",
    "has_traces",
}


def _integer_dict(values: dict[Any, Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in values.items()}


def summarise_rcaeval(path: Path) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "RCAEval index parsing requires requirements-data.txt"
        ) from exc

    frame = pd.read_parquet(path)
    missing = RCA_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"RCAEval index is missing columns: {sorted(missing)}")

    dataset_counts = _integer_dict(
        frame["dataset"].value_counts().sort_index().to_dict()
    )
    fault_counts = _integer_dict(frame["fault"].value_counts().sort_index().to_dict())
    system_counts = _integer_dict(
        frame["system_name"].value_counts().sort_index().to_dict()
    )

    re2_ob = frame[frame["dataset"] == "RE2-OB"]
    re3_ob = frame[frame["dataset"] == "RE3-OB"]
    re2_ob_socket = re2_ob[re2_ob["fault"] == "socket"]

    return {
        "cases": int(len(frame)),
        "columns": list(frame.columns),
        "dataset_counts": dataset_counts,
        "system_counts": system_counts,
        "fault_counts": fault_counts,
        "cases_with_logs": int(frame["has_logs"].sum()),
        "cases_with_traces": int(frame["has_traces"].sum()),
        "candidate_subsets": {
            "RE2-OB": {
                "cases": int(len(re2_ob)),
                "socket_cases": int(len(re2_ob_socket)),
                "services": sorted(re2_ob["root_cause_service"].unique().tolist()),
                "faults": sorted(re2_ob["fault"].unique().tolist()),
            },
            "RE3-OB": {
                "cases": int(len(re3_ob)),
                "services": sorted(re3_ob["root_cause_service"].unique().tolist()),
                "faults": sorted(re3_ob["fault"].unique().tolist()),
            },
        },
    }


def _status_code(test_name: str) -> str | None:
    match = re.search(r"(?:^|_)([1-5][0-9]{2})(?:_|$)", test_name)
    return match.group(1) if match else None


def summarise_lo2v2(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "LO2v2" not in payload or not isinstance(payload["LO2v2"], list):
        raise ValueError("LO2v2 index must contain a list under the 'LO2v2' key")

    runs = payload["LO2v2"]
    tests = [test for run in runs for test in run.get("tests", [])]
    names = Counter(str(test.get("test", "")) for test in tests)
    status_counts = Counter(
        status for name in names.elements() if (status := _status_code(name))
    )
    correct_tests = names.get("correct", 0)

    total_log_lines = sum(
        int(log_file.get("line_count", 0))
        for test in tests
        for log_file in test.get("logs", [])
    )
    total_trace_lines = sum(
        int(trace_file.get("line_count", 0))
        for test in tests
        for trace_file in test.get("traces", [])
    )

    return {
        "runs": len(runs),
        "tests": len(tests),
        "unique_test_types": len(names),
        "correct_tests": correct_tests,
        "designed_error_tests": len(tests) - correct_tests,
        "response_status_counts": _integer_dict(dict(sorted(status_counts.items()))),
        "contains_500_test": "500" in status_counts,
        "total_log_lines": total_log_lines,
        "total_trace_lines": total_trace_lines,
        "test_names": sorted(names),
    }


def build_audit(rcaeval_path: Path, lo2v2_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "rcaeval": summarise_rcaeval(rcaeval_path),
        "lo2v2": summarise_lo2v2(lo2v2_path),
    }


def render_markdown(audit: dict[str, Any]) -> str:
    rca = audit["rcaeval"]
    lo2 = audit["lo2v2"]
    re2_ob = rca["candidate_subsets"]["RE2-OB"]
    re3_ob = rca["candidate_subsets"]["RE3-OB"]
    status_counts = ", ".join(
        f"HTTP {status}: {count}"
        for status, count in lo2["response_status_counts"].items()
    )

    return f"""# Public dataset compatibility audit

Audit generated from locally downloaded metadata indexes on
{audit['generated_at']}.

## Project target

The local experiment classifies three operational failures:

1. `database_connection_failure`;
2. `service_stopped`;
3. `http_500_failure`.

The public datasets must not be merged into this label space unless the fault
mechanism and ground-truth meaning are equivalent.

## RCAEval

- Index size: **{rca['cases']} failure cases** across nine dataset/system
  combinations.
- Cases with logs: **{rca['cases_with_logs']}**.
- Cases with traces: **{rca['cases_with_traces']}**.
- Labels describe root-cause service and public fault type, not our local
  three-class labels.
- `RE2-OB` contains **{re2_ob['cases']} cases**, including
  **{re2_ob['socket_cases']} socket cases** across five services.
- `RE3-OB` contains **{re3_ob['cases']} code-level cases**.

Decision: **accept for a separate root-cause localisation benchmark**. Do not
directly relabel `socket` as database failure. A socket case may be used only
as a domain-transfer proxy, with results reported separately. RE3 cases may
support log/stack-trace representation experiments, but their F1-F5 meanings
must be confirmed case by case before any mapping.

Recommended first telemetry sample:

- three `RE2-OB` socket cases from one service, to validate the importer;
- three `RE3-OB cartservice_f1` cases, because the public tutorial documents
  the injected incorrect-parameter fault and its stack trace;
- retain the public root-cause labels unchanged.

Source: <https://github.com/phamquiluan/RCAEval>

## LO2v2

- Runs: **{lo2['runs']}**.
- Tests: **{lo2['tests']}**, consisting of **{lo2['correct_tests']}** `correct`
  tests and **{lo2['designed_error_tests']}** deliberately invalid API tests.
- Response-code distribution encoded in test names: {status_counts}.
- Explicit HTTP 500 test in the index: **{lo2['contains_500_test']}**.
- Indexed volume: **{lo2['total_log_lines']:,} log lines** and
  **{lo2['total_trace_lines']:,} trace lines**.

Decision: **do not use LO2v2 to train the local three-fault classifier**. Its
error tests primarily represent expected API validation/authentication
responses (400/401/404), not a stopped service, lost database connection, or
internal HTTP 500 failure. It can later provide an external robustness test:
the detector should distinguish expected client errors from operational
incidents. Downloading its full telemetry is deferred because the index alone
shows a task mismatch and a very large data volume.

Source: <https://zenodo.org/records/18937117>

## AIOps Challenge 2020

The official repository describes business metrics, infrastructure metrics,
trace data, and a fault table containing time, type, and location. It does not
describe application log text as a primary modality.

Decision: **defer**. It is a possible external metric/trace root-cause
benchmark, but not appropriate for the planned TF-IDF log classifier. Its
non-commercial research/teaching licence condition must be recorded if it is
used.

Source: <https://github.com/NetManAIOps/AIOps-Challenge-2020-Data>

## Final integration decision

1. Keep locally generated incidents as the only training and test source for
   the three operational fault classes.
2. Use RCAEval as a separate public root-cause benchmark and, optionally, a
   transfer-learning source.
3. Keep LO2v2 as a later robustness dataset rather than a fault-label source.
4. Split all datasets by complete incident/case, never by individual log line.
5. Report local and public results separately to expose domain shift.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit public dataset indexes.")
    parser.add_argument(
        "--rcaeval",
        type=Path,
        default=Path("data/external/rcaeval/cases.parquet"),
    )
    parser.add_argument(
        "--lo2v2",
        type=Path,
        default=Path("data/external/lo2v2/LO2v2_index.json"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data/results/public-dataset-audit.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/data/2026-08-20-public-dataset-compatibility.md"),
    )
    args = parser.parse_args()

    audit = build_audit(args.rcaeval, args.lo2v2)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(audit), encoding="utf-8")
    print(f"Wrote JSON audit -> {args.json_output}")
    print(f"Wrote compatibility report -> {args.markdown_output}")


if __name__ == "__main__":
    main()
