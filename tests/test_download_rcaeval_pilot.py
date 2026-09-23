from experiment.data_import.download_rcaeval_pilot import (
    CORE_FILES,
    PILOT_CASES,
    remote_url,
)


def test_pilot_is_balanced_between_selected_public_case_types() -> None:
    socket_cases = [case for case in PILOT_CASES if "_socket_" in case]
    code_cases = [case for case in PILOT_CASES if "_f1_" in case]

    assert len(socket_cases) == 3
    assert len(code_cases) == 3


def test_pilot_defaults_to_logs_and_metrics_without_traces() -> None:
    assert CORE_FILES == ("inject_time.txt", "logs.parquet", "metrics.parquet")
    assert remote_url(PILOT_CASES[0], "logs.parquet").endswith(
        "/logs.parquet?download=true"
    )
