from __future__ import annotations

from collections import Counter

from experiment.rcaeval_shift_audit import (
    AuditRecord,
    log_fold_diagnostics,
    metric_fold_diagnostics,
    normalise_log_line,
)


def _record(identifier: str, label: str, value: float, template: str) -> AuditRecord:
    return AuditRecord(
        incident_id=identifier,
        label=label,
        fault="cpu",
        repetition=1,
        metrics={
            "a_cpu__pre_mean": value,
            "a_cpu__post_mean": value,
            "a_cpu__mean_delta": value,
            "b_cpu__pre_mean": 0.0,
            "b_cpu__post_mean": 0.0,
            "b_cpu__mean_delta": 0.0,
        },
        templates=Counter({template: 2}),
        log_services=Counter({label: 2}),
    )


def test_log_normalisation_replaces_volatile_values() -> None:
    line = "[svc] Request 192.168.1.2 id=550e8400-e29b-41d4-a716-446655440000 took 12.5ms"

    assert normalise_log_line(line) == "[svc] request <ip> id=<uuid> took <num>ms"


def test_metric_audit_uses_training_only_range() -> None:
    records = [
        _record("train-1", "a", 1.0, "known"),
        _record("train-2", "b", 2.0, "known"),
        _record("test", "a", 10.0, "new"),
    ]
    features = sorted(records[0].metrics)

    rows, shifted = metric_fold_diagnostics(
        records,
        [0, 1],
        [2],
        features,
        epsilon=1e-9,
        z_cap=20.0,
        services=["a", "b"],
    )

    assert rows[0]["metric_range_breach_fraction"] == 0.5
    assert rows[0]["metric_signal"]["true_service_rank"] == 1
    assert shifted[0]["mean_absolute_z"] > 0


def test_log_audit_marks_only_test_templates_as_unseen() -> None:
    records = [
        _record("train-1", "a", 1.0, "known"),
        _record("train-2", "b", 2.0, "known"),
        _record("test", "a", 3.0, "new"),
    ]

    rows, top = log_fold_diagnostics(records, [0, 1], [2])

    assert rows[0]["log_unseen_line_fraction"] == 1.0
    assert top == [{"template": "new", "test_lines": 2}]
