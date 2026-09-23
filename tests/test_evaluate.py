from datetime import UTC, datetime, timedelta

from experiment.evaluate import IncidentInterval, evaluate_intervals


def test_evaluation_matches_incidents_and_calculates_metrics() -> None:
    start = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    truth = [
        IncidentInterval(
            "INC-1",
            "database_connection_failure",
            start,
            start + timedelta(seconds=10),
        ),
        IncidentInterval(
            "INC-2",
            "service_stopped",
            start + timedelta(seconds=30),
            start + timedelta(seconds=40),
        ),
    ]
    detections = [
        IncidentInterval(
            "DET-1",
            "database_connection_failure",
            start + timedelta(seconds=2),
            start + timedelta(seconds=12),
        ),
        IncidentInterval(
            "DET-FP",
            "database_connection_failure",
            start + timedelta(seconds=80),
            start + timedelta(seconds=85),
        ),
    ]

    result = evaluate_intervals(truth, detections)
    summary = result["summary"]

    assert summary["matched_incidents"] == 1
    assert summary["missed_incidents"] == 1
    assert summary["false_positive_detections"] == 1
    assert summary["event_precision"] == 0.5
    assert summary["event_recall"] == 0.5
    assert summary["event_f1"] == 0.5
    assert summary["mean_detection_latency_seconds"] == 2.0
    assert summary["mean_recovery_detection_latency_seconds"] == 2.0


def test_evaluation_records_wrong_fault_class() -> None:
    start = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    truth = [IncidentInterval("INC-1", "database_failure", start, None)]
    detections = [
        IncidentInterval(
            "DET-1",
            "service_stopped",
            start + timedelta(seconds=1),
            None,
        )
    ]

    result = evaluate_intervals(truth, detections)

    assert result["summary"]["matched_incidents"] == 1
    assert result["summary"]["classification_accuracy"] == 0.0
    assert result["matches"][0]["classification_correct"] is False
    assert result["confusion_matrix"]["database_failure"]["service_stopped"] == 1
    assert result["per_class"]["database_failure"]["recall"] == 0.0


def test_small_negative_latency_is_preserved_and_adjusted_to_zero() -> None:
    start = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    truth = [IncidentInterval("INC-1", "service_stopped", start, None)]
    detections = [
        IncidentInterval(
            "DET-1",
            "service_stopped",
            start - timedelta(milliseconds=15),
            None,
        )
    ]

    result = evaluate_intervals(truth, detections)
    match = result["matches"][0]

    assert match["raw_detection_latency_seconds"] == -0.015
    assert match["detection_latency_seconds"] == 0.0
    assert match["timing_boundary_adjusted"] is True
    assert result["summary"]["timing_boundary_adjustments"] == 1
