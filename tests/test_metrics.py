from app.metrics import OrderMetrics


def test_order_metrics_calculates_http_500_rate() -> None:
    metrics = OrderMetrics()
    metrics.record(201)
    metrics.record(500)
    metrics.record(500)

    snapshot = metrics.snapshot(5)

    assert snapshot["total_requests"] == 3
    assert snapshot["successful_requests"] == 1
    assert snapshot["http_500_count"] == 2
    assert snapshot["http_500_rate"] == 0.6667

