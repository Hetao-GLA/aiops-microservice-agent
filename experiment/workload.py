"""Continuous, low-volume user activity for the experimental service."""

import logging
import os
import random
import time

import httpx

from app.logging_config import configure_logging


configure_logging()
logger = logging.getLogger("workload-generator")


def run_workload(
    *,
    base_url: str,
    interval_seconds: float,
    max_requests: int | None = None,
) -> None:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")

    items = ["keyboard", "monitor", "mouse", "headset"]
    request_number = 0

    with httpx.Client(base_url=base_url, timeout=5.0) as client:
        while max_requests is None or request_number < max_requests:
            request_number += 1
            payload = {
                "customer_id": f"customer-{random.randint(1, 20):03d}",
                "item": random.choice(items),
                "quantity": random.randint(1, 3),
            }

            try:
                response = client.post("/orders", json=payload)
                response.raise_for_status()
                logger.info(
                    "workload_request_succeeded",
                    extra={
                        "service": "workload-generator",
                        "target": base_url,
                        "request_number": request_number,
                        "status_code": response.status_code,
                    },
                )
            except httpx.HTTPError as exc:
                logger.error(
                    "workload_request_failed",
                    extra={
                        "service": "workload-generator",
                        "target": base_url,
                        "request_number": request_number,
                        "error_type": type(exc).__name__,
                    },
                )

            if max_requests is None or request_number < max_requests:
                time.sleep(interval_seconds)


def main() -> None:
    run_workload(
        base_url=os.getenv("TARGET_BASE_URL", "http://localhost:8000"),
        interval_seconds=float(os.getenv("WORKLOAD_INTERVAL_SECONDS", "1")),
    )


if __name__ == "__main__":
    main()

