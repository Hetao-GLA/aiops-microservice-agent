"""FastAPI entry point for the first experimental business service."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.database import check_database, create_order, initialise_database
from app.faults import fault_state
from app.logging_config import configure_logging
from app.metrics import order_metrics


configure_logging()
logger = logging.getLogger(settings.service_name)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialise_database()
    logger.info(
        "service_started",
        extra={"service": settings.service_name, "environment": settings.environment},
    )
    yield
    logger.info("service_stopped", extra={"service": settings.service_name})


app = FastAPI(
    title="AI-assisted Operations Experiment Service",
    version="0.1.0",
    lifespan=lifespan,
)


class OrderRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=64)
    item: str = Field(min_length=1, max_length=128)
    quantity: int = Field(ge=1, le=100)


class FaultToggleRequest(BaseModel):
    enabled: bool


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": settings.service_name,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "healthy", "service": settings.service_name}


@app.get("/health/database")
def database_health():
    healthy, detail = check_database()
    if not healthy:
        logger.error(
            "database_health_check_failed",
            extra={
                "service": settings.service_name,
                "component": "database",
                "error_type": detail,
            },
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": settings.service_name,
                "component": "database",
                "error_type": detail,
            },
        )

    return {
        "status": "healthy",
        "service": settings.service_name,
        "component": "database",
    }


@app.get("/metrics/orders")
def order_metric_snapshot(window_seconds: float = 5.0) -> dict[str, object]:
    try:
        return order_metrics.snapshot(window_seconds)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/internal/faults/http-500")
def http_500_fault_status() -> dict[str, object]:
    return fault_state.status()


@app.post("/internal/faults/http-500")
def toggle_http_500_fault(request: FaultToggleRequest) -> dict[str, object]:
    result = fault_state.set_http_500(request.enabled)
    logger.warning(
        "controlled_fault_toggled",
        extra={
            "service": settings.service_name,
            "fault_type": "http_500_failure",
            "enabled": request.enabled,
        },
    )
    return result


@app.post("/orders", status_code=status.HTTP_201_CREATED)
def submit_order(order: OrderRequest) -> dict[str, object]:
    if fault_state.http_500_enabled():
        order_metrics.record(500)
        logger.error(
            "forced_http_500",
            extra={
                "service": settings.service_name,
                "event_type": "order_creation",
                "fault_type": "http_500_failure",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="controlled application failure",
        )

    try:
        created = create_order(order.customer_id, order.item, order.quantity)
    except SQLAlchemyError as exc:
        order_metrics.record(503)
        logger.exception(
            "order_creation_failed",
            extra={
                "service": settings.service_name,
                "component": "database",
                "event_type": "order_creation",
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="order storage is temporarily unavailable",
        ) from exc

    order_metrics.record(201)
    logger.info(
        "order_created",
        extra={
            "service": settings.service_name,
            "event_type": "order_creation",
            "order_id": created["id"],
        },
    )
    return created
