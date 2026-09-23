"""Database connection, schema setup, and health checks."""

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings


engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=5,
    connect_args={"connect_timeout": 2},
)


def initialise_database(database_engine: Engine = engine) -> None:
    statement = text(
        """
        CREATE TABLE IF NOT EXISTS orders (
            id BIGSERIAL PRIMARY KEY,
            customer_id VARCHAR(64) NOT NULL,
            item VARCHAR(128) NOT NULL,
            quantity INTEGER NOT NULL CHECK (quantity > 0),
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    with database_engine.begin() as connection:
        connection.execute(statement)


def check_database(database_engine: Engine = engine) -> tuple[bool, str]:
    try:
        with database_engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True, "database connection is healthy"
    except SQLAlchemyError as exc:
        # Return only the exception type. Full connection errors can expose
        # credentials and are still available through the structured log.
        return False, type(exc).__name__


def create_order(
    customer_id: str,
    item: str,
    quantity: int,
    database_engine: Engine = engine,
) -> dict[str, object]:
    statement = text(
        """
        INSERT INTO orders (customer_id, item, quantity)
        VALUES (:customer_id, :item, :quantity)
        RETURNING id, customer_id, item, quantity, created_at
        """
    )
    with database_engine.begin() as connection:
        row = connection.execute(
            statement,
            {"customer_id": customer_id, "item": item, "quantity": quantity},
        ).mappings().one()
    return dict(row)

