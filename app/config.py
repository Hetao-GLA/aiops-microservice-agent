"""Environment-based application configuration."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    service_name: str
    environment: str
    database_url: str


def get_settings() -> Settings:
    return Settings(
        service_name=os.getenv("SERVICE_NAME", "order-service"),
        environment=os.getenv("ENVIRONMENT", "development"),
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@localhost:5432/operations",
        ),
    )


settings = get_settings()

