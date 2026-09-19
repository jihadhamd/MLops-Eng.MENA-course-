"""Configuration management via environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, overridable via environment variables."""

    model_path: Path = Path("models/model.pkl")
    data_path: Path = Path("data/green_tripdata_2026-05.parquet")
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
