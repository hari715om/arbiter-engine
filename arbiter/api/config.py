from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/arbiter"
    redis_url: str = "redis://localhost:6379/0"
    scheduler_type: str = "utility"
    execution_mode: str = "simulated"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    heartbeat_timeout: float = 30.0
    schedule_interval: float = 5.0

    model_config = {"env_prefix": "ARBITER_"}


settings = Settings()
