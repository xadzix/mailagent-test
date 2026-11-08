"""Конфигурация почтового агента из переменных окружения."""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, SecretStr


class Settings(BaseModel):
    """Хранит валидированные настройки приложения без раскрытия API-ключа."""

    model_config = ConfigDict(frozen=True)

    llm_api_key: SecretStr | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    embedding_model: str | None = None
    database_path: Path = Path("mail_agent.sqlite3")
    chroma_path: Path = Path("chroma_data")


def load_settings() -> Settings:
    """Читает поддерживаемые переменные окружения и валидирует настройки."""
    return Settings(
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_model=os.getenv("LLM_MODEL"),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        embedding_model=os.getenv("EMBEDDING_MODEL"),
        database_path=Path(
            os.getenv("MAIL_AGENT_DATABASE_PATH", "mail_agent.sqlite3")
        ),
        chroma_path=Path(os.getenv("MAIL_AGENT_CHROMA_PATH", "chroma_data")),
    )
