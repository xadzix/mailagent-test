"""Общие fixtures для тестов почтового хранилища и инструментов."""

from pathlib import Path

import pytest
from pydantic import TypeAdapter

from mail_agent.email_models import EmailThread
from mail_agent.email_repository import SqliteEmailRepository
from mail_agent.mail_tools import MailTools


@pytest.fixture
def email_threads() -> list[EmailThread]:
    """Загружает и валидирует все вымышленные цепочки из JSON-файла."""
    data_path = Path(__file__).parents[1] / "data" / "emails.json"
    return TypeAdapter(list[EmailThread]).validate_json(
        data_path.read_text(encoding="utf-8")
    )


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    """Возвращает отдельный SQLite-файл для одного теста."""
    return tmp_path / "mail_agent.sqlite3"


@pytest.fixture
def email_repository(
    database_path: Path,
    email_threads: list[EmailThread],
) -> SqliteEmailRepository:
    """Создаёт схему и сохраняет в неё проверенные учебные письма."""
    email_repository = SqliteEmailRepository(database_path)
    email_repository.initialize_schema()
    email_repository.save_email_threads(email_threads)
    return email_repository


@pytest.fixture
def mail_tools(email_repository: SqliteEmailRepository) -> MailTools:
    """Создаёт три почтовых инструмента над временным хранилищем."""
    return MailTools(email_repository)
