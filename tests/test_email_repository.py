"""Проверки SQLite-репозитория писем и черновиков."""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mail_agent.email_models import EmailThread
from mail_agent.email_repository import (
    EmailThreadNotFoundError,
    SqliteEmailRepository,
)


def test_initialize_schema_creates_tables(database_path: Path) -> None:
    """Проверяет создание трёх согласованных таблиц хранилища."""
    email_repository = SqliteEmailRepository(database_path)
    email_repository.initialize_schema()

    with sqlite3.connect(database_path) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert {"email_threads", "email_messages", "draft_replies"} <= table_names


def test_save_email_threads_persists_messages(
    database_path: Path,
    email_threads: list[EmailThread],
) -> None:
    """Проверяет атомарное сохранение всех учебных цепочек и писем."""
    email_repository = SqliteEmailRepository(database_path)
    email_repository.initialize_schema()
    email_repository.save_email_threads(email_threads)

    with sqlite3.connect(database_path) as connection:
        thread_count = connection.execute(
            "SELECT count(*) FROM email_threads"
        ).fetchone()[0]
        message_count = connection.execute(
            "SELECT count(*) FROM email_messages"
        ).fetchone()[0]
        stored_sent_at = connection.execute(
            "SELECT sent_at FROM email_messages WHERE message_id = 'email-001'"
        ).fetchone()[0]

    assert thread_count == 3
    assert message_count == 7
    assert stored_sent_at == "2026-10-01T08:00:00.100000Z"


def test_save_email_threads_rejects_duplicate_ids(
    email_repository: SqliteEmailRepository,
    email_threads: list[EmailThread],
) -> None:
    """Проверяет запрет неявной перезаписи существующих идентификаторов."""
    with pytest.raises(sqlite3.IntegrityError):
        email_repository.save_email_threads(email_threads)


def test_search_emails_by_query(
    email_repository: SqliteEmailRepository,
) -> None:
    """Проверяет регистронезависимый поиск нескольких русских слов."""
    search_results = email_repository.search_emails(
        "МОДЕРНИЗАЦИИ характеристики"
    )

    assert [result.message_id for result in search_results] == ["email-001"]


def test_search_emails_by_sender(
    email_repository: SqliteEmailRepository,
) -> None:
    """Проверяет точную фильтрацию писем по адресу отправителя."""
    search_results = email_repository.search_emails(
        query="условия",
        sender="SALES@INDUSTRIA-KOMPLEKT.EXAMPLE.COM",
    )

    assert [result.message_id for result in search_results] == ["email-007"]


def test_search_emails_by_date_range(
    email_repository: SqliteEmailRepository,
) -> None:
    """Проверяет включённые UTC-границы и микросекундную точность поиска."""
    search_results = email_repository.search_emails(
        query="узлов",
        date_from=datetime(
            2026,
            10,
            1,
            11,
            0,
            0,
            500000,
            tzinfo=timezone(timedelta(hours=3)),
        ),
        date_to=datetime(
            2026,
            10,
            1,
            11,
            0,
            0,
            900000,
            tzinfo=timezone(timedelta(hours=3)),
        ),
    )

    assert [result.message_id for result in search_results] == ["email-002"]


def test_search_emails_returns_empty_list(
    email_repository: SqliteEmailRepository,
) -> None:
    """Проверяет пустой результат при отсутствии совпадающих слов."""
    assert email_repository.search_emails("компрессор") == []


def test_list_email_messages_returns_chronological_messages(
    email_repository: SqliteEmailRepository,
) -> None:
    """Проверяет устойчивый порядок архива для индексации."""
    email_messages = email_repository.list_email_messages()

    assert [message.message_id for message in email_messages] == [
        "email-001",
        "email-002",
        "email-003",
        "email-004",
        "email-005",
        "email-006",
        "email-007",
    ]


def test_read_thread_returns_chronological_messages(
    email_repository: SqliteEmailRepository,
) -> None:
    """Проверяет хронологию и связи писем с разными микросекундами."""
    email_thread = email_repository.read_thread("thread-001")

    assert [message.message_id for message in email_thread.messages] == [
        "email-001",
        "email-002",
        "email-003",
    ]
    assert email_thread.messages[1].in_reply_to_message_id == "email-001"
    assert email_thread.messages[2].in_reply_to_message_id == "email-002"
    assert all(
        message.sent_at.tzinfo is not None for message in email_thread.messages
    )


def test_read_thread_rejects_unknown_thread_id(
    email_repository: SqliteEmailRepository,
) -> None:
    """Проверяет явную ошибку при чтении отсутствующей цепочки."""
    with pytest.raises(EmailThreadNotFoundError):
        email_repository.read_thread("thread-999")


def test_create_draft_persists_draft(
    email_repository: SqliteEmailRepository,
    database_path: Path,
) -> None:
    """Проверяет сохранение всех полей черновика в SQLite без отправки."""
    draft_reply = email_repository.create_draft(
        thread_id="thread-001",
        recipient="procurement@uralmechanika.example.com",
        subject="Подтверждение резерва",
        body="Оборудование зарезервировано.",
    )

    with sqlite3.connect(database_path) as connection:
        stored_draft = connection.execute(
            """
            SELECT thread_id, recipient, subject, body, created_at
            FROM draft_replies
            WHERE draft_id = ?
            """,
            (draft_reply.draft_id,),
        ).fetchone()

    assert stored_draft is not None
    assert stored_draft[:4] == (
        draft_reply.thread_id,
        draft_reply.recipient,
        draft_reply.subject,
        draft_reply.body,
    )
    assert stored_draft[4].endswith("Z")
    assert draft_reply.created_at.tzinfo is not None


def test_create_draft_rejects_unknown_thread_id(
    email_repository: SqliteEmailRepository,
) -> None:
    """Проверяет запрет черновика для отсутствующей почтовой цепочки."""
    with pytest.raises(EmailThreadNotFoundError):
        email_repository.create_draft(
            thread_id="thread-999",
            recipient="client@example.com",
            subject="Ответ",
            body="Текст ответа",
        )
