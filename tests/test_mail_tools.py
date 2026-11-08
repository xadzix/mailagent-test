"""Проверки контрактов трёх внешних почтовых инструментов."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mail_agent.mail_tools import MailTools


def test_search_emails_returns_repository_results(
    mail_tools: MailTools,
) -> None:
    """Проверяет возврат найденных писем через внешний инструмент."""
    search_results = mail_tools.search_emails("редуктора")

    assert [result.message_id for result in search_results] == [
        "email-005",
        "email-004",
    ]


def test_search_emails_rejects_blank_query(mail_tools: MailTools) -> None:
    """Проверяет отклонение пустого текстового запроса."""
    with pytest.raises(ValidationError):
        mail_tools.search_emails("   ")


def test_search_emails_rejects_blank_sender(mail_tools: MailTools) -> None:
    """Проверяет отклонение пустого переданного адреса отправителя."""
    with pytest.raises(ValidationError):
        mail_tools.search_emails(query="насос", sender="   ")


def test_search_emails_rejects_invalid_date_range(
    mail_tools: MailTools,
) -> None:
    """Проверяет отклонение диапазона с началом после окончания."""
    with pytest.raises(ValueError):
        mail_tools.search_emails(
            query="насос",
            date_from=datetime(2026, 10, 9, tzinfo=UTC),
            date_to=datetime(2026, 10, 8, tzinfo=UTC),
        )


def test_search_emails_rejects_invalid_limit(mail_tools: MailTools) -> None:
    """Проверяет допустимый диапазон количества результатов от 1 до 100."""
    for invalid_limit in (0, 101):
        with pytest.raises(ValidationError):
            mail_tools.search_emails(query="насос", limit=invalid_limit)


def test_read_thread_rejects_blank_thread_id(mail_tools: MailTools) -> None:
    """Проверяет отклонение пустого идентификатора цепочки."""
    with pytest.raises(ValidationError):
        mail_tools.read_thread("   ")


def test_create_draft_returns_saved_draft(mail_tools: MailTools) -> None:
    """Проверяет создание только черновика без инструмента отправки письма."""
    draft_reply = mail_tools.create_draft(
        thread_id="thread-003",
        recipient="finance@neva-process.example.com",
        subject="Счёт на насосы НЦ-80",
        body="Счёт подготовлен и приложен к ответу.",
    )

    assert draft_reply.thread_id == "thread-003"
    assert not hasattr(mail_tools, "send_email")


def test_create_draft_rejects_blank_fields(mail_tools: MailTools) -> None:
    """Проверяет отклонение каждого пустого обязательного поля черновика."""
    invalid_fields = [
        ("", "client@example.com", "Тема", "Текст"),
        ("thread-001", "", "Тема", "Текст"),
        ("thread-001", "client@example.com", "", "Текст"),
        ("thread-001", "client@example.com", "Тема", ""),
    ]

    for thread_id, recipient, subject, body in invalid_fields:
        with pytest.raises(ValidationError):
            mail_tools.create_draft(
                thread_id=thread_id,
                recipient=recipient,
                subject=subject,
                body=body,
            )
