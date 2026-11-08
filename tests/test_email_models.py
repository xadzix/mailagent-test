"""Проверки согласованности моделей почтовых цепочек."""

import pytest
from pydantic import ValidationError

from mail_agent.email_models import EmailThread


def test_email_thread_accepts_consistent_messages(
    email_threads: list[EmailThread],
) -> None:
    """Проверяет, что согласованные письма образуют корректную цепочку."""
    assert email_threads[0].thread_id == "thread-001"
    assert len(email_threads[0].messages) == 3


def test_email_thread_rejects_foreign_message(
    email_threads: list[EmailThread],
) -> None:
    """Проверяет отклонение письма с идентификатором другой цепочки."""
    thread_data = email_threads[0].model_dump()
    thread_data["messages"][0]["thread_id"] = "thread-999"

    with pytest.raises(ValidationError):
        EmailThread.model_validate(thread_data)


def test_email_thread_rejects_duplicate_message_id(
    email_threads: list[EmailThread],
) -> None:
    """Проверяет отклонение повторяющегося идентификатора письма."""
    thread_data = email_threads[0].model_dump()
    thread_data["messages"][1]["message_id"] = "email-001"

    with pytest.raises(ValidationError):
        EmailThread.model_validate(thread_data)


def test_email_thread_rejects_external_reply_reference(
    email_threads: list[EmailThread],
) -> None:
    """Проверяет отклонение внешней ссылки и ссылки письма на самого себя."""
    external_reference_data = email_threads[0].model_dump()
    external_reference_data["messages"][0]["in_reply_to_message_id"] = (
        "email-999"
    )

    with pytest.raises(ValidationError):
        EmailThread.model_validate(external_reference_data)

    self_reference_data = email_threads[0].model_dump()
    self_reference_data["messages"][0]["in_reply_to_message_id"] = "email-001"

    with pytest.raises(ValidationError):
        EmailThread.model_validate(self_reference_data)
