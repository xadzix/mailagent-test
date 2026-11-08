"""Три разрешённых инструмента для работы с локальной почтой."""

from typing import Annotated

from pydantic import AwareDatetime, Field, StringConstraints, validate_call

from mail_agent.email_models import DraftReply, EmailSearchResult, EmailThread
from mail_agent.email_repository import SqliteEmailRepository


class MailTools:
    """Предоставляет контракты поиска, чтения и черновика."""

    def __init__(self, email_repository: SqliteEmailRepository) -> None:
        """Связывает инструменты с единственным SQLite-репозиторием."""
        self.email_repository = email_repository

    @validate_call
    def search_emails(
        self,
        query: Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1),
        ],
        sender: Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1),
        ]
        | None = None,
        date_from: AwareDatetime | None = None,
        date_to: AwareDatetime | None = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
    ) -> list[EmailSearchResult]:
        """Проверяет параметры и ищет письма через локальное хранилище."""
        if (
            date_from is not None
            and date_to is not None
            and date_from > date_to
        ):
            raise ValueError(
                "Начало диапазона не может быть позже его окончания"
            )
        return self.email_repository.search_emails(
            query=query,
            sender=sender,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )

    @validate_call
    def read_thread(
        self,
        thread_id: Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1),
        ],
    ) -> EmailThread:
        """Проверяет идентификатор и читает полную почтовую цепочку."""
        return self.email_repository.read_thread(thread_id)

    @validate_call
    def create_draft(
        self,
        thread_id: Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1),
        ],
        recipient: Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1),
        ],
        subject: Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1),
        ],
        body: Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1),
        ],
    ) -> DraftReply:
        """Проверяет обязательные поля и сохраняет черновик без отправки."""
        return self.email_repository.create_draft(
            thread_id=thread_id,
            recipient=recipient,
            subject=subject,
            body=body,
        )
