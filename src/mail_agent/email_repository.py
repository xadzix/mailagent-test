"""SQLite-хранилище почтовых цепочек, сообщений и черновиков."""

import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from mail_agent.email_models import (
    EMAIL_PREVIEW_LENGTH,
    DraftReply,
    EmailMessage,
    EmailSearchResult,
    EmailThread,
)


class EmailThreadNotFoundError(LookupError):
    """Указывает, что запрошенная почтовая цепочка отсутствует в хранилище."""


class SqliteEmailRepository:
    """Сохраняет письма и черновики в одном локальном файле SQLite."""

    def __init__(self, database_path: Path) -> None:
        """Запоминает путь к SQLite-файлу для последующих операций."""
        self.database_path = database_path

    def initialize_schema(self) -> None:
        """Создаёт таблицы и индексы, не изменяя уже существующие данные."""
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS email_threads (
                    thread_id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL CHECK (trim(subject) <> '')
                );

                CREATE TABLE IF NOT EXISTS email_messages (
                    message_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    sender TEXT NOT NULL CHECK (trim(sender) <> ''),
                    recipients_json TEXT NOT NULL,
                    subject TEXT NOT NULL CHECK (trim(subject) <> ''),
                    body TEXT NOT NULL CHECK (trim(body) <> ''),
                    sent_at TEXT NOT NULL,
                    in_reply_to_message_id TEXT,
                    FOREIGN KEY (thread_id)
                        REFERENCES email_threads (thread_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (in_reply_to_message_id)
                        REFERENCES email_messages (message_id)
                        ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS draft_replies (
                    draft_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    recipient TEXT NOT NULL CHECK (trim(recipient) <> ''),
                    subject TEXT NOT NULL CHECK (trim(subject) <> ''),
                    body TEXT NOT NULL CHECK (trim(body) <> ''),
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (thread_id)
                        REFERENCES email_threads (thread_id)
                        ON DELETE RESTRICT
                );

                CREATE INDEX IF NOT EXISTS idx_email_messages_thread_sent_at
                    ON email_messages (thread_id, sent_at, message_id);

                CREATE INDEX IF NOT EXISTS idx_email_messages_sender
                    ON email_messages (sender);

                CREATE INDEX IF NOT EXISTS idx_email_messages_sent_at
                    ON email_messages (sent_at);
                """
            )

    def save_email_threads(self, email_threads: Sequence[EmailThread]) -> None:
        """Атомарно сохраняет проверенные цепочки и связи между их письмами."""
        with closing(self._connect()) as connection, connection:
            for email_thread in email_threads:
                validated_thread = EmailThread.model_validate(
                    email_thread.model_dump()
                )
                connection.execute(
                    "INSERT INTO email_threads (thread_id, subject) "
                    "VALUES (?, ?)",
                    (validated_thread.thread_id, validated_thread.subject),
                )

                for message in validated_thread.messages:
                    connection.execute(
                        """
                        INSERT INTO email_messages (
                            message_id,
                            thread_id,
                            sender,
                            recipients_json,
                            subject,
                            body,
                            sent_at,
                            in_reply_to_message_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                        """,
                        (
                            message.message_id,
                            message.thread_id,
                            message.sender,
                            json.dumps(message.recipients, ensure_ascii=False),
                            message.subject,
                            message.body,
                            self._format_datetime(message.sent_at),
                        ),
                    )

                for message in validated_thread.messages:
                    if message.in_reply_to_message_id is not None:
                        connection.execute(
                            """
                            UPDATE email_messages
                            SET in_reply_to_message_id = ?
                            WHERE message_id = ?
                            """,
                            (
                                message.in_reply_to_message_id,
                                message.message_id,
                            ),
                        )

    def search_emails(
        self,
        query: str,
        sender: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 20,
    ) -> list[EmailSearchResult]:
        """Ищет письма по словам, отправителю и диапазону дат."""
        where_clauses: list[str] = []
        parameters: list[str] = []

        if sender is not None:
            where_clauses.append("lower(sender) = lower(?)")
            parameters.append(sender)
        if date_from is not None:
            where_clauses.append("sent_at >= ?")
            parameters.append(self._format_datetime(date_from))
        if date_to is not None:
            where_clauses.append("sent_at <= ?")
            parameters.append(self._format_datetime(date_to))

        where_sql = ""
        if where_clauses:
            where_sql = f"WHERE {' AND '.join(where_clauses)}"

        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                f"""
                SELECT message_id, thread_id, sender, subject, body, sent_at
                FROM email_messages
                {where_sql}
                ORDER BY sent_at DESC, message_id ASC
                """,  # noqa: S608
                parameters,
            ).fetchall()

        search_terms = query.casefold().split()
        search_results: list[EmailSearchResult] = []
        for row in rows:
            searchable_text = f"{row['subject']} {row['body']}".casefold()
            if not all(
                search_term in searchable_text for search_term in search_terms
            ):
                continue

            normalized_body = " ".join(row["body"].split())
            search_results.append(
                EmailSearchResult(
                    message_id=row["message_id"],
                    thread_id=row["thread_id"],
                    sender=row["sender"],
                    subject=row["subject"],
                    sent_at=self._parse_datetime(row["sent_at"]),
                    body_preview=normalized_body[:EMAIL_PREVIEW_LENGTH],
                )
            )
            if len(search_results) == limit:
                break

        return search_results

    def list_email_messages(self) -> list[EmailMessage]:
        """Возвращает архив в устойчивом хронологическом порядке."""
        with closing(self._connect()) as connection, connection:
            message_rows = connection.execute(
                """
                SELECT
                    message_id,
                    thread_id,
                    sender,
                    recipients_json,
                    subject,
                    body,
                    sent_at,
                    in_reply_to_message_id
                FROM email_messages
                ORDER BY sent_at ASC, message_id ASC
                """
            ).fetchall()

        return [self._row_to_email_message(row) for row in message_rows]

    def read_thread(self, thread_id: str) -> EmailThread:
        """Возвращает цепочку от раннего письма к позднему."""
        with closing(self._connect()) as connection, connection:
            thread_row = connection.execute(
                "SELECT thread_id, subject FROM email_threads "
                "WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if thread_row is None:
                raise EmailThreadNotFoundError(
                    f"Неизвестная цепочка: {thread_id}"
                )

            message_rows = connection.execute(
                """
                SELECT
                    message_id,
                    thread_id,
                    sender,
                    recipients_json,
                    subject,
                    body,
                    sent_at,
                    in_reply_to_message_id
                FROM email_messages
                WHERE thread_id = ?
                ORDER BY sent_at ASC, message_id ASC
                """,
                (thread_id,),
            ).fetchall()

        if not message_rows:
            raise EmailThreadNotFoundError(
                f"Цепочка не содержит писем: {thread_id}"
            )

        messages = [self._row_to_email_message(row) for row in message_rows]
        return EmailThread(
            thread_id=thread_row["thread_id"],
            subject=thread_row["subject"],
            messages=messages,
        )

    def create_draft(
        self,
        thread_id: str,
        recipient: str,
        subject: str,
        body: str,
    ) -> DraftReply:
        """Сохраняет черновик для существующей цепочки без отправки письма."""
        with closing(self._connect()) as connection, connection:
            thread_exists = connection.execute(
                "SELECT 1 FROM email_threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if thread_exists is None:
                raise EmailThreadNotFoundError(
                    f"Неизвестная цепочка: {thread_id}"
                )

            draft_id = str(uuid4())
            created_at = datetime.now(UTC)
            connection.execute(
                """
                INSERT INTO draft_replies (
                    draft_id, thread_id, recipient, subject, body, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    draft_id,
                    thread_id,
                    recipient,
                    subject,
                    body,
                    self._format_datetime(created_at),
                ),
            )

        return DraftReply(
            draft_id=draft_id,
            thread_id=thread_id,
            recipient=recipient,
            subject=subject,
            body=body,
            created_at=created_at,
        )

    def _connect(self) -> sqlite3.Connection:
        """Открывает соединение с внешними ключами и именами полей."""
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _row_to_email_message(self, row: sqlite3.Row) -> EmailMessage:
        """Преобразует строку SQLite в полную модель письма с UTC-датой."""
        return EmailMessage(
            message_id=row["message_id"],
            thread_id=row["thread_id"],
            sender=row["sender"],
            recipients=json.loads(row["recipients_json"]),
            subject=row["subject"],
            body=row["body"],
            sent_at=self._parse_datetime(row["sent_at"]),
            in_reply_to_message_id=row["in_reply_to_message_id"],
        )

    def _format_datetime(self, value: datetime) -> str:
        """Нормализует дату в UTC и формат с шестью знаками микросекунд."""
        if value.utcoffset() is None:
            raise ValueError(
                "Дата должна содержать информацию о часовом поясе"
            )
        return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def _parse_datetime(self, value: str) -> datetime:
        """Читает UTC-формат SQLite как timezone-aware datetime."""
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=UTC
        )
