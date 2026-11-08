"""Модели писем, цепочек, результатов поиска и черновиков."""

from typing import Self

from pydantic import AwareDatetime, BaseModel, Field, model_validator

EMAIL_PREVIEW_LENGTH = 200


class EmailMessage(BaseModel):
    """Представляет полное сообщение с адресатами и связью внутри цепочки."""

    message_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    sender: str = Field(min_length=1)
    recipients: list[str] = Field(min_length=1)
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)
    sent_at: AwareDatetime
    in_reply_to_message_id: str | None = None


class EmailThread(BaseModel):
    """Объединяет согласованные сообщения одной почтовой цепочки."""

    thread_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    messages: list[EmailMessage] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_thread_messages(self) -> Self:
        """Проверяет идентификаторы и ссылки сообщений внутри одной цепочки."""
        message_ids = [message.message_id for message in self.messages]
        if len(message_ids) != len(set(message_ids)):
            raise ValueError(
                "Идентификаторы сообщений цепочки должны быть уникальны"
            )

        known_message_ids = set(message_ids)
        for message in self.messages:
            if message.thread_id != self.thread_id:
                raise ValueError(
                    "Все сообщения должны принадлежать указанной цепочке"
                )
            if message.in_reply_to_message_id == message.message_id:
                raise ValueError("Письмо не может ссылаться само на себя")
            if (
                message.in_reply_to_message_id is not None
                and message.in_reply_to_message_id not in known_message_ids
            ):
                raise ValueError(
                    "Ответ может ссылаться только на письмо своей цепочки"
                )

        return self


class EmailSearchResult(BaseModel):
    """Содержит краткие сведения о письме, найденном поиском."""

    message_id: str
    thread_id: str
    sender: str
    subject: str
    sent_at: AwareDatetime
    body_preview: str


class DraftReply(BaseModel):
    """Описывает сохранённый, но не отправленный ответ на цепочку."""

    draft_id: str
    thread_id: str
    recipient: str
    subject: str
    body: str
    created_at: AwareDatetime
