"""Клиент OpenAI Responses API и модели его проверяемого результата."""

import json
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field


class LlmToolCall(BaseModel):
    """Описывает предложенный моделью вызов инструмента и его аргументы."""

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments_json: str
    arguments: dict[str, object] | None


class LlmResponse(BaseModel):
    """Хранит нормализованный ответ модели для последующей проверки агентом."""

    response_id: str
    status: str
    output_text: str
    refusal_text: str | None
    output_items: list[dict[str, object]]
    tool_calls: list[LlmToolCall]
    raw_response_json: str


MAIL_TOOL_SCHEMAS: list[dict[str, object]] = [
    {
        "type": "function",
        "name": "search_emails",
        "description": "Ищет письма по словам, отправителю и диапазону дат.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "sender": {"type": ["string", "null"]},
                "date_from": {
                    "type": ["string", "null"],
                    "format": "date-time",
                },
                "date_to": {
                    "type": ["string", "null"],
                    "format": "date-time",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["query", "sender", "date_from", "date_to", "limit"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_thread",
        "description": "Возвращает полную почтовую цепочку по её ID.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"thread_id": {"type": "string", "minLength": 1}},
            "required": ["thread_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "create_draft",
        "description": "Сохраняет черновик ответа без отправки письма.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "thread_id": {"type": "string", "minLength": 1},
                "recipient": {"type": "string", "minLength": 1},
                "subject": {"type": "string", "minLength": 1},
                "body": {"type": "string", "minLength": 1},
            },
            "required": ["thread_id", "recipient", "subject", "body"],
            "additionalProperties": False,
        },
    },
]


class OpenAILlmClient:
    """Вызывает Responses API без хранения и параллельных tools."""

    def __init__(self, openai_client: OpenAI, model: str) -> None:
        """Сохраняет готовый SDK-клиент и имя выбранной модели."""
        self._openai_client = openai_client
        self._model = model

    def generate_response(
        self,
        instructions: str,
        input_items: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
    ) -> LlmResponse:
        """Получает ответ и декодирует вызовы без исполнения."""
        if tool_schemas:
            response = self._openai_client.responses.create(
                model=self._model,
                instructions=instructions,
                input=input_items,
                tools=tool_schemas,
                tool_choice="auto",
                parallel_tool_calls=False,
                store=False,
            )
        else:
            response = self._openai_client.responses.create(
                model=self._model,
                instructions=instructions,
                input=input_items,
                tools=[],
                store=False,
            )
        output_items = [
            item.model_dump(mode="json") for item in response.output
        ]
        tool_calls: list[LlmToolCall] = []
        refusal_parts: list[str] = []

        for item in response.output:
            if item.type == "function_call":
                arguments: dict[str, object] | None = None
                try:
                    decoded_arguments: Any = json.loads(item.arguments)
                except (json.JSONDecodeError, TypeError):
                    decoded_arguments = None
                if isinstance(decoded_arguments, dict):
                    arguments = decoded_arguments
                tool_calls.append(
                    LlmToolCall(
                        call_id=item.call_id,
                        name=item.name,
                        arguments_json=item.arguments,
                        arguments=arguments,
                    )
                )
            if item.type == "message":
                for content_item in item.content:
                    if content_item.type == "refusal":
                        refusal_parts.append(content_item.refusal)

        refusal_text = "\n".join(refusal_parts) or None
        return LlmResponse(
            response_id=response.id,
            status=response.status,
            output_text=response.output_text,
            refusal_text=refusal_text,
            output_items=output_items,
            tool_calls=tool_calls,
            raw_response_json=response.model_dump_json(),
        )
