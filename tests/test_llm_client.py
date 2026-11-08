"""Тесты безопасной подготовки ответов OpenAI Responses API."""

import json
from unittest.mock import Mock

from mail_agent.llm_client import MAIL_TOOL_SCHEMAS, OpenAILlmClient


def test_generate_response_uses_safe_responses_api_options() -> None:
    """Проверяет отключение хранения и параллельных вызовов инструментов."""
    openai_client = Mock()
    response = Mock()
    response.id = "response-001"
    response.status = "completed"
    response.output_text = "Готово"
    response.output = []
    response.model_dump_json.return_value = '{"id":"response-001"}'
    openai_client.responses.create.return_value = response

    llm_client = OpenAILlmClient(openai_client=openai_client, model="gpt-test")
    result = llm_client.generate_response(
        instructions="Инструкция",
        input_items=[{"role": "user", "content": "Найди письмо"}],
        tool_schemas=MAIL_TOOL_SCHEMAS,
    )

    openai_client.responses.create.assert_called_once_with(
        model="gpt-test",
        instructions="Инструкция",
        input=[{"role": "user", "content": "Найди письмо"}],
        tools=MAIL_TOOL_SCHEMAS,
        tool_choice="auto",
        parallel_tool_calls=False,
        store=False,
    )
    assert result.output_text == "Готово"


def test_generate_response_uses_empty_tools_without_tool_controls() -> None:
    """Передаёт пустой tools без лишних ограничителей Ollama."""
    openai_client = Mock()
    response = Mock(
        id="response-final",
        status="completed",
        output_text="Итоговая сводка",
        output=[],
    )
    response.model_dump_json.return_value = '{"id":"response-final"}'
    openai_client.responses.create.return_value = response

    result = OpenAILlmClient(openai_client, "gpt-test").generate_response(
        instructions="Инструкция",
        input_items=[{"role": "user", "content": "Подготовь итог"}],
        tool_schemas=[],
    )

    openai_client.responses.create.assert_called_once_with(
        model="gpt-test",
        instructions="Инструкция",
        input=[{"role": "user", "content": "Подготовь итог"}],
        tools=[],
        store=False,
    )
    assert result.output_text == "Итоговая сводка"


def test_generate_response_preserves_output_items_and_tool_call() -> None:
    """Сохраняет output items и декодирует function call."""
    openai_client = Mock()
    function_call_item = Mock()
    function_call_item.type = "function_call"
    function_call_item.call_id = "call-001"
    function_call_item.name = "read_thread"
    function_call_item.arguments = '{"thread_id":"thread-001"}'
    function_call_item.model_dump.return_value = {
        "type": "function_call",
        "call_id": "call-001",
        "name": "read_thread",
        "arguments": function_call_item.arguments,
    }
    message_item = Mock()
    message_item.type = "message"
    message_item.content = []
    message_item.model_dump.return_value = {
        "type": "message",
        "content": [{"type": "output_text", "text": "Читаю цепочку"}],
    }
    response = Mock(
        id="response-002",
        status="completed",
        output_text="Читаю цепочку",
        output=[function_call_item, message_item],
    )
    response.model_dump_json.return_value = '{"id":"response-002"}'
    openai_client.responses.create.return_value = response

    result = OpenAILlmClient(openai_client, "gpt-test").generate_response(
        instructions="Инструкция",
        input_items=[],
        tool_schemas=MAIL_TOOL_SCHEMAS,
    )

    assert len(result.output_items) == 2
    assert result.tool_calls[0].call_id == "call-001"
    assert result.tool_calls[0].arguments == {"thread_id": "thread-001"}


def test_generate_response_preserves_invalid_function_arguments() -> None:
    """Не исполняет и не скрывает факт невалидного JSON в аргументах модели."""
    openai_client = Mock()
    function_call_item = Mock()
    function_call_item.type = "function_call"
    function_call_item.call_id = "call-invalid"
    function_call_item.name = "read_thread"
    function_call_item.arguments = "{invalid"
    function_call_item.model_dump.return_value = {"type": "function_call"}
    response = Mock(
        id="response-invalid",
        status="completed",
        output_text="",
        output=[function_call_item],
    )
    response.model_dump_json.return_value = "{}"
    openai_client.responses.create.return_value = response

    result = OpenAILlmClient(openai_client, "gpt-test").generate_response(
        instructions="Инструкция",
        input_items=[],
        tool_schemas=MAIL_TOOL_SCHEMAS,
    )

    assert result.tool_calls[0].arguments is None
    assert result.tool_calls[0].arguments_json == "{invalid"


def test_generate_response_collects_refusal_text() -> None:
    """Извлекает отказ модели из message content без потери output item."""
    openai_client = Mock()
    refusal_item = Mock(type="refusal", refusal="Не могу выполнить запрос")
    message_item = Mock(type="message", content=[refusal_item])
    message_item.model_dump.return_value = {
        "type": "message",
        "content": [{"type": "refusal", "refusal": refusal_item.refusal}],
    }
    response = Mock(
        id="response-refusal",
        status="completed",
        output_text="",
        output=[message_item],
    )
    response.model_dump_json.return_value = "{}"
    openai_client.responses.create.return_value = response

    result = OpenAILlmClient(openai_client, "gpt-test").generate_response(
        instructions="Инструкция",
        input_items=[],
        tool_schemas=MAIL_TOOL_SCHEMAS,
    )

    assert result.refusal_text == "Не могу выполнить запрос"


def test_mail_tool_schemas_are_strict_and_limited_to_three_tools() -> None:
    """Проверяет строгие JSON-схемы ровно трёх утверждённых инструментов."""
    assert [schema["name"] for schema in MAIL_TOOL_SCHEMAS] == [
        "search_emails",
        "read_thread",
        "create_draft",
    ]
    for schema in MAIL_TOOL_SCHEMAS:
        parameters = schema["parameters"]
        assert isinstance(parameters, dict)
        assert schema["strict"] is True
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(parameters["properties"])

    serialized_schemas = json.dumps(MAIL_TOOL_SCHEMAS, ensure_ascii=False)
    assert "send_email" not in serialized_schemas
