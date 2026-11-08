"""Тесты детерминированной canary-защиты и безопасного JSONL-журнала."""

import json
from pathlib import Path

import pytest
from openai.types.responses import ResponseReasoningItem

from mail_agent.canary_guard import CanaryBlockedError, CanaryGuard
from mail_agent.llm_client import LlmResponse, LlmToolCall


def test_token_in_long_response_creates_one_safe_event(tmp_path: Path) -> None:
    """Блокирует совпадения и пишет одно событие без секретов."""
    canary_token = "secret-canary-token"
    blocked_text = f"До {canary_token} после {canary_token}"
    events_path = tmp_path / "security_events.jsonl"
    guard = CanaryGuard(events_path)
    response = LlmResponse(
        response_id="response-1",
        status="completed",
        output_text=blocked_text,
        refusal_text=f"Отказ: {canary_token}",
        output_items=[{"type": "message", "text": blocked_text}],
        tool_calls=[],
        raw_response_json=json.dumps({"output": [{"text": blocked_text}]}),
    )

    with pytest.raises(CanaryBlockedError):
        guard.validate_response(
            response, canary_token, "run-1", "instructions"
        )

    event_lines = events_path.read_text(encoding="utf-8").splitlines()
    assert len(event_lines) == 1
    event = json.loads(event_lines[0])
    assert event.keys() == {
        "timestamp",
        "run_id",
        "location",
        "reason",
        "action",
        "event_type",
    }
    assert event["run_id"] == "run-1"
    assert event["location"] == "LLM_RESPONSE"
    assert event["reason"] == "CANARY_MATCH"
    assert event["action"] == "BLOCKED"
    assert event["event_type"] == "SYSTEM_PROMPT_LEAK"
    assert canary_token not in event_lines[0]
    assert blocked_text not in event_lines[0]


def test_token_only_in_arguments_has_tool_arguments_location(
    tmp_path: Path,
) -> None:
    """Не относит копию аргументов в output и raw JSON к тексту ответа."""
    canary_token = "arguments-canary"
    arguments = {"body": f"Текст {canary_token}"}
    arguments_json = json.dumps(arguments)
    function_call = {
        "type": "function_call",
        "call_id": "call-1",
        "name": "unknown_tool",
        "arguments": arguments_json,
    }
    events_path = tmp_path / "security_events.jsonl"
    guard = CanaryGuard(events_path)
    response = LlmResponse(
        response_id="response-2",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[function_call],
        tool_calls=[
            LlmToolCall(
                call_id="call-1",
                name="unknown_tool",
                arguments_json=arguments_json,
                arguments=arguments,
            )
        ],
        raw_response_json=json.dumps({"output": [function_call]}),
    )

    with pytest.raises(CanaryBlockedError):
        guard.validate_response(
            response, canary_token, "run-2", "instructions"
        )

    event = json.loads(events_path.read_text(encoding="utf-8"))
    assert event["location"] == "TOOL_ARGUMENTS"


def test_response_location_has_priority_over_arguments(tmp_path: Path) -> None:
    """Выбирает LLM_RESPONSE при одновременном совпадении в двух категориях."""
    canary_token = "shared-canary"
    events_path = tmp_path / "security_events.jsonl"
    guard = CanaryGuard(events_path)
    response = LlmResponse(
        response_id="response-3",
        status="completed",
        output_text=f"Ответ {canary_token}",
        refusal_text=None,
        output_items=[],
        tool_calls=[
            LlmToolCall(
                call_id="call-3",
                name="create_draft",
                arguments_json=json.dumps({"body": canary_token}),
                arguments={"body": canary_token},
            )
        ],
        raw_response_json="{}",
    )

    with pytest.raises(CanaryBlockedError):
        guard.validate_response(
            response, canary_token, "run-3", "instructions"
        )

    event = json.loads(events_path.read_text(encoding="utf-8"))
    assert event["location"] == "LLM_RESPONSE"


def test_nested_json_and_unicode_escapes_are_decoded(tmp_path: Path) -> None:
    """Находит токен во вложенной строке с Unicode escapes."""
    canary_token = "unicode-token"
    escaped_token = "".join(
        f"\\u{ord(character):04x}" for character in canary_token
    )
    nested_json = json.dumps({"payload": f'{{"items":["{escaped_token}"]}}'})
    events_path = tmp_path / "security_events.jsonl"
    guard = CanaryGuard(events_path)
    response = LlmResponse(
        response_id="response-4",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[],
        tool_calls=[
            LlmToolCall(
                call_id="call-4",
                name="create_draft",
                arguments_json=nested_json,
                arguments=json.loads(nested_json),
            )
        ],
        raw_response_json="{}",
    )

    with pytest.raises(CanaryBlockedError):
        guard.validate_response(
            response, canary_token, "run-4", "instructions"
        )

    event = json.loads(events_path.read_text(encoding="utf-8"))
    assert event["location"] == "TOOL_ARGUMENTS"


def test_invalid_unknown_tool_arguments_cannot_bypass_guard(
    tmp_path: Path,
) -> None:
    """Блокирует токен до разбора повреждённых аргументов."""
    canary_token = "invalid-json-canary"
    arguments_json = f'{{"body":"{canary_token}"'
    function_call = {
        "type": "function_call",
        "call_id": "call-invalid",
        "name": "unknown_tool",
        "arguments": arguments_json,
    }
    events_path = tmp_path / "security_events.jsonl"
    guard = CanaryGuard(events_path)
    response = LlmResponse(
        response_id="response-invalid",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[function_call],
        tool_calls=[
            LlmToolCall(
                call_id="call-invalid",
                name="unknown_tool",
                arguments_json=arguments_json,
                arguments=None,
            )
        ],
        raw_response_json=json.dumps({"output": [function_call]}),
    )

    with pytest.raises(CanaryBlockedError):
        guard.validate_response(
            response,
            canary_token,
            "run-invalid",
            "instructions",
        )

    event = json.loads(events_path.read_text(encoding="utf-8"))
    assert event["location"] == "TOOL_ARGUMENTS"


def test_partial_token_and_safe_response_are_allowed(tmp_path: Path) -> None:
    """Не блокирует частичное совпадение и не создаёт журнал без утечки."""
    events_path = tmp_path / "security_events.jsonl"
    guard = CanaryGuard(events_path)
    response = LlmResponse(
        response_id="response-safe",
        status="completed",
        output_text="Начало secret-canary",
        refusal_text=None,
        output_items=[{"type": "message", "text": "Безопасно"}],
        tool_calls=[],
        raw_response_json="{}",
    )

    guard.validate_response(
        response,
        "secret-canary-token",
        "run-safe",
        "instructions",
    )

    assert not events_path.exists()


def test_token_only_in_decoded_raw_response_is_blocked(tmp_path: Path) -> None:
    """Проверяет raw response вне нормализованных полей."""
    canary_token = "raw-only-token"
    escaped_token = "".join(
        f"\\u{ord(character):04x}" for character in canary_token
    )
    events_path = tmp_path / "security_events.jsonl"
    guard = CanaryGuard(events_path)
    response = LlmResponse(
        response_id="response-raw",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[],
        tool_calls=[],
        raw_response_json=f'{{"hidden":"{escaped_token}"}}',
    )

    with pytest.raises(CanaryBlockedError):
        guard.validate_response(
            response, canary_token, "run-raw", "instructions"
        )

    event = json.loads(events_path.read_text(encoding="utf-8"))
    assert event["location"] == "LLM_RESPONSE"


def test_token_in_reasoning_output_item_is_blocked(tmp_path: Path) -> None:
    """Блокирует canary в reasoning item реальной структуры OpenAI SDK."""
    canary_token = "reasoning-canary"
    expected_instructions = f"Доверенные instructions: {canary_token}"
    reasoning_item = ResponseReasoningItem.model_validate(
        {
            "id": "reasoning-1",
            "type": "reasoning",
            "summary": [
                {"type": "summary_text", "text": f"Утечка {canary_token}"}
            ],
        }
    )
    output_item = reasoning_item.model_dump(mode="json")
    events_path = tmp_path / "security_events.jsonl"
    response = LlmResponse(
        response_id="response-reasoning",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[output_item],
        tool_calls=[],
        raw_response_json=json.dumps(
            {"instructions": expected_instructions, "output": [output_item]},
            ensure_ascii=False,
        ),
    )

    with pytest.raises(CanaryBlockedError):
        CanaryGuard(events_path).validate_response(
            response,
            canary_token,
            "run-reasoning",
            expected_instructions,
        )

    event = json.loads(events_path.read_text(encoding="utf-8"))
    assert event["location"] == "LLM_RESPONSE"


def test_changed_returned_instructions_without_token_are_blocked(
    tmp_path: Path,
) -> None:
    """Останавливает запуск при изменённом request-echo без ложного события."""
    events_path = tmp_path / "security_events.jsonl"
    response = LlmResponse(
        response_id="response-changed-instructions",
        status="completed",
        output_text="Безопасный текст",
        refusal_text=None,
        output_items=[],
        tool_calls=[],
        raw_response_json=json.dumps(
            {"instructions": "Изменённые instructions"}
        ),
    )

    with pytest.raises(CanaryBlockedError, match="instructions изменены"):
        CanaryGuard(events_path).validate_response(
            response,
            "expected-canary",
            "run-changed-instructions",
            "Исходные instructions expected-canary",
        )

    assert not events_path.exists()


def test_changed_returned_instructions_with_token_create_one_event(
    tmp_path: Path,
) -> None:
    """Журналирует реальный canary в недоверенных изменённых instructions."""
    canary_token = "changed-instructions-canary"
    events_path = tmp_path / "security_events.jsonl"
    response = LlmResponse(
        response_id="response-changed-canary",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[],
        tool_calls=[],
        raw_response_json=json.dumps(
            {"instructions": f"Изменено: {canary_token}"},
            ensure_ascii=False,
        ),
    )

    with pytest.raises(CanaryBlockedError):
        CanaryGuard(events_path).validate_response(
            response,
            canary_token,
            "run-changed-canary",
            f"Исходные instructions: {canary_token}",
        )

    event_lines = events_path.read_text(encoding="utf-8").splitlines()
    assert len(event_lines) == 1
    assert json.loads(event_lines[0])["location"] == "LLM_RESPONSE"
    assert canary_token not in event_lines[0]


def test_nested_instructions_with_token_are_not_excluded(
    tmp_path: Path,
) -> None:
    """Проверяет вложенные instructions при точном request-echo."""
    canary_token = "nested-instructions-canary"
    expected_instructions = f"Исходные instructions: {canary_token}"
    output_item = {
        "type": "message",
        "instructions": f"Сгенерировано: {canary_token}",
    }
    events_path = tmp_path / "security_events.jsonl"
    response = LlmResponse(
        response_id="response-nested-instructions",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[output_item],
        tool_calls=[],
        raw_response_json=json.dumps(
            {"instructions": expected_instructions, "output": [output_item]},
            ensure_ascii=False,
        ),
    )

    with pytest.raises(CanaryBlockedError):
        CanaryGuard(events_path).validate_response(
            response,
            canary_token,
            "run-nested-instructions",
            expected_instructions,
        )

    assert json.loads(events_path.read_text(encoding="utf-8"))["location"] == (
        "LLM_RESPONSE"
    )


@pytest.mark.parametrize(
    "raw_response_json",
    ["{}", '{"instructions": null}'],
)
def test_missing_or_null_returned_instructions_are_allowed(
    tmp_path: Path,
    raw_response_json: str,
) -> None:
    """Проверяет ответ при отсутствующем или null request-echo."""
    events_path = tmp_path / "security_events.jsonl"
    response = LlmResponse(
        response_id="response-compatible",
        status="completed",
        output_text="Безопасный ответ",
        refusal_text=None,
        output_items=[],
        tool_calls=[],
        raw_response_json=raw_response_json,
    )

    CanaryGuard(events_path).validate_response(
        response,
        "compatible-canary",
        "run-compatible",
        "Исходные instructions compatible-canary",
    )

    assert not events_path.exists()


def test_invalid_raw_response_json_is_blocked_without_false_event(
    tmp_path: Path,
) -> None:
    """Блокирует невалидный raw JSON без ложной утечки."""
    events_path = tmp_path / "security_events.jsonl"
    response = LlmResponse(
        response_id="response-invalid-raw",
        status="completed",
        output_text="Безопасный ответ",
        refusal_text=None,
        output_items=[],
        tool_calls=[],
        raw_response_json="{invalid",
    )

    with pytest.raises(CanaryBlockedError, match="raw response"):
        CanaryGuard(events_path).validate_response(
            response,
            "invalid-raw-canary",
            "run-invalid-raw",
            "Исходные instructions invalid-raw-canary",
        )

    assert not events_path.exists()


def test_log_write_failure_still_blocks(tmp_path: Path) -> None:
    """Сохраняет блокировку, когда путь журнала является каталогом."""
    guard = CanaryGuard(tmp_path)
    response = LlmResponse(
        response_id="response-fail-closed",
        status="completed",
        output_text="leaked-token",
        refusal_text=None,
        output_items=[],
        tool_calls=[],
        raw_response_json="{}",
    )

    with pytest.raises(CanaryBlockedError, match="записать не удалось"):
        guard.validate_response(
            response,
            "leaked-token",
            "run-fail-closed",
            "instructions",
        )
