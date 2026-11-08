"""Детерминированная защита от раскрытия активного canary-токена."""

import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from mail_agent.llm_client import LlmResponse

CANARY_TOKEN_BYTES = 32
BLOCKED_RESPONSE_MESSAGE = (
    "Не удалось безопасно завершить запрос. Текущий запуск остановлен."
)


class CanaryBlockedError(RuntimeError):
    """Сигнализирует о безусловной блокировке текущего запуска агента."""


class CanaryGuard:
    """Проверяет полный ответ LLM и журналирует одно событие утечки."""

    def __init__(self, security_events_path: Path) -> None:
        """Сохраняет путь JSONL-журнала без создания файла заранее."""
        self._security_events_path = security_events_path

    def validate_response(
        self,
        llm_response: LlmResponse,
        canary_token: str,
        run_id: str,
        expected_instructions: str,
    ) -> None:
        """Разрешает ответ либо журналирует и блокирует запуск."""
        if not canary_token:
            raise ValueError("Canary-токен не может быть пустым")
        if not expected_instructions:
            raise ValueError("Ожидаемые instructions не могут быть пустыми")
        location = self._detect_location(
            llm_response,
            canary_token,
            expected_instructions,
        )
        if location is None:
            return
        try:
            self._write_security_event(run_id, location)
        except Exception:
            raise CanaryBlockedError(
                "Ответ заблокирован; событие безопасности записать не удалось"
            ) from None
        raise CanaryBlockedError("Ответ LLM заблокирован CanaryGuard")

    def _detect_location(
        self,
        llm_response: LlmResponse,
        canary_token: str,
        expected_instructions: str,
    ) -> str | None:
        """Отделяет точный request-echo от недоверенного ответа."""
        response_match = self._contains_token(
            (llm_response.response_id, llm_response.status),
            canary_token,
        )
        response_match = response_match or self._contains_token(
            llm_response.output_text,
            canary_token,
        )
        response_match = response_match or self._contains_token(
            llm_response.refusal_text,
            canary_token,
        )
        arguments_match = False
        structured_values: list[object] = list(llm_response.output_items)
        instructions_mismatch = False
        invalid_raw_response = False

        try:
            raw_response_value = json.loads(llm_response.raw_response_json)
        except (json.JSONDecodeError, TypeError):
            raw_response_value = llm_response.raw_response_json
            invalid_raw_response = True
        if isinstance(raw_response_value, Mapping):
            checked_raw_response = dict(raw_response_value)
            if "instructions" in checked_raw_response:
                returned_instructions = checked_raw_response["instructions"]
                if returned_instructions == expected_instructions:
                    checked_raw_response.pop("instructions")
                elif returned_instructions is not None:
                    instructions_mismatch = True
            structured_values.append(checked_raw_response)
        else:
            structured_values.append(raw_response_value)
            invalid_raw_response = True

        pending_values = structured_values.copy()
        while pending_values:
            current_value = pending_values.pop()
            if isinstance(current_value, Mapping):
                is_function_call = current_value.get("type") == "function_call"
                for key, nested_value in current_value.items():
                    response_match = response_match or self._contains_token(
                        key,
                        canary_token,
                    )
                    if is_function_call and key == "arguments":
                        arguments_match = (
                            arguments_match
                            or self._contains_token(
                                nested_value,
                                canary_token,
                            )
                        )
                    elif isinstance(nested_value, Mapping) or (
                        isinstance(nested_value, Sequence)
                        and not isinstance(
                            nested_value, (str, bytes, bytearray)
                        )
                    ):
                        pending_values.append(nested_value)
                    else:
                        response_match = (
                            response_match
                            or self._contains_token(
                                nested_value,
                                canary_token,
                            )
                        )
            elif isinstance(current_value, Sequence) and not isinstance(
                current_value,
                (str, bytes, bytearray),
            ):
                pending_values.extend(current_value)
            else:
                response_match = response_match or self._contains_token(
                    current_value,
                    canary_token,
                )

        for tool_call in llm_response.tool_calls:
            response_match = response_match or self._contains_token(
                (tool_call.call_id, tool_call.name),
                canary_token,
            )
            arguments_match = arguments_match or self._contains_token(
                tool_call.arguments_json,
                canary_token,
            )
            arguments_match = arguments_match or self._contains_token(
                tool_call.arguments,
                canary_token,
            )

        if response_match:
            return "LLM_RESPONSE"
        if arguments_match:
            return "TOOL_ARGUMENTS"
        if instructions_mismatch:
            raise CanaryBlockedError(
                "Ответ заблокирован: возвращённые instructions изменены"
            )
        if invalid_raw_response:
            raise CanaryBlockedError(
                "Ответ заблокирован: raw response не является JSON-объектом"
            )
        return None

    def _contains_token(self, value: object, canary_token: str) -> bool:
        """Ищет полный токен во всех строках и декодирует вложенный JSON."""
        if isinstance(value, str):
            decoded_value: object = value
            seen_strings: set[str] = set()
            while isinstance(decoded_value, str):
                if canary_token in decoded_value:
                    return True
                if decoded_value in seen_strings:
                    return False
                seen_strings.add(decoded_value)
                try:
                    next_value = json.loads(decoded_value)
                except (json.JSONDecodeError, TypeError):
                    return False
                if next_value == decoded_value:
                    return False
                decoded_value = next_value
            return self._contains_token(decoded_value, canary_token)
        if isinstance(value, Mapping):
            return any(
                self._contains_token(key, canary_token)
                or self._contains_token(nested_value, canary_token)
                for key, nested_value in value.items()
            )
        if isinstance(value, Sequence) and not isinstance(
            value,
            (bytes, bytearray),
        ):
            return any(
                self._contains_token(nested_value, canary_token)
                for nested_value in value
            )
        return False

    def _write_security_event(self, run_id: str, location: str) -> None:
        """Добавляет одну строку журнала только с разрешёнными полями."""
        security_event = {
            "timestamp": datetime.now(UTC)
            .isoformat(timespec="microseconds")
            .replace(
                "+00:00",
                "Z",
            ),
            "run_id": run_id,
            "location": location,
            "reason": "CANARY_MATCH",
            "action": "BLOCKED",
            "event_type": "SYSTEM_PROMPT_LEAK",
        }
        with self._security_events_path.open(
            "a", encoding="utf-8"
        ) as event_file:
            event_file.write(
                json.dumps(security_event, ensure_ascii=False) + "\n"
            )
            event_file.flush()
            os.fsync(event_file.fileno())
