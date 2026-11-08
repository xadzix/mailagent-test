"""Оркестрация почтового агента с защищённым публичным запуском."""

import json
import logging
import re
import secrets
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from mail_agent.canary_guard import (
    BLOCKED_RESPONSE_MESSAGE,
    CANARY_TOKEN_BYTES,
    CanaryBlockedError,
    CanaryGuard,
)
from mail_agent.context_store import (
    DOCUMENT_FILE_NAMES,
    ChromaContextStore,
    ContextSearchResults,
)
from mail_agent.email_models import EmailThread
from mail_agent.llm_client import (
    MAIL_TOOL_SCHEMAS,
    LlmResponse,
    LlmToolCall,
    OpenAILlmClient,
)
from mail_agent.mail_tools import MailTools

SYSTEM_PROMPT_CANARY_PLACEHOLDER = "{{CANARY_TOKEN}}"
MAX_LLM_RESPONSES = 8
MAX_TOOL_CALLS = 6
LOGGER = logging.getLogger(__name__)
EMAIL_ADDRESS_PATTERN = re.compile(
    r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@"
    r"[\w-]+(?:\.[\w-]+)+(?![\w.-])",
    re.IGNORECASE,
)


class AgentRunState(BaseModel):
    """Содержит изолированное изменяемое состояние одного запроса менеджера."""

    manager_request: str = Field(min_length=1)
    conversation_items: list[dict[str, object]]
    selected_thread: EmailThread | None = None
    context_results: ContextSearchResults | None = None
    allow_draft: bool = False
    draft_id: str | None = None
    draft_thread_id: str | None = None
    draft_follow_up_used: bool = False
    search_result_count: int | None = None
    read_thread_follow_up_used: bool = False
    llm_response_count: int = 0
    tool_call_count: int = 0
    processed_call_ids: set[str] = Field(default_factory=set)
    explicit_email_addresses: set[str] = Field(default_factory=set)


class AgentResult(BaseModel):
    """Возвращает текст агента и проверяемые идентификаторы результата."""

    text: str
    thread_id: str | None
    source_ids: list[str]
    draft_id: str | None = None
    draft_status: Literal["not_authorized", "not_created", "created"] = (
        "not_authorized"
    )


def build_system_prompt(system_prompt_template: str, canary_token: str) -> str:
    """Подставляет canary-токен в предусмотренное место шаблона."""
    if system_prompt_template.count(SYSTEM_PROMPT_CANARY_PLACEHOLDER) != 1:
        raise ValueError(
            "Шаблон должен содержать ровно одно место для canary-токена"
        )
    if not canary_token:
        raise ValueError("Canary-токен не может быть пустым")
    return system_prompt_template.replace(
        SYSTEM_PROMPT_CANARY_PLACEHOLDER,
        canary_token,
    )


class MailAgent:
    """Координирует LLM, почтовые инструменты и обязательный внутренний RAG."""

    def __init__(
        self,
        llm_client: OpenAILlmClient,
        mail_tools: MailTools,
        context_store: ChromaContextStore,
        canary_guard: CanaryGuard,
        system_prompt_template: str,
    ) -> None:
        """Принимает существующие компоненты без запуска внешних операций."""
        self._llm_client = llm_client
        self._mail_tools = mail_tools
        self._context_store = context_store
        self._canary_guard = canary_guard
        self._system_prompt_template = system_prompt_template

    def run(
        self, manager_request: str, allow_draft: bool = False
    ) -> AgentResult:
        """Выполняет защищённый цикл LLM с проверкой каждого нового ответа."""
        run_id = str(uuid4())
        canary_token = secrets.token_urlsafe(CANARY_TOKEN_BYTES)
        system_prompt = build_system_prompt(
            self._system_prompt_template,
            canary_token,
        )
        run_state = self._create_run_state(manager_request, allow_draft)

        while run_state.llm_response_count < MAX_LLM_RESPONSES:
            available_tool_schemas = self._get_available_tool_schemas(
                run_state
            )
            llm_response = self._llm_client.generate_response(
                instructions=system_prompt,
                input_items=run_state.conversation_items,
                tool_schemas=available_tool_schemas,
            )
            try:
                self._canary_guard.validate_response(
                    llm_response=llm_response,
                    canary_token=canary_token,
                    run_id=run_id,
                    expected_instructions=system_prompt,
                )
            except CanaryBlockedError:
                return AgentResult(
                    text=BLOCKED_RESPONSE_MESSAGE,
                    thread_id=None,
                    source_ids=[],
                    draft_id=None,
                )
            agent_result = self._process_approved_response(
                run_state, llm_response
            )
            if agent_result is not None:
                return agent_result

        if (
            run_state.allow_draft
            and run_state.draft_follow_up_used
            and run_state.selected_thread is not None
            and run_state.context_results is not None
            and run_state.draft_id is None
        ):
            return AgentResult(
                text=(
                    "Черновик в текущем запуске для выбранной цепочки "
                    "не создан; "
                    "ранее сохранённые черновики в SQLite не изменялись."
                ),
                thread_id=run_state.selected_thread.thread_id,
                source_ids=self._build_verified_source_ids(run_state),
                draft_id=None,
                draft_status="not_created",
            )
        raise RuntimeError("Превышен лимит ответов LLM")

    def _create_run_state(
        self,
        manager_request: str,
        allow_draft: bool = False,
    ) -> AgentRunState:
        """Создаёт состояние и извлекает только явно заданных получателей."""
        normalized_request = manager_request.strip()
        if not normalized_request:
            raise ValueError("Запрос менеджера не может быть пустым")
        return AgentRunState(
            manager_request=normalized_request,
            conversation_items=[
                {"role": "user", "content": normalized_request}
            ],
            allow_draft=allow_draft,
            explicit_email_addresses=self._extract_explicit_email_addresses(
                normalized_request
            ),
        )

    def _get_available_tool_schemas(
        self,
        run_state: AgentRunState,
    ) -> list[dict[str, object]]:
        """Возвращает инструменты текущего этапа запуска агента."""
        draft_created_for_selected_thread = (
            run_state.selected_thread is not None
            and run_state.draft_id is not None
            and run_state.draft_thread_id
            == run_state.selected_thread.thread_id
        )
        if (
            run_state.selected_thread is None
            or run_state.context_results is None
        ):
            available_tool_names = {"search_emails", "read_thread"}
        elif not run_state.allow_draft:
            available_tool_names = {"search_emails", "read_thread"}
        elif not draft_created_for_selected_thread:
            available_tool_names = {"create_draft"}
        else:
            available_tool_names = set()
        return [
            tool_schema
            for tool_schema in MAIL_TOOL_SCHEMAS
            if tool_schema["name"] in available_tool_names
        ]

    def _process_approved_response(
        self,
        run_state: AgentRunState,
        llm_response: LlmResponse,
    ) -> AgentResult | None:
        """Обрабатывает проверенный CanaryGuard ответ.

        Выполняет разрешённый инструмент либо формирует результат.
        """
        run_state.llm_response_count += 1
        safe_tool_names: list[str] = [
            tool_call.name
            if tool_call.name
            in {"search_emails", "read_thread", "create_draft"}
            else "UNKNOWN_TOOL"
            for tool_call in llm_response.tool_calls
        ]
        LOGGER.info(
            "llm_response_processed "
            "llm_response_count=%d has_function_call=%s "
            "response_tool_call_count=%d tool_names=%s",
            run_state.llm_response_count,
            "true" if llm_response.tool_calls else "false",
            len(llm_response.tool_calls),
            ",".join(safe_tool_names) if safe_tool_names else "NONE",
        )
        if run_state.llm_response_count > MAX_LLM_RESPONSES:
            raise RuntimeError("Превышен лимит ответов LLM")
        if llm_response.status != "completed":
            raise RuntimeError("LLM не завершила формирование ответа")
        if len(llm_response.tool_calls) > 1:
            raise RuntimeError(
                "Ответ содержит более одного вызова инструмента"
            )

        run_state.conversation_items.extend(llm_response.output_items)
        if llm_response.tool_calls:
            tool_call = llm_response.tool_calls[0]
            available_tool_names = {
                str(tool_schema["name"])
                for tool_schema in self._get_available_tool_schemas(run_state)
            }
            safe_tool_name = (
                tool_call.name
                if tool_call.name
                in {"search_emails", "read_thread", "create_draft"}
                else "UNKNOWN_TOOL"
            )
            if run_state.tool_call_count >= MAX_TOOL_CALLS:
                LOGGER.info(
                    "tool_call_rejected tool_call_number=%d tool_name=%s "
                    "success=false error_category=MAX_TOOL_CALLS "
                    "used_tool_calls=%d llm_response_count=%d",
                    run_state.tool_call_count + 1,
                    safe_tool_name,
                    run_state.tool_call_count,
                    run_state.llm_response_count,
                )
                raise RuntimeError("Превышен лимит вызовов инструментов")
            if tool_call.call_id in run_state.processed_call_ids:
                raise RuntimeError(
                    "Повторный call_id инструмента заблокирован"
                )
            run_state.tool_call_count += 1
            run_state.processed_call_ids.add(tool_call.call_id)
            if (
                tool_call.name
                in {"search_emails", "read_thread", "create_draft"}
                and tool_call.name not in available_tool_names
            ):
                tool_result = {
                    "ok": False,
                    "error": "TOOL_VALIDATION_ERROR",
                    "message": (
                        "Инструмент недоступен на текущем этапе запуска"
                    ),
                }
            elif tool_call.arguments is None:
                tool_result: dict[str, object] = {
                    "ok": False,
                    "error": "INVALID_ARGUMENTS",
                    "message": (
                        "Аргументы инструмента не являются корректным "
                        "JSON-объектом"
                    ),
                }
            else:
                try:
                    tool_result = self._execute_approved_tool_call(
                        run_state, tool_call
                    )
                except Exception:
                    error_category = (
                        "UNKNOWN_TOOL"
                        if safe_tool_name == "UNKNOWN_TOOL"
                        else "TOOL_EXECUTION_ERROR"
                    )
                    LOGGER.info(
                        "tool_call_completed tool_call_number=%d tool_name=%s "
                        "success=false error_category=%s used_tool_calls=%d "
                        "llm_response_count=%d",
                        run_state.tool_call_count,
                        safe_tool_name,
                        error_category,
                        run_state.tool_call_count,
                        run_state.llm_response_count,
                    )
                    if run_state.tool_call_count == MAX_TOOL_CALLS:
                        LOGGER.info(
                            "tool_call_limit_reached used_tool_calls=%d "
                            "max_tool_calls=%d llm_response_count=%d",
                            run_state.tool_call_count,
                            MAX_TOOL_CALLS,
                            run_state.llm_response_count,
                        )
                    raise
            tool_succeeded = tool_result.get("ok") is True
            error_category = "NONE"
            if not tool_succeeded:
                returned_error = tool_result.get("error")
                error_category = (
                    returned_error
                    if returned_error
                    in {"INVALID_ARGUMENTS", "TOOL_VALIDATION_ERROR"}
                    else "TOOL_EXECUTION_ERROR"
                )
            LOGGER.info(
                "tool_call_completed tool_call_number=%d tool_name=%s "
                "success=%s error_category=%s used_tool_calls=%d "
                "llm_response_count=%d",
                run_state.tool_call_count,
                safe_tool_name,
                "true" if tool_succeeded else "false",
                error_category,
                run_state.tool_call_count,
                run_state.llm_response_count,
            )
            if run_state.tool_call_count == MAX_TOOL_CALLS:
                LOGGER.info(
                    "tool_call_limit_reached used_tool_calls=%d "
                    "max_tool_calls=%d llm_response_count=%d",
                    run_state.tool_call_count,
                    MAX_TOOL_CALLS,
                    run_state.llm_response_count,
                )
            run_state.conversation_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": json.dumps(tool_result, ensure_ascii=False),
                }
            )
            return None

        if (
            run_state.selected_thread is None
            or run_state.context_results is None
        ):
            if run_state.search_result_count == 0:
                return AgentResult(
                    text=(
                        "Поиск не нашёл писем; данных для выбора и чтения "
                        "почтовой цепочки недостаточно."
                    ),
                    thread_id=None,
                    source_ids=[],
                    draft_id=None,
                    draft_status=(
                        "not_created"
                        if run_state.allow_draft
                        else "not_authorized"
                    ),
                )
            if run_state.search_result_count is not None:
                if (
                    not run_state.read_thread_follow_up_used
                    and run_state.llm_response_count < MAX_LLM_RESPONSES
                ):
                    run_state.read_thread_follow_up_used = True
                    run_state.conversation_items.append(
                        {
                            "role": "developer",
                            "content": (
                                "Выбери подходящую цепочку из результатов "
                                "search_emails и прочитай её через "
                                "read_thread. "
                                "Итоговый ответ до чтения цепочки запрещён."
                            ),
                        }
                    )
                    return None
                raise RuntimeError(
                    "После search_emails модель не выбрала и не прочитала "
                    "цепочку через read_thread"
                )
            run_state.conversation_items.append(
                {
                    "role": "developer",
                    "content": (
                        "Итоговый ответ пока запрещён: сначала прочитай "
                        "выбранную цепочку через read_thread; внутренний RAG "
                        "будет выполнен "
                        "Python-кодом автоматически."
                    ),
                }
            )
            return None

        response_text = llm_response.output_text
        if not response_text.strip() and llm_response.refusal_text is not None:
            response_text = llm_response.refusal_text
        if not response_text.strip():
            raise RuntimeError(
                "Ошибка протокола LLM: итоговый ответ и текст отказа пусты"
            )
        draft_is_current = (
            run_state.draft_id is not None
            and run_state.draft_thread_id
            == run_state.selected_thread.thread_id
        )
        if run_state.allow_draft and not draft_is_current:
            if (
                not run_state.draft_follow_up_used
                and run_state.llm_response_count < MAX_LLM_RESPONSES
            ):
                run_state.draft_follow_up_used = True
                run_state.conversation_items.append(
                    {
                        "role": "developer",
                        "content": (
                            "Менеджер явно разрешил создание черновика, но в "
                            "текущем запуске черновик ещё не сохранён. Если "
                            "данных достаточно, вызови create_draft. Иначе "
                            "заверши ответ "
                            "без утверждения о создании черновика."
                        ),
                    }
                )
                return None
            return AgentResult(
                text=(
                    "Черновик в текущем запуске для выбранной цепочки "
                    "не создан; "
                    "ранее сохранённые черновики в SQLite не изменялись."
                ),
                thread_id=run_state.selected_thread.thread_id,
                source_ids=self._build_verified_source_ids(run_state),
                draft_id=None,
                draft_status="not_created",
            )

        return AgentResult(
            text=response_text,
            thread_id=run_state.selected_thread.thread_id,
            source_ids=self._build_verified_source_ids(run_state),
            draft_id=run_state.draft_id if draft_is_current else None,
            draft_status="created" if draft_is_current else "not_authorized",
        )

    def _execute_approved_tool_call(
        self,
        run_state: AgentRunState,
        tool_call: LlmToolCall,
    ) -> dict[str, object]:
        """Валидирует и исполняет один проверенный разрешённый вызов."""
        arguments = tool_call.arguments
        if arguments is None:
            raise ValueError(
                "Аргументы должны быть декодированы до исполнения"
            )

        try:
            if tool_call.name == "search_emails":
                search_results = self._mail_tools.search_emails(**arguments)
                run_state.search_result_count = len(search_results)
                return {
                    "ok": True,
                    "results": [
                        result.model_dump(mode="json")
                        for result in search_results
                    ],
                }
            if tool_call.name == "read_thread":
                email_thread = self._mail_tools.read_thread(**arguments)
                if (
                    run_state.draft_thread_id is not None
                    and run_state.draft_thread_id != email_thread.thread_id
                ):
                    run_state.draft_id = None
                    run_state.draft_thread_id = None
                run_state.selected_thread = email_thread
                run_state.context_results = None
                context_results = self._context_store.search_context(
                    user_query=run_state.manager_request,
                    email_thread=email_thread,
                )
                if any(
                    (
                        match.source_type == "email"
                        and match.thread_id != email_thread.thread_id
                    )
                    or match.source_type not in {"email", "document"}
                    for match in context_results.matches
                ):
                    raise RuntimeError(
                        "RAG вернул источник, недопустимый для выбранной "
                        "цепочки"
                    )
                if context_results.source_ids != list(
                    dict.fromkeys(
                        match.source_id for match in context_results.matches
                    )
                ):
                    raise RuntimeError(
                        "RAG вернул source_ids, не соответствующие найденным "
                        "фрагментам"
                    )
                run_state.context_results = context_results
                return {
                    "ok": True,
                    "thread": email_thread.model_dump(mode="json"),
                    "rag_context": {
                        "untrusted": True,
                        **context_results.model_dump(mode="json"),
                    },
                }
            if tool_call.name == "create_draft":
                if not run_state.allow_draft:
                    raise ValueError(
                        "Создание черновика не разрешено менеджером для "
                        "этого запуска"
                    )
                if run_state.selected_thread is None:
                    raise ValueError(
                        "Сначала необходимо прочитать выбранную цепочку"
                    )
                if run_state.context_results is None:
                    raise ValueError(
                        "Перед черновиком обязателен внутренний RAG-поиск"
                    )
                if run_state.draft_id is not None:
                    raise ValueError("Черновик уже создан в текущем запуске")
                thread_id = arguments.get("thread_id")
                if thread_id != run_state.selected_thread.thread_id:
                    raise ValueError(
                        "thread_id черновика не совпадает с выбранной цепочкой"
                    )
                recipient = arguments.get("recipient")
                if not isinstance(recipient, str):
                    raise ValueError("Получатель должен быть строкой")
                self._validate_draft_recipient(run_state, recipient)
                subject = arguments.get("subject")
                body = arguments.get("body")
                if not isinstance(subject, str) or not isinstance(body, str):
                    raise ValueError(
                        "Тема и текст черновика должны быть строками"
                    )
                self._validate_draft_source_references(
                    run_state, subject, body
                )
                draft = self._mail_tools.create_draft(**arguments)
                run_state.draft_id = draft.draft_id
                run_state.draft_thread_id = draft.thread_id
                return {"ok": True, "draft": draft.model_dump(mode="json")}
        except (TypeError, ValueError, ValidationError, LookupError) as error:
            return {
                "ok": False,
                "error": "TOOL_VALIDATION_ERROR",
                "message": str(error),
            }
        raise RuntimeError(f"Неизвестный инструмент: {tool_call.name}")

    def _extract_explicit_email_addresses(
        self, manager_request: str
    ) -> set[str]:
        """Находит адреса, однозначно названные получателями без отрицания."""
        recipient_line_pattern = re.compile(
            rf"^\s*получатель\s*:\s*"
            rf"(?P<address>{EMAIL_ADDRESS_PATTERN.pattern})\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        return {
            address_match.group("address").casefold()
            for address_match in recipient_line_pattern.finditer(
                manager_request
            )
        }

    def _validate_draft_recipient(
        self,
        run_state: AgentRunState,
        recipient: str,
    ) -> None:
        """Разрешает участника цепочки или явно заданного адресата."""
        if run_state.selected_thread is None:
            raise ValueError("Выбранная цепочка отсутствует")
        participant_addresses: set[str] = set()
        for message in run_state.selected_thread.messages:
            participant_addresses.add(message.sender.casefold())
            participant_addresses.update(
                address.casefold() for address in message.recipients
            )
        normalized_recipient = recipient.strip().casefold()
        if (
            normalized_recipient not in participant_addresses
            and normalized_recipient not in run_state.explicit_email_addresses
        ):
            raise ValueError(
                "Сторонний получатель не был однозначно указан менеджером; "
                "необходимо запросить уточнение"
            )

    def _validate_draft_source_references(
        self,
        run_state: AgentRunState,
        subject: str,
        body: str,
    ) -> None:
        """Запрещает внутренние ID в клиентском тексте черновика."""
        if (
            run_state.selected_thread is None
            or run_state.context_results is None
        ):
            raise ValueError(
                "Нельзя проверить источники без выбранной цепочки и RAG"
            )
        draft_text = f"{subject}\n{body}".casefold()
        internal_source_references = {
            run_state.selected_thread.thread_id,
            *(
                message.message_id
                for message in run_state.selected_thread.messages
            ),
            *run_state.context_results.source_ids,
            *(match.chunk_id for match in run_state.context_results.matches),
            *DOCUMENT_FILE_NAMES,
        }
        if any(
            source_reference.casefold() in draft_text
            for source_reference in internal_source_references
            if source_reference
        ):
            raise ValueError(
                "Клиентский черновик не должен содержать внутренние "
                "идентификаторы источников"
            )

    def _build_verified_source_ids(
        self, run_state: AgentRunState
    ) -> list[str]:
        """Объединяет ID писем выбранной цепочки и источников текущего RAG."""
        if (
            run_state.selected_thread is None
            or run_state.context_results is None
        ):
            raise ValueError("Нельзя сформировать источники без цепочки и RAG")
        source_ids = [
            message.message_id
            for message in run_state.selected_thread.messages
        ]
        source_ids.extend(run_state.context_results.source_ids)
        return list(dict.fromkeys(source_ids))
