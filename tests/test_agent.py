"""Тесты оркестрации агента на синтетических ответах LLM."""

import json
import logging
import sqlite3
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

import pytest

from mail_agent.agent import (
    MAX_LLM_RESPONSES,
    MAX_TOOL_CALLS,
    AgentResult,
    MailAgent,
    build_system_prompt,
)
from mail_agent.canary_guard import (
    BLOCKED_RESPONSE_MESSAGE,
    CanaryBlockedError,
    CanaryGuard,
)
from mail_agent.context_store import (
    DOCUMENT_FILE_NAMES,
    ChromaContextStore,
    ContextSearchResult,
    ContextSearchResults,
)
from mail_agent.email_models import EmailThread
from mail_agent.llm_client import LlmResponse, LlmToolCall, OpenAILlmClient
from mail_agent.mail_tools import MailTools


def test_run_blocks_response_before_processing_and_tools() -> None:
    """Не обрабатывает и не сохраняет ответ, отклонённый CanaryGuard."""
    llm_client = Mock(spec=OpenAILlmClient)
    mail_tools = Mock(spec=MailTools)
    context_store = Mock(spec=ChromaContextStore)
    canary_guard = Mock(spec=CanaryGuard)
    canary_guard.validate_response.side_effect = CanaryBlockedError()
    llm_client.generate_response.return_value = LlmResponse(
        response_id="blocked",
        status="completed",
        output_text="Заблокированный секрет",
        refusal_text=None,
        output_items=[{"type": "message", "text": "Заблокированный секрет"}],
        tool_calls=[],
        raw_response_json="{}",
    )
    agent = MailAgent(
        llm_client,
        mail_tools,
        context_store,
        canary_guard,
        "Токен: {{CANARY_TOKEN}}",
    )
    process_response = Mock()
    agent._process_approved_response = process_response

    result = agent.run("Подготовь сводку")

    assert result == AgentResult(
        text=BLOCKED_RESPONSE_MESSAGE,
        thread_id=None,
        source_ids=[],
        draft_id=None,
    )
    process_response.assert_not_called()
    mail_tools.assert_not_called()


def test_canary_blocked_response_creates_no_tool_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Не журналирует содержимое ответа, отклонённого CanaryGuard."""
    blocked_content = "blocked-sensitive-model-content"
    llm_client = Mock(spec=OpenAILlmClient)
    llm_client.generate_response.return_value = LlmResponse(
        response_id="blocked-diagnostics",
        status="completed",
        output_text=blocked_content,
        refusal_text=None,
        output_items=[{"type": "function_call", "text": blocked_content}],
        tool_calls=[
            LlmToolCall(
                call_id="blocked-call",
                name="create_draft",
                arguments_json=json.dumps({"body": blocked_content}),
                arguments={"body": blocked_content},
            )
        ],
        raw_response_json="{}",
    )
    canary_guard = Mock(spec=CanaryGuard)
    canary_guard.validate_response.side_effect = CanaryBlockedError()
    agent = MailAgent(
        llm_client, Mock(), Mock(), canary_guard, "{{CANARY_TOKEN}}"
    )
    caplog.set_level(logging.INFO, logger="mail_agent.agent")

    result = agent.run("Запрос")

    assert result.text == BLOCKED_RESPONSE_MESSAGE
    assert caplog.records == []
    assert blocked_content not in caplog.text


def test_run_is_fail_closed_when_security_event_cannot_be_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Возвращает нейтральный результат даже при ошибке записи журнала."""
    canary_token = "fixed-canary-token"
    monkeypatch.setattr(
        "mail_agent.agent.secrets.token_urlsafe",
        Mock(return_value=canary_token),
    )
    llm_client = Mock(spec=OpenAILlmClient)
    llm_client.generate_response.return_value = LlmResponse(
        response_id="blocked-write",
        status="completed",
        output_text=f"Утечка {canary_token}",
        refusal_text=None,
        output_items=[],
        tool_calls=[],
        raw_response_json="{}",
    )
    agent = MailAgent(
        llm_client,
        Mock(),
        Mock(),
        CanaryGuard(tmp_path),
        "Токен: {{CANARY_TOKEN}}",
    )
    process_response = Mock()
    agent._process_approved_response = process_response

    result = agent.run("Запрос")

    assert result.text == BLOCKED_RESPONSE_MESSAGE
    assert result.thread_id is None
    assert result.source_ids == []
    assert result.draft_id is None
    process_response.assert_not_called()


def test_build_system_prompt_replaces_single_canary_placeholder() -> None:
    """Подставляет токен только в единственное согласованное место шаблона."""
    prompt = build_system_prompt("Токен: {{CANARY_TOKEN}}", "canary-value")

    assert prompt == "Токен: canary-value"
    with pytest.raises(ValueError, match="ровно одно"):
        build_system_prompt("Шаблон без места", "canary-value")


def test_final_response_requires_thread_and_rag() -> None:
    """Не возвращает сводку до чтения цепочки и обязательного RAG-поиска."""
    agent = MailAgent(Mock(), Mock(), Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Подготовь сводку")
    llm_response = LlmResponse(
        response_id="response-001",
        status="completed",
        output_text="Итоговая сводка",
        refusal_text=None,
        output_items=[],
        tool_calls=[],
        raw_response_json="{}",
    )

    result = agent._process_approved_response(run_state, llm_response)

    assert result is None
    assert "read_thread" in str(run_state.conversation_items[-1]["content"])


def test_search_follow_up_is_used_once_and_preserves_tool_output(
    mail_tools: MailTools,
) -> None:
    """После одного напоминания прекращает цикл поиска."""
    agent = MailAgent(Mock(), mail_tools, Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Найди переписку о насосах")
    search_arguments = {
        "query": "насос",
        "sender": None,
        "date_from": None,
        "date_to": None,
        "limit": 5,
    }
    search_response = LlmResponse(
        response_id="search-response",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[
            {
                "type": "function_call",
                "call_id": "search-call",
                "name": "search_emails",
                "arguments": json.dumps(search_arguments),
            }
        ],
        tool_calls=[
            LlmToolCall(
                call_id="search-call",
                name="search_emails",
                arguments_json=json.dumps(search_arguments),
                arguments=search_arguments,
            )
        ],
        raw_response_json="{}",
    )
    premature_response = LlmResponse(
        response_id="premature-response",
        status="completed",
        output_text="Неподтверждённая сводка",
        refusal_text=None,
        output_items=[{"type": "message", "text": "Неподтверждённая сводка"}],
        tool_calls=[],
        raw_response_json="{}",
    )

    assert agent._process_approved_response(run_state, search_response) is None
    function_output = run_state.conversation_items[-1]
    decoded_output = json.loads(str(function_output["output"]))
    assert run_state.search_result_count == len(decoded_output["results"]) > 0
    assert function_output["call_id"] == "search-call"
    assert all(result["thread_id"] for result in decoded_output["results"])
    assert [
        schema["name"]
        for schema in agent._get_available_tool_schemas(run_state)
    ] == ["search_emails", "read_thread"]

    assert (
        agent._process_approved_response(run_state, premature_response) is None
    )
    assert run_state.read_thread_follow_up_used is True
    follow_up_count = sum(
        item.get("role") == "developer"
        and "search_emails" in str(item.get("content"))
        for item in run_state.conversation_items
    )
    assert follow_up_count == 1

    repeated_search = search_response.model_copy(
        update={
            "response_id": "repeated-search-response",
            "output_items": [
                {
                    "type": "function_call",
                    "call_id": "repeated-search-call",
                    "name": "search_emails",
                    "arguments": json.dumps(search_arguments),
                }
            ],
            "tool_calls": [
                search_response.tool_calls[0].model_copy(
                    update={"call_id": "repeated-search-call"}
                )
            ],
        }
    )
    assert agent._process_approved_response(run_state, repeated_search) is None
    assert run_state.read_thread_follow_up_used is True

    with pytest.raises(RuntimeError, match="не выбрала и не прочитала"):
        agent._process_approved_response(run_state, premature_response)


def test_empty_search_returns_deterministic_result_without_thread() -> None:
    """Не требует чтения несуществующей цепочки после пустого поиска."""
    mail_tools = Mock(spec=MailTools)
    mail_tools.search_emails.return_value = []
    agent = MailAgent(Mock(), mail_tools, Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state(
        "Найди переписку и создай черновик",
        allow_draft=True,
    )
    search_arguments = {
        "query": "несуществующее оборудование",
        "sender": None,
        "date_from": None,
        "date_to": None,
        "limit": 5,
    }
    search_response = LlmResponse(
        response_id="empty-search",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[
            {"type": "function_call", "call_id": "empty-search-call"}
        ],
        tool_calls=[
            LlmToolCall(
                call_id="empty-search-call",
                name="search_emails",
                arguments_json=json.dumps(search_arguments),
                arguments=search_arguments,
            )
        ],
        raw_response_json="{}",
    )
    final_response = LlmResponse(
        response_id="empty-search-final",
        status="completed",
        output_text="Найдена цепочка thread-invented",
        refusal_text=None,
        output_items=[{"type": "message", "text": "Выдуманный результат"}],
        tool_calls=[],
        raw_response_json="{}",
    )

    assert agent._process_approved_response(run_state, search_response) is None
    result = agent._process_approved_response(run_state, final_response)

    assert run_state.search_result_count == 0
    assert run_state.read_thread_follow_up_used is False
    assert result is not None
    assert result.text == (
        "Поиск не нашёл писем; данных для выбора и чтения почтовой цепочки "
        "недостаточно."
    )
    assert result.thread_id is None
    assert result.source_ids == []
    assert result.draft_id is None
    assert result.draft_status == "not_created"


def test_read_thread_succeeds_after_search_follow_up(
    mail_tools: MailTools,
    email_threads: list[EmailThread],
) -> None:
    """После единственного напоминания читает цепочку и выполняет RAG."""
    context_store = Mock(spec=ChromaContextStore)
    context_store.search_context.return_value = ContextSearchResults(
        matches=[],
        source_ids=[],
    )
    agent = MailAgent(Mock(), mail_tools, context_store, Mock(), "Шаблон")
    run_state = agent._create_run_state("Найди и прочитай переписку о насосах")
    run_state.search_result_count = 2
    premature_response = LlmResponse(
        response_id="premature-before-read",
        status="completed",
        output_text="Предварительная сводка",
        refusal_text=None,
        output_items=[{"type": "message"}],
        tool_calls=[],
        raw_response_json="{}",
    )

    assert (
        agent._process_approved_response(run_state, premature_response) is None
    )
    selected_thread = email_threads[0]
    read_response = LlmResponse(
        response_id="read-after-follow-up",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[
            {"type": "function_call", "call_id": "read-after-follow-up"}
        ],
        tool_calls=[
            LlmToolCall(
                call_id="read-after-follow-up",
                name="read_thread",
                arguments_json=json.dumps(
                    {"thread_id": selected_thread.thread_id}
                ),
                arguments={"thread_id": selected_thread.thread_id},
            )
        ],
        raw_response_json="{}",
    )

    assert agent._process_approved_response(run_state, read_response) is None
    assert run_state.selected_thread == selected_thread
    assert run_state.context_results == ContextSearchResults(
        matches=[], source_ids=[]
    )
    context_store.search_context.assert_called_once()


def test_available_tool_schemas_follow_agent_state(
    email_threads: list[EmailThread],
) -> None:
    """Меняет доступные схемы только по подтверждённому состоянию запуска."""
    agent = MailAgent(Mock(), Mock(), Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Подготовь ответ")

    initial_names = [
        schema["name"]
        for schema in agent._get_available_tool_schemas(run_state)
    ]
    run_state.selected_thread = email_threads[0]
    run_state.context_results = ContextSearchResults(matches=[], source_ids=[])
    summary_names = [
        schema["name"]
        for schema in agent._get_available_tool_schemas(run_state)
    ]
    run_state.allow_draft = True
    draft_names = [
        schema["name"]
        for schema in agent._get_available_tool_schemas(run_state)
    ]
    run_state.draft_id = "00000000-0000-4000-8000-000000000001"
    run_state.draft_thread_id = email_threads[0].thread_id
    final_names = [
        schema["name"]
        for schema in agent._get_available_tool_schemas(run_state)
    ]

    assert initial_names == ["search_emails", "read_thread"]
    assert summary_names == ["search_emails", "read_thread"]
    assert draft_names == ["create_draft"]
    assert final_names == []


def test_read_thread_runs_rag_and_returns_verified_sources(
    mail_tools: MailTools,
    email_threads: list[EmailThread],
) -> None:
    """После чтения запускает RAG и объединяет проверенные ID источников."""
    context_store = Mock(spec=ChromaContextStore)
    context_store.search_context.return_value = ContextSearchResults(
        matches=[
            ContextSearchResult(
                chunk_id="document:product_catalog.md:0000",
                source_id="product_catalog.md",
                source_type="document",
                date="2026-10-01",
                text="Каталог",
                distance=0.1,
            )
        ],
        source_ids=["product_catalog.md"],
    )
    agent = MailAgent(Mock(), mail_tools, context_store, Mock(), "Шаблон")
    run_state = agent._create_run_state("Подготовь сводку по поставке")
    assert [
        schema["name"]
        for schema in agent._get_available_tool_schemas(run_state)
    ] == ["search_emails", "read_thread"]
    thread_id = email_threads[0].thread_id
    read_response = LlmResponse(
        response_id="response-read",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[{"type": "function_call", "call_id": "call-read"}],
        tool_calls=[
            LlmToolCall(
                call_id="call-read",
                name="read_thread",
                arguments_json=json.dumps({"thread_id": thread_id}),
                arguments={"thread_id": thread_id},
            )
        ],
        raw_response_json="{}",
    )

    assert agent._process_approved_response(run_state, read_response) is None
    final_response = LlmResponse(
        response_id="response-final",
        status="completed",
        output_text="Краткая сводка",
        refusal_text=None,
        output_items=[{"type": "message"}],
        tool_calls=[],
        raw_response_json="{}",
    )
    result = agent._process_approved_response(run_state, final_response)

    context_store.search_context.assert_called_once()
    assert result is not None
    assert result.thread_id == thread_id
    assert result.draft_status == "not_authorized"
    assert result.source_ids == [
        *(message.message_id for message in email_threads[0].messages),
        "product_catalog.md",
    ]


@pytest.mark.parametrize("invalid_thread_id", ["thread-001", None, ""])
def test_invalid_email_rag_match_is_rejected_before_model_context(
    invalid_thread_id: str | None,
    mail_tools: MailTools,
    email_threads: list[EmailThread],
) -> None:
    """Не передаёт модели email-фрагмент без текущего thread_id."""
    foreign_text = "Содержимое другой клиентской переписки"
    context_store = Mock(spec=ChromaContextStore)
    context_store.search_context.return_value = ContextSearchResults(
        matches=[
            ContextSearchResult(
                chunk_id="email:email-003:0000",
                source_id="email-003",
                source_type="email",
                date="2026-10-01T09:30:00.000000Z",
                thread_id=invalid_thread_id,
                text=foreign_text,
                distance=0.1,
            )
        ],
        source_ids=["email-003"],
    )
    agent = MailAgent(Mock(), mail_tools, context_store, Mock(), "Шаблон")
    run_state = agent._create_run_state("Прочитай условия оплаты")
    selected_thread = email_threads[2]
    read_response = LlmResponse(
        response_id="invalid-rag-thread",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[{"type": "function_call", "call_id": "read-thread"}],
        tool_calls=[
            LlmToolCall(
                call_id="read-thread",
                name="read_thread",
                arguments_json="{}",
                arguments={"thread_id": selected_thread.thread_id},
            )
        ],
        raw_response_json="{}",
    )

    with pytest.raises(RuntimeError, match="недопустимый"):
        agent._process_approved_response(run_state, read_response)

    assert run_state.context_results is None
    assert foreign_text not in str(run_state.conversation_items)


def test_rag_source_ids_must_match_returned_results(
    mail_tools: MailTools,
    email_threads: list[EmailThread],
) -> None:
    """Отклоняет список source_ids, не соответствующий найденным фрагментам."""
    context_store = Mock(spec=ChromaContextStore)
    context_store.search_context.return_value = ContextSearchResults(
        matches=[
            ContextSearchResult(
                chunk_id="document:payment_terms.md:0000",
                source_id="payment_terms.md",
                source_type="document",
                date="2026-10-03",
                text="Условия оплаты",
                distance=0.1,
            )
        ],
        source_ids=["delivery_terms.md"],
    )
    agent = MailAgent(Mock(), mail_tools, context_store, Mock(), "Шаблон")
    run_state = agent._create_run_state("Прочитай условия оплаты")
    selected_thread = email_threads[2]

    with pytest.raises(RuntimeError, match="source_ids"):
        agent._execute_approved_tool_call(
            run_state,
            LlmToolCall(
                call_id="read-thread",
                name="read_thread",
                arguments_json="{}",
                arguments={"thread_id": selected_thread.thread_id},
            ),
        )

    assert run_state.context_results is None


def test_reading_another_thread_replaces_rag_context(
    mail_tools: MailTools,
    email_threads: list[EmailThread],
) -> None:
    """Повторное чтение иной цепочки заново выполняет и заменяет RAG."""
    context_store = Mock(spec=ChromaContextStore)
    context_store.search_context.side_effect = [
        ContextSearchResults(
            matches=[
                ContextSearchResult(
                    chunk_id="document:first-document:0000",
                    source_id="first-document",
                    source_type="document",
                    date="2026-10-01",
                    text="Первый документ",
                    distance=0.1,
                )
            ],
            source_ids=["first-document"],
        ),
        ContextSearchResults(
            matches=[
                ContextSearchResult(
                    chunk_id="document:second-document:0000",
                    source_id="second-document",
                    source_type="document",
                    date="2026-10-02",
                    text="Второй документ",
                    distance=0.1,
                )
            ],
            source_ids=["second-document"],
        ),
    ]
    agent = MailAgent(Mock(), mail_tools, context_store, Mock(), "Шаблон")
    run_state = agent._create_run_state("Сравни переписки")

    for call_index, email_thread in enumerate(email_threads[:2], start=1):
        tool_call = LlmToolCall(
            call_id=f"call-{call_index}",
            name="read_thread",
            arguments_json="{}",
            arguments={"thread_id": email_thread.thread_id},
        )
        agent._execute_approved_tool_call(run_state, tool_call)

    assert context_store.search_context.call_count == 2
    assert run_state.selected_thread == email_threads[1]
    assert run_state.context_results is not None
    assert run_state.context_results.source_ids == ["second-document"]
    assert [
        schema["name"]
        for schema in agent._get_available_tool_schemas(run_state)
    ] == ["search_emails", "read_thread"]


def test_unavailable_create_draft_is_rejected_before_execution(
    email_threads: list[EmailThread],
) -> None:
    """Отклоняет черновик до RAG и после RAG без явного разрешения."""
    mail_tools = Mock(spec=MailTools)
    agent = MailAgent(Mock(), mail_tools, Mock(), Mock(), "Шаблон")
    states = [
        agent._create_run_state("Создай черновик", allow_draft=True),
        agent._create_run_state("Подготовь сводку"),
    ]
    states[1].selected_thread = email_threads[0]
    states[1].context_results = ContextSearchResults(matches=[], source_ids=[])

    for response_number, run_state in enumerate(states, start=1):
        llm_response = LlmResponse(
            response_id=f"unavailable-draft-{response_number}",
            status="completed",
            output_text="",
            refusal_text=None,
            output_items=[{"type": "function_call"}],
            tool_calls=[
                LlmToolCall(
                    call_id=f"unavailable-call-{response_number}",
                    name="create_draft",
                    arguments_json="{}",
                    arguments={
                        "thread_id": email_threads[0].thread_id,
                        "recipient": email_threads[0].messages[0].sender,
                        "subject": "Ответ",
                        "body": "Текст",
                    },
                )
            ],
            raw_response_json="{}",
        )

        assert (
            agent._process_approved_response(run_state, llm_response) is None
        )
        tool_output = json.loads(
            str(run_state.conversation_items[-1]["output"])
        )
        assert tool_output["error"] == "TOOL_VALIDATION_ERROR"

    mail_tools.create_draft.assert_not_called()


def test_allow_draft_locks_selected_thread_after_rag(
    mail_tools: MailTools,
    email_threads: list[EmailThread],
) -> None:
    """Не переключает выбранную цепочку после перехода к этапу черновика."""
    context_store = Mock(spec=ChromaContextStore)
    context_store.search_context.return_value = ContextSearchResults(
        matches=[],
        source_ids=[],
    )
    agent = MailAgent(Mock(), mail_tools, context_store, Mock(), "Шаблон")
    run_state = agent._create_run_state("Создай черновик", allow_draft=True)
    first_thread = email_threads[0]
    second_thread = email_threads[1]
    agent._execute_approved_tool_call(
        run_state,
        LlmToolCall(
            call_id="read-first",
            name="read_thread",
            arguments_json="{}",
            arguments={"thread_id": first_thread.thread_id},
        ),
    )
    unexpected_read = LlmResponse(
        response_id="unexpected-read",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[{"type": "function_call"}],
        tool_calls=[
            LlmToolCall(
                call_id="read-second",
                name="read_thread",
                arguments_json="{}",
                arguments={"thread_id": second_thread.thread_id},
            )
        ],
        raw_response_json="{}",
    )

    assert agent._process_approved_response(run_state, unexpected_read) is None

    assert run_state.selected_thread == first_thread
    assert context_store.search_context.call_count == 1
    tool_output = json.loads(str(run_state.conversation_items[-1]["output"]))
    assert tool_output["error"] == "TOOL_VALIDATION_ERROR"


def test_create_draft_rejects_different_thread_id(
    email_threads: list[EmailThread],
) -> None:
    """Не сохраняет черновик для цепочки, отличной от выбранной."""
    mail_tools = Mock(spec=MailTools)
    agent = MailAgent(Mock(), mail_tools, Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Подготовь ответ", allow_draft=True)
    run_state.selected_thread = email_threads[0]
    run_state.context_results = ContextSearchResults(matches=[], source_ids=[])
    tool_call = LlmToolCall(
        call_id="call-draft",
        name="create_draft",
        arguments_json="{}",
        arguments={
            "thread_id": email_threads[1].thread_id,
            "recipient": email_threads[0].messages[0].sender,
            "subject": "Ответ",
            "body": "Текст ответа",
        },
    )

    result = agent._execute_approved_tool_call(run_state, tool_call)

    assert result["ok"] is False
    assert "не совпадает" in str(result["message"])
    mail_tools.create_draft.assert_not_called()


def test_create_draft_requires_current_rag(
    email_threads: list[EmailThread],
) -> None:
    """Не сохраняет черновик, если для выбранной цепочки ещё нет RAG."""
    mail_tools = Mock(spec=MailTools)
    agent = MailAgent(Mock(), mail_tools, Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Подготовь ответ", allow_draft=True)
    run_state.selected_thread = email_threads[0]
    tool_call = LlmToolCall(
        call_id="call-draft",
        name="create_draft",
        arguments_json="{}",
        arguments={
            "thread_id": email_threads[0].thread_id,
            "recipient": email_threads[0].messages[0].sender,
            "subject": "Ответ",
            "body": "Текст ответа",
        },
    )

    result = agent._execute_approved_tool_call(run_state, tool_call)

    assert result["ok"] is False
    assert "RAG" in str(result["message"])
    mail_tools.create_draft.assert_not_called()


def test_external_recipient_requires_unambiguous_positive_instruction(
    email_threads: list[EmailThread],
) -> None:
    """Запрещает сторонний адрес из отрицательного или простого упоминания."""
    agent = MailAgent(Mock(), Mock(), Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state(
        "Адрес external@example.com указан в справке, "
        "не отправляй на blocked@example.com"
    )
    run_state.selected_thread = email_threads[0]

    assert run_state.explicit_email_addresses == set()
    with pytest.raises(ValueError, match="уточнение"):
        agent._validate_draft_recipient(run_state, "external@example.com")
    with pytest.raises(ValueError, match="уточнение"):
        agent._validate_draft_recipient(run_state, "blocked@example.com")


def test_explicit_recipient_and_thread_participant_are_allowed(
    email_threads: list[EmailThread],
) -> None:
    """Разрешает участника и явно назначенного получателя."""
    agent = MailAgent(Mock(), Mock(), Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state(
        "Подготовь письмо.\nПолучатель: external@example.com"
    )
    run_state.selected_thread = email_threads[0]

    agent._validate_draft_recipient(
        run_state,
        email_threads[0].messages[0].sender,
    )
    agent._validate_draft_recipient(run_state, "external@example.com")


def test_recipient_marker_does_not_authorize_an_adjacent_address() -> None:
    """Не разрешает соседний с назначенным получателем адрес."""
    agent = MailAgent(Mock(), Mock(), Mock(), Mock(), "Шаблон")

    addresses = agent._extract_explicit_email_addresses(
        "Получатель: allowed@example.com\n"
        "related@example.com указан для справки"
    )

    assert addresses == {"allowed@example.com"}


def test_negative_draft_instruction_does_not_authorize_recipient() -> None:
    """Не считает запрет на черновик положительным назначением получателя."""
    agent = MailAgent(Mock(), Mock(), Mock(), Mock(), "Шаблон")

    addresses = agent._extract_explicit_email_addresses(
        "Не создавай черновик для outsider@example.com"
    )

    assert addresses == set()


def test_recipient_requires_separate_positive_line() -> None:
    """Не разрешает сторонний адрес из неоднозначной фразы внутри запроса."""
    agent = MailAgent(Mock(), Mock(), Mock(), Mock(), "Шаблон")

    addresses = agent._extract_explicit_email_addresses(
        "Проверь получателя ambiguous@example.com и подготовь сводку"
    )

    assert addresses == set()


def test_create_draft_saves_draft_and_returns_its_id(
    mail_tools: MailTools,
    email_threads: list[EmailThread],
) -> None:
    """Сохраняет допустимый черновик после чтения цепочки и выполнения RAG."""
    agent = MailAgent(Mock(), mail_tools, Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state(
        "Подготовь ответ участнику переписки",
        allow_draft=True,
    )
    run_state.selected_thread = email_threads[0]
    run_state.context_results = ContextSearchResults(matches=[], source_ids=[])
    tool_call = LlmToolCall(
        call_id="call-valid-draft",
        name="create_draft",
        arguments_json="{}",
        arguments={
            "thread_id": email_threads[0].thread_id,
            "recipient": email_threads[0].messages[0].sender,
            "subject": "Ответ по поставке",
            "body": "Подтверждаем получение запроса.",
        },
    )

    result = agent._execute_approved_tool_call(run_state, tool_call)

    assert result["ok"] is True
    assert run_state.draft_id
    assert run_state.draft_thread_id == email_threads[0].thread_id
    assert result["draft"]["draft_id"] == run_state.draft_id
    assert agent._get_available_tool_schemas(run_state) == []

    repeated_result = agent._execute_approved_tool_call(
        run_state,
        tool_call.model_copy(update={"call_id": "call-repeated-draft"}),
    )

    assert repeated_result["ok"] is False
    assert "уже создан" in str(repeated_result["message"])
    with sqlite3.connect(
        mail_tools.email_repository.database_path
    ) as connection:
        draft_count = connection.execute(
            "SELECT COUNT(*) FROM draft_replies WHERE thread_id = ?",
            (email_threads[0].thread_id,),
        ).fetchone()[0]
    assert draft_count == 1


def test_draft_rejects_internal_references_and_saves_corrected_text(
    mail_tools: MailTools,
    email_threads: list[EmailThread],
) -> None:
    """Блокирует внутренние ID и сохраняет исправленный деловой текст."""
    agent = MailAgent(Mock(), mail_tools, Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Подготовь ответ", allow_draft=True)
    selected_thread = email_threads[2]
    context_match = ContextSearchResult(
        chunk_id="document:payment_terms.md:0000",
        source_id="payment_terms.md",
        source_type="document",
        date="2026-10-03",
        text="Счёт действует пять банковских дней.",
        distance=0.1,
    )
    run_state.selected_thread = selected_thread
    run_state.context_results = ContextSearchResults(
        matches=[context_match],
        source_ids=["payment_terms.md"],
    )
    internal_references = [
        selected_thread.messages[0].message_id,
        selected_thread.thread_id,
        context_match.chunk_id,
        *DOCUMENT_FILE_NAMES,
    ]

    for reference_number, internal_reference in enumerate(
        internal_references,
        start=1,
    ):
        blocked_result = agent._execute_approved_tool_call(
            run_state,
            LlmToolCall(
                call_id=f"blocked-draft-{reference_number}",
                name="create_draft",
                arguments_json="{}",
                arguments={
                    "thread_id": selected_thread.thread_id,
                    "recipient": selected_thread.messages[0].sender,
                    "subject": "Ответ по условиям оплаты",
                    "body": f"Внутренняя ссылка: {internal_reference.upper()}",
                },
            ),
        )
        assert blocked_result["ok"] is False
        assert "внутренние идентификаторы" in str(blocked_result["message"])

    with sqlite3.connect(
        mail_tools.email_repository.database_path
    ) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM draft_replies"
            ).fetchone()[0]
            == 0
        )

    corrected_body = (
        "Подтверждаем согласованные условия оплаты. Направьте реквизиты для "
        "счёта; срок его действия составит пять банковских дней."
    )
    saved_result = agent._execute_approved_tool_call(
        run_state,
        LlmToolCall(
            call_id="corrected-draft",
            name="create_draft",
            arguments_json="{}",
            arguments={
                "thread_id": selected_thread.thread_id,
                "recipient": selected_thread.messages[0].sender,
                "subject": "Ответ по условиям оплаты",
                "body": corrected_body,
            },
        ),
    )

    assert saved_result["ok"] is True
    with sqlite3.connect(
        mail_tools.email_repository.database_path
    ) as connection:
        saved_rows = connection.execute(
            "SELECT draft_id, thread_id, body FROM draft_replies"
        ).fetchall()
    assert saved_rows == [
        (run_state.draft_id, selected_thread.thread_id, corrected_body)
    ]


def test_create_draft_requires_explicit_run_authorization(
    email_threads: list[EmailThread],
) -> None:
    """Не исполняет create_draft без явного разрешения текущего запуска."""
    mail_tools = Mock(spec=MailTools)
    agent = MailAgent(Mock(), mail_tools, Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Подготовь сводку")
    run_state.selected_thread = email_threads[0]
    run_state.context_results = ContextSearchResults(matches=[], source_ids=[])
    tool_call = LlmToolCall(
        call_id="call-unauthorized-draft",
        name="create_draft",
        arguments_json="{}",
        arguments={
            "thread_id": email_threads[0].thread_id,
            "recipient": email_threads[0].messages[0].sender,
            "subject": "Ответ",
            "body": "Текст ответа",
        },
    )

    result = agent._execute_approved_tool_call(run_state, tool_call)

    assert result["ok"] is False
    assert "не разрешено" in str(result["message"])
    assert run_state.draft_id is None
    mail_tools.create_draft.assert_not_called()


def test_draft_state_follows_only_the_current_thread(
    mail_tools: MailTools,
    email_threads: list[EmailThread],
) -> None:
    """Сохраняет UUID в своей цепочке и сбрасывает без удаления при смене."""
    context_store = Mock(spec=ChromaContextStore)
    context_store.search_context.return_value = ContextSearchResults(
        matches=[],
        source_ids=[],
    )
    agent = MailAgent(Mock(), mail_tools, context_store, Mock(), "Шаблон")
    run_state = agent._create_run_state("Подготовь ответ", allow_draft=True)
    first_thread = email_threads[0]
    second_thread = email_threads[1]
    read_first = LlmToolCall(
        call_id="read-first",
        name="read_thread",
        arguments_json="{}",
        arguments={"thread_id": first_thread.thread_id},
    )
    create_draft = LlmToolCall(
        call_id="create-first-draft",
        name="create_draft",
        arguments_json="{}",
        arguments={
            "thread_id": first_thread.thread_id,
            "recipient": first_thread.messages[0].sender,
            "subject": "Ответ",
            "body": "Подтверждаем получение запроса.",
        },
    )
    read_second = LlmToolCall(
        call_id="read-second",
        name="read_thread",
        arguments_json="{}",
        arguments={"thread_id": second_thread.thread_id},
    )

    agent._execute_approved_tool_call(run_state, read_first)
    draft_result = agent._execute_approved_tool_call(run_state, create_draft)
    saved_draft_id = str(draft_result["draft"]["draft_id"])
    UUID(saved_draft_id)

    agent._execute_approved_tool_call(run_state, read_first)
    assert run_state.draft_id == saved_draft_id
    assert run_state.draft_thread_id == first_thread.thread_id
    same_thread_result = agent._process_approved_response(
        run_state,
        LlmResponse(
            response_id="final-for-first-thread",
            status="completed",
            output_text="Краткая сводка выбранной цепочки.",
            refusal_text=None,
            output_items=[],
            tool_calls=[],
            raw_response_json="{}",
        ),
    )
    assert same_thread_result is not None
    assert same_thread_result.draft_status == "created"
    assert same_thread_result.draft_id == saved_draft_id
    assert same_thread_result.source_ids

    agent._execute_approved_tool_call(run_state, read_second)
    assert run_state.draft_id is None
    assert run_state.draft_thread_id is None
    with sqlite3.connect(
        mail_tools.email_repository.database_path
    ) as connection:
        saved_row = connection.execute(
            "SELECT thread_id FROM draft_replies WHERE draft_id = ?",
            (saved_draft_id,),
        ).fetchone()
    assert saved_row == (first_thread.thread_id,)

    agent._execute_approved_tool_call(run_state, read_first)
    assert run_state.draft_id is None
    assert run_state.draft_thread_id is None
    run_state.draft_follow_up_used = True
    final_result = agent._process_approved_response(
        run_state,
        LlmResponse(
            response_id="final-after-return",
            status="completed",
            output_text="Черновик создан с draft_id: fake-id",
            refusal_text=None,
            output_items=[],
            tool_calls=[],
            raw_response_json="{}",
        ),
    )
    assert final_result is not None
    assert final_result.thread_id == first_thread.thread_id
    assert final_result.draft_status == "not_created"
    assert final_result.draft_id is None
    assert "fake-id" not in final_result.text


def test_draft_follow_up_is_used_once_and_hides_unverified_claim(
    email_threads: list[EmailThread],
) -> None:
    """Один раз продолжает диалог, затем возвращает проверенный неуспех."""
    mail_tools = Mock(spec=MailTools)
    agent = MailAgent(Mock(), mail_tools, Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Создай черновик", allow_draft=True)
    run_state.selected_thread = email_threads[0]
    run_state.context_results = ContextSearchResults(
        matches=[],
        source_ids=["delivery_terms.md"],
    )
    failed_draft_result = agent._execute_approved_tool_call(
        run_state,
        LlmToolCall(
            call_id="failed-draft",
            name="create_draft",
            arguments_json="{}",
            arguments={
                "thread_id": email_threads[1].thread_id,
                "recipient": email_threads[0].messages[0].sender,
                "subject": "Ответ",
                "body": "Текст ответа",
            },
        ),
    )
    false_final_response = LlmResponse(
        response_id="false-final",
        status="completed",
        output_text='Черновик создан: {"draft_id": 1}',
        refusal_text=None,
        output_items=[],
        tool_calls=[],
        raw_response_json="{}",
    )

    first_result = agent._process_approved_response(
        run_state, false_final_response
    )
    second_result = agent._process_approved_response(
        run_state,
        false_final_response.model_copy(
            update={"response_id": "false-final-again"}
        ),
    )

    assert failed_draft_result["ok"] is False
    mail_tools.create_draft.assert_not_called()
    assert first_result is None
    assert run_state.draft_follow_up_used is True
    assert second_result is not None
    assert second_result.draft_status == "not_created"
    assert second_result.draft_id is None
    assert '"draft_id": 1' not in second_result.text
    follow_up_items = [
        item
        for item in run_state.conversation_items
        if item.get("role") == "developer"
        and "create_draft" in str(item.get("content"))
    ]
    assert len(follow_up_items) == 1


def test_draft_follow_up_respects_llm_response_limit(
    email_threads: list[EmailThread],
) -> None:
    """Не начинает продолжение, если исчерпан лимит ответов модели."""
    agent = MailAgent(Mock(), Mock(), Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Создай черновик", allow_draft=True)
    run_state.selected_thread = email_threads[0]
    run_state.context_results = ContextSearchResults(matches=[], source_ids=[])
    run_state.llm_response_count = MAX_LLM_RESPONSES - 1
    final_response = LlmResponse(
        response_id="last-response",
        status="completed",
        output_text="Черновик создан с draft_id: 1",
        refusal_text=None,
        output_items=[],
        tool_calls=[],
        raw_response_json="{}",
    )

    result = agent._process_approved_response(run_state, final_response)

    assert result is not None
    assert result.draft_status == "not_created"
    assert result.draft_id is None
    assert run_state.draft_follow_up_used is False


def test_unknown_tool_is_rejected_without_execution() -> None:
    """Останавливает обработку неизвестного имени инструмента."""
    mail_tools = Mock(spec=MailTools)
    agent = MailAgent(Mock(), mail_tools, Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Выполни запрос")
    tool_call = LlmToolCall(
        call_id="call-unknown",
        name="send_email",
        arguments_json="{}",
        arguments={},
    )

    with pytest.raises(RuntimeError, match="Неизвестный инструмент"):
        agent._execute_approved_tool_call(run_state, tool_call)

    mail_tools.assert_not_called()


def test_invalid_json_arguments_do_not_execute_tools() -> None:
    """Возвращает контролируемую ошибку и не вызывает инструмент без JSON."""
    mail_tools = Mock(spec=MailTools)
    agent = MailAgent(Mock(), mail_tools, Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Прочитай цепочку")
    llm_response = LlmResponse(
        response_id="response-invalid",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[{"type": "function_call", "call_id": "call-invalid"}],
        tool_calls=[
            LlmToolCall(
                call_id="call-invalid",
                name="read_thread",
                arguments_json="{invalid",
                arguments=None,
            )
        ],
        raw_response_json="{}",
    )

    assert agent._process_approved_response(run_state, llm_response) is None
    output = json.loads(str(run_state.conversation_items[-1]["output"]))
    assert output["error"] == "INVALID_ARGUMENTS"
    mail_tools.assert_not_called()


def test_llm_response_limit_is_enforced() -> None:
    """Останавливает обработку после утверждённого числа ответов модели."""
    agent = MailAgent(Mock(), Mock(), Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Запрос")
    run_state.llm_response_count = MAX_LLM_RESPONSES
    llm_response = LlmResponse(
        response_id="response-limit",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[],
        tool_calls=[],
        raw_response_json="{}",
    )

    with pytest.raises(RuntimeError, match="лимит ответов"):
        agent._process_approved_response(run_state, llm_response)


def test_empty_final_response_is_protocol_error(
    email_threads: list[EmailThread],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Не создаёт успешный результат из пустого итогового ответа модели."""
    agent = MailAgent(Mock(), Mock(), Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Подготовь сводку")
    run_state.selected_thread = email_threads[0]
    run_state.context_results = ContextSearchResults(matches=[], source_ids=[])
    llm_response = LlmResponse(
        response_id="response-empty",
        status="completed",
        output_text="  ",
        refusal_text=None,
        output_items=[],
        tool_calls=[],
        raw_response_json="{}",
    )
    caplog.set_level(logging.INFO, logger="mail_agent.agent")

    with pytest.raises(RuntimeError, match="Ошибка протокола LLM"):
        agent._process_approved_response(run_state, llm_response)

    assert "response_tool_call_count=0" in caplog.text
    assert "tool_names=NONE" in caplog.text


def test_tool_call_limit_is_enforced() -> None:
    """Не исполняет инструмент после утверждённого лимита вызовов."""
    mail_tools = Mock(spec=MailTools)
    agent = MailAgent(Mock(), mail_tools, Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Запрос")
    run_state.tool_call_count = MAX_TOOL_CALLS
    llm_response = LlmResponse(
        response_id="response-limit",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[],
        tool_calls=[
            LlmToolCall(
                call_id="call-limit",
                name="search_emails",
                arguments_json="{}",
                arguments={
                    "query": "насос",
                    "sender": None,
                    "date_from": None,
                    "date_to": None,
                    "limit": 5,
                },
            )
        ],
        raw_response_json="{}",
    )

    with pytest.raises(RuntimeError, match="лимит вызовов"):
        agent._process_approved_response(run_state, llm_response)

    mail_tools.assert_not_called()


def test_repeated_failed_tool_calls_are_logged_until_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Показывает шесть повторных ошибок и отклонение седьмой попытки."""
    mail_tools = Mock(spec=MailTools)
    agent = MailAgent(Mock(), mail_tools, Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Запрос")
    caplog.set_level(logging.INFO, logger="mail_agent.agent")

    for call_number in range(1, MAX_TOOL_CALLS + 1):
        failed_response = LlmResponse(
            response_id=f"response-{call_number}",
            status="completed",
            output_text="",
            refusal_text=None,
            output_items=[{"type": "function_call"}],
            tool_calls=[
                LlmToolCall(
                    call_id=f"call-{call_number}",
                    name="read_thread",
                    arguments_json="{invalid",
                    arguments=None,
                )
            ],
            raw_response_json="{}",
        )
        assert (
            agent._process_approved_response(run_state, failed_response)
            is None
        )

    rejected_response = failed_response.model_copy(
        update={
            "response_id": "response-rejected",
            "tool_calls": [
                failed_response.tool_calls[0].model_copy(
                    update={"call_id": "call-rejected"}
                )
            ],
        }
    )
    with pytest.raises(RuntimeError, match="лимит вызовов"):
        agent._process_approved_response(run_state, rejected_response)

    diagnostic_messages = [record.getMessage() for record in caplog.records]
    assert (
        sum(
            "llm_response_processed" in message
            for message in diagnostic_messages
        )
        == 7
    )
    assert all(
        "response_tool_call_count=1" in message
        and "tool_names=read_thread" in message
        for message in diagnostic_messages
        if "llm_response_processed" in message
    )
    assert (
        sum(
            "tool_call_completed" in message for message in diagnostic_messages
        )
        == 6
    )
    assert all(
        "tool_name=read_thread" in message
        and "success=false" in message
        and "error_category=INVALID_ARGUMENTS" in message
        for message in diagnostic_messages
        if "tool_call_completed" in message
    )
    assert any(
        "tool_call_limit_reached" in message
        and "used_tool_calls=6" in message
        and "llm_response_count=6" in message
        for message in diagnostic_messages
    )
    assert any(
        "tool_call_rejected" in message
        and "tool_call_number=7" in message
        and "error_category=MAX_TOOL_CALLS" in message
        and "llm_response_count=7" in message
        for message in diagnostic_messages
    )
    mail_tools.assert_not_called()


def test_tool_diagnostics_do_not_expose_sensitive_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Оставляет в диагностике только статическую категорию и счётчики."""
    sensitive_content = "secret-canary-prompt-email-rag-arguments"
    agent = MailAgent(Mock(), Mock(), Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Запрос")
    llm_response = LlmResponse(
        response_id="response-sensitive",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[{"type": "function_call", "content": sensitive_content}],
        tool_calls=[
            LlmToolCall(
                call_id="call-sensitive",
                name=f"unknown-{sensitive_content}",
                arguments_json=json.dumps({"body": sensitive_content}),
                arguments={"body": sensitive_content},
            )
        ],
        raw_response_json=json.dumps({"output": sensitive_content}),
    )
    caplog.set_level(logging.INFO, logger="mail_agent.agent")

    with pytest.raises(RuntimeError, match="Неизвестный инструмент"):
        agent._process_approved_response(run_state, llm_response)

    assert "tool_name=UNKNOWN_TOOL" in caplog.text
    assert "response_tool_call_count=1" in caplog.text
    assert "tool_names=UNKNOWN_TOOL" in caplog.text
    assert "error_category=UNKNOWN_TOOL" in caplog.text
    assert sensitive_content not in caplog.text


@pytest.mark.parametrize(
    ("tool_names", "expected_safe_names"),
    [
        (["read_thread", "create_draft"], "read_thread,create_draft"),
        (["read_thread", "read_thread"], "read_thread,read_thread"),
    ],
)
def test_multiple_tool_calls_are_rejected_atomically(
    tool_names: list[str],
    expected_safe_names: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Диагностирует порядок вызовов и атомарно отклоняет весь ответ."""
    mail_tools = Mock(spec=MailTools)
    agent = MailAgent(Mock(), mail_tools, Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Запрос")
    tool_calls = [
        LlmToolCall(
            call_id=f"call-{call_number}",
            name=tool_name,
            arguments_json="{}",
            arguments={"thread_id": "thread-001"},
        )
        for call_number, tool_name in enumerate(tool_names, start=1)
    ]
    llm_response = LlmResponse(
        response_id="response-multiple",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[],
        tool_calls=tool_calls,
        raw_response_json="{}",
    )
    caplog.set_level(logging.INFO, logger="mail_agent.agent")

    with pytest.raises(RuntimeError, match="более одного"):
        agent._process_approved_response(run_state, llm_response)

    assert "response_tool_call_count=2" in caplog.text
    assert f"tool_names={expected_safe_names}" in caplog.text
    mail_tools.assert_not_called()


def test_verified_source_ids_remove_duplicates(
    email_threads: list[EmailThread],
) -> None:
    """Возвращает уникальные ID писем и документов в стабильном порядке."""
    agent = MailAgent(Mock(), Mock(), Mock(), Mock(), "Шаблон")
    run_state = agent._create_run_state("Сводка")
    run_state.selected_thread = email_threads[0]
    run_state.context_results = ContextSearchResults(
        matches=[
            ContextSearchResult(
                chunk_id="email-001:0",
                source_id="email-001",
                source_type="email",
                date="2026-10-01T08:00:00.100000Z",
                thread_id="thread-001",
                text="Фрагмент",
                distance=0.1,
            )
        ],
        source_ids=["email-001", "delivery_terms.md"],
    )

    source_ids = agent._build_verified_source_ids(run_state)

    assert source_ids.count("email-001") == 1
    assert source_ids[-1] == "delivery_terms.md"


def test_run_generates_new_canary_token_for_each_run() -> None:
    """Создаёт новый токен для запуска и сохраняет его на один ответ цикла."""
    llm_client = Mock(spec=OpenAILlmClient)
    llm_client.generate_response.return_value = LlmResponse(
        response_id="safe",
        status="completed",
        output_text="Безопасный ответ",
        refusal_text=None,
        output_items=[],
        tool_calls=[],
        raw_response_json="{}",
    )
    canary_guard = Mock(spec=CanaryGuard)
    agent = MailAgent(
        llm_client,
        Mock(),
        Mock(),
        canary_guard,
        "Активный токен: {{CANARY_TOKEN}}",
    )
    agent._process_approved_response = Mock(
        return_value=AgentResult(
            text="Безопасный ответ",
            thread_id="thread-001",
            source_ids=[],
        )
    )

    agent.run("Первый запрос")
    agent.run("Второй запрос")

    first_token = canary_guard.validate_response.call_args_list[0].kwargs[
        "canary_token"
    ]
    second_token = canary_guard.validate_response.call_args_list[1].kwargs[
        "canary_token"
    ]
    assert first_token != second_token
    assert (
        first_token
        in llm_client.generate_response.call_args_list[0].kwargs[
            "instructions"
        ]
    )
    assert (
        second_token
        in llm_client.generate_response.call_args_list[1].kwargs[
            "instructions"
        ]
    )


def test_run_creates_draft_after_each_response_passes_guard(
    mail_tools: MailTools,
    email_threads: list[EmailThread],
) -> None:
    """Создаёт черновик после трёх последовательных безопасных проверок."""
    email_thread = email_threads[0]
    recipient = email_thread.messages[0].sender
    read_arguments = {"thread_id": email_thread.thread_id}
    draft_arguments = {
        "thread_id": email_thread.thread_id,
        "recipient": recipient,
        "subject": "Ответ по поставке",
        "body": "Подтверждаем получение запроса.",
    }
    llm_client = Mock(spec=OpenAILlmClient)
    llm_client.generate_response.side_effect = [
        LlmResponse(
            response_id="read",
            status="completed",
            output_text="",
            refusal_text=None,
            output_items=[{"type": "function_call", "call_id": "call-read"}],
            tool_calls=[
                LlmToolCall(
                    call_id="call-read",
                    name="read_thread",
                    arguments_json=json.dumps(read_arguments),
                    arguments=read_arguments,
                )
            ],
            raw_response_json="{}",
        ),
        LlmResponse(
            response_id="draft",
            status="completed",
            output_text="",
            refusal_text=None,
            output_items=[{"type": "function_call", "call_id": "call-draft"}],
            tool_calls=[
                LlmToolCall(
                    call_id="call-draft",
                    name="create_draft",
                    arguments_json=json.dumps(
                        draft_arguments, ensure_ascii=False
                    ),
                    arguments=draft_arguments,
                )
            ],
            raw_response_json="{}",
        ),
        LlmResponse(
            response_id="final",
            status="completed",
            output_text="Черновик сохранён",
            refusal_text=None,
            output_items=[{"type": "message", "text": "Черновик сохранён"}],
            tool_calls=[],
            raw_response_json="{}",
        ),
    ]
    context_store = Mock(spec=ChromaContextStore)
    context_store.search_context.return_value = ContextSearchResults(
        matches=[
            ContextSearchResult(
                chunk_id="document:client_communication.md:0000",
                source_id="client_communication.md",
                source_type="document",
                date="2026-10-04",
                text="Правила общения",
                distance=0.1,
            )
        ],
        source_ids=["client_communication.md"],
    )
    canary_guard = Mock(spec=CanaryGuard)
    agent = MailAgent(
        llm_client,
        mail_tools,
        context_store,
        canary_guard,
        "Активный токен: {{CANARY_TOKEN}}",
    )

    result = agent.run(
        "Подготовь черновик участнику переписки",
        allow_draft=True,
    )

    assert result.text == "Черновик сохранён"
    assert result.draft_id is not None
    UUID(result.draft_id)
    assert result.draft_status == "created"
    assert result.thread_id == email_thread.thread_id
    assert "client_communication.md" in result.source_ids
    assert canary_guard.validate_response.call_count == 3
    tool_name_sequences = [
        [schema["name"] for schema in call.kwargs["tool_schemas"]]
        for call in llm_client.generate_response.call_args_list
    ]
    assert tool_name_sequences == [
        ["search_emails", "read_thread"],
        ["create_draft"],
        [],
    ]
    run_tokens = {
        call.kwargs["canary_token"]
        for call in canary_guard.validate_response.call_args_list
    }
    assert len(run_tokens) == 1
