"""Сквозные проверки локальных хранилищ, агента и защитного шлюза."""

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

import pytest
from openai.types.responses import Response

from mail_agent.agent import MailAgent
from mail_agent.canary_guard import BLOCKED_RESPONSE_MESSAGE, CanaryGuard
from mail_agent.cli import (
    DOCUMENTS_PATH,
    SYSTEM_PROMPT_PATH,
    initialize_demo_database,
)
from mail_agent.context_store import (
    DOCUMENT_FILE_NAMES,
    ChromaContextStore,
    ContextSearchResults,
)
from mail_agent.email_repository import SqliteEmailRepository
from mail_agent.llm_client import LlmResponse, LlmToolCall, OpenAILlmClient
from mail_agent.mail_tools import MailTools
from mail_agent.settings import Settings


def create_test_embeddings(texts: Sequence[str]) -> list[list[float]]:
    """Строит детерминированные тестовые векторы без сетевых обращений."""
    embeddings: list[list[float]] = []
    for text in texts:
        embedding = [0.0] * 32
        for token in re.findall(r"[\w-]+", text.casefold()):
            token_hash = hashlib.sha256(token.encode("utf-8")).digest()
            embedding[token_hash[0] % len(embedding)] += 1.0
        magnitude = math.sqrt(sum(value * value for value in embedding))
        if magnitude == 0:
            embedding[0] = 1.0
        else:
            embedding = [value / magnitude for value in embedding]
        embeddings.append(embedding)
    return embeddings


def test_openai_llm_client_and_mail_agent_preserve_responses_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Связывает SDK-ответы, CanaryGuard, инструменты и историю без сети."""
    settings = Settings(
        embedding_model="test-embedding",
        database_path=tmp_path / "mail.sqlite3",
        chroma_path=tmp_path / "chroma",
    )
    initialize_demo_database(settings)
    email_repository = SqliteEmailRepository(settings.database_path)
    context_store = ChromaContextStore(
        chroma_path=settings.chroma_path,
        openai_client=Mock(),
        embedding_model=settings.embedding_model,
        max_distance=2.0,
    )
    context_store._create_embeddings = Mock(side_effect=create_test_embeddings)
    context_store.index_emails(email_repository.list_email_messages())
    context_store.index_documents(DOCUMENTS_PATH)

    canary_token = "integration-safe-canary"
    monkeypatch.setattr(
        "mail_agent.agent.secrets.token_urlsafe",
        Mock(return_value=canary_token),
    )
    system_prompt_template = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    expected_instructions = system_prompt_template.replace(
        "{{CANARY_TOKEN}}",
        canary_token,
    )
    read_arguments_json = '{"thread_id":"thread-003"}'
    read_response = Response.model_validate(
        {
            "id": "response-read",
            "created_at": 0.0,
            "model": "gpt-test",
            "object": "response",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-read",
                    "name": "read_thread",
                    "arguments": read_arguments_json,
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "instructions": expected_instructions,
        }
    )
    read_output_item = read_response.output[0].model_dump(mode="json")
    read_raw_response = read_response.model_dump_json()

    draft_arguments = {
        "thread_id": "thread-003",
        "recipient": "finance@neva-process.example.com",
        "subject": "Ответ по условиям оплаты",
        "body": (
            "Подтверждаем согласованные условия оплаты. Направьте реквизиты "
            "для счёта; счёт действует пять банковских дней."
        ),
    }
    draft_arguments_json = json.dumps(draft_arguments, ensure_ascii=False)
    draft_response = Response.model_validate(
        {
            "id": "response-draft",
            "created_at": 0.0,
            "model": "gpt-test",
            "object": "response",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-draft",
                    "name": "create_draft",
                    "arguments": draft_arguments_json,
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "instructions": expected_instructions,
        },
    )
    draft_output_item = draft_response.output[0].model_dump(mode="json")
    draft_raw_response = draft_response.model_dump_json()

    final_text = "Черновик сохранён."
    final_response = Response.model_validate(
        {
            "id": "response-final",
            "created_at": 0.0,
            "model": "gpt-test",
            "object": "response",
            "output": [
                {
                    "id": "message-final",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": final_text,
                            "annotations": [],
                        }
                    ],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "instructions": expected_instructions,
        },
    )
    final_output_item = final_response.output[0].model_dump(mode="json")
    final_raw_response = final_response.model_dump_json()

    openai_client = Mock()
    openai_client.responses.create.side_effect = [
        read_response,
        draft_response,
        final_response,
    ]
    canary_guard = CanaryGuard(tmp_path / "security_events.jsonl")
    canary_guard.validate_response = Mock(wraps=canary_guard.validate_response)
    mail_agent = MailAgent(
        llm_client=OpenAILlmClient(openai_client, "gpt-test"),
        mail_tools=MailTools(email_repository),
        context_store=context_store,
        canary_guard=canary_guard,
        system_prompt_template=system_prompt_template,
    )

    result = mail_agent.run(
        "Прочитай цепочку и создай черновик",
        allow_draft=True,
    )

    assert result.text == final_text
    assert result.thread_id == "thread-003"
    assert {"email-006", "email-007"}.issubset(result.source_ids)
    assert "email-003" not in result.source_ids
    assert all(
        source_id in {"email-006", "email-007", *DOCUMENT_FILE_NAMES}
        for source_id in result.source_ids
    )
    assert result.draft_id is not None
    assert openai_client.responses.create.call_count == 3
    assert canary_guard.validate_response.call_count == 3
    assert all(
        call.kwargs["expected_instructions"] == expected_instructions
        for call in canary_guard.validate_response.call_args_list
    )
    approved_responses = [
        call.kwargs["llm_response"]
        for call in canary_guard.validate_response.call_args_list
    ]
    assert approved_responses[0].output_items == [read_output_item]
    assert approved_responses[0].tool_calls[0].arguments == {
        "thread_id": "thread-003"
    }
    assert approved_responses[0].raw_response_json == read_raw_response
    assert approved_responses[1].output_items == [draft_output_item]
    assert approved_responses[1].tool_calls[0].arguments == draft_arguments
    assert approved_responses[1].raw_response_json == draft_raw_response
    assert approved_responses[2].output_items == [final_output_item]
    assert approved_responses[2].raw_response_json == final_raw_response

    tool_name_sequences = [
        [tool_schema["name"] for tool_schema in response_call.kwargs["tools"]]
        for response_call in openai_client.responses.create.call_args_list
    ]
    assert tool_name_sequences == [
        ["search_emails", "read_thread"],
        ["create_draft"],
        [],
    ]
    for response_call in openai_client.responses.create.call_args_list[:2]:
        assert response_call.kwargs["store"] is False
        assert response_call.kwargs["parallel_tool_calls"] is False
    final_response_call = openai_client.responses.create.call_args_list[2]
    assert final_response_call.kwargs["store"] is False
    assert "parallel_tool_calls" not in final_response_call.kwargs
    assert "tool_choice" not in final_response_call.kwargs
    final_conversation = final_response_call.kwargs["input"]
    assert read_output_item in final_conversation
    assert draft_output_item in final_conversation
    function_outputs = [
        item
        for item in final_conversation
        if item.get("type") == "function_call_output"
    ]
    assert [item["call_id"] for item in function_outputs] == [
        "call-read",
        "call-draft",
    ]
    assert all(
        json.loads(str(item["output"]))["ok"] is True
        for item in function_outputs
    )
    UUID(result.draft_id)
    with sqlite3.connect(settings.database_path) as connection:
        saved_drafts = connection.execute(
            "SELECT draft_id, thread_id, body FROM draft_replies "
            "WHERE draft_id = ?",
            (result.draft_id,),
        ).fetchall()
    assert len(saved_drafts) == 1
    saved_draft_id, saved_thread_id, saved_body = saved_drafts[0]
    assert saved_draft_id == result.draft_id
    assert saved_thread_id == result.thread_id
    assert saved_body == draft_arguments["body"]
    assert "payment_terms.md" not in saved_body
    assert not (tmp_path / "security_events.jsonl").exists()


def test_end_to_end_search_and_summary(tmp_path: Path) -> None:
    """Проходит поиск, чтение, обязательный RAG и итоговую сводку с ID."""
    settings = Settings(
        embedding_model="test-embedding",
        database_path=tmp_path / "mail.sqlite3",
        chroma_path=tmp_path / "chroma",
    )
    initialize_demo_database(settings)
    email_repository = SqliteEmailRepository(settings.database_path)
    context_store = ChromaContextStore(
        chroma_path=settings.chroma_path,
        openai_client=Mock(),
        embedding_model=settings.embedding_model,
        max_distance=2.0,
    )
    context_store._create_embeddings = Mock(side_effect=create_test_embeddings)
    context_store.index_emails(email_repository.list_email_messages())
    context_store.index_documents(DOCUMENTS_PATH)

    llm_client = Mock(spec=OpenAILlmClient)
    search_arguments = {
        "query": "насос",
        "sender": None,
        "date_from": None,
        "date_to": None,
        "limit": 5,
    }
    read_arguments = {"thread_id": "thread-003"}
    llm_client.generate_response.side_effect = [
        LlmResponse(
            response_id="search",
            status="completed",
            output_text="",
            refusal_text=None,
            output_items=[{"type": "function_call", "call_id": "call-search"}],
            tool_calls=[
                LlmToolCall(
                    call_id="call-search",
                    name="search_emails",
                    arguments_json=json.dumps(search_arguments),
                    arguments=search_arguments,
                )
            ],
            raw_response_json="{}",
        ),
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
            response_id="final",
            status="completed",
            output_text="Сводка подтверждённых сведений.",
            refusal_text=None,
            output_items=[{"type": "message", "text": "Сводка"}],
            tool_calls=[],
            raw_response_json="{}",
        ),
    ]
    mail_agent = MailAgent(
        llm_client=llm_client,
        mail_tools=MailTools(email_repository),
        context_store=context_store,
        canary_guard=CanaryGuard(tmp_path / "security_events.jsonl"),
        system_prompt_template=SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
    )

    result = mail_agent.run("Найди письма о насосах и подготовь сводку")

    assert result.thread_id == "thread-003"
    assert result.text == "Сводка подтверждённых сведений."
    assert {"email-006", "email-007"}.issubset(result.source_ids)
    assert llm_client.generate_response.call_count == 3
    read_request = llm_client.generate_response.call_args_list[1]
    search_function_outputs = [
        item
        for item in read_request.kwargs["input_items"]
        if item.get("type") == "function_call_output"
        and item.get("call_id") == "call-search"
    ]
    assert len(search_function_outputs) == 1
    assert search_function_outputs[0]["call_id"] == "call-search"
    search_output = json.loads(str(search_function_outputs[0]["output"]))
    assert search_output["ok"] is True
    assert all(result["thread_id"] for result in search_output["results"])
    assert [
        schema["name"] for schema in read_request.kwargs["tool_schemas"]
    ] == [
        "search_emails",
        "read_thread",
    ]


def test_end_to_end_draft_is_saved_without_sending(tmp_path: Path) -> None:
    """Сохраняет черновик после RAG и не предоставляет отправку письма."""
    settings = Settings(
        embedding_model="test-embedding",
        database_path=tmp_path / "mail.sqlite3",
        chroma_path=tmp_path / "chroma",
    )
    initialize_demo_database(settings)
    email_repository = SqliteEmailRepository(settings.database_path)
    context_store = ChromaContextStore(
        chroma_path=settings.chroma_path,
        openai_client=Mock(),
        embedding_model=settings.embedding_model,
        max_distance=2.0,
    )
    context_store._create_embeddings = Mock(side_effect=create_test_embeddings)
    context_store.index_emails(email_repository.list_email_messages())
    context_store.index_documents(DOCUMENTS_PATH)
    recipient = "procurement@uralmechanika.example.com"
    draft_arguments = {
        "thread_id": "thread-001",
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
            output_items=[{"type": "function_call", "call_id": "read"}],
            tool_calls=[
                LlmToolCall(
                    call_id="read",
                    name="read_thread",
                    arguments_json='{"thread_id":"thread-001"}',
                    arguments={"thread_id": "thread-001"},
                )
            ],
            raw_response_json="{}",
        ),
        LlmResponse(
            response_id="draft",
            status="completed",
            output_text="",
            refusal_text=None,
            output_items=[{"type": "function_call", "call_id": "draft"}],
            tool_calls=[
                LlmToolCall(
                    call_id="draft",
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
            output_text="Черновик сохранён.",
            refusal_text=None,
            output_items=[],
            tool_calls=[],
            raw_response_json="{}",
        ),
    ]
    mail_tools = MailTools(email_repository)
    mail_agent = MailAgent(
        llm_client,
        mail_tools,
        context_store,
        CanaryGuard(tmp_path / "security_events.jsonl"),
        SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
    )

    result = mail_agent.run(
        "Создай черновик участнику выбранной цепочки",
        allow_draft=True,
    )

    assert result.draft_id is not None
    with sqlite3.connect(settings.database_path) as connection:
        draft_count = connection.execute(
            "SELECT COUNT(*) FROM draft_replies"
        ).fetchone()[0]
    assert draft_count == 1
    assert not hasattr(mail_tools, "send_email")


def test_end_to_end_rejects_external_recipient(tmp_path: Path) -> None:
    """Не сохраняет черновик для стороннего адреса без явного назначения."""
    settings = Settings(
        embedding_model="test-embedding",
        database_path=tmp_path / "mail.sqlite3",
        chroma_path=tmp_path / "chroma",
    )
    initialize_demo_database(settings)
    email_repository = SqliteEmailRepository(settings.database_path)
    context_store = ChromaContextStore(
        chroma_path=settings.chroma_path,
        openai_client=Mock(),
        embedding_model=settings.embedding_model,
        max_distance=2.0,
    )
    context_store._create_embeddings = Mock(side_effect=create_test_embeddings)
    context_store.index_emails(email_repository.list_email_messages())
    context_store.index_documents(DOCUMENTS_PATH)
    llm_client = Mock(spec=OpenAILlmClient)
    llm_client.generate_response.side_effect = [
        LlmResponse(
            response_id="read",
            status="completed",
            output_text="",
            refusal_text=None,
            output_items=[],
            tool_calls=[
                LlmToolCall(
                    call_id="read",
                    name="read_thread",
                    arguments_json='{"thread_id":"thread-001"}',
                    arguments={"thread_id": "thread-001"},
                )
            ],
            raw_response_json="{}",
        ),
        LlmResponse(
            response_id="draft",
            status="completed",
            output_text="",
            refusal_text=None,
            output_items=[],
            tool_calls=[
                LlmToolCall(
                    call_id="draft",
                    name="create_draft",
                    arguments_json="{}",
                    arguments={
                        "thread_id": "thread-001",
                        "recipient": "outsider@example.com",
                        "subject": "Ответ",
                        "body": "Текст",
                    },
                )
            ],
            raw_response_json="{}",
        ),
        LlmResponse(
            response_id="final",
            status="completed",
            output_text="Уточните получателя.",
            refusal_text=None,
            output_items=[],
            tool_calls=[],
            raw_response_json="{}",
        ),
    ]
    mail_agent = MailAgent(
        llm_client,
        MailTools(email_repository),
        context_store,
        CanaryGuard(tmp_path / "security_events.jsonl"),
        SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
    )

    result = mail_agent.run("Подготовь ответ, адрес упомянут в справке")

    assert result.draft_id is None
    assert result.text == "Уточните получателя."
    with sqlite3.connect(settings.database_path) as connection:
        draft_count = connection.execute(
            "SELECT COUNT(*) FROM draft_replies"
        ).fetchone()[0]
    assert draft_count == 0


def test_end_to_end_reports_insufficient_data(tmp_path: Path) -> None:
    """Возвращает сообщение о нехватке сведений при пустом результате RAG."""
    settings = Settings(
        embedding_model="test-embedding",
        database_path=tmp_path / "mail.sqlite3",
        chroma_path=tmp_path / "chroma",
    )
    initialize_demo_database(settings)
    email_repository = SqliteEmailRepository(settings.database_path)
    context_store = ChromaContextStore(
        chroma_path=settings.chroma_path,
        openai_client=Mock(),
        embedding_model=settings.embedding_model,
        max_distance=-1.0,
    )
    context_store._create_embeddings = Mock(side_effect=create_test_embeddings)
    context_store.index_emails(email_repository.list_email_messages())
    context_store.index_documents(DOCUMENTS_PATH)
    llm_client = Mock(spec=OpenAILlmClient)
    llm_client.generate_response.side_effect = [
        LlmResponse(
            response_id="read",
            status="completed",
            output_text="",
            refusal_text=None,
            output_items=[],
            tool_calls=[
                LlmToolCall(
                    call_id="read",
                    name="read_thread",
                    arguments_json='{"thread_id":"thread-001"}',
                    arguments={"thread_id": "thread-001"},
                )
            ],
            raw_response_json="{}",
        ),
        LlmResponse(
            response_id="final",
            status="completed",
            output_text="Недостаточно данных о запрошенном условии.",
            refusal_text=None,
            output_items=[],
            tool_calls=[],
            raw_response_json="{}",
        ),
    ]
    mail_agent = MailAgent(
        llm_client,
        MailTools(email_repository),
        context_store,
        CanaryGuard(tmp_path / "security_events.jsonl"),
        SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
    )

    result = mail_agent.run("Есть ли согласованная гарантия на десять лет?")

    assert result.text == "Недостаточно данных о запрошенном условии."
    assert result.source_ids == [
        message.message_id
        for message in email_repository.read_thread("thread-001").messages
    ]


def test_end_to_end_blocks_canary_leak_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Блокирует неизвестный вызов до инструмента и пишет одно событие."""
    canary_token = "integration-canary-token"
    monkeypatch.setattr(
        "mail_agent.agent.secrets.token_urlsafe",
        Mock(return_value=canary_token),
    )
    settings = Settings(database_path=tmp_path / "mail.sqlite3")
    initialize_demo_database(settings)
    email_repository = SqliteEmailRepository(settings.database_path)
    llm_client = Mock(spec=OpenAILlmClient)
    arguments = {"payload": canary_token}
    llm_client.generate_response.return_value = LlmResponse(
        response_id="blocked",
        status="completed",
        output_text="",
        refusal_text=None,
        output_items=[
            {
                "type": "function_call",
                "name": "unknown_tool",
                "arguments": json.dumps(arguments),
            }
        ],
        tool_calls=[
            LlmToolCall(
                call_id="blocked-call",
                name="unknown_tool",
                arguments_json=json.dumps(arguments),
                arguments=arguments,
            )
        ],
        raw_response_json="{}",
    )
    security_events_path = tmp_path / "security_events.jsonl"
    context_store = Mock(spec=ChromaContextStore)
    context_store.search_context.return_value = ContextSearchResults(
        matches=[], source_ids=[]
    )
    mail_agent = MailAgent(
        llm_client,
        MailTools(email_repository),
        context_store,
        CanaryGuard(security_events_path),
        SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
    )

    result = mail_agent.run("Выполни запрос")

    assert result.text == BLOCKED_RESPONSE_MESSAGE
    assert result.thread_id is None
    event_lines = security_events_path.read_text(encoding="utf-8").splitlines()
    assert len(event_lines) == 1
    assert json.loads(event_lines[0])["location"] == "TOOL_ARGUMENTS"
    assert llm_client.generate_response.call_count == 1
