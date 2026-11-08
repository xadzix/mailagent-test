"""Проверки точки входа командной строки."""

import logging
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from mail_agent.agent import LOGGER, AgentResult
from mail_agent.canary_guard import BLOCKED_RESPONSE_MESSAGE
from mail_agent.cli import (
    create_openai_client,
    format_agent_result,
    index_context,
    initialize_demo_database,
    main,
    run_manager_request,
)
from mail_agent.context_store import DOCUMENT_FILE_NAMES, ChromaContextStore
from mail_agent.settings import Settings, load_settings


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Проверяет успешный вывод справки без ключа API и внешних обращений."""
    with pytest.raises(SystemExit) as exit_context:
        main(["--help"])

    captured_output = capsys.readouterr()

    assert exit_context.value.code == 0
    assert "mail-agent" in captured_output.out


def test_load_settings_uses_neutral_environment_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Читает нейтральные имена настроек без участия OpenAI SDK."""
    database_path = tmp_path / "mail.sqlite3"
    chroma_path = tmp_path / "chroma"
    monkeypatch.setenv("LLM_API_KEY", "neutral-key")
    monkeypatch.setenv("LLM_MODEL", "qwen3:8b")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "embeddinggemma")
    monkeypatch.setenv("MAIL_AGENT_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("MAIL_AGENT_CHROMA_PATH", str(chroma_path))

    settings = load_settings()

    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "neutral-key"
    assert settings.llm_model == "qwen3:8b"
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.embedding_model == "embeddinggemma"
    assert settings.database_path == database_path
    assert settings.chroma_path == chroma_path


def test_load_settings_ignores_legacy_openai_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Не использует старые OPENAI-переменные как скрытый fallback."""
    for variable_name in (
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(variable_name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_MODEL", "legacy-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://legacy.invalid/v1")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "legacy-embedding")

    settings = load_settings()

    assert settings.llm_api_key is None
    assert settings.llm_model is None
    assert settings.llm_base_url is None
    assert settings.embedding_model is None


@pytest.mark.parametrize(
    "base_url",
    ["http://localhost:11434/v1", "https://api.openai.com/v1"],
)
def test_create_openai_client_passes_explicit_key_and_base_url(
    base_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Передаёт SDK явные настройки для локального и облачного endpoint."""
    openai_constructor = Mock()
    monkeypatch.setattr("mail_agent.cli.OpenAI", openai_constructor)
    settings = Settings(llm_api_key="explicit-key", llm_base_url=base_url)

    created_client = create_openai_client(settings)

    assert created_client == openai_constructor.return_value
    openai_constructor.assert_called_once_with(
        api_key="explicit-key",
        base_url=base_url,
    )


@pytest.mark.parametrize(
    ("api_key", "base_url", "expected_error"),
    [
        (None, "http://localhost:11434/v1", "LLM_API_KEY"),
        ("   ", "http://localhost:11434/v1", "LLM_API_KEY"),
        ("test-key", None, "LLM_BASE_URL"),
        ("test-key", "   ", "LLM_BASE_URL"),
    ],
)
def test_create_openai_client_rejects_missing_explicit_configuration(
    api_key: str | None,
    base_url: str | None,
    expected_error: str,
) -> None:
    """Не создаёт SDK-клиент с отсутствующим ключом или неявным endpoint."""
    settings = Settings(llm_api_key=api_key, llm_base_url=base_url)

    with pytest.raises(ValueError, match=expected_error):
        create_openai_client(settings)


def test_cli_initializes_demo_database_without_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Создаёт SQLite и загружает письма без обращения к OpenAI."""
    database_path = tmp_path / "mail.sqlite3"
    monkeypatch.setenv("MAIL_AGENT_DATABASE_PATH", str(database_path))
    for variable_name in (
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(variable_name, raising=False)

    exit_code = main(["init-db"])

    captured_output = capsys.readouterr()
    assert exit_code == 0
    assert database_path.is_file()
    assert "писем — 7" in captured_output.out


def test_cli_repeated_init_db_preserves_existing_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Повторно подтверждает данные без удаления или повторной вставки."""
    database_path = tmp_path / "mail.sqlite3"
    monkeypatch.setenv("MAIL_AGENT_DATABASE_PATH", str(database_path))

    assert main(["init-db"]) == 0
    assert main(["init-db"]) == 0

    captured_output = capsys.readouterr()
    assert "уже присутствуют" in captured_output.out


def test_cli_index_context_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Отклоняет индексацию до любых действий без API-ключа."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    exit_code = main(["index-context"])

    captured_output = capsys.readouterr()
    assert exit_code == 1
    assert "LLM_API_KEY" in captured_output.err


def test_cli_index_context_requires_explicit_base_url(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Не подключается к неявному облачному endpoint при отсутствии URL."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    exit_code = main(["index-context"])

    captured_output = capsys.readouterr()
    assert exit_code == 1
    assert "LLM_BASE_URL" in captured_output.err


@pytest.mark.parametrize("embedding_model", [None, "   "])
def test_cli_index_context_requires_embedding_model(
    embedding_model: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Отклоняет отсутствующее или пустое имя embedding-модели."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    if embedding_model is None:
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    else:
        monkeypatch.setenv("EMBEDDING_MODEL", embedding_model)

    exit_code = main(["index-context"])

    captured_output = capsys.readouterr()
    assert exit_code == 1
    assert "EMBEDDING_MODEL" in captured_output.err


@pytest.mark.parametrize("llm_model", [None, "   "])
def test_cli_run_requires_model_without_exposing_key(
    llm_model: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Отклоняет отсутствующую модель и не печатает заданный секрет."""
    api_key = "test-secret-api-key"
    monkeypatch.setenv("LLM_API_KEY", api_key)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    if llm_model is None:
        monkeypatch.delenv("LLM_MODEL", raising=False)
    else:
        monkeypatch.setenv("LLM_MODEL", llm_model)

    exit_code = main(["run", "Подготовь сводку"])

    captured_output = capsys.readouterr()
    assert exit_code == 1
    assert "LLM_MODEL" in captured_output.err
    assert api_key not in captured_output.err


def test_cli_run_requires_initialized_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Подсказывает init-db, если SQLite ещё не создана."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setenv(
        "MAIL_AGENT_DATABASE_PATH", str(tmp_path / "missing.sqlite3")
    )

    exit_code = main(["run", "Подготовь сводку"])

    captured_output = capsys.readouterr()
    assert exit_code == 1
    assert "mail-agent init-db" in captured_output.err


def test_cli_run_requires_prepared_context_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Не запускает агента с отсутствующим Chroma-индексом."""
    database_path = tmp_path / "mail.sqlite3"
    monkeypatch.setenv("MAIL_AGENT_DATABASE_PATH", str(database_path))
    monkeypatch.setenv(
        "MAIL_AGENT_CHROMA_PATH", str(tmp_path / "missing-chroma")
    )
    assert main(["init-db"]) == 0
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding")

    exit_code = main(["run", "Подготовь сводку"])

    captured_output = capsys.readouterr()
    assert exit_code == 1
    assert "mail-agent index-context" in captured_output.err


def test_run_rejects_context_index_with_missing_source_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Не запускает агента без обязательного источника Chroma."""
    settings = Settings(
        llm_model="test-model",
        embedding_model="test-embedding",
        database_path=tmp_path / "mail.sqlite3",
        chroma_path=tmp_path / "chroma",
    )
    initialize_demo_database(settings)
    settings.chroma_path.mkdir()
    context_store = Mock(spec=ChromaContextStore)
    context_store.list_indexed_source_ids.return_value = {
        *(f"email-{message_number:03d}" for message_number in range(2, 8)),
        *DOCUMENT_FILE_NAMES,
    }
    monkeypatch.setattr("mail_agent.cli.create_openai_client", Mock())
    monkeypatch.setattr(
        "mail_agent.cli.ChromaContextStore",
        Mock(return_value=context_store),
    )
    mail_agent = Mock()
    monkeypatch.setattr("mail_agent.cli.MailAgent", mail_agent)

    with pytest.raises(RuntimeError, match="email-001.*index-context"):
        run_manager_request(settings, "Подготовь сводку")

    mail_agent.assert_not_called()


def test_run_accepts_complete_context_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Принимает все обязательные ID и дополнительный источник."""
    settings = Settings(
        llm_model="test-model",
        embedding_model="test-embedding",
        database_path=tmp_path / "mail.sqlite3",
        chroma_path=tmp_path / "chroma",
    )
    initialize_demo_database(settings)
    settings.chroma_path.mkdir()
    context_store = Mock(spec=ChromaContextStore)
    context_store.list_indexed_source_ids.return_value = {
        *(f"email-{message_number:03d}" for message_number in range(1, 8)),
        *DOCUMENT_FILE_NAMES,
        "additional-source",
    }
    monkeypatch.setattr("mail_agent.cli.create_openai_client", Mock())
    monkeypatch.setattr(
        "mail_agent.cli.ChromaContextStore",
        Mock(return_value=context_store),
    )
    expected_result = AgentResult(
        text="Сводка",
        thread_id="thread-001",
        source_ids=["email-001"],
    )
    mail_agent = Mock()
    mail_agent.return_value.run.return_value = expected_result
    monkeypatch.setattr("mail_agent.cli.MailAgent", mail_agent)

    result = run_manager_request(
        settings,
        "Подготовь сводку",
        allow_draft=True,
    )

    assert result == expected_result
    mail_agent.return_value.run.assert_called_once_with(
        "Подготовь сводку",
        allow_draft=True,
    )


def test_cli_canary_block_prints_only_neutral_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Возвращает ненулевой код и не добавляет метаданные при блокировке."""
    blocked_result = AgentResult(
        text=BLOCKED_RESPONSE_MESSAGE,
        thread_id=None,
        source_ids=[],
        draft_id=None,
    )
    run_request_mock = Mock(return_value=blocked_result)
    monkeypatch.setattr("mail_agent.cli.run_manager_request", run_request_mock)

    exit_code = main(["run", "Запрос"])

    captured_output = capsys.readouterr()
    assert exit_code == 2
    assert captured_output.out.strip() == BLOCKED_RESPONSE_MESSAGE
    assert "thread_id" not in captured_output.out
    assert "source_ids" not in captured_output.out
    assert "draft_id" not in captured_output.out


def test_cli_diagnostics_flag_enables_safe_logging(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Включает отдельный stderr-handler только на время запуска с флагом."""
    expected_result = AgentResult(
        text="Сводка",
        thread_id="thread-001",
        source_ids=["email-001"],
    )
    run_request_mock = Mock(return_value=expected_result)
    diagnostics_handler = Mock(spec=logging.Handler)
    stream_handler_mock = Mock(return_value=diagnostics_handler)
    monkeypatch.setattr("mail_agent.cli.run_manager_request", run_request_mock)
    monkeypatch.setattr(
        "mail_agent.cli.logging.StreamHandler", stream_handler_mock
    )
    previous_handlers = LOGGER.handlers.copy()
    previous_level = LOGGER.level
    previous_propagate = LOGGER.propagate

    diagnostics_exit_code = main(["run", "--diagnostics", "Запрос"])
    ordinary_exit_code = main(["run", "Запрос"])

    assert diagnostics_exit_code == 0
    assert ordinary_exit_code == 0
    stream_handler_mock.assert_called_once_with(sys.stderr)
    diagnostics_handler.setFormatter.assert_called_once()
    diagnostics_handler.close.assert_called_once()
    assert LOGGER.handlers == previous_handlers
    assert LOGGER.level == previous_level
    assert LOGGER.propagate == previous_propagate
    captured_output = capsys.readouterr()
    assert "llm_response_processed" not in captured_output.err
    assert run_request_mock.call_count == 2


def test_cli_allow_draft_flag_enables_draft_authorization(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Передаёт явное разрешение на черновик из CLI в запуск агента."""
    expected_result = AgentResult(
        text="Сводка условий",
        thread_id="thread-001",
        source_ids=["email-001"],
        draft_id="00000000-0000-4000-8000-000000000001",
        draft_status="created",
    )
    run_request_mock = Mock(return_value=expected_result)
    monkeypatch.setattr("mail_agent.cli.run_manager_request", run_request_mock)

    exit_code = main(["run", "--allow-draft", "Подготовь ответ"])

    assert exit_code == 0
    assert run_request_mock.call_args.kwargs["allow_draft"] is True
    captured_output = capsys.readouterr()
    assert "draft_status: created" in captured_output.out
    assert f"draft_id: {expected_result.draft_id}" in captured_output.out


def test_cli_without_allow_draft_preserves_summary_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Сохраняет обычный режим сводки без разрешения создавать черновик."""
    expected_result = AgentResult(
        text="Сводка без создания черновика",
        thread_id="thread-001",
        source_ids=["email-001"],
        draft_status="not_authorized",
    )
    run_request_mock = Mock(return_value=expected_result)
    monkeypatch.setattr("mail_agent.cli.run_manager_request", run_request_mock)

    exit_code = main(["run", "Подготовь сводку"])

    assert exit_code == 0
    assert run_request_mock.call_args.kwargs["allow_draft"] is False
    captured_output = capsys.readouterr()
    assert "Сводка без создания черновика" in captured_output.out
    assert "draft_status: not_authorized" in captured_output.out
    assert "draft_id:" not in captured_output.out


def test_format_agent_result_hides_id_without_created_status() -> None:
    """Не отображает идентификатор без подтверждённого статуса created."""
    formatted_result = format_agent_result(
        AgentResult(
            text=(
                "Черновик в текущем запуске для выбранной цепочки не создан; "
                "ранее сохранённые черновики в SQLite не изменялись."
            ),
            thread_id="thread-001",
            source_ids=["email-001"],
            draft_id=None,
            draft_status="not_created",
        )
    )

    assert "draft_status: not_created" in formatted_result
    assert "draft_id:" not in formatted_result


def test_recreate_checks_documents_before_opening_chroma(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Не создаёт и не удаляет коллекцию до проверки четырёх документов."""
    settings = Settings(
        llm_api_key="test-key",
        llm_base_url="http://localhost:11434/v1",
        embedding_model="test-embedding",
        database_path=tmp_path / "mail.sqlite3",
        chroma_path=tmp_path / "chroma",
    )
    initialize_demo_database(settings)
    context_store_mock = Mock(spec=ChromaContextStore)
    monkeypatch.setattr(
        "mail_agent.cli.ChromaContextStore", context_store_mock
    )
    monkeypatch.setattr(
        "mail_agent.cli.DOCUMENTS_PATH", tmp_path / "missing-documents"
    )

    with pytest.raises(FileNotFoundError, match="справочные документы"):
        index_context(settings, recreate_collection=True)

    context_store_mock.assert_not_called()
