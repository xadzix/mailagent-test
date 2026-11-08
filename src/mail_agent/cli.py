"""Командная строка для подготовки данных и защищённого запуска агента."""

import argparse
import logging
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from openai import OpenAI, OpenAIError
from pydantic import TypeAdapter

from mail_agent.agent import LOGGER, AgentResult, MailAgent
from mail_agent.canary_guard import BLOCKED_RESPONSE_MESSAGE, CanaryGuard
from mail_agent.context_store import DOCUMENT_FILE_NAMES, ChromaContextStore
from mail_agent.email_models import EmailThread
from mail_agent.email_repository import SqliteEmailRepository
from mail_agent.llm_client import OpenAILlmClient
from mail_agent.mail_tools import MailTools
from mail_agent.settings import Settings, load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_EMAILS_PATH = PROJECT_ROOT / "data" / "emails.json"
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "documents"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.txt"
SECURITY_EVENTS_PATH = PROJECT_ROOT / "security_events.jsonl"


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт парсер трёх утверждённых CLI-команд и их аргументов."""
    argument_parser = argparse.ArgumentParser(
        prog="mail-agent",
        description="Почтовый AI-агент.",
    )
    subparsers = argument_parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "init-db",
        help="Создать SQLite-схему и загрузить демонстрационные письма.",
    )
    index_parser = subparsers.add_parser(
        "index-context",
        help="Индексировать письма и справочные документы в ChromaDB.",
    )
    index_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Удалить существующую коллекцию и построить индекс заново.",
    )
    run_parser = subparsers.add_parser(
        "run",
        help="Передать запрос менеджера защищённому агенту.",
    )
    run_parser.add_argument("request", help="Текст запроса менеджера.")
    run_parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Вывести в stderr безопасные счётчики цикла агента.",
    )
    run_parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Явно разрешить сохранение черновика в текущем запуске.",
    )
    return argument_parser


def load_demo_email_threads(emails_path: Path) -> list[EmailThread]:
    """Читает и полностью валидирует демонстрационные цепочки из JSON."""
    if not emails_path.is_file():
        raise FileNotFoundError(
            f"Не найден файл демонстрационных писем: {emails_path}"
        )
    return TypeAdapter(list[EmailThread]).validate_json(
        emails_path.read_text(encoding="utf-8")
    )


def create_openai_client(settings: Settings) -> OpenAI:
    """Создаёт SDK-клиент только с явно заданными ключом и адресом API."""
    if settings.llm_api_key is None:
        raise ValueError("Не задан LLM_API_KEY")
    api_key = settings.llm_api_key.get_secret_value().strip()
    if not api_key:
        raise ValueError("Не задан LLM_API_KEY")
    if settings.llm_base_url is None or not settings.llm_base_url.strip():
        raise ValueError("Не задан LLM_BASE_URL")
    return OpenAI(
        api_key=api_key,
        base_url=settings.llm_base_url.strip(),
    )


def initialize_demo_database(settings: Settings) -> tuple[int, int, bool]:
    """Создаёт схему и загружает полный набор писем без перезаписи данных."""
    email_threads = load_demo_email_threads(DEMO_EMAILS_PATH)
    expected_message_ids = {
        message.message_id
        for email_thread in email_threads
        for message in email_thread.messages
    }
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    email_repository = SqliteEmailRepository(settings.database_path)
    email_repository.initialize_schema()
    existing_messages = email_repository.list_email_messages()
    existing_message_ids = {
        message.message_id for message in existing_messages
    }

    if existing_message_ids:
        if not expected_message_ids.issubset(existing_message_ids):
            raise RuntimeError(
                "SQLite уже содержит неполный или другой набор писем; "
                "существующие данные не изменены"
            )
        return len(email_threads), len(existing_messages), False

    email_repository.save_email_threads(email_threads)
    return len(email_threads), len(expected_message_ids), True


def index_context(
    settings: Settings,
    recreate_collection: bool,
) -> tuple[int, int]:
    """Проверяет источники и индексирует полный архив писем и документов."""
    openai_client = create_openai_client(settings)
    if (
        settings.embedding_model is None
        or not settings.embedding_model.strip()
    ):
        raise ValueError("Не задан EMBEDDING_MODEL")
    if not settings.database_path.is_file():
        raise RuntimeError(
            "SQLite не инициализирована; выполните mail-agent init-db"
        )

    email_repository = SqliteEmailRepository(settings.database_path)
    email_messages = email_repository.list_email_messages()
    expected_threads = load_demo_email_threads(DEMO_EMAILS_PATH)
    expected_message_ids = {
        message.message_id
        for email_thread in expected_threads
        for message in email_thread.messages
    }
    actual_message_ids = {message.message_id for message in email_messages}
    if not expected_message_ids.issubset(actual_message_ids):
        raise RuntimeError(
            "SQLite не содержит полный набор демонстрационных писем; "
            "выполните mail-agent init-db"
        )

    missing_documents = [
        DOCUMENTS_PATH / document_file_name
        for document_file_name in DOCUMENT_FILE_NAMES
        if not (DOCUMENTS_PATH / document_file_name).is_file()
    ]
    if missing_documents:
        missing_names = ", ".join(path.name for path in missing_documents)
        raise FileNotFoundError(
            f"Не найдены обязательные справочные документы: {missing_names}"
        )

    context_store = ChromaContextStore(
        chroma_path=settings.chroma_path,
        openai_client=openai_client,
        embedding_model=settings.embedding_model,
        recreate_collection=recreate_collection,
    )
    indexed_email_count = context_store.index_emails(email_messages)
    indexed_document_count = context_store.index_documents(DOCUMENTS_PATH)
    return indexed_email_count, indexed_document_count


def run_manager_request(
    settings: Settings,
    manager_request: str,
    allow_draft: bool = False,
) -> AgentResult:
    """Проверяет данные и запускает защищённый путь агента."""
    openai_client = create_openai_client(settings)
    if settings.llm_model is None or not settings.llm_model.strip():
        raise ValueError("Не задан LLM_MODEL")
    if (
        settings.embedding_model is None
        or not settings.embedding_model.strip()
    ):
        raise ValueError("Не задан EMBEDDING_MODEL")
    if not settings.database_path.is_file():
        raise RuntimeError(
            "SQLite не инициализирована; выполните mail-agent init-db"
        )

    email_repository = SqliteEmailRepository(settings.database_path)
    email_messages = email_repository.list_email_messages()
    expected_threads = load_demo_email_threads(DEMO_EMAILS_PATH)
    expected_message_ids = {
        message.message_id
        for email_thread in expected_threads
        for message in email_thread.messages
    }
    if not expected_message_ids.issubset(
        {message.message_id for message in email_messages}
    ):
        raise RuntimeError(
            "SQLite не содержит демонстрационные письма; выполните "
            "mail-agent init-db"
        )
    if not settings.chroma_path.exists():
        raise RuntimeError(
            "Chroma-индекс не подготовлен; выполните mail-agent index-context"
        )

    context_store = ChromaContextStore(
        chroma_path=settings.chroma_path,
        openai_client=openai_client,
        embedding_model=settings.embedding_model,
    )
    expected_source_ids = expected_message_ids | set(DOCUMENT_FILE_NAMES)
    missing_source_ids = (
        expected_source_ids - context_store.list_indexed_source_ids()
    )
    if missing_source_ids:
        raise RuntimeError(
            "Chroma-индекс не содержит все обязательные источники: "
            f"{', '.join(sorted(missing_source_ids))}; "
            "выполните mail-agent index-context"
        )
    if not SYSTEM_PROMPT_PATH.is_file():
        raise FileNotFoundError(
            f"Не найден системный промпт: {SYSTEM_PROMPT_PATH}"
        )

    mail_agent = MailAgent(
        llm_client=OpenAILlmClient(openai_client, settings.llm_model),
        mail_tools=MailTools(email_repository),
        context_store=context_store,
        canary_guard=CanaryGuard(SECURITY_EVENTS_PATH),
        system_prompt_template=SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
    )
    return mail_agent.run(manager_request, allow_draft=allow_draft)


def _is_blocked_agent_result(agent_result: AgentResult) -> bool:
    """Распознаёт нейтральный результ полной CanaryGuard-блокировки."""
    return (
        agent_result.text == BLOCKED_RESPONSE_MESSAGE
        and agent_result.thread_id is None
        and not agent_result.source_ids
        and agent_result.draft_id is None
    )


def format_agent_result(agent_result: AgentResult) -> str:
    """Форматирует результат без метаданных заблокированного ответа."""
    if _is_blocked_agent_result(agent_result):
        return agent_result.text

    output_lines = ["Сводка:", agent_result.text]
    if agent_result.thread_id is not None:
        output_lines.extend(("", f"thread_id: {agent_result.thread_id}"))
    output_lines.append("source_ids:")
    output_lines.extend(
        f"- {source_id}" for source_id in agent_result.source_ids
    )
    output_lines.append(f"draft_status: {agent_result.draft_status}")
    if (
        agent_result.draft_status == "created"
        and agent_result.draft_id is not None
    ):
        output_lines.append(f"draft_id: {agent_result.draft_id}")
    return "\n".join(output_lines)


def main(arguments: Sequence[str] | None = None) -> int:
    """Выполняет выбранную команду и возвращает устойчивый код завершения."""
    argument_parser = build_argument_parser()
    parsed_arguments = argument_parser.parse_args(arguments)
    settings = load_settings()

    try:
        if parsed_arguments.command == "init-db":
            thread_count, message_count, data_loaded = (
                initialize_demo_database(settings)
            )
            status = "загружены" if data_loaded else "уже присутствуют"
            print(
                f"Демонстрационные данные {status}: "
                f"цепочек — {thread_count}, писем — {message_count}."
            )
            return 0
        if parsed_arguments.command == "index-context":
            email_count, document_count = index_context(
                settings,
                recreate_collection=parsed_arguments.recreate,
            )
            print(
                "Индексация завершена: "
                f"писем обновлено — {email_count}, "
                f"документов обновлено — {document_count}."
            )
            return 0
        if parsed_arguments.command == "run":
            previous_handlers = LOGGER.handlers.copy()
            previous_level = LOGGER.level
            previous_propagate = LOGGER.propagate
            diagnostics_handler: logging.Handler | None = None
            if parsed_arguments.diagnostics:
                diagnostics_handler = logging.StreamHandler(sys.stderr)
                diagnostics_handler.setFormatter(
                    logging.Formatter("%(message)s")
                )
                LOGGER.handlers = [diagnostics_handler]
                LOGGER.setLevel(logging.INFO)
                LOGGER.propagate = False
            try:
                agent_result = run_manager_request(
                    settings,
                    parsed_arguments.request,
                    allow_draft=parsed_arguments.allow_draft,
                )
            finally:
                if diagnostics_handler is not None:
                    LOGGER.handlers = previous_handlers
                    LOGGER.setLevel(previous_level)
                    LOGGER.propagate = previous_propagate
                    diagnostics_handler.close()
            print(format_agent_result(agent_result))
            return 2 if _is_blocked_agent_result(agent_result) else 0
    except OpenAIError:
        print(
            "Ошибка OpenAI API. Проверьте сеть, API-ключ и "
            "доступность модели.",
            file=sys.stderr,
        )
        return 1
    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
        sqlite3.Error,
    ) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1

    argument_parser.error("Неизвестная команда")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
