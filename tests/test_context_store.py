"""Проверки индексации и поиска контекста без платных API-вызовов."""

import hashlib
import math
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import chromadb
import pytest

from mail_agent.context_store import (
    CHUNK_MAX_CHARACTERS,
    CHUNK_OVERLAP_CHARACTERS,
    CHUNKING_VERSION,
    CONTEXT_COLLECTION_NAME,
    ChromaContextStore,
)
from mail_agent.email_repository import SqliteEmailRepository


def create_test_embeddings(texts: Sequence[str]) -> list[list[float]]:
    """Строит детерминированные vectors для проверки pipeline."""
    embeddings: list[list[float]] = []
    for text in texts:
        embedding = [0.0] * 16
        tokens = re.findall(r"[\w-]+", text.casefold())
        for token in tokens:
            token_hash = hashlib.sha256(token.encode("utf-8")).digest()
            embedding[token_hash[0] % len(embedding)] += 1.0
        magnitude = math.sqrt(sum(value * value for value in embedding))
        if magnitude == 0:
            embedding[0] = 1.0
        else:
            embedding = [value / magnitude for value in embedding]
        embeddings.append(embedding)
    return embeddings


@pytest.fixture
def openai_client() -> Mock:
    """Возвращает mock клиента без сетевых запросов."""
    return Mock()


@pytest.fixture
def documents_path(tmp_path: Path) -> Path:
    """Копирует утверждённые документы во временный изменяемый каталог."""
    source_path = Path(__file__).parents[1] / "data" / "documents"
    destination_path = tmp_path / "documents"
    shutil.copytree(source_path, destination_path)
    return destination_path


@pytest.fixture
def context_store(
    tmp_path: Path,
    openai_client: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> ChromaContextStore:
    """Создаёт cosine-хранилище с детерминированными тестовыми embeddings."""
    context_store = ChromaContextStore(
        chroma_path=tmp_path / "chroma",
        openai_client=openai_client,
        embedding_model="text-embedding-3-small",
    )
    monkeypatch.setattr(
        context_store,
        "_create_embeddings",
        Mock(side_effect=create_test_embeddings),
    )
    return context_store


def test_index_emails_stores_required_metadata(
    context_store: ChromaContextStore,
    email_repository: SqliteEmailRepository,
) -> None:
    """Проверяет ID, тип, дату и thread_id индексированных писем."""
    indexed_count = context_store.index_emails(
        email_repository.list_email_messages()
    )
    records = context_store._context_collection.get(
        where={"source_type": {"$eq": "email"}},
        include=["metadatas"],
    )

    assert indexed_count == 7
    assert len(records["ids"]) == 7
    assert all(
        record_id.startswith("email:email-") for record_id in records["ids"]
    )
    assert (
        context_store._context_collection.configuration["hnsw"]["space"]
        == "cosine"
    )
    assert (
        context_store._context_collection.configuration["embedding_function"]
        is None
    )
    assert records["metadatas"] is not None
    assert all(
        metadata is not None
        and metadata["source_type"] == "email"
        and str(metadata["date"]).endswith("Z")
        and str(metadata["thread_id"]).startswith("thread-")
        for metadata in records["metadatas"]
    )


def test_index_documents_stores_required_metadata(
    context_store: ChromaContextStore,
    documents_path: Path,
) -> None:
    """Проверяет metadata документов и отсутствие фиктивного thread_id."""
    indexed_count = context_store.index_documents(documents_path)
    records = context_store._context_collection.get(
        where={"source_type": {"$eq": "document"}},
        include=["metadatas"],
    )

    assert indexed_count == 4
    assert records["metadatas"] is not None
    assert {
        str(metadata["source_id"])
        for metadata in records["metadatas"]
        if metadata is not None
    } == {
        "product_catalog.md",
        "delivery_terms.md",
        "payment_terms.md",
        "client_communication.md",
    }
    assert all(
        metadata is not None and "thread_id" not in metadata
        for metadata in records["metadatas"]
    )


def test_list_indexed_source_ids_returns_unique_sources(
    context_store: ChromaContextStore,
    documents_path: Path,
    email_repository: SqliteEmailRepository,
) -> None:
    """Возвращает каждый ID один раз при нескольких chunks одного источника."""
    context_store.index_emails(email_repository.list_email_messages())
    document_path = documents_path / "product_catalog.md"
    document_path.write_text(
        document_path.read_text(encoding="utf-8")
        + "\n\n"
        + "Дополнение. " * 500,
        encoding="utf-8",
    )
    context_store.index_documents(documents_path)

    source_ids = context_store.list_indexed_source_ids()

    assert source_ids == {
        *(f"email-{message_number:03d}" for message_number in range(1, 8)),
        "product_catalog.md",
        "delivery_terms.md",
        "payment_terms.md",
        "client_communication.md",
    }
    assert len(source_ids) == 11


def test_missing_document_does_not_change_collection(
    context_store: ChromaContextStore,
    documents_path: Path,
) -> None:
    """Проверяет все файлы до индексации и не оставляет частичных данных."""
    (documents_path / "client_communication.md").unlink()

    with pytest.raises(FileNotFoundError, match="client_communication.md"):
        context_store.index_documents(documents_path)

    assert context_store._context_collection.count() == 0
    context_store._create_embeddings.assert_not_called()


def test_indexing_unchanged_sources_skips_embeddings(
    context_store: ChromaContextStore,
    email_repository: SqliteEmailRepository,
) -> None:
    """Проверяет отсутствие API-работы при совпадающих SHA-256 и chunk ID."""
    email_messages = email_repository.list_email_messages()
    context_store.index_emails(email_messages)
    embedding_mock = context_store._create_embeddings
    first_call_count = embedding_mock.call_count

    indexed_count = context_store.index_emails(email_messages)

    assert indexed_count == 0
    assert embedding_mock.call_count == first_call_count


def test_reindexing_changed_source_replaces_chunks(
    context_store: ChromaContextStore,
    documents_path: Path,
) -> None:
    """Проверяет замену изменённого документа без повторяющихся chunk ID."""
    context_store.index_documents(documents_path)
    document_path = documents_path / "product_catalog.md"
    original_text = document_path.read_text(encoding="utf-8")
    document_path.write_text(
        f"{original_text}\n\n## Новая позиция\n\n{'Описание. ' * 500}",
        encoding="utf-8",
    )

    indexed_count = context_store.index_documents(documents_path)
    records = context_store._context_collection.get(
        where={"source_id": {"$eq": "product_catalog.md"}},
        include=["metadatas"],
    )

    assert indexed_count == 1
    assert len(records["ids"]) == len(set(records["ids"]))
    assert len(records["ids"]) > 1


def test_partial_email_indexing_does_not_delete_other_messages(
    context_store: ChromaContextStore,
    email_repository: SqliteEmailRepository,
) -> None:
    """Проверяет безопасность повторной индексации неполного набора писем."""
    email_messages = email_repository.list_email_messages()
    context_store.index_emails(email_messages)

    context_store.index_emails(email_messages[:1])
    records = context_store._context_collection.get(
        where={"source_type": {"$eq": "email"}},
        include=[],
    )

    assert len(records["ids"]) == 7


def test_split_text_preserves_context_and_limit(
    context_store: ChromaContextStore,
) -> None:
    """Проверяет повторение заголовка и полный лимит длины каждого chunk."""
    text = "\n\n".join(
        f"Абзац {index}. " + "данные " * 120 for index in range(8)
    )

    chunks = context_store._split_text(
        text, "Документ: test.md\nРаздел: Проверка"
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= CHUNK_MAX_CHARACTERS for chunk in chunks)
    assert all(chunk.startswith("Документ: test.md") for chunk in chunks)


def test_build_context_query_uses_user_request_and_thread(
    context_store: ChromaContextStore,
    email_repository: SqliteEmailRepository,
) -> None:
    """Проверяет запрос, тему и последние письма цепочки."""
    email_thread = email_repository.read_thread("thread-001")

    query_text = context_store.build_context_query(
        "Когда возможна поставка?",
        email_thread,
    )

    assert "Когда возможна поставка?" in query_text
    assert email_thread.subject in query_text
    assert email_thread.messages[-1].sender in query_text


def test_search_context_returns_top_k_and_source_ids(
    context_store: ChromaContextStore,
    email_repository: SqliteEmailRepository,
    documents_path: Path,
) -> None:
    """Ограничивает письма цепочкой и сохраняет общий top-k с документами."""
    context_store.index_emails(email_repository.list_email_messages())
    context_store.index_documents(documents_path)
    email_thread = email_repository.read_thread("thread-003")

    search_results = context_store.search_context(
        user_query="Какие условия оплаты насосов?",
        email_thread=email_thread,
        top_k=3,
    )

    assert 0 < len(search_results.matches) <= 3
    assert len({match.chunk_id for match in search_results.matches}) == len(
        search_results.matches
    )
    assert search_results.source_ids == list(
        dict.fromkeys(match.source_id for match in search_results.matches)
    )
    assert all(
        match.source_type == "document" or match.thread_id == "thread-003"
        for match in search_results.matches
    )
    assert "email-003" not in search_results.source_ids

    context_store._max_distance = 2.0
    all_allowed_results = context_store.search_context(
        user_query="Какие условия оплаты насосов?",
        email_thread=email_thread,
        top_k=100,
    )

    assert {"email-006", "email-007"}.issubset(all_allowed_results.source_ids)
    assert {
        "product_catalog.md",
        "delivery_terms.md",
        "payment_terms.md",
        "client_communication.md",
    }.issubset(all_allowed_results.source_ids)
    assert all(
        match.source_type == "document" or match.thread_id == "thread-003"
        for match in all_allowed_results.matches
    )


def test_search_context_returns_empty_results_beyond_distance(
    context_store: ChromaContextStore,
    email_repository: SqliteEmailRepository,
) -> None:
    """Проверяет пустой результат, когда все distances превышают порог."""
    context_store.index_emails(email_repository.list_email_messages())
    context_store._max_distance = -1.0

    search_results = context_store.search_context(
        user_query="Любой запрос",
        email_thread=email_repository.read_thread("thread-001"),
    )

    assert search_results.matches == []
    assert search_results.source_ids == []


def test_create_embeddings_uses_configured_model(
    tmp_path: Path,
    openai_client: Mock,
) -> None:
    """Проверяет вызов API и порядок vectors по index."""
    openai_client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[0.0, 1.0]),
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
        ]
    )
    context_store = ChromaContextStore(
        chroma_path=tmp_path / "api-chroma",
        openai_client=openai_client,
        embedding_model="text-embedding-3-small",
    )

    embeddings = context_store._create_embeddings(["первый", "второй"])

    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]
    openai_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["первый", "второй"],
        encoding_format="float",
    )


def test_create_embeddings_rejects_wrong_vector_count(
    tmp_path: Path,
    openai_client: Mock,
) -> None:
    """Отклоняет ответ OpenAI с неполным количеством embedding-векторов."""
    openai_client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])]
    )
    context_store = ChromaContextStore(
        chroma_path=tmp_path / "wrong-count-chroma",
        openai_client=openai_client,
        embedding_model="text-embedding-3-small",
    )

    with pytest.raises(RuntimeError, match="количество embeddings"):
        context_store._create_embeddings(["первый", "второй"])


def test_create_embeddings_rejects_duplicate_indices(
    tmp_path: Path,
    openai_client: Mock,
) -> None:
    """Отклоняет ответ OpenAI с повторяющимися индексами векторов."""
    openai_client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            SimpleNamespace(index=0, embedding=[0.0, 1.0]),
        ]
    )
    context_store = ChromaContextStore(
        chroma_path=tmp_path / "duplicate-indices-chroma",
        openai_client=openai_client,
        embedding_model="text-embedding-3-small",
    )

    with pytest.raises(RuntimeError, match="повторяющиеся индексы"):
        context_store._create_embeddings(["первый", "второй"])


def test_create_embeddings_rejects_incomplete_index_range(
    tmp_path: Path,
    openai_client: Mock,
) -> None:
    """Отклоняет уникальные индексы вне полного диапазона от нуля до N-1."""
    openai_client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            SimpleNamespace(index=2, embedding=[0.0, 1.0]),
        ]
    )
    context_store = ChromaContextStore(
        chroma_path=tmp_path / "incomplete-indices-chroma",
        openai_client=openai_client,
        embedding_model="text-embedding-3-small",
    )

    with pytest.raises(RuntimeError, match="диапазон от 0 до N-1"):
        context_store._create_embeddings(["первый", "второй"])


def test_existing_incompatible_collection_raises_error(
    tmp_path: Path,
    openai_client: Mock,
) -> None:
    """Проверяет явную диагностику коллекции с несовместимой метрикой."""
    chroma_path = tmp_path / "incompatible-chroma"
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    chroma_client.create_collection(
        name=CONTEXT_COLLECTION_NAME,
        configuration={"hnsw": {"space": "l2"}},
        metadata={
            "embedding_model": "text-embedding-3-small",
            "chunking_version": CHUNKING_VERSION,
            "chunk_max_characters": CHUNK_MAX_CHARACTERS,
            "chunk_overlap_characters": CHUNK_OVERLAP_CHARACTERS,
        },
        embedding_function=None,
    )

    with pytest.raises(RuntimeError, match="несовместима"):
        ChromaContextStore(
            chroma_path=chroma_path,
            openai_client=openai_client,
            embedding_model="text-embedding-3-small",
        )

    embedding_path = tmp_path / "incompatible-embedding-chroma"
    embedding_client = chromadb.PersistentClient(path=embedding_path)
    embedding_client.create_collection(
        name=CONTEXT_COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={
            "embedding_model": "another-embedding-model",
            "chunking_version": CHUNKING_VERSION,
            "chunk_max_characters": CHUNK_MAX_CHARACTERS,
            "chunk_overlap_characters": CHUNK_OVERLAP_CHARACTERS,
        },
        embedding_function=None,
    )

    with pytest.raises(RuntimeError, match="несовместима"):
        ChromaContextStore(
            chroma_path=embedding_path,
            openai_client=openai_client,
            embedding_model="text-embedding-3-small",
        )


def test_recreate_collection_replaces_incompatible_configuration(
    tmp_path: Path,
    openai_client: Mock,
) -> None:
    """Проверяет явную полную переиндексацию через пересоздание коллекции."""
    chroma_path = tmp_path / "recreated-chroma"
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    chroma_client.create_collection(
        name=CONTEXT_COLLECTION_NAME,
        configuration={"hnsw": {"space": "l2"}},
        embedding_function=None,
    )

    context_store = ChromaContextStore(
        chroma_path=chroma_path,
        openai_client=openai_client,
        embedding_model="text-embedding-3-small",
        recreate_collection=True,
    )

    assert (
        context_store._context_collection.configuration["hnsw"]["space"]
        == "cosine"
    )
