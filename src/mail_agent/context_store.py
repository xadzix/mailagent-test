"""Локальная индексация и семантический поиск почтового контекста."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date
from pathlib import Path
from typing import Literal

import chromadb
from openai import OpenAI
from pydantic import BaseModel, Field

from mail_agent.email_models import EmailMessage, EmailThread

CONTEXT_COLLECTION_NAME = "mail_agent_context"
CHUNK_MAX_CHARACTERS = 2000
CHUNK_OVERLAP_CHARACTERS = 200
CHUNKING_VERSION = 1
DEFAULT_TOP_K = 5
MAX_CONTEXT_DISTANCE = 0.75
SEARCH_QUERY_MESSAGE_LIMIT = 3
SEARCH_QUERY_BODY_LENGTH = 500
DOCUMENT_FILE_NAMES = (
    "product_catalog.md",
    "delivery_terms.md",
    "payment_terms.md",
    "client_communication.md",
)

SourceType = Literal["email", "document"]


class ContextChunk(BaseModel):
    """Описывает проверяемый фрагмент письма или справочного документа."""

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_type: SourceType
    date: str = Field(min_length=1)
    thread_id: str | None = None
    text: str = Field(min_length=1)
    source_hash: str = Field(min_length=64, max_length=64)
    chunk_index: int = Field(ge=0)


class ContextSearchResult(BaseModel):
    """Содержит найденный фрагмент и его косинусное расстояние до запроса."""

    chunk_id: str
    source_id: str
    source_type: SourceType
    date: str
    thread_id: str | None = None
    text: str
    distance: float


class ContextSearchResults(BaseModel):
    """Объединяет релевантные фрагменты и проверенные ID их источников."""

    matches: list[ContextSearchResult]
    source_ids: list[str]


class ChromaContextStore:
    """Индексирует контекст в ChromaDB с embeddings от OpenAI API."""

    def __init__(
        self,
        chroma_path: Path,
        openai_client: OpenAI,
        embedding_model: str,
        max_distance: float = MAX_CONTEXT_DISTANCE,
        recreate_collection: bool = False,
    ) -> None:
        """Открывает cosine-коллекцию или явно создаёт её заново."""
        self._chroma_client = chromadb.PersistentClient(path=chroma_path)
        self._openai_client = openai_client
        self._embedding_model = embedding_model
        self._max_distance = max_distance

        collection_names = {
            collection.name
            for collection in self._chroma_client.list_collections()
        }
        collection_exists = CONTEXT_COLLECTION_NAME in collection_names
        if collection_exists and recreate_collection:
            self._chroma_client.delete_collection(CONTEXT_COLLECTION_NAME)
            collection_exists = False

        expected_metadata = {
            "embedding_model": embedding_model,
            "chunking_version": CHUNKING_VERSION,
            "chunk_max_characters": CHUNK_MAX_CHARACTERS,
            "chunk_overlap_characters": CHUNK_OVERLAP_CHARACTERS,
        }
        if collection_exists:
            self._context_collection = self._chroma_client.get_collection(
                CONTEXT_COLLECTION_NAME,
                embedding_function=None,
            )
            configuration = self._context_collection.configuration
            hnsw_configuration = configuration.get("hnsw") or {}
            collection_metadata = self._context_collection.metadata or {}
            metadata_matches = all(
                collection_metadata.get(key) == value
                for key, value in expected_metadata.items()
            )
            if (
                hnsw_configuration.get("space") != "cosine"
                or configuration.get("embedding_function") is not None
                or not metadata_matches
            ):
                raise RuntimeError(
                    "Коллекция mail_agent_context несовместима с текущей "
                    "embedding-моделью или алгоритмом chunking. Создайте "
                    "ChromaContextStore с recreate_collection=True для полной "
                    "переиндексации."
                )
        else:
            self._context_collection = self._chroma_client.create_collection(
                name=CONTEXT_COLLECTION_NAME,
                configuration={"hnsw": {"space": "cosine"}},
                metadata=expected_metadata,
                embedding_function=None,
            )

    def index_emails(self, email_messages: Sequence[EmailMessage]) -> int:
        """Индексирует письма без удаления отсутствующих записей."""
        indexed_source_count = 0
        for email_message in email_messages:
            chunks = self._build_email_chunks(email_message)
            if self._index_source(chunks):
                indexed_source_count += 1
        return indexed_source_count

    def index_documents(self, documents_path: Path) -> int:
        """Индексирует четыре документа и требует наличия каждого."""
        document_paths = [
            documents_path / document_file_name
            for document_file_name in DOCUMENT_FILE_NAMES
        ]
        missing_document_paths = [
            document_path
            for document_path in document_paths
            if not document_path.is_file()
        ]
        if missing_document_paths:
            missing_documents = ", ".join(
                str(document_path) for document_path in missing_document_paths
            )
            raise FileNotFoundError(
                "Не найдены обязательные справочные документы: "
                f"{missing_documents}"
            )

        indexed_source_count = 0
        for document_path in document_paths:
            chunks = self._build_document_chunks(document_path)
            if self._index_source(chunks):
                indexed_source_count += 1
        return indexed_source_count

    def list_indexed_source_ids(self) -> set[str]:
        """Возвращает уникальные ID источников без изменения индекса."""
        records = self._context_collection.get(include=["metadatas"])
        metadatas = records.get("metadatas") or []
        return {
            str(metadata["source_id"])
            for metadata in metadatas
            if metadata is not None and "source_id" in metadata
        }

    def build_context_query(
        self,
        user_query: str,
        email_thread: EmailThread,
    ) -> str:
        """Дополняет запрос темой и последними письмами цепочки."""
        normalized_user_query = " ".join(user_query.split())
        if not normalized_user_query:
            raise ValueError("Поисковый запрос не может быть пустым")

        query_parts = [
            f"Запрос пользователя: {normalized_user_query}",
            f"Тема переписки: {email_thread.subject}",
        ]
        for message in email_thread.messages[-SEARCH_QUERY_MESSAGE_LIMIT:]:
            normalized_body = " ".join(message.body.split())
            recipients = ", ".join(message.recipients)
            query_parts.append(
                "\n".join(
                    (
                        f"Отправитель: {message.sender}",
                        f"Получатели: {recipients}",
                        "Сообщение: "
                        f"{normalized_body[:SEARCH_QUERY_BODY_LENGTH]}",
                    )
                )
            )
        return "\n\n".join(query_parts)

    def search_context(
        self,
        user_query: str,
        email_thread: EmailThread,
        top_k: int = DEFAULT_TOP_K,
    ) -> ContextSearchResults:
        """Ищет по письмам выбранной цепочки и всем общим документам."""
        if top_k < 1:
            raise ValueError(
                "Количество результатов должно быть положительным"
            )

        query_text = self.build_context_query(user_query, email_thread)
        metadata_filter: Mapping[str, object] = {
            "$or": [
                {"source_type": {"$eq": "document"}},
                {
                    "$and": [
                        {"source_type": {"$eq": "email"}},
                        {"thread_id": {"$eq": email_thread.thread_id}},
                    ]
                },
            ]
        }
        matches = self._query_context(
            query_text=query_text,
            result_limit=top_k,
            metadata_filter=metadata_filter,
        )

        unique_matches: list[ContextSearchResult] = []
        seen_chunk_ids: set[str] = set()
        for match in matches:
            if match.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(match.chunk_id)
            unique_matches.append(match)

        source_ids = list(
            dict.fromkeys(match.source_id for match in unique_matches)
        )
        return ContextSearchResults(
            matches=unique_matches,
            source_ids=source_ids,
        )

    def _build_email_chunks(
        self, email_message: EmailMessage
    ) -> list[ContextChunk]:
        """Сохраняет письмо целиком и делит только длинное тело."""
        sent_at = email_message.sent_at.astimezone(UTC).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        recipients = ", ".join(email_message.recipients)
        context_prefix = "\n".join(
            (
                f"Тема: {email_message.subject}",
                f"Отправитель: {email_message.sender}",
                f"Получатели: {recipients}",
                f"Дата: {sent_at}",
            )
        )
        canonical_source = json.dumps(
            email_message.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        source_hash = hashlib.sha256(
            canonical_source.encode("utf-8")
        ).hexdigest()
        chunk_texts = self._split_text(email_message.body, context_prefix)
        return [
            ContextChunk(
                chunk_id=f"email:{email_message.message_id}:{chunk_index:04d}",
                source_id=email_message.message_id,
                source_type="email",
                date=sent_at,
                thread_id=email_message.thread_id,
                text=chunk_text,
                source_hash=source_hash,
                chunk_index=chunk_index,
            )
            for chunk_index, chunk_text in enumerate(chunk_texts)
        ]

    def _build_document_chunks(
        self, document_path: Path
    ) -> list[ContextChunk]:
        """Читает дату редакции и повторяет путь Markdown-раздела в chunks."""
        raw_text = document_path.read_text(encoding="utf-8")
        lines = raw_text.splitlines()
        if not lines or lines[0].strip() != "---":
            raise ValueError(
                f"Документ не содержит front matter: {document_path}"
            )
        try:
            front_matter_end = lines.index("---", 1)
        except ValueError as error:
            raise ValueError(
                f"Документ содержит незакрытый front matter: {document_path}"
            ) from error

        front_matter = lines[1:front_matter_end]
        date_values = [
            line.partition(":")[2].strip()
            for line in front_matter
            if line.partition(":")[0].strip() == "date"
        ]
        if len(date_values) != 1:
            raise ValueError(
                f"Документ должен содержать одну дату: {document_path}"
            )
        document_date = date_values[0]
        try:
            date.fromisoformat(document_date)
        except ValueError as error:
            raise ValueError(
                f"Документ содержит некорректную дату: {document_path}"
            ) from error

        document_lines = lines[front_matter_end + 1 :]
        document_text = "\n".join(document_lines).strip()
        if not document_text:
            raise ValueError(f"Документ не содержит текста: {document_path}")

        base_prefix = "\n".join(
            (
                f"Документ: {document_path.name}",
                f"Дата редакции: {document_date}",
            )
        )
        section_chunks: list[str] = []
        heading_path: list[str] = []
        section_lines: list[str] = []

        def append_section() -> None:
            """Добавляет раздел с полным путём заголовков."""
            section_text = "\n".join(section_lines).strip()
            if not section_text:
                return
            context_prefix = base_prefix
            if heading_path:
                context_prefix += f"\nРаздел: {' > '.join(heading_path)}"
            section_chunks.extend(
                self._split_text(section_text, context_prefix)
            )

        for line in document_lines:
            heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if heading_match is None:
                section_lines.append(line)
                continue

            append_section()
            section_lines.clear()
            heading_level = len(heading_match.group(1))
            heading_path = heading_path[: heading_level - 1]
            heading_path.append(heading_match.group(2))
        append_section()

        if not section_chunks:
            section_chunks = self._split_text(document_text, base_prefix)

        canonical_source = "\0".join(
            ("document", document_path.name, document_date, document_text)
        )
        source_hash = hashlib.sha256(
            canonical_source.encode("utf-8")
        ).hexdigest()
        return [
            ContextChunk(
                chunk_id=f"document:{document_path.name}:{chunk_index:04d}",
                source_id=document_path.name,
                source_type="document",
                date=document_date,
                text=chunk_text,
                source_hash=source_hash,
                chunk_index=chunk_index,
            )
            for chunk_index, chunk_text in enumerate(section_chunks)
        ]

    def _split_text(self, text: str, context_prefix: str) -> list[str]:
        """Делит текст по абзацам и предложениям с ограниченным перекрытием."""
        normalized_prefix = "\n".join(
            " ".join(line.split())
            for line in context_prefix.splitlines()
            if line.strip()
        )
        if len(normalized_prefix) > CHUNK_MAX_CHARACTERS // 2:
            normalized_prefix = normalized_prefix[: CHUNK_MAX_CHARACTERS // 2]

        separator = "\n\n" if normalized_prefix else ""
        body_limit = (
            CHUNK_MAX_CHARACTERS - len(normalized_prefix) - len(separator)
        )
        normalized_paragraphs = [
            " ".join(paragraph.split())
            for paragraph in re.split(r"\n\s*\n+", text.strip())
            if paragraph.strip()
        ]
        semantic_units: list[str] = []
        for paragraph in normalized_paragraphs:
            if len(paragraph) <= body_limit:
                semantic_units.append(paragraph)
                continue

            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
                if sentence.strip()
            ]
            for sentence in sentences:
                if len(sentence) <= body_limit:
                    semantic_units.append(sentence)
                    continue
                step = max(1, body_limit - CHUNK_OVERLAP_CHARACTERS)
                semantic_units.extend(
                    sentence[offset : offset + body_limit]
                    for offset in range(0, len(sentence), step)
                )

        if not semantic_units and normalized_prefix:
            return [normalized_prefix]

        body_chunks: list[str] = []
        current_body = ""
        for semantic_unit in semantic_units:
            candidate = (
                f"{current_body}\n\n{semantic_unit}"
                if current_body
                else semantic_unit
            )
            if len(candidate) <= body_limit:
                current_body = candidate
                continue

            if current_body:
                body_chunks.append(current_body)
            overlap_length = min(
                CHUNK_OVERLAP_CHARACTERS,
                max(0, body_limit - len(semantic_unit) - 2),
            )
            overlap = current_body[-overlap_length:] if overlap_length else ""
            current_body = (
                f"{overlap}\n\n{semantic_unit}" if overlap else semantic_unit
            )
        if current_body:
            body_chunks.append(current_body)

        return [
            f"{normalized_prefix}{separator}{body_chunk}"
            for body_chunk in body_chunks
        ]

    def _index_source(self, chunks: Sequence[ContextChunk]) -> bool:
        """Пропускает источник без изменений или обновляет chunks."""
        if not chunks:
            return False

        first_chunk = chunks[0]
        source_filter = {
            "$and": [
                {"source_id": {"$eq": first_chunk.source_id}},
                {"source_type": {"$eq": first_chunk.source_type}},
            ]
        }
        existing_records = self._context_collection.get(
            where=source_filter,
            include=["metadatas"],
        )
        existing_ids = set(existing_records["ids"])
        expected_ids = {chunk.chunk_id for chunk in chunks}
        existing_metadatas = existing_records.get("metadatas") or []
        source_is_current = (
            existing_ids == expected_ids
            and bool(existing_metadatas)
            and all(
                metadata is not None
                and metadata.get("source_hash") == first_chunk.source_hash
                for metadata in existing_metadatas
            )
        )
        if source_is_current:
            return False

        embeddings = self._create_embeddings([chunk.text for chunk in chunks])
        metadatas: list[dict[str, str | int]] = []
        for chunk in chunks:
            metadata: dict[str, str | int] = {
                "source_id": chunk.source_id,
                "source_type": chunk.source_type,
                "date": chunk.date,
                "source_hash": chunk.source_hash,
                "chunk_index": chunk.chunk_index,
            }
            if chunk.thread_id is not None:
                metadata["thread_id"] = chunk.thread_id
            metadatas.append(metadata)

        self._context_collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
            documents=[chunk.text for chunk in chunks],
        )
        stale_ids = sorted(existing_ids - expected_ids)
        if stale_ids:
            self._context_collection.delete(ids=stale_ids)
        return True

    def _create_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """Получает embeddings и восстанавливает порядок текстов."""
        if not texts:
            return []
        response = self._openai_client.embeddings.create(
            model=self._embedding_model,
            input=list(texts),
            encoding_format="float",
        )
        if len(response.data) != len(texts):
            raise RuntimeError(
                "OpenAI вернул количество embeddings, не совпадающее "
                "с количеством входных текстов"
            )
        response_indices = [item.index for item in response.data]
        if len(response_indices) != len(set(response_indices)):
            raise RuntimeError(
                "OpenAI вернул повторяющиеся индексы embeddings"
            )
        expected_indices = set(range(len(texts)))
        if set(response_indices) != expected_indices:
            raise RuntimeError(
                "Индексы embeddings должны полностью покрывать диапазон "
                "от 0 до N-1"
            )
        ordered_embeddings = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered_embeddings]

    def _query_context(
        self,
        query_text: str,
        result_limit: int,
        metadata_filter: Mapping[str, object] | None,
    ) -> list[ContextSearchResult]:
        """Преобразует один cosine-запрос ChromaDB в проверенные результаты."""
        collection_count = self._context_collection.count()
        if collection_count == 0:
            return []

        query_embedding = self._create_embeddings([query_text])[0]
        query_result = self._context_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(result_limit, collection_count),
            where=metadata_filter,
            include=["documents", "metadatas", "distances"],
        )
        ids = query_result.get("ids") or [[]]
        documents = query_result.get("documents") or [[]]
        metadatas = query_result.get("metadatas") or [[]]
        distances = query_result.get("distances") or [[]]

        search_results: list[ContextSearchResult] = []
        for chunk_id, document, metadata, distance_value in zip(
            ids[0],
            documents[0],
            metadatas[0],
            distances[0],
            strict=True,
        ):
            if (
                document is None
                or metadata is None
                or distance_value > self._max_distance
            ):
                continue
            source_type = metadata.get("source_type")
            if source_type not in ("email", "document"):
                raise RuntimeError(
                    f"Некорректный source_type у chunk: {chunk_id}"
                )
            search_results.append(
                ContextSearchResult(
                    chunk_id=chunk_id,
                    source_id=str(metadata["source_id"]),
                    source_type=source_type,
                    date=str(metadata["date"]),
                    thread_id=(
                        str(metadata["thread_id"])
                        if "thread_id" in metadata
                        else None
                    ),
                    text=document,
                    distance=float(distance_value),
                )
            )
        return search_results
