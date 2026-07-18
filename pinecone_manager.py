"""
Модуль для работы с векторной базой данных Pinecone.

Класс PineconeManager инкапсулирует операции записи и чтения:
векторов, документов и текстовых запросов, а также управление
долговременной памятью чат-бота с проверкой дубликатов.
"""

from __future__ import annotations

import math
import os
import uuid
from typing import Any, Literal, Mapping, Sequence

from dotenv import load_dotenv
import httpx
from openai import OpenAI
from pinecone import Pinecone

# ---------------------------------------------------------------------------
# Настройка порога косинусного сходства для долговременной памяти чат-бота.
# ---------------------------------------------------------------------------
# score >= COSINE_SIMILARITY_THRESHOLD — высокое сходство (дубликат/вариация):
#   запись пропускается или обновляется существующий слот памяти.
# score < COSINE_SIMILARITY_THRESHOLD — низкое сходство (новая информация):
#   сообщение сохраняется как новый фрагмент памяти.
COSINE_SIMILARITY_THRESHOLD = 0.85

# Префикс namespace для изоляции памяти каждого пользователя Telegram.
# Формат: user_{telegram_id}. Общий/default namespace для диалогов не используется.
USER_NAMESPACE_PREFIX = "user_"
FORBIDDEN_SHARED_NAMESPACES = frozenset({"", "default", "shared", "common"})

MemoryDuplicateAction = Literal["skip", "update"]
MemoryAction = Literal["saved", "skipped", "updated"]


class PineconeManager:
    """
    Менеджер для чтения и записи данных в Pinecone.

    Поддерживает:
    - запись готовых векторов и документов (с эмбеддингом на клиенте);
    - запись записей для индексов с серверным эмбеддингом;
    - поиск по вектору и по тексту;
    - получение записей по идентификатору и метаданным;
    - сохранение в долговременную память с фильтрацией дубликатов.
    """

    def __init__(
        self,
        namespace: str = "",
        load_env: bool = True,
    ) -> None:
        """
        Инициализирует подключение к Pinecone и OpenAI.

        Args:
            namespace: Пространство имён по умолчанию для операций.
            load_env: Загружать ли переменные из файла .env.
        """
        if load_env:
            load_dotenv()

        self._pinecone_api_key = self._require_env("PINECONE_API_KEY")
        self._index_name = self._require_env("PINECONE_INDEX_NAME")
        self._openai_api_key = self._require_env("OPENAI_API_KEY")
        self.base_url = self._require_env("OPENAI_BASE_URL")
        self._embedding_model = os.getenv(
            "EMBEDDING_MODEL",
            "text-embedding-3-small",
        )

        self._namespace = namespace

        # Клиент Pinecone для работы с индексом.
        self._pc = Pinecone(api_key=self._pinecone_api_key)
        self._index = self._pc.index(self._index_name)

        # Клиент OpenAI с base_url из переменной окружения.
        self._openai = OpenAI(
            api_key=self._openai_api_key,
            base_url=self.base_url,
            http_client=httpx.Client(trust_env=False),
        )

    @property
    def index_name(self) -> str:
        """Имя подключённого индекса Pinecone."""
        return self._index_name

    @property
    def namespace(self) -> str:
        """Пространство имён, используемое по умолчанию."""
        return self._namespace

    @staticmethod
    def _require_env(name: str) -> str:
        """
        Возвращает значение обязательной переменной окружения.

        Raises:
            ValueError: Если переменная не задана или пуста.
        """
        value = os.getenv(name, "").strip()
        if not value:
            raise ValueError(
                f"Переменная окружения {name} не задана. "
                f"Добавьте её в файл .env.",
            )
        return value

    def _resolve_namespace(self, namespace: str | None) -> str:
        """Возвращает namespace из аргумента или значение по умолчанию."""
        if namespace is None:
            return self._namespace
        return namespace

    @staticmethod
    def build_user_namespace(telegram_id: str | int) -> str:
        """
        Формирует персональный namespace пользователя Telegram.

        Пример: telegram_id=12345 → "user_12345".
        Каждый пользователь работает только в своём namespace —
        чужая память недоступна.

        Args:
            telegram_id: Идентификатор пользователя в Telegram.

        Returns:
            Имя namespace вида user_{telegram_id}.

        Raises:
            ValueError: Если telegram_id пустой или недопустимый.
        """
        if telegram_id is None:
            raise ValueError(
                "telegram_id обязателен: каждый пользователь работает "
                "в своём namespace Pinecone.",
            )
        user_id = str(telegram_id).strip()
        if not user_id:
            raise ValueError("telegram_id не может быть пустым.")
        if user_id.lower() in FORBIDDEN_SHARED_NAMESPACES:
            raise ValueError(
                f"Недопустимый telegram_id={user_id!r}: "
                "это имя общего namespace.",
            )
        namespace = f"{USER_NAMESPACE_PREFIX}{user_id}"
        if namespace.lower() in FORBIDDEN_SHARED_NAMESPACES:
            raise ValueError(f"Запрещён общий namespace: {namespace}")
        return namespace

    def _resolve_user_namespace(
        self,
        telegram_id: str | int | None = None,
        namespace: str | None = None,
    ) -> str:
        """
        Определяет namespace для операций с памятью пользователя.

        Приоритет:
        1. telegram_id → user_{telegram_id};
        2. явный namespace;
        3. namespace по умолчанию класса.

        Raises:
            ValueError: Если не удалось определить namespace.
        """
        if telegram_id is not None:
            return self.build_user_namespace(telegram_id)

        resolved = self._resolve_namespace(namespace)
        if not resolved:
            raise ValueError(
                "Укажите telegram_id или namespace для работы с памятью "
                "пользователя.",
            )
        return resolved

    @staticmethod
    def _enrich_user_metadata(
        metadata: Mapping[str, Any] | None,
        telegram_id: str | int,
    ) -> dict[str, Any]:
        """Добавляет telegram_id в метаданные фрагмента памяти."""
        result = dict(metadata or {})
        result["telegram_id"] = str(telegram_id)
        return result

    def create_embedding(self, text: str) -> list[float]:
        """
        Создаёт эмбеддинг для одного текста через OpenAI API.

        Использует base_url и модель эмбеддингов из переменных окружения.

        Args:
            text: Исходный текст.

        Returns:
            Вектор эмбеддинга.
        """
        response = self._openai.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def create_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Создаёт эмбеддинги для списка текстов одним запросом.

        Args:
            texts: Список текстов.

        Returns:
            Список векторов в том же порядке, что и входные тексты.
        """
        response = self._openai.embeddings.create(
            model=self._embedding_model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]

    def _get_embedding(self, text: str) -> list[float]:
        """Внутренний алиас для create_embedding."""
        return self.create_embedding(text)

    def _get_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """Внутренний алиас для create_embeddings."""
        return self.create_embeddings(texts)

    @staticmethod
    def cosine_similarity(
        vector_a: Sequence[float],
        vector_b: Sequence[float],
    ) -> float:
        """
        Вычисляет косинусное сходство между двумя векторами.

        Args:
            vector_a: Первый вектор.
            vector_b: Второй вектор.

        Returns:
            Значение от -1.0 до 1.0. Для нормализованных эмбеддингов OpenAI
            обычно находится в диапазоне 0.0–1.0.

        Raises:
            ValueError: Если векторы разной длины или нулевые.
        """
        if len(vector_a) != len(vector_b):
            raise ValueError("Векторы должны иметь одинаковую размерность.")

        dot_product = sum(a * b for a, b in zip(vector_a, vector_b, strict=True))
        norm_a = math.sqrt(sum(value * value for value in vector_a))
        norm_b = math.sqrt(sum(value * value for value in vector_b))

        if norm_a == 0 or norm_b == 0:
            raise ValueError("Нулевой вектор не поддерживается.")

        return dot_product / (norm_a * norm_b)

    def find_similar_memory(
        self,
        text: str,
        telegram_id: str | int | None = None,
        top_k: int = 1,
        namespace: str | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        text_field: str = "text",
        similarity_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Ищет наиболее похожий фрагмент памяти для текста сообщения.

        Сравнение выполняется только внутри namespace пользователя
        (по умолчанию user_{telegram_id}).

        Args:
            text: Текущее сообщение пользователя или факт для запоминания.
            telegram_id: ID пользователя Telegram. Рекомендуемый способ
                изоляции памяти.
            top_k: Сколько ближайших фрагментов вернуть.
            namespace: Явный namespace (если telegram_id не передан).
            metadata_filter: Дополнительный фильтр по метаданным.
            text_field: Поле metadata, где хранится исходный текст.
            similarity_threshold: Порог для определения дубликата. По умолчанию —
                COSINE_SIMILARITY_THRESHOLD.

        Returns:
            Словарь с полями:
            - namespace: namespace, в котором выполнялся поиск;
            - best_match: ближайший фрагмент или None;
            - similarity: косинусное сходство с ближайшим фрагментом или None;
            - is_duplicate: True, если сходство >= порога;
            - matches: список всех найденных совпадений.
        """
        user_namespace = self._resolve_user_namespace(
            telegram_id=telegram_id,
            namespace=namespace,
        )
        embedding = self.create_embedding(text)
        matches = self.query_by_vector(
            vector=embedding,
            top_k=top_k,
            namespace=user_namespace,
            metadata_filter=metadata_filter,
            include_metadata=True,
        )

        if not matches:
            return {
                "namespace": user_namespace,
                "best_match": None,
                "similarity": None,
                "is_duplicate": False,
                "matches": [],
            }

        best_match = matches[0]
        similarity = float(best_match["score"])
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else COSINE_SIMILARITY_THRESHOLD
        )

        return {
            "namespace": user_namespace,
            "best_match": best_match,
            "similarity": similarity,
            "is_duplicate": similarity >= threshold,
            "matches": matches,
            "threshold": threshold,
            "text_field": text_field,
        }

    def save_to_long_term_memory(
        self,
        text: str,
        telegram_id: str | int | None = None,
        memory_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        text_field: str = "text",
        on_duplicate: MemoryDuplicateAction = "update",
        similarity_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Сохраняет сообщение в долговременную память чат-бота.

        Память каждого пользователя хранится в отдельном namespace
        user_{telegram_id}. Сравнение на дубликаты выполняется только
        внутри памяти этого пользователя.

        Для изоляции пользователей передавайте telegram_id (рекомендуется).
        """
        if telegram_id is None and not namespace:
            raise ValueError(
                "Укажите telegram_id, чтобы сохранить память в личный "
                "namespace пользователя (user_{telegram_id}).",
            )
        user_namespace = self._resolve_user_namespace(
            telegram_id=telegram_id,
            namespace=namespace,
        )
        if not user_namespace.startswith(USER_NAMESPACE_PREFIX):
            raise ValueError(
                f"Namespace {user_namespace!r} не персональный: "
                f"ожидался префикс {USER_NAMESPACE_PREFIX!r}.",
            )
        if user_namespace.lower() in FORBIDDEN_SHARED_NAMESPACES:
            raise ValueError(
                f"Запрещено писать память пользователей в общий "
                f"namespace {user_namespace!r}.",
            )
        if telegram_id is not None:
            metadata = self._enrich_user_metadata(metadata, telegram_id)

        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else COSINE_SIMILARITY_THRESHOLD
        )
        similarity_info = self.find_similar_memory(
            text=text,
            telegram_id=telegram_id,
            top_k=1,
            namespace=user_namespace,
            metadata_filter=metadata_filter,
            text_field=text_field,
            similarity_threshold=threshold,
        )

        best_match = similarity_info["best_match"]
        similarity = similarity_info["similarity"]

        # Высокое сходство — дубликат или вариация уже известной информации.
        if best_match is not None and similarity is not None and similarity >= threshold:
            existing_id = str(best_match["id"])
            existing_metadata = dict(best_match.get("metadata") or {})

            if on_duplicate == "skip":
                return {
                    "action": "skipped",
                    "namespace": user_namespace,
                    "memory_id": existing_id,
                    "similarity": similarity,
                    "threshold": threshold,
                    "message": (
                        "Фрагмент похож на уже сохранённую память. "
                        "Запись пропущена."
                    ),
                    "existing_text": existing_metadata.get(text_field),
                }

            # Обновляем существующий слот: новый текст и эмбеддинг.
            updated_metadata = dict(existing_metadata)
            updated_metadata.update(metadata or {})
            updated_metadata[text_field] = text

            embedding = self.create_embedding(text)
            self.upsert_vector(
                vector_id=existing_id,
                values=embedding,
                metadata=updated_metadata,
                namespace=user_namespace,
            )

            return {
                "action": "updated",
                "namespace": user_namespace,
                "memory_id": existing_id,
                "similarity": similarity,
                "threshold": threshold,
                "message": (
                    "Найден похожий фрагмент памяти. "
                    "Существующий слот обновлён."
                ),
                "previous_text": existing_metadata.get(text_field),
                "updated_text": text,
            }

        # Низкое сходство — новая информация, сохраняем отдельным фрагментом.
        new_memory_id = memory_id or str(uuid.uuid4())
        doc_metadata = dict(metadata or {})
        doc_metadata[text_field] = text

        self.upsert_document(
            document_id=new_memory_id,
            text=text,
            metadata=doc_metadata,
            text_field=text_field,
            namespace=user_namespace,
        )

        return {
            "action": "saved",
            "namespace": user_namespace,
            "memory_id": new_memory_id,
            "similarity": similarity,
            "threshold": threshold,
            "message": "Новая информация сохранена в долговременную память.",
            "text": text,
        }

    def recall_user_memory(
        self,
        query_text: str,
        telegram_id: str | int,
        top_k: int = 5,
        text_field: str = "text",
    ) -> list[dict[str, Any]]:
        """
        Возвращает релевантные фрагменты памяти пользователя по запросу.

        Поиск выполняется только в namespace user_{telegram_id}.

        Args:
            query_text: Текст запроса (вопрос или контекст диалога).
            telegram_id: ID пользователя Telegram.
            top_k: Сколько фрагментов вернуть.
            text_field: Поле metadata с текстом фрагмента.

        Returns:
            Список найденных фрагментов памяти с score и metadata.
        """
        user_namespace = self.build_user_namespace(telegram_id)
        return self.query_by_text(
            text=query_text,
            top_k=top_k,
            namespace=user_namespace,
            include_metadata=True,
        )

    def clear_user_memory(self, telegram_id: str | int) -> None:
        """
        Полностью очищает долговременную память пользователя.

        Args:
            telegram_id: ID пользователя Telegram.
        """
        user_namespace = self.build_user_namespace(telegram_id)
        self.delete_all(namespace=user_namespace)

    # ------------------------------------------------------------------
    # Запись данных
    # ------------------------------------------------------------------

    def upsert_vector(
        self,
        vector_id: str,
        values: Sequence[float],
        metadata: Mapping[str, Any] | None = None,
        namespace: str | None = None,
    ) -> int:
        """
        Записывает один вектор в индекс.

        Args:
            vector_id: Уникальный идентификатор записи.
            values: Числовой вектор.
            metadata: Дополнительные метаданные.
            namespace: Пространство имён (если не указано — используется default).

        Returns:
            Количество успешно записанных векторов.
        """
        record: dict[str, Any] = {
            "id": vector_id,
            "values": list(values),
        }
        if metadata is not None:
            record["metadata"] = dict(metadata)

        response = self._index.upsert(
            vectors=[record],
            namespace=self._resolve_namespace(namespace),
        )
        return response.upserted_count

    def upsert_vectors(
        self,
        vectors: Sequence[Mapping[str, Any]],
        namespace: str | None = None,
        batch_size: int | None = None,
    ) -> int:
        """
        Записывает пакет готовых векторов.

        Каждый элемент должен содержать ключи:
        - id (str) — идентификатор;
        - values (list[float]) — вектор;
        - metadata (dict, опционально) — метаданные.

        Args:
            vectors: Список словарей с данными векторов.
            namespace: Пространство имён.
            batch_size: Размер пакета для отправки (None — одним запросом).

        Returns:
            Количество успешно записанных векторов.
        """
        response = self._index.upsert(
            vectors=list(vectors),
            namespace=self._resolve_namespace(namespace),
            batch_size=batch_size,
        )
        return response.upserted_count

    def upsert_document(
        self,
        document_id: str,
        text: str,
        metadata: Mapping[str, Any] | None = None,
        text_field: str = "text",
        namespace: str | None = None,
    ) -> int:
        """
        Записывает документ: текст преобразуется в вектор на клиенте.

        Текст сохраняется в metadata, чтобы его можно было вернуть при поиске.

        Args:
            document_id: Уникальный идентификатор документа.
            text: Текстовое содержимое.
            metadata: Дополнительные метаданные.
            text_field: Имя поля в metadata, куда сохраняется текст.
            namespace: Пространство имён.

        Returns:
            Количество успешно записанных векторов.
        """
        embedding = self._get_embedding(text)
        doc_metadata = dict(metadata or {})
        doc_metadata[text_field] = text

        return self.upsert_vector(
            vector_id=document_id,
            values=embedding,
            metadata=doc_metadata,
            namespace=namespace,
        )

    def upsert_documents(
        self,
        documents: Sequence[Mapping[str, Any]],
        text_field: str = "text",
        namespace: str | None = None,
        batch_size: int | None = 100,
    ) -> int:
        """
        Записывает пакет документов с клиентским эмбеддингом.

        Каждый документ должен содержать:
        - id (str) — идентификатор;
        - text (str) — текст для эмбеддинга;
        - metadata (dict, опционально) — дополнительные поля.

        Args:
            documents: Список документов.
            text_field: Имя поля в metadata для хранения текста.
            namespace: Пространство имён.
            batch_size: Размер пакета при upsert векторов.

        Returns:
            Количество успешно записанных векторов.
        """
        texts = [str(doc["text"]) for doc in documents]
        embeddings = self._get_embeddings(texts)

        vectors: list[dict[str, Any]] = []
        for doc, embedding in zip(documents, embeddings, strict=True):
            doc_metadata = dict(doc.get("metadata", {}))
            doc_metadata[text_field] = str(doc["text"])
            vectors.append(
                {
                    "id": str(doc["id"]),
                    "values": embedding,
                    "metadata": doc_metadata,
                },
            )

        return self.upsert_vectors(
            vectors=vectors,
            namespace=namespace,
            batch_size=batch_size,
        )

    def upsert_records(
        self,
        records: list[dict[str, Any]],
        namespace: str,
    ) -> int:
        """
        Записывает записи в индекс с серверным эмбеддингом (integrated inference).

        Используйте для индексов Pinecone, где эмбеддинг создаётся на стороне
        сервера. Каждая запись должна содержать поле _id или id.

        Args:
            records: Список записей для upsert.
            namespace: Пространство имён (обязательно, не может быть пустым).

        Returns:
            Количество отправленных записей.
        """
        response = self._index.upsert_records(
            records=records,
            namespace=namespace,
        )
        return response.record_count

    # ------------------------------------------------------------------
    # Чтение и поиск
    # ------------------------------------------------------------------

    def query_by_vector(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        namespace: str | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Ищет ближайшие записи по готовому вектору.

        Args:
            vector: Вектор запроса.
            top_k: Сколько результатов вернуть.
            namespace: Пространство имён.
            metadata_filter: Фильтр по метаданным Pinecone.
            include_metadata: Включать ли метаданные в ответ.
            include_values: Включать ли значения вектора в ответ.

        Returns:
            Список словарей с полями id, score, metadata, values.
        """
        response = self._index.query(
            vector=list(vector),
            top_k=top_k,
            namespace=self._resolve_namespace(namespace),
            filter=metadata_filter,
            include_metadata=include_metadata,
            include_values=include_values,
        )
        return self._format_query_matches(response.matches)

    def query_by_text(
        self,
        text: str,
        top_k: int = 5,
        namespace: str | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Ищет ближайшие записи по тексту.

        Текст сначала преобразуется в эмбеддинг через OpenAI (base_url из .env),
        затем выполняется векторный поиск.

        Args:
            text: Текстовый запрос.
            top_k: Сколько результатов вернуть.
            namespace: Пространство имён.
            metadata_filter: Фильтр по метаданным.
            include_metadata: Включать ли метаданные.
            include_values: Включать ли значения вектора.

        Returns:
            Список найденных записей.
        """
        embedding = self._get_embedding(text)
        return self.query_by_vector(
            vector=embedding,
            top_k=top_k,
            namespace=namespace,
            metadata_filter=metadata_filter,
            include_metadata=include_metadata,
            include_values=include_values,
        )

    def search_by_text(
        self,
        text: str,
        namespace: str,
        top_k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
        fields: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Ищет записи по тексту в индексе с серверным эмбеддингом.

        Использует метод search Pinecone — эмбеддинг создаётся на сервере.
        Подходит для индексов с integrated inference.

        Args:
            text: Текстовый запрос.
            namespace: Пространство имён (обязательно).
            top_k: Сколько результатов вернуть.
            metadata_filter: Фильтр по метаданным.
            fields: Какие поля включить в ответ.

        Returns:
            Список найденных записей.
        """
        response = self._index.search(
            namespace=namespace,
            top_k=top_k,
            inputs={"text": text},
            filter=metadata_filter,
            fields=list(fields) if fields is not None else None,
        )
        return self._format_search_hits(response.result.hits)

    def search_by_vector(
        self,
        vector: Sequence[float],
        namespace: str,
        top_k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
        fields: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Ищет записи по вектору в индексе с integrated inference API.

        Args:
            vector: Вектор запроса.
            namespace: Пространство имён (обязательно).
            top_k: Сколько результатов вернуть.
            metadata_filter: Фильтр по метаданным.
            fields: Какие поля включить в ответ.

        Returns:
            Список найденных записей.
        """
        response = self._index.search(
            namespace=namespace,
            top_k=top_k,
            vector=list(vector),
            filter=metadata_filter,
            fields=list(fields) if fields is not None else None,
        )
        return self._format_search_hits(response.result.hits)

    def fetch_vectors(
        self,
        ids: Sequence[str],
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Получает векторы по списку идентификаторов.

        Args:
            ids: Список id для загрузки.
            namespace: Пространство имён.

        Returns:
            Список найденных векторов.
        """
        response = self._index.fetch(
            ids=list(ids),
            namespace=self._resolve_namespace(namespace),
        )

        result: list[dict[str, Any]] = []
        for vector_id, vector in response.vectors.items():
            result.append(
                {
                    "id": vector_id,
                    "values": vector.values,
                    "metadata": vector.metadata,
                },
            )
        return result

    def fetch_by_ids(
        self,
        ids: Sequence[str],
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """Алиас для fetch_vectors (обратная совместимость)."""
        return self.fetch_vectors(ids=ids, namespace=namespace)

    def fetch_by_metadata(
        self,
        metadata_filter: Mapping[str, Any],
        namespace: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Получает записи по фильтру метаданных.

        Args:
            metadata_filter: Выражение фильтра Pinecone.
            namespace: Пространство имён.
            limit: Максимум записей на страницу.

        Returns:
            Список подходящих векторов.
        """
        response = self._index.fetch_by_metadata(
            filter=metadata_filter,
            namespace=self._resolve_namespace(namespace),
            limit=limit,
        )

        result: list[dict[str, Any]] = []
        for vector_id, vector in response.vectors.items():
            result.append(
                {
                    "id": vector_id,
                    "values": vector.values,
                    "metadata": vector.metadata,
                },
            )
        return result

    # ------------------------------------------------------------------
    # Удаление и статистика
    # ------------------------------------------------------------------

    def delete(
        self,
        ids: Sequence[str],
        namespace: str | None = None,
    ) -> None:
        """
        Удаляет векторы по списку идентификаторов.

        Args:
            ids: Список id для удаления.
            namespace: Пространство имён.
        """
        self._index.delete(
            ids=list(ids),
            namespace=self._resolve_namespace(namespace),
        )

    def delete_by_ids(
        self,
        ids: Sequence[str],
        namespace: str | None = None,
    ) -> None:
        """Алиас для delete (обратная совместимость)."""
        self.delete(ids=ids, namespace=namespace)

    def delete_by_filter(
        self,
        metadata_filter: Mapping[str, Any],
        namespace: str | None = None,
    ) -> None:
        """
        Удаляет записи, соответствующие фильтру метаданных.

        Args:
            metadata_filter: Выражение фильтра Pinecone.
            namespace: Пространство имён.
        """
        self._index.delete(
            filter=metadata_filter,
            namespace=self._resolve_namespace(namespace),
        )

    def delete_all(self, namespace: str | None = None) -> None:
        """
        Удаляет все векторы в указанном пространстве имён.

        Args:
            namespace: Пространство имён. По умолчанию — namespace класса.
        """
        self._index.delete(
            delete_all=True,
            namespace=self._resolve_namespace(namespace),
        )

    def describe_index_stats(
        self,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Возвращает статистику индекса.

        Args:
            metadata_filter: Опциональный фильтр для подсчёта записей.

        Returns:
            Словарь с dimension, total_vector_count и namespaces.
        """
        response = self._index.describe_index_stats(filter=metadata_filter)
        return {
            "dimension": response.dimension,
            "total_vector_count": response.total_vector_count,
            "namespaces": response.namespaces,
        }

    def describe_stats(
        self,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Алиас для describe_index_stats (обратная совместимость)."""
        return self.describe_index_stats(metadata_filter=metadata_filter)

    def update_metadata(
        self,
        vector_id: str,
        metadata: Mapping[str, Any],
        namespace: str | None = None,
    ) -> int:
        """
        Обновляет метаданные вектора по его идентификатору.

        Существующие поля метаданных перезаписываются переданными значениями.

        Args:
            vector_id: Идентификатор вектора.
            metadata: Новые или обновляемые поля метаданных.
            namespace: Пространство имён.

        Returns:
            Количество затронутых записей (если доступно в ответе API).
        """
        response = self._index.update(
            id=vector_id,
            set_metadata=dict(metadata),
            namespace=self._resolve_namespace(namespace),
        )
        return response.matched_records or 0

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _format_query_matches(matches: Sequence[Any]) -> list[dict[str, Any]]:
        """Преобразует ответ query в удобный список словарей."""
        formatted: list[dict[str, Any]] = []
        for match in matches:
            formatted.append(
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata,
                    "values": match.values,
                },
            )
        return formatted

    @staticmethod
    def _format_search_hits(hits: Sequence[Any]) -> list[dict[str, Any]]:
        """Преобразует ответ search в удобный список словарей."""
        formatted: list[dict[str, Any]] = []
        for hit in hits:
            formatted.append(
                {
                    "id": hit.id,
                    "score": hit.score,
                    "fields": hit.fields,
                },
            )
        return formatted


if __name__ == "__main__":
    pinecone_manager = PineconeManager()

    # Демо: сохраняем тестовый фрагмент, затем ищем по тексту.
    # Без записи query_by_text вернёт [] — в индексе ещё нет данных.
    pinecone_manager.upsert_document(
        document_id="demo-hello",
        text="Привет!",
        metadata={"type": "demo"},
    )

    result = pinecone_manager.query_by_text(text="Привет!")
    print(result)
