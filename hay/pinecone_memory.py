"""
Долговременная память пользователя через PineconeDocumentStore (Haystack).

Интеграция как в 03doc_integration_pinecone.md:
- metric=cosine
- OpenAI-эмбеддинги (ProxyAPI)
- namespace на каждого пользователя Telegram
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from haystack import Document
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

logger = logging.getLogger(__name__)

# Порог косинусного сходства: выше — дубликат/вариация, ниже — новая память.
COSINE_SIMILARITY_THRESHOLD = 0.85

# Каждый пользователь Telegram — отдельный namespace Pinecone.
# Формат: hay_user_{telegram_id} (не пересекается с базовым ботом user_{id}).
USER_NAMESPACE_PREFIX = "hay_user_"
FORBIDDEN_SHARED_NAMESPACES = frozenset({"", "default", "shared", "common"})

EMBEDDING_DIMENSION = 1536  # text-embedding-3-small


class HaystackPineconeMemory:
    """Память диалога в Pinecone с поиском по косинусному сходству."""

    def __init__(
        self,
        index_name: str,
        openai_api_key: str,
        openai_base_url: str,
        embedding_model: str = "text-embedding-3-small",
        dimension: int = EMBEDDING_DIMENSION,
        similarity_threshold: float = COSINE_SIMILARITY_THRESHOLD,
    ) -> None:
        self._index_name = index_name
        self._dimension = dimension
        self._similarity_threshold = similarity_threshold
        self._api_key = Secret.from_token(openai_api_key)
        self._base_url = openai_base_url
        self._embedding_model = embedding_model
        self._http_client_kwargs = {"trust_env": False}

        self._doc_embedder = OpenAIDocumentEmbedder(
            api_key=self._api_key,
            api_base_url=self._base_url,
            model=self._embedding_model,
            http_client_kwargs=self._http_client_kwargs,
            progress_bar=False,
            # Эмбеддинг только по content (= текст пользователя), не по metadata.
            meta_fields_to_embed=[],
        )
        self._text_embedder = OpenAITextEmbedder(
            api_key=self._api_key,
            api_base_url=self._base_url,
            model=self._embedding_model,
            http_client_kwargs=self._http_client_kwargs,
        )
        self._assert_proxy_base_url(self._base_url)
        self._assert_proxy_base_url(str(self._doc_embedder.client.base_url))
        self._assert_proxy_base_url(str(self._text_embedder.client.base_url))
        self._stores: dict[str, PineconeDocumentStore] = {}

    @staticmethod
    def _assert_proxy_base_url(base_url: str) -> None:
        """Все эмбеддинги должны идти через OPENAI_BASE_URL (ProxyAPI)."""
        normalized = (base_url or "").strip().rstrip("/")
        if not normalized:
            raise ValueError("OPENAI_BASE_URL пуст.")
        if "api.openai.com" in normalized.lower():
            raise ValueError(
                f"Прямой OpenAI URL запрещён: {normalized}. "
                "Используйте OPENAI_BASE_URL.",
            )

    @staticmethod
    def _require_telegram_id(telegram_id: str | int) -> str:
        """
        Валидирует Telegram user id.

        Память пользователей нельзя писать в общий namespace —
        всегда нужен персональный идентификатор.
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
        return user_id

    @classmethod
    def build_user_namespace(cls, telegram_id: str | int) -> str:
        """
        Персональный namespace пользователя: hay_user_{telegram_id}.

        Пример: telegram_id=12345 → \"hay_user_12345\".
        Другие пользователи в этот namespace не попадают.
        """
        user_id = cls._require_telegram_id(telegram_id)
        namespace = f"{USER_NAMESPACE_PREFIX}{user_id}"
        if namespace.lower() in FORBIDDEN_SHARED_NAMESPACES:
            raise ValueError(f"Запрещён общий namespace: {namespace}")
        if not namespace.startswith(USER_NAMESPACE_PREFIX):
            raise ValueError(
                f"Namespace {namespace!r} должен начинаться с "
                f"{USER_NAMESPACE_PREFIX!r}.",
            )
        return namespace

    def get_document_store(self, telegram_id: str | int) -> PineconeDocumentStore:
        """
        PineconeDocumentStore, привязанный к namespace этого пользователя.

        Изоляция: index общий, namespace = hay_user_{telegram_id}.
        """
        namespace = self.build_user_namespace(telegram_id)
        if namespace not in self._stores:
            logger.info(
                "[memory] isolation: create store index=%s namespace=%s "
                "(personal for telegram_id=%s) metric=cosine dimension=%s",
                self._index_name,
                namespace,
                telegram_id,
                self._dimension,
            )
            try:
                store = PineconeDocumentStore(
                    index=self._index_name,
                    namespace=namespace,
                    metric="cosine",
                    dimension=self._dimension,
                    spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
                    show_progress=False,
                )
            except Exception:
                logger.exception(
                    "[memory.get_document_store] FAILED index=%s namespace=%s",
                    self._index_name,
                    namespace,
                )
                raise

            # Контроль: store действительно смотрит в личный namespace.
            store_ns = getattr(store, "namespace", namespace)
            if store_ns != namespace:
                raise RuntimeError(
                    f"PineconeDocumentStore.namespace={store_ns!r}, "
                    f"ожидался личный {namespace!r}.",
                )
            self._stores[namespace] = store
            logger.info(
                "[memory] user isolation ok | telegram_id=%s namespace=%s",
                telegram_id,
                namespace,
            )
        return self._stores[namespace]

    def recall(
        self,
        query_text: str,
        telegram_id: str | int,
        top_k: int = 5,
    ) -> list[Document]:
        """Ищет релевантные фрагменты памяти по косинусному сходству."""
        namespace = self.build_user_namespace(telegram_id)
        logger.info(
            "[memory.recall] start namespace=%s top_k=%s text_len=%s",
            namespace,
            top_k,
            len(query_text),
        )
        try:
            store = self.get_document_store(telegram_id)
            logger.info("[memory.recall] step=embed_query namespace=%s", namespace)
            embedding = self._text_embedder.run(text=query_text)["embedding"]
            logger.info(
                "[memory.recall] step=embed_query ok dim=%s",
                len(embedding) if embedding is not None else None,
            )
            retriever = PineconeEmbeddingRetriever(document_store=store, top_k=top_k)
            logger.info("[memory.recall] step=retrieve namespace=%s", namespace)
            documents = retriever.run(query_embedding=embedding)["documents"]
            logger.info(
                "[memory.recall] ok namespace=%s found=%s",
                namespace,
                len(documents),
            )
            return documents
        except Exception:
            logger.exception(
                "[memory.recall] FAILED namespace=%s text_preview=%r",
                namespace,
                query_text[:120],
            )
            raise

    def save(
        self,
        text: str,
        telegram_id: str | int,
        metadata: dict[str, Any] | None = None,
        on_duplicate: str = "skip",
    ) -> dict[str, Any]:
        """
        Сохраняет в Pinecone только переданный текст (вектор = embedding(text)).

        Служебные поля — только в metadata, в эмбеддинг не входят.
        Для ответов бота этот метод не использовать (см. save_user_message).
        """
        user_text = (text or "").strip()
        if not user_text:
            logger.info("[memory.save] skip: пустой текст")
            return {
                "action": "skipped",
                "namespace": self.build_user_namespace(telegram_id),
                "similarity": None,
                "message": "Пустой текст — ничего не сохранено.",
            }

        namespace = self.build_user_namespace(telegram_id)
        logger.info(
            "[memory.save] start namespace=%s on_duplicate=%s text_len=%s text_preview=%r",
            namespace,
            on_duplicate,
            len(user_text),
            user_text[:120],
        )
        try:
            logger.info("[memory.save] step=get_store namespace=%s", namespace)
            store = self.get_document_store(telegram_id)

            # content = только текст пользователя; служебное — в metadata.
            meta = {
                "type": "user_message",
                "role": "user",
                "telegram_id": str(telegram_id),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "text": user_text,
            }
            if metadata:
                for key, value in metadata.items():
                    if key in {"type", "role", "content", "text"}:
                        continue
                    meta[key] = value

            logger.info("[memory.save] step=duplicate_check namespace=%s", namespace)
            existing = self.recall(
                query_text=user_text,
                telegram_id=telegram_id,
                top_k=1,
            )
            best = existing[0] if existing else None
            similarity = (
                float(best.score)
                if best is not None and best.score is not None
                else None
            )
            logger.info(
                "[memory.save] step=duplicate_check ok similarity=%s threshold=%s "
                "best_id=%s",
                similarity,
                self._similarity_threshold,
                getattr(best, "id", None),
            )

            if (
                best is not None
                and similarity is not None
                and similarity >= self._similarity_threshold
            ):
                if on_duplicate == "skip":
                    result = {
                        "action": "skipped",
                        "namespace": namespace,
                        "similarity": similarity,
                        "message": "Похожий фрагмент уже есть — запись пропущена.",
                    }
                    logger.info("[memory.save] done (skip) %s", result)
                    return result

                logger.info(
                    "[memory.save] step=update_delete id=%s namespace=%s",
                    best.id,
                    namespace,
                )
                try:
                    store.delete_documents([best.id])
                except Exception:
                    logger.exception(
                        "[memory.save] step=update_delete FAILED id=%s",
                        best.id,
                    )

                doc = Document(content=user_text, meta=meta, id=best.id)
                logger.info("[memory.save] step=embed_document (update)")
                embedded = self._doc_embedder.run(documents=[doc])["documents"]
                logger.info("[memory.save] step=write_documents (update)")
                store.write_documents(embedded)
                result = {
                    "action": "updated",
                    "namespace": namespace,
                    "similarity": similarity,
                    "message": "Похожий фрагмент обновлён.",
                }
                logger.info("[memory.save] done (update) %s", result)
                return result

            doc_id = str(uuid.uuid4())
            doc = Document(content=user_text, meta=meta, id=doc_id)
            logger.info(
                "[memory.save] step=embed_document (new) id=%s content=%r",
                doc_id,
                user_text[:120],
            )
            embedded = self._doc_embedder.run(documents=[doc])["documents"]
            emb_dim = (
                len(embedded[0].embedding)
                if embedded and embedded[0].embedding is not None
                else None
            )
            logger.info(
                "[memory.save] step=embed_document ok dim=%s id=%s",
                emb_dim,
                doc_id,
            )
            logger.info(
                "[memory.save] step=write_documents (new) namespace=%s id=%s",
                namespace,
                doc_id,
            )
            store.write_documents(embedded)
            result = {
                "action": "saved",
                "namespace": namespace,
                "similarity": similarity,
                "message": "Текст сообщения пользователя сохранён в память.",
            }
            logger.info("[memory.save] done (saved) %s", result)
            return result
        except Exception:
            logger.exception(
                "[memory.save] FAILED namespace=%s text_preview=%r",
                namespace,
                user_text[:120],
            )
            raise

    def save_user_message(
        self,
        text: str,
        telegram_id: str | int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Сохраняет только оригинальный текст сообщения пользователя."""
        return self.save(
            text=text,
            telegram_id=telegram_id,
            metadata=metadata,
            on_duplicate="skip",
        )

    def clear(self, telegram_id: str | int) -> None:
        """Полностью очищает namespace пользователя."""
        store = self.get_document_store(telegram_id)
        store.delete_all_documents()
        namespace = self.build_user_namespace(telegram_id)
        self._stores.pop(namespace, None)

    @staticmethod
    def format_context(documents: list[Document]) -> str:
        """Собирает контекст только из сообщений пользователя."""
        if not documents:
            return "Пока нет сохранённых данных о пользователе."

        lines: list[str] = []
        for index, doc in enumerate(documents, start=1):
            meta = doc.meta or {}
            if meta.get("role") == "assistant":
                continue
            if meta.get("type") in {"bot_response", "assistant", "system"}:
                continue

            text = (doc.content or meta.get("text") or "").strip()
            if not text:
                continue
            score = doc.score
            if score is not None:
                lines.append(f"{index}. {text} (релевантность: {score:.2f})")
            else:
                lines.append(f"{index}. {text}")

        return (
            "\n".join(lines)
            if lines
            else "Пока нет сохранённых данных о пользователе."
        )


def memory_from_env() -> HaystackPineconeMemory:
    """Создаёт память из переменных окружения."""
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    missing = [
        name
        for name, value in [
            ("PINECONE_INDEX_NAME", index_name),
            ("OPENAI_API_KEY", api_key),
            ("OPENAI_BASE_URL", base_url),
        ]
        if not value
    ]
    if missing:
        raise ValueError(f"Не заданы переменные окружения: {', '.join(missing)}")

    return HaystackPineconeMemory(
        index_name=index_name,
        openai_api_key=api_key,
        openai_base_url=base_url,
        embedding_model=embedding_model,
    )
