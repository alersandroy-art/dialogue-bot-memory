"""Фабрики PineconeDocumentStore: память диалога и документы пользователя."""

from __future__ import annotations

import logging

from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from .config import (
    DOCS_NAMESPACE_PREFIX,
    EMBEDDING_DIMENSION,
    FORBIDDEN_SHARED_NAMESPACES,
    MEMORY_NAMESPACE_PREFIX,
    Settings,
)

logger = logging.getLogger(__name__)


def _require_telegram_id(telegram_id: str | int) -> str:
    if telegram_id is None:
        raise ValueError("telegram_id обязателен для персонального namespace.")
    user_id = str(telegram_id).strip()
    if not user_id:
        raise ValueError("telegram_id не может быть пустым.")
    if user_id.lower() in FORBIDDEN_SHARED_NAMESPACES:
        raise ValueError(f"Недопустимый telegram_id={user_id!r}.")
    return user_id


def memory_namespace(telegram_id: str | int) -> str:
    """Персональный namespace памяти: hay_v2_user_{telegram_id}."""
    return f"{MEMORY_NAMESPACE_PREFIX}{_require_telegram_id(telegram_id)}"


def docs_namespace(telegram_id: str | int) -> str:
    """Персональный namespace документов: hay_v2_docs_{telegram_id}."""
    return f"{DOCS_NAMESPACE_PREFIX}{_require_telegram_id(telegram_id)}"


class PineconeStoreFactory:
    """Кэш PineconeDocumentStore по namespace (изоляция пользователей)."""

    def __init__(self, settings: Settings, dimension: int = EMBEDDING_DIMENSION) -> None:
        self._settings = settings
        self._dimension = dimension
        self._stores: dict[str, PineconeDocumentStore] = {}

    def get(self, namespace: str) -> PineconeDocumentStore:
        if namespace in self._stores:
            return self._stores[namespace]

        logger.info(
            "[store] create index=%s namespace=%s metric=cosine dim=%s",
            self._settings.pinecone_index_name,
            namespace,
            self._dimension,
        )
        store = PineconeDocumentStore(
            index=self._settings.pinecone_index_name,
            namespace=namespace,
            metric="cosine",
            dimension=self._dimension,
            spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
            show_progress=False,
        )
        store_ns = getattr(store, "namespace", namespace)
        if store_ns != namespace:
            raise RuntimeError(
                f"PineconeDocumentStore.namespace={store_ns!r}, ожидался {namespace!r}.",
            )
        self._stores[namespace] = store
        return store

    def get_memory_store(self, telegram_id: str | int) -> PineconeDocumentStore:
        return self.get(memory_namespace(telegram_id))

    def get_docs_store(self, telegram_id: str | int) -> PineconeDocumentStore:
        return self.get(docs_namespace(telegram_id))

    def drop_cached(self, namespace: str) -> None:
        self._stores.pop(namespace, None)
