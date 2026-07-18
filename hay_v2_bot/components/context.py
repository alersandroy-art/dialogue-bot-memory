"""
Контекст пользователя в Pinecone: память диалога и чанки документов.

Единая точка для:
- обновления контекста (запись сообщений пользователя);
- сборки релевантного контекста (recall + retrieve);
- очистки namespace памяти и документов.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from haystack import Document
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever

from .config import (
    COSINE_SIMILARITY_THRESHOLD,
    DOCS_TOP_K,
    MEMORY_TOP_K,
    Settings,
    assert_proxy_base_url,
)
from .document_store import (
    PineconeStoreFactory,
    docs_namespace,
    memory_namespace,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextBundle:
    """Собранный контекст для промпта генерации."""

    memory_documents: list[Document]
    document_chunks: list[Document]
    memory_text: str
    documents_text: str


class PineconeContext:
    """
    Работа с контекстом пользователя в Pinecone.

    Память  → namespace hay_v2_user_{telegram_id}
    Файлы   → namespace hay_v2_docs_{telegram_id}
    """

    def __init__(
        self,
        settings: Settings,
        store_factory: PineconeStoreFactory,
        similarity_threshold: float = COSINE_SIMILARITY_THRESHOLD,
    ) -> None:
        self._settings = settings
        self._stores = store_factory
        self._similarity_threshold = similarity_threshold
        api_key = Secret.from_token(settings.openai_api_key)

        self._doc_embedder = OpenAIDocumentEmbedder(
            api_key=api_key,
            api_base_url=settings.openai_base_url,
            model=settings.embedding_model,
            http_client_kwargs=settings.http_client_kwargs,
            progress_bar=False,
            meta_fields_to_embed=[],
        )
        self._text_embedder = OpenAITextEmbedder(
            api_key=api_key,
            api_base_url=settings.openai_base_url,
            model=settings.embedding_model,
            http_client_kwargs=settings.http_client_kwargs,
        )
        assert_proxy_base_url(settings.openai_base_url)
        assert_proxy_base_url(str(self._doc_embedder.client.base_url))
        assert_proxy_base_url(str(self._text_embedder.client.base_url))

    # --- namespaces ---------------------------------------------------------

    def memory_ns(self, telegram_id: str | int) -> str:
        return memory_namespace(telegram_id)

    def docs_ns(self, telegram_id: str | int) -> str:
        return docs_namespace(telegram_id)

    # --- update (запись в Pinecone) ----------------------------------------

    def update_memory(
        self,
        text: str,
        telegram_id: str | int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Обновляет контекст памяти: сохраняет текст сообщения пользователя.

        Ответы бота сюда не передавать. Дубликаты (cosine ≥ threshold) пропускаются.
        """
        user_text = (text or "").strip()
        namespace = memory_namespace(telegram_id)
        if not user_text:
            return {
                "action": "skipped",
                "namespace": namespace,
                "similarity": None,
                "message": "Пустой текст — ничего не сохранено.",
            }

        store = self._stores.get_memory_store(telegram_id)
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

        existing = self.recall_memory(
            query_text=user_text,
            telegram_id=telegram_id,
            top_k=1,
        )
        best = existing[0] if existing else None
        similarity = (
            float(best.score) if best is not None and best.score is not None else None
        )

        if (
            best is not None
            and similarity is not None
            and similarity >= self._similarity_threshold
        ):
            logger.info(
                "[context.update_memory] skip duplicate namespace=%s similarity=%s",
                namespace,
                similarity,
            )
            return {
                "action": "skipped",
                "namespace": namespace,
                "similarity": similarity,
                "message": "Похожий фрагмент уже есть — запись пропущена.",
            }

        doc = Document(content=user_text, meta=meta, id=str(uuid.uuid4()))
        embedded = self._doc_embedder.run(documents=[doc])["documents"]
        store.write_documents(embedded)
        logger.info("[context.update_memory] saved namespace=%s", namespace)
        return {
            "action": "saved",
            "namespace": namespace,
            "similarity": similarity,
            "message": "Текст сообщения пользователя сохранён в память.",
        }

    # алиас под прежнее API DialogMemory
    save_user_message = update_memory

    # --- read / build context ----------------------------------------------

    def recall_memory(
        self,
        query_text: str,
        telegram_id: str | int,
        top_k: int = MEMORY_TOP_K,
    ) -> list[Document]:
        namespace = memory_namespace(telegram_id)
        logger.info("[context.recall_memory] namespace=%s top_k=%s", namespace, top_k)
        store = self._stores.get_memory_store(telegram_id)
        embedding = self._text_embedder.run(text=query_text)["embedding"]
        retriever = PineconeEmbeddingRetriever(document_store=store, top_k=top_k)
        documents = retriever.run(query_embedding=embedding)["documents"]
        logger.info("[context.recall_memory] found=%s", len(documents))
        return documents

    def retrieve_documents(
        self,
        query_text: str,
        telegram_id: str | int,
        top_k: int = DOCS_TOP_K,
    ) -> list[Document]:
        namespace = docs_namespace(telegram_id)
        logger.info("[context.retrieve_documents] namespace=%s top_k=%s", namespace, top_k)
        store = self._stores.get_docs_store(telegram_id)
        embedding = self._text_embedder.run(text=query_text)["embedding"]
        retriever = PineconeEmbeddingRetriever(document_store=store, top_k=top_k)
        documents = retriever.run(query_embedding=embedding)["documents"]
        logger.info("[context.retrieve_documents] found=%s", len(documents))
        return documents

    def build(
        self,
        query_text: str,
        telegram_id: str | int,
        memory_top_k: int = MEMORY_TOP_K,
        docs_top_k: int = DOCS_TOP_K,
    ) -> ContextBundle:
        """Собирает актуальный контекст (память + документы) для генерации."""
        memory_docs = self.recall_memory(
            query_text=query_text,
            telegram_id=telegram_id,
            top_k=memory_top_k,
        )
        doc_chunks = self.retrieve_documents(
            query_text=query_text,
            telegram_id=telegram_id,
            top_k=docs_top_k,
        )
        return ContextBundle(
            memory_documents=memory_docs,
            document_chunks=doc_chunks,
            memory_text=self.format_memory(memory_docs),
            documents_text=self.format_documents(doc_chunks),
        )

    # --- clear -------------------------------------------------------------

    def clear_memory(self, telegram_id: str | int) -> None:
        store = self._stores.get_memory_store(telegram_id)
        store.delete_all_documents()
        self._stores.drop_cached(memory_namespace(telegram_id))
        logger.info("[context.clear_memory] namespace=%s", memory_namespace(telegram_id))

    def clear_documents(self, telegram_id: str | int) -> None:
        store = self._stores.get_docs_store(telegram_id)
        store.delete_all_documents()
        self._stores.drop_cached(docs_namespace(telegram_id))
        logger.info("[context.clear_documents] namespace=%s", docs_namespace(telegram_id))

    def clear_all(self, telegram_id: str | int) -> None:
        """Полная очистка контекста пользователя (память + документы)."""
        self.clear_memory(telegram_id)
        self.clear_documents(telegram_id)

    # --- formatters --------------------------------------------------------

    @staticmethod
    def format_memory(documents: list[Document]) -> str:
        if not documents:
            return "Пока нет сохранённых данных о пользователе."

        lines: list[str] = []
        for index, doc in enumerate(documents, start=1):
            meta = doc.meta or {}
            if meta.get("role") == "assistant":
                continue
            if meta.get("type") in {"bot_response", "assistant", "system", "file_chunk"}:
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

    @staticmethod
    def format_documents(documents: list[Document]) -> str:
        if not documents:
            return "Пока нет загруженных документов."

        lines: list[str] = []
        for index, doc in enumerate(documents, start=1):
            meta = doc.meta or {}
            text = (doc.content or "").strip()
            if not text:
                continue
            filename = meta.get("filename") or "файл"
            chunk_index = meta.get("chunk_index")
            page = meta.get("page")
            section = meta.get("section")
            parts = [f"файл: {filename}"]
            if chunk_index is not None:
                parts.append(f"чанк: {chunk_index}")
            if page is not None:
                parts.append(f"стр.: {page}")
            if section:
                parts.append(f"секция: {section}")
            header = ", ".join(parts)
            score = doc.score
            score_s = f" (релевантность: {score:.2f})" if score is not None else ""
            lines.append(f"{index}. [{header}]{score_s}\n{text}")

        return "\n\n".join(lines) if lines else "Пока нет загруженных документов."
