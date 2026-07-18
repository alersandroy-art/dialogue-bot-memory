"""
Ingestion pipeline: файл → Docling → метаданные → эмбеддинги → Pinecone.

DocLing анализирует документ, HybridChunker режет на чанки,
каждый чанк пишется в персональный namespace hay_v2_docs_{telegram_id}.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from haystack import Document, Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from ..components.config import Settings, assert_proxy_base_url
from ..components.meta_enricher import DocumentMetaEnricher
from ..components.proxy_sanitize import prepare_docling_network

logger = logging.getLogger(__name__)

# Лимит токенов чанка под OpenAI embeddings (tiktoken, без Hugging Face).
CHUNK_MAX_TOKENS = 512


def _import_docling_converter():
    """Импорт DoclingConverter из актуального или legacy-пакета."""
    try:
        from haystack_integrations.components.converters.docling import (
            DoclingConverter,
            ExportType,
        )

        return DoclingConverter, ExportType
    except ImportError:
        pass

    try:
        from docling_haystack.converter import DoclingConverter, ExportType

        return DoclingConverter, ExportType
    except ImportError as exc:
        raise ImportError(
            "Не установлен Docling для разбора файлов. "
            "Выполните: pip install docling-haystack docling"
        ) from exc


def _extract_pdf_pages_pypdfium(path: Path) -> list[tuple[int, str]]:
    """Простой текстовый слой PDF без моделей Hugging Face."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    pages: list[tuple[int, str]] = []
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            textpage = page.get_textpage()
            try:
                text = (textpage.get_text_bounded() or "").strip()
            finally:
                textpage.close()
                page.close()
            if text:
                pages.append((index + 1, text))
    finally:
        pdf.close()
    return pages


def _chunk_page_text(text: str, max_chars: int = 1800) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split_at = text.rfind("\n", start, end)
            if split_at <= start:
                split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end if end > start else end + 1
    return chunks


def _is_network_or_hf_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "huggingface",
        "xet-read-token",
        "network error",
        "socks4",
        "401",
        "403",
        "connection",
        "timeout",
        "snapshot_download",
        "layout-heron",
    )
    return any(marker in text for marker in markers)


def _build_chunker():
    """
    HybridChunker на tiktoken (cl100k_base) — без скачивания моделей с Hugging Face.
    """
    import tiktoken
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

    encoding = tiktoken.get_encoding("cl100k_base")
    tokenizer = OpenAITokenizer(tokenizer=encoding, max_tokens=CHUNK_MAX_TOKENS)
    return HybridChunker(tokenizer=tokenizer)


class IngestionPipeline:
    """Индексация файла пользователя через Docling + Pinecone."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        assert_proxy_base_url(settings.openai_base_url)
        # Docling подгружается лениво при первом файле — бот стартует без него.
        self._DoclingConverter = None
        self._ExportType = None
        self._doc_embedder = OpenAIDocumentEmbedder(
            api_key=Secret.from_token(settings.openai_api_key),
            api_base_url=settings.openai_base_url,
            model=settings.embedding_model,
            http_client_kwargs=settings.http_client_kwargs,
            progress_bar=False,
            meta_fields_to_embed=[],
        )
        assert_proxy_base_url(str(self._doc_embedder.client.base_url))

    def _ensure_docling(self) -> None:
        if self._DoclingConverter is not None:
            return
        self._DoclingConverter, self._ExportType = _import_docling_converter()

    def _build_pipeline(self, document_store: PineconeDocumentStore) -> Pipeline:
        self._ensure_docling()
        pipe = Pipeline()
        pipe.add_component(
            "converter",
            self._DoclingConverter(
                export_type=self._ExportType.DOC_CHUNKS,
                chunker=_build_chunker(),
            ),
        )
        pipe.add_component("enricher", DocumentMetaEnricher())
        pipe.add_component("embedder", self._doc_embedder)
        pipe.add_component(
            "writer",
            DocumentWriter(
                document_store=document_store,
                policy=DuplicatePolicy.OVERWRITE,
            ),
        )
        pipe.connect("converter.documents", "enricher.documents")
        pipe.connect("enricher.documents", "embedder.documents")
        pipe.connect("embedder.documents", "writer.documents")
        return pipe

    def _run_pypdfium_fallback(
        self,
        path: Path,
        document_store: PineconeDocumentStore,
        filename: str,
        telegram_id: str | int,
    ) -> dict[str, Any]:
        """Fallback: текст PDF через pypdfium2 → embed → Pinecone (без HF-моделей)."""
        logger.warning(
            "[ingestion] Docling/HF недоступен — fallback pypdfium2 для %s",
            filename,
        )
        pages = _extract_pdf_pages_pypdfium(path)
        if not pages:
            raise RuntimeError(
                "Не удалось извлечь текст из PDF (пустой текстовый слой / скан без OCR)."
            )

        documents: list[Document] = []
        chunk_index = 0
        for page_no, page_text in pages:
            for piece in _chunk_page_text(page_text):
                documents.append(
                    Document(
                        content=piece,
                        meta={
                            "type": "file_chunk",
                            "filename": filename,
                            "chunk_index": chunk_index,
                            "page": page_no,
                            "telegram_id": str(telegram_id),
                            "source": "pypdfium2_fallback",
                        },
                    ),
                )
                chunk_index += 1

        embedded = self._doc_embedder.run(documents=documents)["documents"]
        written = document_store.write_documents(
            embedded,
            policy=DuplicatePolicy.OVERWRITE,
        )
        # write_documents может вернуть int или None в разных версиях.
        written_count = int(written) if isinstance(written, int) else len(embedded)
        logger.info(
            "[ingestion] fallback done documents_written=%s file=%s",
            written_count,
            filename,
        )
        return {
            "documents_written": written_count,
            "documents": embedded,
            "filename": filename,
            "result": {"writer": {"documents_written": written_count}, "fallback": "pypdfium2"},
        }

    def run(
        self,
        file_path: str | Path,
        document_store: PineconeDocumentStore,
        filename: str,
        telegram_id: str | int,
    ) -> dict[str, Any]:
        """
        Запускает ingestion для одного файла.

        Returns:
            documents_written, documents (список чанков до/после записи).
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Файл не найден: {path}")

        logger.info(
            "[ingestion] start file=%s telegram_id=%s",
            filename,
            telegram_id,
        )
        prepare_docling_network()

        try:
            pipe = self._build_pipeline(document_store)
            run_input: dict[str, Any] = {
                "enricher": {
                    "filename": filename,
                    "telegram_id": str(telegram_id),
                },
                "converter": {"sources": [str(path)]},
            }
            include_from = {"enricher", "embedder", "writer"}
            try:
                result = pipe.run(run_input, include_outputs_from=include_from)
            except TypeError:
                try:
                    result = pipe.run(run_input)
                except Exception:
                    run_input["converter"] = {"paths": [str(path)]}
                    result = pipe.run(run_input)
            except Exception as first_exc:
                logger.warning(
                    "[ingestion] sources failed (%s), retry with paths",
                    first_exc,
                )
                run_input["converter"] = {"paths": [str(path)]}
                try:
                    result = pipe.run(run_input, include_outputs_from=include_from)
                except TypeError:
                    result = pipe.run(run_input)

            written = result.get("writer", {}).get("documents_written", 0)
            documents = (
                result.get("embedder", {}).get("documents")
                or result.get("enricher", {}).get("documents")
                or []
            )
            logger.info(
                "[ingestion] done documents_written=%s chunks=%s file=%s",
                written,
                len(documents),
                filename,
            )
            return {
                "documents_written": written,
                "documents": documents,
                "filename": filename,
                "result": result,
            }
        except Exception as exc:
            if path.suffix.lower() == ".pdf" and _is_network_or_hf_error(exc):
                logger.exception(
                    "[ingestion] Docling failed (network/HF), using pypdfium2 fallback",
                )
                return self._run_pypdfium_fallback(
                    path=path,
                    document_store=document_store,
                    filename=filename,
                    telegram_id=telegram_id,
                )
            raise
