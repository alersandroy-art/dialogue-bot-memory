"""Обогащение метаданных чанков после Docling."""

from __future__ import annotations

import logging
from typing import Any

from haystack import Document, component

logger = logging.getLogger(__name__)


def _extract_page(meta: dict[str, Any]) -> int | None:
    """Достаёт номер страницы из Docling dl_meta, если есть."""
    dl_meta = meta.get("dl_meta")
    if not isinstance(dl_meta, dict):
        return None
    try:
        doc_items = dl_meta.get("meta", {}).get("doc_items") or []
        if not doc_items:
            return None
        prov = doc_items[0].get("prov") or []
        if not prov:
            return None
        page = prov[0].get("page_no")
        return int(page) if page is not None else None
    except (TypeError, ValueError, KeyError, IndexError):
        return None


def _extract_headings(meta: dict[str, Any]) -> str | None:
    dl_meta = meta.get("dl_meta")
    if not isinstance(dl_meta, dict):
        return None
    headings = dl_meta.get("meta", {}).get("headings")
    if isinstance(headings, list) and headings:
        return " / ".join(str(h) for h in headings)
    return None


@component
class DocumentMetaEnricher:
    """
    Добавляет обязательные поля: filename, chunk_index, page (если доступно).

    Также помечает документ как file_chunk для отделения от памяти диалога.
    """

    @component.output_types(documents=list[Document])
    def run(
        self,
        documents: list[Document],
        filename: str = "",
        telegram_id: str = "",
    ) -> dict[str, list[Document]]:
        enriched: list[Document] = []
        safe_name = (filename or "document").strip() or "document"

        for index, doc in enumerate(documents):
            meta = dict(doc.meta or {})
            meta["type"] = "file_chunk"
            meta["filename"] = safe_name
            meta["chunk_index"] = index
            if telegram_id:
                meta["telegram_id"] = str(telegram_id)

            page = _extract_page(meta)
            if page is not None:
                meta["page"] = page

            section = _extract_headings(meta)
            if section:
                meta["section"] = section

            enriched.append(
                Document(
                    id=doc.id,
                    content=doc.content,
                    meta=meta,
                    embedding=doc.embedding,
                ),
            )

        logger.info(
            "[DocumentMetaEnricher] enriched=%s filename=%s",
            len(enriched),
            safe_name,
        )
        return {"documents": enriched}
