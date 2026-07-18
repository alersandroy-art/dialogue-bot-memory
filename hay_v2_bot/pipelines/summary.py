"""Пайплайн однопредложного резюме загруженного файла."""

from __future__ import annotations

import logging
import re

from haystack import Document
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret

from ..components.config import Settings, assert_proxy_base_url
from ..components.prompts import FILE_SUMMARY_PROMPT

logger = logging.getLogger(__name__)

# Сколько чанков брать для резюме (начало файла обычно информативнее).
SUMMARY_CHUNK_LIMIT = 8
SUMMARY_CHARS_PER_CHUNK = 500


class SummaryPipeline:
    """Генерирует ровно одно короткое предложение о содержимом файла."""

    def __init__(self, settings: Settings) -> None:
        assert_proxy_base_url(settings.openai_base_url)
        self._generator = OpenAIChatGenerator(
            api_key=Secret.from_token(settings.openai_api_key),
            api_base_url=settings.openai_base_url,
            model=settings.llm_model,
            http_client_kwargs=settings.http_client_kwargs,
            generation_kwargs={"temperature": 0.3, "max_tokens": 120},
        )
        assert_proxy_base_url(str(self._generator.api_base_url or ""))
        if getattr(self._generator, "client", None) is not None:
            assert_proxy_base_url(str(self._generator.client.base_url))

    @staticmethod
    def _build_chunks_text(documents: list[Document], filename: str) -> str:
        parts: list[str] = [f"Имя файла: {filename}"]
        for doc in documents[:SUMMARY_CHUNK_LIMIT]:
            text = (doc.content or "").strip()
            if not text:
                continue
            meta = doc.meta or {}
            page = meta.get("page")
            chunk_index = meta.get("chunk_index")
            prefix_bits = []
            if chunk_index is not None:
                prefix_bits.append(f"чанк {chunk_index}")
            if page is not None:
                prefix_bits.append(f"стр. {page}")
            prefix = f"[{', '.join(prefix_bits)}] " if prefix_bits else ""
            parts.append(prefix + text[:SUMMARY_CHARS_PER_CHUNK])
        return "\n\n".join(parts)

    @staticmethod
    def _one_sentence(text: str) -> str:
        cleaned = (text or "").strip().strip('"«»')
        # Берём первое предложение, если модель всё же выдала больше.
        match = re.split(r"(?<=[.!?…])\s+", cleaned, maxsplit=1)
        sentence = match[0].strip() if match else cleaned
        if sentence and sentence[-1] not in ".!?…":
            sentence += "."
        return sentence

    def run(self, documents: list[Document], filename: str) -> str:
        if not documents:
            return f"Файл «{filename}» обработан, но извлечь текст не удалось."

        chunks = self._build_chunks_text(documents, filename)
        prompt = FILE_SUMMARY_PROMPT.format(chunks=chunks)
        logger.info("[summary] start filename=%s chunks=%s", filename, len(documents))

        result = self._generator.run(messages=prompt)
        replies = result.get("replies") or []
        raw = ""
        if replies:
            first = replies[0]
            raw = getattr(first, "text", None) or str(first)
        summary = self._one_sentence(raw)
        logger.info("[summary] done preview=%r", summary[:160])
        return summary
