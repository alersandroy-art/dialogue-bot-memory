"""
Generation pipeline:

1. PineconeContext.build — релевантная память + чанки документов
2. Agent (LLM + generation tools) — ответ с учётом контекста
"""

from __future__ import annotations

import logging
from typing import Any

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from openai import OpenAI

from ..components.config import DOCS_TOP_K, MEMORY_TOP_K, Settings, assert_proxy_base_url
from ..components.context import PineconeContext
from ..components.prompts import SYSTEM_PROMPT
from ..components.tools import build_generation_tools

logger = logging.getLogger(__name__)


class GenerationPipeline:
    """Оркестрация context.build + Haystack Agent (тулы генерации)."""

    def __init__(
        self,
        settings: Settings,
        context: PineconeContext,
        openai_client: OpenAI,
        last_tool_result: dict[str, Any],
    ) -> None:
        self._settings = settings
        self._context = context
        self._last_tool_result = last_tool_result

        assert_proxy_base_url(settings.openai_base_url)

        tools = build_generation_tools(
            openai_client=openai_client,
            vision_model=settings.vision_model,
            last_result=last_tool_result,
        )
        chat_generator = OpenAIChatGenerator(
            api_key=Secret.from_token(settings.openai_api_key),
            api_base_url=settings.openai_base_url,
            model=settings.llm_model,
            http_client_kwargs=settings.http_client_kwargs,
        )
        assert_proxy_base_url(str(chat_generator.api_base_url or ""))
        if getattr(chat_generator, "client", None) is not None:
            assert_proxy_base_url(str(chat_generator.client.base_url))

        self.agent = Agent(
            chat_generator=chat_generator,
            system_prompt="Ты персональный помощник.",
            tools=tools,
            exit_conditions=["text"],
            max_agent_steps=8,
        )
        self.agent.warm_up()

    def run(
        self,
        user_message: str,
        telegram_id: str | int,
        user_name: str,
        memory_top_k: int = MEMORY_TOP_K,
        docs_top_k: int = DOCS_TOP_K,
    ) -> dict[str, Any]:
        """
        Полный цикл генерации ответа.

        Returns:
            reply, memory_documents, document_chunks
        """
        logger.info("[generation] start telegram_id=%s", telegram_id)

        bundle = self._context.build(
            query_text=user_message,
            telegram_id=telegram_id,
            memory_top_k=memory_top_k,
            docs_top_k=docs_top_k,
        )

        self.agent.system_prompt = SYSTEM_PROMPT.format(
            user_name=user_name,
            memory_context=bundle.memory_text,
            documents_context=bundle.documents_text,
        )
        result = self.agent.run(messages=[ChatMessage.from_user(user_message)])

        last = result.get("last_message")
        if last is None:
            messages = result.get("messages") or []
            last = messages[-1] if messages else None
        text = getattr(last, "text", None) if last is not None else None
        if not text:
            text = "Извините, не удалось сформировать ответ. Попробуйте ещё раз."

        logger.info(
            "[generation] done reply_len=%s memory_hits=%s doc_hits=%s",
            len(text),
            len(bundle.memory_documents),
            len(bundle.document_chunks),
        )
        return {
            "reply": text.strip(),
            "memory_documents": bundle.memory_documents,
            "document_chunks": bundle.document_chunks,
        }
