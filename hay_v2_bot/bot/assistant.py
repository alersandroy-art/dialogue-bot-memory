"""
Telegram-бот v2 — Haystack Agent + Docling + Pinecone.

Запуск из корня проекта:
    python -m hay_v2_bot.main
    python hay_v2_bot/main.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import telebot
from openai import OpenAI
from telebot import apihelper, types

from ..components.config import Settings, assert_proxy_base_url, load_settings
from ..components.context import PineconeContext
from ..components.document_store import PineconeStoreFactory
from ..pipelines.generation import GenerationPipeline
from ..pipelines.ingestion import IngestionPipeline
from ..pipelines.summary import SummaryPipeline
from . import handlers

logger = logging.getLogger(__name__)


def _configure_telegram_proxy(proxy_url: str | None) -> None:
    """Направляет запросы TeleBot к api.telegram.org через прокси, если задан."""
    if not proxy_url:
        apihelper.proxy = None
        return
    apihelper.proxy = {"http": proxy_url, "https": proxy_url}
    logger.info("[telegram] proxy enabled: %s", proxy_url.split("@")[-1])


class HaystackV2Bot:
    """Персональный помощник v2: текст, файлы Docling, Agent, тулы."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        assert_proxy_base_url(self.settings.openai_base_url)
        _configure_telegram_proxy(self.settings.telegram_proxy)

        self._store_factory = PineconeStoreFactory(self.settings)
        self.context = PineconeContext(self.settings, self._store_factory)

        self._openai = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            http_client=httpx.Client(trust_env=False),
        )
        assert_proxy_base_url(str(self._openai.base_url))

        self._last_tool_result: dict = {}
        self.generation = GenerationPipeline(
            settings=self.settings,
            context=self.context,
            openai_client=self._openai,
            last_tool_result=self._last_tool_result,
        )
        self.ingestion = IngestionPipeline(self.settings)
        self.summary = SummaryPipeline(self.settings)

        self.bot = telebot.TeleBot(self.settings.telegram_bot_token, parse_mode="HTML")
        handlers.register_handlers(self)

    def handle_start(self, message: types.Message) -> None:
        user = message.from_user
        if user is None:
            return
        self.bot.reply_to(
            message,
            handlers.format_start_text(user.first_name or "друг", user.id),
        )

    def handle_help(self, message: types.Message) -> None:
        self.bot.reply_to(message, handlers.HELP_TEXT)

    def handle_memory(self, message: types.Message) -> None:
        user = message.from_user
        if user is None:
            return

        self.bot.send_chat_action(message.chat.id, "typing")
        try:
            query = (message.text or "").replace("/memory", "").strip()
            if not query:
                query = "Информация о пользователе и наш диалог"

            bundle = self.context.build(query_text=query, telegram_id=user.id)
            reply = (
                f"<b>Память</b> (<code>{self.context.memory_ns(user.id)}</code>):\n\n"
                f"{bundle.memory_text}\n\n"
                f"<b>Документы</b> (<code>{self.context.docs_ns(user.id)}</code>):\n\n"
                f"{bundle.documents_text}"
            )
            self.bot.reply_to(message, reply)
        except Exception:
            logger.exception("Ошибка /memory для %s", user.id)
            self.bot.reply_to(message, "Не удалось прочитать память. Попробуй позже.")

    def handle_forget(self, message: types.Message) -> None:
        user = message.from_user
        if user is None:
            return
        try:
            self.context.clear_all(telegram_id=user.id)
            self.bot.reply_to(
                message,
                "Память и загруженные документы очищены. Можем начать с чистого листа.",
            )
        except Exception:
            logger.exception("Ошибка /forget для %s", user.id)
            self.bot.reply_to(message, "Не удалось очистить данные. Попробуй позже.")

    def handle_text(self, message: types.Message) -> None:
        user = message.from_user
        if user is None or not message.text:
            return

        user_text = message.text.strip()
        if not user_text:
            return

        logger.info(
            "[handle_text] user_id=%s text_preview=%r",
            user.id,
            user_text[:120],
        )
        self.bot.send_chat_action(message.chat.id, "typing")
        self._last_tool_result.clear()

        try:
            if not user_text.startswith("/"):
                self.context.update_memory(
                    text=user_text,
                    telegram_id=user.id,
                    metadata={
                        "username": user.username or "",
                        "first_name": user.first_name or "",
                    },
                )

            result = self.generation.run(
                user_message=user_text,
                telegram_id=user.id,
                user_name=handlers.get_user_display_name(user),
            )
            reply_text = result["reply"]

            sent_photo = handlers.send_dog_tool_photo(
                self.bot,
                message,
                self._last_tool_result,
            )
            if not sent_photo:
                self.bot.reply_to(message, reply_text)
        except Exception:
            logger.exception("[handle_text] FAILED user_id=%s", user.id)
            self.bot.reply_to(
                message,
                "Произошла ошибка при обработке сообщения. Попробуй ещё раз.",
            )

    def handle_document(self, message: types.Message) -> None:
        user = message.from_user
        document = message.document
        if user is None or document is None:
            return

        filename = document.file_name or f"document_{document.file_id}"
        size_bytes = getattr(document, "file_size", None)

        # 1) Сразу по file_size: без getFile, без typing, без анализа.
        if handlers.is_document_too_big(document):
            logger.warning(
                "[handle_document] SKIP download (>20MB) user_id=%s file=%s size=%s",
                user.id,
                filename,
                size_bytes,
            )
            self.bot.reply_to(
                message,
                handlers.telegram_file_too_big_message(filename, size_bytes),
            )
            return

        if not handlers.is_supported_document(filename):
            self.bot.reply_to(
                message,
                "Этот формат пока не поддерживается. "
                "Пришлите PDF, DOCX, PPTX, HTML, MD, TXT и т.п.",
            )
            return

        self.bot.send_chat_action(message.chat.id, "typing")

        try:
            with handlers.make_temp_dir() as tmp:
                local_path = handlers.download_telegram_document(
                    self.bot,
                    document,
                    Path(tmp),
                )
                self.bot.reply_to(
                    message,
                    "Файл получен. Запускаю анализ и сохранение. "
                    "Это может занять немного времени…",
                )
                self.bot.send_chat_action(message.chat.id, "typing")

                store = self._store_factory.get_docs_store(user.id)
                ingest_result = self.ingestion.run(
                    file_path=local_path,
                    document_store=store,
                    filename=filename,
                    telegram_id=user.id,
                )

            chunks = list(ingest_result.get("documents") or [])
            written = ingest_result.get("documents_written", 0)
            if not chunks and written:
                chunks = self.context.retrieve_documents(
                    query_text=filename,
                    telegram_id=user.id,
                    top_k=min(8, max(written, 1)),
                )
            if written == 0 and not chunks:
                self.bot.reply_to(
                    message,
                    "Не удалось извлечь текст из файла. Попробуйте другой документ.",
                )
                return

            self.bot.reply_to(
                message,
                "Готово. Я изучил этот файл, теперь можем его обсудить.",
            )

            summary = self.summary.run(documents=chunks, filename=filename)
            self.bot.reply_to(message, summary)
            logger.info(
                "[handle_document] ok user_id=%s file=%s written=%s",
                user.id,
                filename,
                written,
            )
        except handlers.TelegramFileTooBigError:
            logger.warning(
                "[handle_document] SKIP download (API too big) user_id=%s file=%s size=%s",
                user.id,
                filename,
                size_bytes,
            )
            self.bot.reply_to(
                message,
                handlers.telegram_file_too_big_message(filename, size_bytes),
            )
        except Exception as exc:
            logger.exception(
                "[handle_document] FAILED user_id=%s file=%s",
                user.id,
                filename,
            )
            hint = str(exc)
            if "socks4" in hint.lower() or (
                "proxy" in hint.lower() and "unknown scheme" in hint.lower()
            ):
                user_msg = (
                    "Не удалось обработать файл: конфликт с системным SOCKS-прокси "
                    "при загрузке моделей. Перезапустите бота после обновления."
                )
            elif "401" in hint or "expired" in hint.lower() or "token" in hint.lower():
                user_msg = (
                    "Не удалось скачать модели Docling с Hugging Face (401 / "
                    "просроченный token). Перезапустите бота — используется "
                    "анонимная загрузка. Либо задайте свежий HF_TOKEN в .env."
                )
            else:
                user_msg = (
                    "Не удалось обработать файл. Проверьте формат и попробуйте ещё раз."
                )
            self.bot.reply_to(message, user_msg)

    def run(self) -> None:
        logger.info(
            "hay_v2_bot запущен. index=%s base_url=%s model=%s",
            self.settings.pinecone_index_name,
            self.settings.openai_base_url,
            self.settings.llm_model,
        )
        logger.info("Ожидание сообщений Telegram (infinity_polling)...")
        self.bot.infinity_polling(timeout=60, long_polling_timeout=60)
