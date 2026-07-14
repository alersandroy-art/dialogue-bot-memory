"""
Telegram-бот — умный персональный помощник на базе Haystack Agent.

- Память: PineconeDocumentStore (cosine), как в 03doc_integration_pinecone.md
- Агент: Haystack Agent + OpenAIChatGenerator (ProxyAPI)
- Тулы: случайный факт о собаках; картинка собаки + vision-описание породы
- Транспорт: pyTelegramBotAPI

Запуск из корня проекта:
    python hay/hay-telegram-bot.py
"""

from __future__ import annotations

import html
import logging
import os
import sys
from pathlib import Path

import httpx
import telebot
from dotenv import load_dotenv
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from openai import OpenAI
from telebot import types

# Чтобы импорты соседних модулей работали при запуске как скрипта.
_HAY_DIR = Path(__file__).resolve().parent
if str(_HAY_DIR) not in sys.path:
    sys.path.insert(0, str(_HAY_DIR))

from dog_tools import build_dog_tools
from loguru import logger as image_log
from pinecone_memory import HaystackPineconeMemory, memory_from_env

MEMORY_TOP_K = 5

SYSTEM_PROMPT = """Ты умный персональный помощник в Telegram.

Имя пользователя: {user_name}

Релевантный контекст из долговременной памяти (косинусное сходство в Pinecone):
{memory_context}

Правила:
- Отвечай на русском, если пользователь пишет по-русски.
- Веди себя как настоящий личный помощник: помни детали из контекста,
  продолжай диалог естественно, уточняй при необходимости.
- Используй факты из памяти, когда они помогают ответить.
- Не выдумывай факты о пользователе, которых нет в памяти.
- Если пользователь просит факт о собаках — вызови инструмент dogFactTool.
- Если нужна только картинка собаки без анализа — dogImageTool.
- Если нужна картинка и описание породы / предыстория — dogImageAnalyzerTool
  (инструмент сам получит картинку с API и отправит её в ChatGPT).
- После dogImageAnalyzerTool / dogImageTool не дублируй длинный текст:
  картинка и подпись уйдут пользователю отдельным фото в Telegram.
  Кратко подтверди, что отправил результат.
- Будь полезным, тёплым и по делу.
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class HaystackAssistantBot:
    """Персональный помощник: Haystack Agent + Pinecone + Telegram."""

    def __init__(self) -> None:
        # .env лежит в корне проекта (на уровень выше hay/).
        load_dotenv(_HAY_DIR.parent / ".env")
        load_dotenv()

        self._token = self._require_env("TELEGRAM_BOT_TOKEN")
        self._llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self._vision_model = os.getenv("VISION_MODEL", self._llm_model)
        self._openai_api_key = self._require_env("OPENAI_API_KEY")
        self._openai_base_url = self._require_env("OPENAI_BASE_URL")
        self._assert_proxy_base_url(self._openai_base_url)

        self.memory: HaystackPineconeMemory = memory_from_env()

        self._openai = OpenAI(
            api_key=self._openai_api_key,
            base_url=self._openai_base_url,
            http_client=httpx.Client(trust_env=False),
        )
        self._assert_proxy_base_url(str(self._openai.base_url))

        # Результат последнего dog-тула (image_url + caption и т.д.).
        self._last_dog_result: dict = {}

        self._tools = build_dog_tools(
            openai_client=self._openai,
            vision_model=self._vision_model,
            last_result=self._last_dog_result,
        )

        self._chat_generator = OpenAIChatGenerator(
            api_key=Secret.from_token(self._openai_api_key),
            api_base_url=self._openai_base_url,
            model=self._llm_model,
            http_client_kwargs={"trust_env": False},
        )
        self._assert_proxy_base_url(str(self._chat_generator.api_base_url or ""))
        if getattr(self._chat_generator, "client", None) is not None:
            self._assert_proxy_base_url(str(self._chat_generator.client.base_url))


        self.agent = Agent(
            chat_generator=self._chat_generator,
            system_prompt="Ты персональный помощник.",
            tools=self._tools,
            exit_conditions=["text"],
            max_agent_steps=8,
        )
        self.agent.warm_up()

        self.bot = telebot.TeleBot(self._token, parse_mode="HTML")
        self._register_handlers()

    @staticmethod
    def _assert_proxy_base_url(base_url: str) -> None:
        """Гарантирует, что запросы идут через OPENAI_BASE_URL, а не на api.openai.com."""
        normalized = (base_url or "").strip().rstrip("/")
        if not normalized:
            raise ValueError("OPENAI_BASE_URL пуст — все вызовы OpenAI должны идти через base_url.")
        if "api.openai.com" in normalized.lower():
            raise ValueError(
                f"Запрещён прямой доступ к OpenAI API: {normalized}. "
                "Укажите прокси в OPENAI_BASE_URL.",
            )

    @staticmethod
    def _require_env(name: str) -> str:
        value = os.getenv(name, "").strip()
        if not value:
            raise ValueError(
                f"Переменная окружения {name} не задана. Добавьте её в .env.",
            )
        return value

    def _register_handlers(self) -> None:
        self.bot.message_handler(commands=["start"])(self.handle_start)
        self.bot.message_handler(commands=["help"])(self.handle_help)
        self.bot.message_handler(commands=["memory"])(self.handle_memory)
        self.bot.message_handler(commands=["forget"])(self.handle_forget)
        self.bot.message_handler(content_types=["text"])(self.handle_text)

    @staticmethod
    def _get_user_display_name(user: types.User) -> str:
        parts = [user.first_name or "", user.last_name or ""]
        name = " ".join(part for part in parts if part).strip()
        if user.username:
            return f"{name} (@{user.username})" if name else f"@{user.username}"
        return name or "Пользователь"

    def _run_agent(self, user_message: str, memory_context: str, user_name: str) -> str:
        """Запускает Haystack Agent с учётом памяти пользователя."""
        self.agent.system_prompt = SYSTEM_PROMPT.format(
            user_name=user_name,
            memory_context=memory_context,
        )
        result = self.agent.run(
            messages=[ChatMessage.from_user(user_message)],
        )
        last = result.get("last_message")
        if last is None:
            messages = result.get("messages") or []
            last = messages[-1] if messages else None
        text = getattr(last, "text", None) if last is not None else None
        if not text:
            return "Извините, не удалось сформировать ответ. Попробуйте ещё раз."
        return text.strip()

    def handle_start(self, message: types.Message) -> None:
        user = message.from_user
        if user is None:
            return
        text = (
            f"Привет, <b>{user.first_name or 'друг'}</b>!\n\n"
            "Я персональный помощник на Haystack Agent с памятью в Pinecone.\n"
            "Я запоминаю наш диалог и могу пользоваться инструментами:\n"
            "• случайный факт о собаках\n"
            "• случайная картинка собаки + описание породы (vision)\n\n"
            "Команды: /help, /memory, /forget"
        )
        self.bot.reply_to(message, text)

    def handle_help(self, message: types.Message) -> None:
        text = (
            "<b>Справка</b>\n\n"
            "В векторной базе Pinecone сохраняется <b>только текст</b> "
            "твоих сообщений. Ответы бота не пишутся.\n\n"
            "Примеры:\n"
            "• «Запомни, что меня зовут Алекс и я люблю кофе»\n"
            "• «Расскажи факт о собаках»\n"
            "• «Покажи случайную собаку и расскажи про породу»\n\n"
            "<b>Команды:</b>\n"
            "/start — начать\n"
            "/memory — что я помню\n"
            "/forget — очистить память"
        )
        self.bot.reply_to(message, text)

    def handle_memory(self, message: types.Message) -> None:
        user = message.from_user
        if user is None:
            return

        self.bot.send_chat_action(message.chat.id, "typing")
        try:
            query = (message.text or "").replace("/memory", "").strip()
            if not query:
                query = "Информация о пользователе и наш диалог"

            documents = self.memory.recall(
                query_text=query,
                telegram_id=user.id,
                top_k=MEMORY_TOP_K,
            )
            context = self.memory.format_context(documents)
            namespace = self.memory.build_user_namespace(user.id)
            reply = (
                f"<b>Память</b> (namespace: <code>{namespace}</code>):\n\n"
                f"{context}"
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
            self.memory.clear(telegram_id=user.id)
            self.bot.reply_to(
                message,
                "Память очищена. Можем начать с чистого листа.",
            )
        except Exception:
            logger.exception("Ошибка /forget для %s", user.id)
            self.bot.reply_to(message, "Не удалось очистить память. Попробуй позже.")

    def handle_text(self, message: types.Message) -> None:
        user = message.from_user
        if user is None or not message.text:
            return

        user_text = message.text.strip()
        if not user_text:
            return

        logger.info(
            "[handle_text] start user_id=%s username=%s text_len=%s text_preview=%r",
            user.id,
            user.username,
            len(user_text),
            user_text[:120],
        )
        self.bot.send_chat_action(message.chat.id, "typing")
        self._last_dog_result.clear()

        try:
            # В Pinecone — только оригинальный текст пользователя.
            # Ответы бота, tool-результаты и команды не сохраняются.
            if user_text.startswith("/"):
                logger.info(
                    "[handle_text] skip save: command text user_id=%s",
                    user.id,
                )
            else:
                logger.info("[handle_text] step=save_memory user_id=%s", user.id)
                save_result = self.memory.save_user_message(
                    text=user_text,
                    telegram_id=user.id,
                    metadata={
                        "username": user.username or "",
                        "first_name": user.first_name or "",
                    },
                )
                logger.info(
                    "[handle_text] step=save_memory ok result=%s",
                    save_result,
                )

            logger.info("[handle_text] step=recall_memory user_id=%s", user.id)
            documents = self.memory.recall(
                query_text=user_text,
                telegram_id=user.id,
                top_k=MEMORY_TOP_K,
            )
            memory_context = self.memory.format_context(documents)
            logger.info(
                "[handle_text] step=recall_memory ok docs=%s context_len=%s",
                len(documents),
                len(memory_context),
            )

            user_name = self._get_user_display_name(user)
            logger.info("[handle_text] step=run_agent user_id=%s", user.id)
            reply_text = self._run_agent(
                user_message=user_text,
                memory_context=memory_context,
                user_name=user_name,
            )
            logger.info(
                "[handle_text] step=run_agent ok reply_len=%s last_dog_result_keys=%s",
                len(reply_text),
                list(self._last_dog_result.keys()),
            )

            sent_photo = self._send_dog_tool_photo(message)
            if not sent_photo:
                logger.info("[handle_text] step=reply_to user_id=%s", user.id)
                self.bot.reply_to(message, reply_text)
            else:
                # Картинка + caption уже ушли; короткий текстовый follow-up опционален.
                logger.info(
                    "[handle_text] photo+caption sent, skip duplicate long reply",
                )
            logger.info("[handle_text] done user_id=%s", user.id)
        except Exception:
            logger.exception(
                "[handle_text] FAILED user_id=%s text_preview=%r",
                user.id,
                user_text[:120],
            )
            self.bot.reply_to(
                message,
                "Произошла ошибка при обработке сообщения. Попробуй ещё раз.",
            )

    def _send_dog_tool_photo(self, message: types.Message) -> bool:
        """
        Отправляет картинку из dogImageTool / dogImageAnalyzerTool.

        Для анализатора caption = ответ ChatGPT.
        """
        image_url = self._last_dog_result.get("image_url")
        if not image_url:
            return False

        caption = (
            self._last_dog_result.get("caption")
            or self._last_dog_result.get("breed_analysis")
            or "Случайная собака"
        )
        # Telegram caption max 1024.
        if len(caption) > 1024:
            caption = caption[:1023].rstrip() + "…"

        tool_name = self._last_dog_result.get("tool", "dog_tool")
        if tool_name == "dogImageAnalyzerTool":
            image_log.info(
                "[Telegram] send_photo analyzer | url={url} caption_len={caption_len}",
                url=image_url,
                caption_len=len(caption),
            )
        else:
            logger.info(
                "[handle_text] step=send_photo tool=%s url=%s caption_len=%s",
                tool_name,
                image_url,
                len(caption),
            )
        try:
            # parse_mode=HTML у бота — экранируем caption, чтобы не сломать разметку.
            self.bot.send_photo(
                message.chat.id,
                photo=image_url,
                caption=html.escape(caption),
                reply_to_message_id=message.message_id,
            )
            if tool_name == "dogImageAnalyzerTool":
                image_log.success(
                    "[Telegram] send_photo analyzer ok | url={url}",
                    url=image_url,
                )
            else:
                logger.info("[handle_text] step=send_photo ok tool=%s", tool_name)
            return True
        except Exception:
            if tool_name == "dogImageAnalyzerTool":
                image_log.exception(
                    "[Telegram] send_photo analyzer FAILED | url={url}",
                    url=image_url,
                )
            else:
                logger.exception(
                    "[handle_text] step=send_photo FAILED tool=%s url=%s",
                    tool_name,
                    image_url,
                )
            # Fallback: текст, если фото не ушло.
            try:
                self.bot.reply_to(
                    message,
                    f"{caption}\n\n{image_url}",
                )
            except Exception:
                logger.exception("[handle_text] caption fallback FAILED")
            return False


    def run(self) -> None:
        logger.info(
            "Haystack-помощник запущен. index=%s base_url=%s model=%s",
            self.memory._index_name,
            self._openai_base_url,
            self._llm_model,
        )
        logger.info("Ожидание сообщений Telegram (infinity_polling)...")
        try:
            self.bot.infinity_polling(timeout=60, long_polling_timeout=60)
        except Exception:
            logger.exception("[run] infinity_polling упал")
            raise



def main() -> None:
    bot = HaystackAssistantBot()
    bot.run()


if __name__ == "__main__":
    main()
