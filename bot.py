"""
Telegram-бот-помощник с долговременной памятью на базе Pinecone.

Использует pyTelegramBotAPI для общения с пользователем и PineconeManager
для хранения всей информации о пользователе в векторной базе.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import httpx
import telebot
from dotenv import load_dotenv
from openai import OpenAI
from telebot import types

from pinecone_manager import PineconeManager

# Количество фрагментов памяти, подставляемых в контекст ответа LLM.
MEMORY_TOP_K = 5

# Системный промпт: бот опирается на сохранённые факты о пользователе.
SYSTEM_PROMPT = """Ты дружелюбный Telegram-помощник с долговременной памятью.

Имя пользователя в Telegram: {user_name}

Известные факты о пользователе из долговременной памяти:
{memory_context}

Правила:
- Отвечай на русском языке, если пользователь пишет по-русски.
- Используй факты из памяти, когда они релевантны вопросу.
- Если в памяти нет нужной информации — честно скажи об этом.
- Будь кратким и полезным.
- Не выдумывай факты о пользователе, которых нет в памяти.
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class AssistantBot:
    """Telegram-бот, который хранит данные пользователя через PineconeManager."""

    def __init__(self) -> None:
        load_dotenv()

        self._token = self._require_env("TELEGRAM_BOT_TOKEN")
        self._llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self._openai_api_key = self._require_env("OPENAI_API_KEY")
        self._openai_base_url = self._require_env("OPENAI_BASE_URL")

        # Менеджер Pinecone — единая точка работы с памятью пользователя.
        self.memory = PineconeManager(load_env=False)

        # Клиент LLM через ProxyAPI (trust_env=False — без системного SOCKS-прокси).
        self._llm = OpenAI(
            api_key=self._openai_api_key,
            base_url=self._openai_base_url,
            http_client=httpx.Client(trust_env=False),
        )

        self.bot = telebot.TeleBot(self._token, parse_mode="HTML")
        self._register_handlers()

    @staticmethod
    def _require_env(name: str) -> str:
        """Возвращает обязательную переменную окружения."""
        value = os.getenv(name, "").strip()
        if not value:
            raise ValueError(
                f"Переменная окружения {name} не задана. "
                f"Добавьте её в файл .env.",
            )
        return value

    def _register_handlers(self) -> None:
        """Регистрирует обработчики команд и сообщений."""
        self.bot.message_handler(commands=["start"])(self.handle_start)
        self.bot.message_handler(commands=["help"])(self.handle_help)
        self.bot.message_handler(commands=["memory"])(self.handle_memory)
        self.bot.message_handler(commands=["forget"])(self.handle_forget)
        self.bot.message_handler(content_types=["text"])(self.handle_text)

    @staticmethod
    def _get_user_display_name(user: types.User) -> str:
        """Формирует отображаемое имя пользователя."""
        parts = [user.first_name or "", user.last_name or ""]
        name = " ".join(part for part in parts if part).strip()
        if user.username:
            return f"{name} (@{user.username})" if name else f"@{user.username}"
        return name or "Пользователь"

    @staticmethod
    def _utc_timestamp() -> str:
        """Возвращает текущее время в UTC в ISO-формате."""
        return datetime.now(timezone.utc).isoformat()

    def _save_user_message_to_memory(
        self,
        user: types.User,
        user_message: str,
    ) -> dict:
        """
        Сохраняет в Pinecone только оригинальный текст сообщения пользователя.

        В векторе — исключительно user_message. Служебные данные
        (username, timestamp и т.д.) хранятся только в metadata.

        Args:
            user: Объект пользователя Telegram.
            user_message: Оригинальный текст сообщения пользователя.

        Returns:
            Результат операции save_to_long_term_memory.
        """
        return self.memory.save_to_long_term_memory(
            text=user_message,
            telegram_id=user.id,
            metadata={
                "type": "user_message",
                "role": "user",
                "username": user.username or "",
                "first_name": user.first_name or "",
                "last_name": user.last_name or "",
                "telegram_id": str(user.id),
                "timestamp": self._utc_timestamp(),
            },
            on_duplicate="update",
        )

    def _build_memory_context(self, memories: list[dict]) -> str:
        """Собирает текстовый блок фактов из найденных фрагментов памяти."""
        if not memories:
            return "Пока нет сохранённых данных о пользователе."

        lines: list[str] = []
        for index, item in enumerate(memories, start=1):
            metadata = item.get("metadata") or {}

            # В контекст попадают только оригинальные сообщения пользователя.
            message_type = metadata.get("type", "")
            message_role = metadata.get("role", "")
            if message_type in {"profile", "message"} and message_role != "user":
                continue
            if message_role == "assistant":
                continue

            text = metadata.get("text", "").strip()
            if not text:
                continue
            score = item.get("score")
            if score is not None:
                lines.append(f"{index}. {text} (релевантность: {score:.2f})")
            else:
                lines.append(f"{index}. {text}")

        if not lines:
            return "Пока нет сохранённых данных о пользователе."

        return "\n".join(lines)

    def _generate_reply(
        self,
        user_message: str,
        memory_context: str,
        user_name: str,
    ) -> str:
        """
        Генерирует ответ помощника через LLM.

        Args:
            user_message: Сообщение пользователя.
            memory_context: Контекст из Pinecone.
            user_name: Отображаемое имя пользователя.

        Returns:
            Текст ответа бота.
        """
        response = self._llm.chat.completions.create(
            model=self._llm_model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(
                        user_name=user_name,
                        memory_context=memory_context,
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
        )
        content = response.choices[0].message.content
        if not content:
            return "Извините, не удалось сформировать ответ. Попробуйте ещё раз."
        return content.strip()

    def handle_start(self, message: types.Message) -> None:
        """Обработчик команды /start."""
        user = message.from_user
        if user is None:
            return

        welcome_text = (
            f"Привет, <b>{user.first_name or 'друг'}</b>! 👋\n\n"
            "Я бот-помощник с долговременной памятью. "
            "Я запоминаю информацию о тебе и использую её в наших разговорах.\n\n"
            "Команды:\n"
            "/help — справка\n"
            "/memory — что я помню о тебе\n"
            "/forget — очистить мою память о тебе\n\n"
            "Просто напиши сообщение — и начнём общение!"
        )
        self.bot.reply_to(message, welcome_text)

    def handle_help(self, message: types.Message) -> None:
        """Обработчик команды /help."""
        help_text = (
            "<b>Справка по боту</b>\n\n"
            "В векторной базе Pinecone сохраняются только твои "
            "оригинальные текстовые сообщения.\n\n"
            "Ответы бота и служебные шаблоны в память не записываются.\n\n"
            "Перед сохранением проверяется косинусное сходство, "
            "чтобы не дублировать одинаковые фрагменты.\n\n"
            "<b>Команды:</b>\n"
            "/start — начать работу\n"
            "/memory — показать релевантную память\n"
            "/forget — полностью очистить память о тебе"
        )
        self.bot.reply_to(message, help_text)

    def handle_memory(self, message: types.Message) -> None:
        """Обработчик команды /memory — показывает сохранённые факты."""
        user = message.from_user
        if user is None:
            return

        self.bot.send_chat_action(message.chat.id, "typing")

        try:
            query = message.text.replace("/memory", "").strip()
            if not query:
                query = "Вся информация о пользователе и наши разговоры"

            memories = self.memory.recall_user_memory(
                query_text=query,
                telegram_id=user.id,
                top_k=MEMORY_TOP_K,
            )
            context = self._build_memory_context(memories)
            namespace = self.memory.build_user_namespace(user.id)

            reply = (
                f"<b>Память о тебе</b> (namespace: <code>{namespace}</code>):\n\n"
                f"{context}"
            )
            self.bot.reply_to(message, reply)
        except Exception:
            logger.exception("Ошибка чтения памяти для пользователя %s", user.id)
            self.bot.reply_to(
                message,
                "Не удалось получить память. Попробуй позже.",
            )

    def handle_forget(self, message: types.Message) -> None:
        """Обработчик команды /forget — очищает память пользователя."""
        user = message.from_user
        if user is None:
            return

        try:
            self.memory.clear_user_memory(telegram_id=user.id)
            self.bot.reply_to(
                message,
                "Память о тебе полностью очищена. Можем начать с чистого листа.",
            )
        except Exception:
            logger.exception("Ошибка очистки памяти для пользователя %s", user.id)
            self.bot.reply_to(
                message,
                "Не удалось очистить память. Попробуй позже.",
            )

    def handle_text(self, message: types.Message) -> None:
        """Обработчик текстовых сообщений — диалог с учётом памяти."""
        user = message.from_user
        if user is None or not message.text:
            return

        user_text = message.text.strip()
        if not user_text:
            return

        self.bot.send_chat_action(message.chat.id, "typing")

        try:
            # В Pinecone сохраняется только оригинальный текст пользователя.
            self._save_user_message_to_memory(user, user_text)

            # Ищем релевантные фрагменты памяти для контекста ответа.
            memories = self.memory.recall_user_memory(
                query_text=user_text,
                telegram_id=user.id,
                top_k=MEMORY_TOP_K,
            )
            memory_context = self._build_memory_context(memories)
            user_name = self._get_user_display_name(user)

            # Генерируем ответ с учётом памяти.
            reply_text = self._generate_reply(
                user_message=user_text,
                memory_context=memory_context,
                user_name=user_name,
            )

            self.bot.reply_to(message, reply_text)
        except Exception:
            logger.exception(
                "Ошибка обработки сообщения пользователя %s",
                user.id,
            )
            self.bot.reply_to(
                message,
                "Произошла ошибка при обработке сообщения. Попробуй ещё раз.",
            )

    def run(self) -> None:
        """Запускает бота в режиме long polling."""
        logger.info("Бот запущен. Ожидание сообщений...")
        self.bot.infinity_polling(timeout=60, long_polling_timeout=60)


def main() -> None:
    """Точка входа при запуске из командной строки."""
    assistant = AssistantBot()
    assistant.run()


if __name__ == "__main__":
    main()
