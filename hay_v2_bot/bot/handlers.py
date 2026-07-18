"""Хендлеры команд, текста и документов Telegram."""

from __future__ import annotations

import html
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from loguru import logger as image_log
from telebot import types

from ..components.config import SUPPORTED_EXTENSIONS, TELEGRAM_MAX_DOWNLOAD_BYTES
from ..components.document_store import docs_namespace, memory_namespace

if TYPE_CHECKING:
    from .assistant import HaystackV2Bot

logger = logging.getLogger(__name__)


class TelegramFileTooBigError(ValueError):
    """Файл превышает лимит скачивания Bot API (~20 MB)."""

    def __init__(self, filename: str, size_bytes: int | None = None) -> None:
        self.filename = filename
        self.size_bytes = size_bytes
        size_mb = (size_bytes / (1024 * 1024)) if size_bytes else None
        detail = f" (~{size_mb:.1f} МБ)" if size_mb is not None else ""
        super().__init__(
            f"Файл «{filename}»{detail} слишком большой для скачивания через Telegram Bot API "
            f"(лимит {TELEGRAM_MAX_DOWNLOAD_BYTES // (1024 * 1024)} МБ).",
        )


def format_file_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "неизвестно"
    mb = num_bytes / (1024 * 1024)
    if mb < 0.1:
        return f"{num_bytes / 1024:.0f} КБ"
    return f"{mb:.1f} МБ"


def assert_telegram_download_allowed(document: types.Document) -> None:
    """Жёсткий отказ ДО getFile / download_file при file_size > 20 МБ."""
    size = getattr(document, "file_size", None)
    filename = document.file_name or "document"
    if size is None:
        logger.info(
            "[download] file_size отсутствует, file=%s — проверим на getFile",
            filename,
        )
        return
    size_int = int(size)
    logger.info(
        "[download] precheck file=%s size=%s (%s) limit=%s",
        filename,
        size_int,
        format_file_size(size_int),
        TELEGRAM_MAX_DOWNLOAD_BYTES,
    )
    if size_int > TELEGRAM_MAX_DOWNLOAD_BYTES:
        raise TelegramFileTooBigError(filename=filename, size_bytes=size_int)


def is_document_too_big(document: types.Document) -> bool:
    """True, если Telegram передал file_size и он больше лимита Bot API."""
    size = getattr(document, "file_size", None)
    return size is not None and int(size) > TELEGRAM_MAX_DOWNLOAD_BYTES


def telegram_file_too_big_message(filename: str, size_bytes: int | None = None) -> str:
    limit_mb = TELEGRAM_MAX_DOWNLOAD_BYTES // (1024 * 1024)
    size_part = (
        f" Размер: {format_file_size(size_bytes)}."
        if size_bytes is not None
        else ""
    )
    return (
        f"Документ «{html.escape(filename)}» больше {limit_mb} МБ."
        f"{size_part}\n"
        "Скачиваться не будет. Пришлите файл до 20 МБ."
    )


def register_handlers(bot_app: "HaystackV2Bot") -> None:
    """Регистрирует все message_handler на TeleBot."""
    tb = bot_app.bot
    tb.message_handler(commands=["start"])(bot_app.handle_start)
    tb.message_handler(commands=["help"])(bot_app.handle_help)
    tb.message_handler(commands=["memory"])(bot_app.handle_memory)
    tb.message_handler(commands=["forget"])(bot_app.handle_forget)
    tb.message_handler(content_types=["document"])(bot_app.handle_document)
    tb.message_handler(content_types=["text"])(bot_app.handle_text)


def get_user_display_name(user: types.User) -> str:
    parts = [user.first_name or "", user.last_name or ""]
    name = " ".join(part for part in parts if part).strip()
    if user.username:
        return f"{name} (@{user.username})" if name else f"@{user.username}"
    return name or "Пользователь"


def extension_of(filename: str) -> str:
    return Path(filename or "").suffix.lower()


def is_supported_document(filename: str) -> bool:
    return extension_of(filename) in SUPPORTED_EXTENSIONS


def send_dog_tool_photo(
    bot: object,
    message: types.Message,
    last_dog_result: dict,
) -> bool:
    """Отправляет картинку из dogImageTool / dogImageAnalyzerTool."""
    image_url = last_dog_result.get("image_url")
    if not image_url:
        return False

    caption = (
        last_dog_result.get("caption")
        or last_dog_result.get("breed_analysis")
        or "Случайная собака"
    )
    if len(caption) > 1024:
        caption = caption[:1023].rstrip() + "…"

    tool_name = last_dog_result.get("tool", "dog_tool")
    send_photo: Callable = bot.send_photo  # type: ignore[attr-defined]
    reply_to: Callable = bot.reply_to  # type: ignore[attr-defined]

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
        send_photo(
            message.chat.id,
            photo=image_url,
            caption=html.escape(caption),
            reply_to_message_id=message.message_id,
        )
        return True
    except Exception:
        logger.exception("[send_dog_tool_photo] FAILED tool=%s", tool_name)
        try:
            reply_to(message, f"{caption}\n\n{image_url}")
        except Exception:
            logger.exception("[send_dog_tool_photo] caption fallback FAILED")
        return False


def download_telegram_document(
    bot: object,
    document: types.Document,
    dest_dir: Path,
) -> Path:
    """Скачивает файл из Telegram. При размере > 20 МБ getFile не вызывается."""
    assert_telegram_download_allowed(document)

    file_name = document.file_name or f"document_{document.file_id}"
    safe_name = Path(file_name).name
    dest = dest_dir / safe_name

    # Повторная страховка: сюда не должны попадать файлы > лимита.
    if is_document_too_big(document):
        raise TelegramFileTooBigError(
            filename=file_name,
            size_bytes=int(document.file_size),
        )

    get_file: Callable = bot.get_file  # type: ignore[attr-defined]
    download_file: Callable = bot.download_file  # type: ignore[attr-defined]

    try:
        file_info = get_file(document.file_id)
    except Exception as exc:
        text = str(exc).lower()
        if "file is too big" in text or "too big" in text:
            raise TelegramFileTooBigError(
                filename=file_name,
                size_bytes=getattr(document, "file_size", None),
            ) from exc
        raise

    payload = download_file(file_info.file_path)
    dest.write_bytes(payload)
    logger.info(
        "[download] saved=%s size=%s",
        dest,
        dest.stat().st_size,
    )
    return dest


def make_temp_dir() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(prefix="hay_v2_doc_")


def format_start_text(first_name: str, telegram_id: int) -> str:
    return (
        f"Привет, <b>{html.escape(first_name or 'друг')}</b>!\n\n"
        "Я персональный помощник <b>v2</b> на Haystack: память, документы (Docling) "
        "и инструменты.\n"
        f"Память: <code>{memory_namespace(telegram_id)}</code>\n"
        f"Документы: <code>{docs_namespace(telegram_id)}</code>\n\n"
        "Могу:\n"
        "• помнить диалог (Pinecone)\n"
        "• разбирать PDF/DOCX и обсуждать файлы\n"
        "• факт / картинка / анализ породы собаки\n\n"
        "Команды: /help, /memory, /forget"
    )


HELP_TEXT = (
    "<b>Справка (hay_v2_bot)</b>\n\n"
    "В Pinecone сохраняется <b>только текст</b> твоих сообщений "
    "(ответы бота не пишутся) и <b>чанки загруженных файлов</b>.\n\n"
    "Каждый пользователь — в своих namespace:\n"
    "• память — <code>hay_v2_user_&lt;id&gt;</code>\n"
    "• документы — <code>hay_v2_docs_&lt;id&gt;</code>\n\n"
    "Пришли PDF, DOCX и др. (до <b>20 МБ</b>) — разберу через Docling, сохраню в базу "
    "и пришлю короткое резюме.\n\n"
    "Примеры:\n"
    "• «Запомни, что меня зовут Алекс»\n"
    "• «О чём мой файл?»\n"
    "• «Расскажи факт о собаках»\n\n"
    "<b>Команды:</b>\n"
    "/start — начать\n"
    "/memory — что я помню\n"
    "/forget — очистить память и документы"
)
