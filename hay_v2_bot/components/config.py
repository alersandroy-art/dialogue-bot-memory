"""Конфигурация из переменных окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Корень репозитория (hay_v2_bot/../).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Namespace-префиксы: не пересекаются с hay/ (hay_user_*) и bot.py (user_*).
MEMORY_NAMESPACE_PREFIX = "hay_v2_user_"
DOCS_NAMESPACE_PREFIX = "hay_v2_docs_"
FORBIDDEN_SHARED_NAMESPACES = frozenset({"", "default", "shared", "common"})

EMBEDDING_DIMENSION = 1536
COSINE_SIMILARITY_THRESHOLD = 0.85
MEMORY_TOP_K = 5
DOCS_TOP_K = 5

# Форматы, которые Docling стабильно обрабатывает в этом боте.
SUPPORTED_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".docx",
        ".doc",
        ".pptx",
        ".html",
        ".htm",
        ".md",
        ".txt",
        ".xlsx",
        ".asciidoc",
    },
)

# Запасной id HF-токенизатора (ingestion использует tiktoken, без Hugging Face).
CHUNKER_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"

# Лимит Bot API на скачивание файла через getFile (~20 MiB).
TELEGRAM_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class Settings:
    """Настройки приложения."""

    telegram_bot_token: str
    openai_api_key: str
    openai_base_url: str
    pinecone_index_name: str
    llm_model: str
    vision_model: str
    embedding_model: str
    # Опционально: HTTP/SOCKS прокси до api.telegram.org (если Telegram недоступен напрямую).
    telegram_proxy: str | None = None

    @property
    def http_client_kwargs(self) -> dict:
        return {"trust_env": False}


def assert_proxy_base_url(base_url: str) -> None:
    """Все вызовы OpenAI должны идти через OPENAI_BASE_URL (ProxyAPI)."""
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("OPENAI_BASE_URL пуст — укажите прокси (ProxyAPI).")
    if "api.openai.com" in normalized.lower():
        raise ValueError(
            f"Запрещён прямой доступ к OpenAI API: {normalized}. "
            "Укажите прокси в OPENAI_BASE_URL.",
        )


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Переменная окружения {name} не задана. Добавьте её в .env.")
    return value


def load_settings() -> Settings:
    """Загружает .env из корня проекта и собирает Settings."""
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()

    openai_base_url = require_env("OPENAI_BASE_URL")
    assert_proxy_base_url(openai_base_url)

    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    vision_model = os.getenv("VISION_MODEL", llm_model).strip() or llm_model
    embedding_model = (
        os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
        or "text-embedding-3-small"
    )

    telegram_proxy = (
        os.getenv("TELEGRAM_PROXY", "").strip()
        or os.getenv("HTTPS_PROXY", "").strip()
        or os.getenv("HTTP_PROXY", "").strip()
        or None
    )

    return Settings(
        telegram_bot_token=require_env("TELEGRAM_BOT_TOKEN"),
        openai_api_key=require_env("OPENAI_API_KEY"),
        openai_base_url=openai_base_url,
        pinecone_index_name=require_env("PINECONE_INDEX_NAME"),
        llm_model=llm_model,
        vision_model=vision_model,
        embedding_model=embedding_model,
        telegram_proxy=telegram_proxy,
    )
