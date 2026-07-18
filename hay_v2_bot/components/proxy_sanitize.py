"""Обход системного SOCKS-прокси и просроченного HF-токена (Docling / Hugging Face)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "SOCKS_PROXY",
    "socks_proxy",
    "SOCKS4_PROXY",
    "socks4_proxy",
    "SOCKS5_PROXY",
    "socks5_proxy",
)

_hf_configured = False


def sanitize_incompatible_proxies(*, clear_all_proxies: bool = True) -> list[str]:
    """
    Убирает proxy из окружения процесса.

    httpx не понимает socks4:// → Docling/HF падают при download_models.
    Telegram ходит через telebot + TELEGRAM_PROXY, не через эти env.
    """
    cleared: list[str] = []
    for key in _PROXY_ENV_KEYS:
        value = os.environ.get(key)
        if value is None or not str(value).strip():
            continue
        text = str(value).strip()
        lowered = text.lower()
        if clear_all_proxies or lowered.startswith("socks"):
            cleared.append(f"{key}={text}")
            del os.environ[key]

    if cleared:
        logger.warning(
            "Очищены proxy-переменные окружения (для Docling/Hugging Face): %s. "
            "Для Telegram задайте TELEGRAM_PROXY в .env.",
            ", ".join(cleared),
        )
    return cleared


def configure_huggingface_auth_for_docling() -> None:
    """
    Просроченный локальный HF-токен («home-download») даёт 401 на публичных моделях.

    Ставим HF_HUB_DISABLE_IMPLICIT_TOKEN до/после импорта constants.
    Явный валидный HF_TOKEN из .env — по-прежнему поддерживается.
    """
    explicit = (
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
        or os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
    )
    if explicit:
        os.environ["HF_TOKEN"] = explicit
        os.environ.pop("HF_HUB_DISABLE_IMPLICIT_TOKEN", None)
        _sync_hf_disable_implicit_constant(False)
        logger.info("Hugging Face: используем HF_TOKEN из окружения/.env")
        return

    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    _sync_hf_disable_implicit_constant(True)
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if key in os.environ and not os.environ.get(key, "").strip():
            del os.environ[key]
    logger.info(
        "Hugging Face: анонимная загрузка (игнор просроченного cached token). "
        "При необходимости задайте свежий HF_TOKEN в .env."
    )


def _sync_hf_disable_implicit_constant(disabled: bool) -> None:
    """constants читаются при импорте — обновляем и runtime-флаг."""
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HUB_DISABLE_IMPLICIT_TOKEN = disabled
    except ImportError:
        pass


def configure_huggingface_http_no_proxy() -> None:
    """httpx-клиент Hugging Face без чтения proxy из env (trust_env=False)."""
    global _hf_configured
    if _hf_configured:
        # Всё равно сбросить клиент — мог подхватить proxy раньше.
        try:
            from huggingface_hub.utils import _http as hf_http

            hf_http._GLOBAL_CLIENT = None
            if hasattr(hf_http, "_GLOBAL_ASYNC_CLIENT"):
                hf_http._GLOBAL_ASYNC_CLIENT = None
        except ImportError:
            pass
        return

    try:
        import httpx
        from huggingface_hub.utils import _http as hf_http
    except ImportError:
        logger.info("huggingface_hub недоступен — пропуск configure_huggingface_http_no_proxy")
        return

    def _client_factory() -> httpx.Client:
        return httpx.Client(
            timeout=httpx.Timeout(None),
            follow_redirects=True,
            trust_env=False,
        )

    def _async_client_factory():
        return httpx.AsyncClient(
            timeout=httpx.Timeout(None),
            follow_redirects=True,
            trust_env=False,
        )

    hf_http.set_client_factory(_client_factory)
    hf_http.set_async_client_factory(_async_client_factory)
    hf_http._GLOBAL_CLIENT = None
    if hasattr(hf_http, "_GLOBAL_ASYNC_CLIENT"):
        hf_http._GLOBAL_ASYNC_CLIENT = None

    _hf_configured = True
    logger.info("Hugging Face httpx: trust_env=False (socks4 из системы игнорируется)")


def configure_huggingface_download_mode() -> None:
    """
    XET-бэкенд HF часто падает сетью (xet-read-token).
    Отключаем XET → классическое HTTP-скачивание файлов моделей.
    """
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        import huggingface_hub.constants as hf_constants

        if hasattr(hf_constants, "HF_HUB_DISABLE_XET"):
            hf_constants.HF_HUB_DISABLE_XET = True
    except ImportError:
        pass
    logger.info("Hugging Face: HF_HUB_DISABLE_XET=1 (без xet-read-token)")


def prepare_docling_network() -> None:
    """Вызывать перед Docling convert / download моделей."""
    configure_huggingface_download_mode()
    configure_huggingface_auth_for_docling()
    sanitize_incompatible_proxies(clear_all_proxies=True)
    configure_huggingface_http_no_proxy()
