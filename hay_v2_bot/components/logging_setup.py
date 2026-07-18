"""Настройка логирования: консоль + ротация файлов в logs/hay_v2_bot/."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "logs" / "hay_v2_bot"
APP_LOG_NAME = "app.log"
ERROR_LOG_NAME = "errors.log"
VISION_LOG_NAME = "vision.log"

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# 5 MB × 5 файлов на каждый лог.
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 5

_configured = False


def _parse_level(value: str | None) -> int:
    name = (value or "INFO").strip().upper()
    return getattr(logging, name, logging.INFO)


def setup_logging(level: str | int | None = None) -> Path:
    """
    Включает логирование в консоль и в файлы.

    Файлы:
      logs/hay_v2_bot/app.log     — INFO и выше
      logs/hay_v2_bot/errors.log  — только ERROR/CRITICAL
      logs/hay_v2_bot/vision.log  — Loguru (dogImageAnalyzer / vision)

    Уровень: аргумент level, иначе LOG_LEVEL из окружения, иначе INFO.
    """
    global _configured
    if _configured:
        return LOG_DIR

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if isinstance(level, int):
        log_level = level
    else:
        log_level = _parse_level(level if level is not None else os.getenv("LOG_LEVEL"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    app_file = RotatingFileHandler(
        LOG_DIR / APP_LOG_NAME,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    app_file.setLevel(log_level)
    app_file.setFormatter(formatter)
    root.addHandler(app_file)

    error_file = RotatingFileHandler(
        LOG_DIR / ERROR_LOG_NAME,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(formatter)
    root.addHandler(error_file)

    # Меньше шума от сторонних HTTP-библиотек.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    _setup_loguru_vision()

    _configured = True
    logging.getLogger(__name__).info(
        "Логирование включено: console + %s + %s (level=%s)",
        LOG_DIR / APP_LOG_NAME,
        LOG_DIR / ERROR_LOG_NAME,
        logging.getLevelName(log_level),
    )
    return LOG_DIR


def _setup_loguru_vision() -> None:
    """Дублирует Loguru (vision/dog analyzer) в отдельный файл."""
    try:
        from loguru import logger as image_log
    except ImportError:
        return

    image_log.remove()
    image_log.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} [{level}] loguru: {message}",
    )
    image_log.add(
        LOG_DIR / VISION_LOG_NAME,
        level="INFO",
        rotation="5 MB",
        retention=5,
        encoding="utf-8",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {message}",
    )
