"""
Точка входа hay_v2_bot.

Из корня репозитория:
    python -m hay_v2_bot.main
    python hay_v2_bot/main.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# До импорта huggingface_hub: не слать просроченный cached token (401 home-download).
# XET часто рвётся по сети — сразу классическое скачивание.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Запуск как скрипта: python hay_v2_bot/main.py
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hay_v2_bot.components.proxy_sanitize import (  # noqa: E402
    configure_huggingface_http_no_proxy,
    prepare_docling_network,
    sanitize_incompatible_proxies,
)

sanitize_incompatible_proxies(clear_all_proxies=True)
configure_huggingface_http_no_proxy()

from hay_v2_bot.bot.assistant import HaystackV2Bot  # noqa: E402
from hay_v2_bot.components.logging_setup import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    prepare_docling_network()
    bot = HaystackV2Bot()
    # После load_dotenv (.env) — ещё раз: proxy + HF auth.
    prepare_docling_network()
    bot.run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("hay_v2_bot упал при старте")
        raise
