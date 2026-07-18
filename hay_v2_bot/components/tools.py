"""
Инструменты генерации для Haystack Agent.

- dogFactTool — случайный факт (dogapi.dog)
- dogImageTool — случайная картинка (dog.ceo)
- dogImageAnalyzerTool — картинка → ChatGPT vision → {image_url, caption}
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from haystack.tools import Tool
from loguru import logger as image_log
from openai import OpenAI

logger = logging.getLogger(__name__)

DOG_FACT_URL = "https://dogapi.dog/api/v2/facts"
DOG_IMAGE_URL = "https://dog.ceo/api/breeds/image/random"
TELEGRAM_CAPTION_MAX = 1024

VISION_PROMPT = """Ты эксперт по породам собак. Посмотри на фотографию и ответь на русском языке.

Структура ответа:
1. Порода (или наиболее вероятная порода / смесь) — кратко и уверенно.
2. Характерные внешние признаки, по которым ты определил породу.
3. Краткая предыстория: как и зачем появилась эта порода (страна, период, назначение).
4. 1–2 интересных факта о породе.

Если порода неуверенна — напиши 2–3 наиболее вероятных варианта.
Будь информативным, но без воды. Без markdown-ссылок на картинку."""


def _truncate_caption(text: str, limit: int = TELEGRAM_CAPTION_MAX) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def build_generation_tools(
    openai_client: OpenAI,
    vision_model: str,
    last_result: dict[str, Any],
) -> list[Tool]:
    """
    Собирает тулы, которыми пользуется Agent при генерации ответа.

    last_result — общий словарь с хендлером Telegram (image_url, caption, tool).
    """
    client_base = str(getattr(openai_client, "base_url", "") or "").rstrip("/")
    if not client_base:
        raise ValueError("openai_client должен быть создан с base_url (OPENAI_BASE_URL).")
    if "api.openai.com" in client_base.lower():
        raise ValueError(
            f"openai_client указывает на прямой OpenAI API ({client_base}). "
            "Нужен OPENAI_BASE_URL.",
        )

    def fetch_random_dog_image_url(caller: str = "dogImageTool") -> str:
        if caller == "dogImageAnalyzerTool":
            image_log.info("[dogImageAnalyzerTool] fetch start | url={url}", url=DOG_IMAGE_URL)
        else:
            logger.info("[%s] fetch start url=%s", caller, DOG_IMAGE_URL)

        with httpx.Client(timeout=30.0, trust_env=False) as client:
            response = client.get(DOG_IMAGE_URL)
            response.raise_for_status()
            payload = response.json()

        image_url = str(payload.get("message") or "").strip()
        if not image_url:
            raise ValueError("dog.ceo вернул пустой URL картинки")

        if caller == "dogImageAnalyzerTool":
            image_log.success(
                "[dogImageAnalyzerTool] fetch ok | image_url={image_url}",
                image_url=image_url,
            )
        else:
            logger.info("[%s] fetch ok image_url=%s", caller, image_url)
        return image_url

    def analyze_image_with_chatgpt(image_url: str, focus: str = "") -> str:
        image_log.info(
            "[dogImageAnalyzerTool] ChatGPT vision start | model={model} "
            "base_url={base_url} image_url={image_url}",
            model=vision_model,
            base_url=client_base,
            image_url=image_url,
        )
        user_prompt = VISION_PROMPT
        if focus and focus.strip():
            user_prompt += f"\n\nДополнительный акцент пользователя: {focus.strip()}"

        vision = openai_client.chat.completions.create(
            model=vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            temperature=0.4,
        )
        analysis = (vision.choices[0].message.content or "").strip()
        if not analysis:
            analysis = "Не удалось получить описание от ChatGPT."
        else:
            image_log.success(
                "[dogImageAnalyzerTool] ChatGPT vision ok | analysis_len={length}",
                length=len(analysis),
            )
        return analysis

    def dog_fact_tool_fn() -> str:
        try:
            with httpx.Client(timeout=20.0, trust_env=False) as client:
                response = client.get(DOG_FACT_URL)
                response.raise_for_status()
                payload = response.json()

            facts = payload.get("data") or []
            if not facts:
                return "Не удалось получить факт о собаках: пустой ответ API."

            body = facts[0].get("attributes", {}).get("body", "").strip()
            if not body:
                return "API вернул факт без текста."

            result = {
                "tool": "dogFactTool",
                "fact": body,
                "message": f"Случайный факт о собаках: {body}",
            }
            last_result.clear()
            last_result.update(result)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            logger.exception("[dogFactTool] FAILED")
            return json.dumps({"tool": "dogFactTool", "error": str(exc)}, ensure_ascii=False)

    def dog_image_tool_fn() -> str:
        try:
            image_url = fetch_random_dog_image_url(caller="dogImageTool")
            result = {
                "tool": "dogImageTool",
                "image_url": image_url,
                "source": "dog.ceo",
                "caption": "Случайная собака с dog.ceo",
                "message": f"Случайная картинка собаки: {image_url}",
            }
            last_result.clear()
            last_result.update(result)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            logger.exception("[dogImageTool] FAILED")
            return json.dumps({"tool": "dogImageTool", "error": str(exc)}, ensure_ascii=False)

    def dog_image_analyzer_tool_fn(focus: str = "порода и история") -> str:
        try:
            image_url = fetch_random_dog_image_url(caller="dogImageAnalyzerTool")
            analysis = analyze_image_with_chatgpt(image_url, focus=focus)
            caption = _truncate_caption(analysis)
            result = {
                "tool": "dogImageAnalyzerTool",
                "image_url": image_url,
                "source": "dog.ceo",
                "breed_analysis": analysis,
                "caption": caption,
                "message": (
                    "Анализ породы готов. Пользователю нужно показать "
                    f"картинку {image_url} и описание породы."
                ),
            }
            last_result.clear()
            last_result.update(result)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            image_log.exception("[dogImageAnalyzerTool] pipeline FAILED")
            return json.dumps(
                {"tool": "dogImageAnalyzerTool", "error": str(exc)},
                ensure_ascii=False,
            )

    tools: list[Tool] = [
        Tool(
            name="dogFactTool",
            description=(
                "Получает случайный факт о собаках из бесплатного API dogapi.dog. "
                "Вызывай, когда пользователь просит факт о собаках."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            function=dog_fact_tool_fn,
        ),
        Tool(
            name="dogImageTool",
            description=(
                "Получает только URL случайной картинки собаки с dog.ceo "
                "(без анализа породы). Вызывай, если нужна просто картинка."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            function=dog_image_tool_fn,
        ),
        Tool(
            name="dogImageAnalyzerTool",
            description=(
                "Получает случайную картинку собаки с dog.ceo, отправляет её в "
                "ChatGPT (vision) и возвращает один объект: image_url + caption "
                "(описание породы и предыстория)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "Акцент для описания (характер, история породы).",
                    },
                },
                "required": [],
            },
            function=dog_image_analyzer_tool_fn,
        ),
    ]
    logger.info(
        "Registered generation tools: %s",
        ", ".join(tool.name for tool in tools),
    )
    return tools


# Обратная совместимость со старым именем.
build_dog_tools = build_generation_tools
