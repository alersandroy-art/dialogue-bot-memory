# Dialogue Bot Memory / Personal Assistant

Telegram-помощник с **долговременной памятью** в [Pinecone](https://www.pinecone.io/) и LLM через [ProxyAPI](https://proxyapi.ru/).

В репозитории три варианта бота:

| Вариант | Запуск | Стек |
|---------|--------|------|
| Базовый | `python bot.py` | OpenAI SDK + `PineconeManager` |
| Haystack-агент (v1) | `python hay/hay-telegram-bot.py` | Haystack Agent + Pinecone + тулы |
| Haystack v2 (модульный) | `python -m hay_v2_bot.main` | Agent + Docling + пайплайны + Pinecone |

Все варианты сохраняют в Pinecone **только текст сообщений пользователя** (ответы бота не пишутся). В v2 дополнительно индексируются **чанки загруженных файлов**. Все вызовы OpenAI идут через `OPENAI_BASE_URL`.

---

## Быстрый старт

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
copy .env.example .env   # заполните ключи
```

Запуск модульного бота v2 (рекомендуется для работы с файлами):

```bash
python -m hay_v2_bot.main
```

Запуск Haystack-агента v1:

```bash
python hay/hay-telegram-bot.py
```

Запуск базового бота:

```bash
python bot.py
```

---

## Haystack v2 (`hay_v2_bot/`)

Модульная архитектура: компоненты, пайплайны, Telegram-слой.

1. Текст → `generation_pipeline`: recall памяти + retrieve документов → Haystack Agent (+ тулы).
2. Файл (PDF/DOCX/…, **до 20 МБ**): проверка размера → скачивание → Docling (или fallback `pypdfium2` для PDF при сбое HF) → Pinecone → «Готово…» → одно предложение-резюме.
3. Изоляция: память `hay_v2_user_{id}`, документы `hay_v2_docs_{id}`.

Подробности: [`hay_v2_bot/README.md`](hay_v2_bot/README.md), архитектура: [`hay_v2_bot/ARCHITECTURE.md`](hay_v2_bot/ARCHITECTURE.md).

### Структура

```
hay_v2_bot/
├── main.py                 # точка входа
├── components/             # config, context, tools, store, logging, proxy_sanitize…
├── pipelines/              # ingestion, generation, summary
└── bot/                    # TeleBot handlers + assistant
```

### Команды

| Команда | Описание |
|---------|----------|
| `/start` | Приветствие |
| `/help` | Справка |
| `/memory` | Фрагменты памяти и документов |
| `/forget` | Очистить память и документы |

---

## Haystack-агент v1 (`hay/`)

Умный персональный помощник:

1. Сообщение пользователя → эмбеддинг → `PineconeDocumentStore` (metric=`cosine`).
2. Память каждого пользователя в **личном** namespace `hay_user_{telegram_id}` — чужие данные недоступны.
3. Retriever поднимает релевантный контекст только из этого namespace.
4. Haystack `Agent` отвечает с учётом памяти и может вызвать тулы.

### Тулы

| Tool | Назначение |
|------|------------|
| `dogFactTool` | Случайный факт о собаках ([dogapi.dog](https://dogapi.dog)) |
| `dogImageTool` | Случайная картинка ([dog.ceo](https://dog.ceo)) |
| `dogImageAnalyzerTool` | Картинка с API → ChatGPT vision → один объект `{image_url, caption}` |

`dogImageAnalyzerTool` логируется через **Loguru**. В Telegram уходит фото с caption = ответ ChatGPT.

### Файлы

| Файл | Назначение |
|------|------------|
| `hay/hay-telegram-bot.py` | Telegram-бот + Haystack Agent |
| `hay/pinecone_memory.py` | Память на `PineconeDocumentStore` |
| `hay/dog_tools.py` | Тулы про собак |

### Команды

| Команда | Описание |
|---------|----------|
| `/start` | Приветствие |
| `/help` | Справка |
| `/memory` | Что бот помнит |
| `/forget` | Очистить память пользователя |

---

## Базовый бот (`bot.py`)

Диалог через OpenAI chat completions и класс `PineconeManager` (`pinecone_manager.py`).

Namespace: **личный** `user_{telegram_id}` (общий `default` для диалогов не используется). Порог косинусного сходства: `0.85`.

---

## Переменные окружения

Скопируйте `.env.example` → `.env` (файл `.env` **не** коммитится).

| Переменная | Описание |
|------------|----------|
| `OPENAI_API_KEY` | Ключ ProxyAPI |
| `OPENAI_BASE_URL` | `https://api.proxyapi.ru/openai/v1` |
| `PINECONE_API_KEY` | Ключ Pinecone |
| `PINECONE_INDEX_NAME` | Имя индекса (cosine, для `text-embedding-3-small` — **1536** dims) |
| `LLM_MODEL` | Модель ответов (например `gpt-4o-mini`) |
| `VISION_MODEL` | Vision-модель для анализа картинок |
| `EMBEDDING_MODEL` | Модель эмбеддингов |
| `TELEGRAM_BOT_TOKEN` | Токен от [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_PROXY` | Опционально: прокси до `api.telegram.org` (v2) |
| `LOG_LEVEL` | Опционально: уровень логов v2 (`INFO` по умолчанию) |
| `HF_TOKEN` | Опционально: токен Hugging Face для моделей Docling (v2) |

---

## Структура проекта

```
Poisk/
├── bot.py
├── pinecone_manager.py
├── hay/                         # v1
├── hay_v2_bot/                  # v2: Docling + пайплайны
│   ├── main.py
│   ├── components/
│   ├── pipelines/
│   └── bot/
├── docs/
├── logs/                        # логи v2 (в git не коммитятся)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Зависимости

См. `requirements.txt`: для v2 дополнительно `docling`, `docling-haystack`, `tiktoken`.

---

## Безопасность

- Не публикуйте `.env` и не добавляйте его в git.
- Используйте `.env.example` только как шаблон без реальных ключей.
- Каждый пользователь Telegram работает в своём namespace Pinecone.
