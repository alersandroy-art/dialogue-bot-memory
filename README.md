# Dialogue Bot Memory / Personal Assistant

Telegram-помощник с **долговременной памятью** в [Pinecone](https://www.pinecone.io/) и LLM через [ProxyAPI](https://proxyapi.ru/).

В репозитории два варианта бота:

| Вариант | Запуск | Стек |
|---------|--------|------|
| Базовый | `python bot.py` | OpenAI SDK + `PineconeManager` |
| Haystack-агент | `python hay/hay-telegram-bot.py` | Haystack Agent + `PineconeDocumentStore` + тулы |

Оба варианта сохраняют в Pinecone **только текст сообщений пользователя**. Ответы бота в память не пишутся. Все вызовы OpenAI идут через `OPENAI_BASE_URL`.

---

## Быстрый старт

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
copy .env.example .env   # заполните ключи
```

Запуск Haystack-агента (основной персональный помощник):

```bash
python hay/hay-telegram-bot.py
```

Запуск базового бота:

```bash
python bot.py
```

---

## Haystack-агент (`hay/`)

Умный персональный помощник:

1. Сообщение пользователя → эмбеддинг → `PineconeDocumentStore` (metric=`cosine`, namespace `hay_user_{telegram_id}`).
2. Retriever поднимает релевантный контекст.
3. Haystack `Agent` отвечает с учётом памяти и может вызвать тулы.

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
| `/memory` | Релевантные фрагменты памяти |
| `/forget` | Очистить память пользователя |

---

## Базовый бот (`bot.py`)

Диалог через OpenAI chat completions и класс `PineconeManager` (`pinecone_manager.py`).

Namespace: `user_{telegram_id}`. Порог косинусного сходства: `0.85`.

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

---

## Структура проекта

```
personal assistant/
├── bot.py                      # Базовый Telegram-бот
├── pinecone_manager.py         # Менеджер Pinecone (базовый бот)
├── hay/
│   ├── hay-telegram-bot.py     # Haystack Agent + Telegram
│   ├── pinecone_memory.py      # Память через PineconeDocumentStore
│   └── dog_tools.py            # Тулы: факт / картинка / vision-анализ
├── 01doc_start_openai.md       # Документация Haystack (старт)
├── 02doc_tutorial_agent.md     # Tutorial: Agent
├── 02doc_tutorial_rag.md       # Tutorial: RAG
├── 03doc_integration_pinecone.md
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Зависимости

См. комментарии в `requirements.txt`: `pinecone`, `openai`, `python-dotenv`, `pyTelegramBotAPI`, `haystack-ai`, `pinecone-haystack`, `httpx`, `loguru`.

---

## Безопасность

- Не публикуйте `.env` и не добавляйте его в git.
- Используйте `.env.example` только как шаблон без реальных ключей.
- Каждый пользователь Telegram работает в своём namespace Pinecone.
