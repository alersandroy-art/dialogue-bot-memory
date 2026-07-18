# hay_v2_bot

Модульный Telegram-бот на **Haystack**: долговременная память, разбор файлов через **Docling**, векторный поиск в **Pinecone**, LLM и vision через **ProxyAPI**.

Версия v2 повторяет возможности `hay/hay-telegram-bot.py` и добавляет полноценную работу с документами (PDF, DOCX и др.).

---

## Возможности

| Функция | Описание |
|---------|----------|
| Диалог | Haystack `Agent` + OpenAI Chat (ProxyAPI) |
| Память | В Pinecone пишется **только текст пользователя** |
| Документы | Docling → чанки → эмбеддинги → Pinecone (для PDF при сбое HF — fallback `pypdfium2`) |
| Резюме файла | После индексации — **одно** короткое предложение |
| RAG | При ответах учитываются релевантные чанки файлов и память |
| Тулы | Факт о собаках, картинка, vision-анализ породы |
| Изоляция | У каждого Telegram-пользователя свои namespace |
| Лимит файла | Telegram Bot API: **до 20 МБ** (больше — отказ без скачивания ботом) |

---

## Быстрый старт

Из **корня репозитория** (не из этой папки):

```bash
# активируйте venv проекта
.\.venv\Scripts\Activate.ps1   # Windows PowerShell

pip install -r requirements.txt
# .env уже должен быть заполнен в корне (см. ../.env.example)

python -m hay_v2_bot.main
```

Альтернатива:

```bash
python hay_v2_bot/main.py
```

---

## Переменные окружения

Используется `.env` в корне проекта:

| Переменная | Назначение |
|------------|------------|
| `TELEGRAM_BOT_TOKEN` | Токен бота |
| `TELEGRAM_PROXY` | Опционально: HTTP/SOCKS прокси до `api.telegram.org` |
| `OPENAI_API_KEY` | Ключ ProxyAPI |
| `OPENAI_BASE_URL` | `https://api.proxyapi.ru/openai/v1` (прямой `api.openai.com` запрещён) |
| `PINECONE_API_KEY` | Ключ Pinecone |
| `PINECONE_INDEX_NAME` | Индекс (cosine, **1536** dims для `text-embedding-3-small`) |
| `LLM_MODEL` | Модель ответов (по умолчанию `gpt-4o-mini`) |
| `VISION_MODEL` | Vision-модель для анализа картинок |
| `EMBEDDING_MODEL` | Модель эмбеддингов |
| `LOG_LEVEL` | Уровень логов: `DEBUG`, `INFO` (по умолчанию), `WARNING`, `ERROR` |
| `HF_TOKEN` | Опционально: валидный токен Hugging Face для моделей Docling |

---

## Логи

Пишутся в консоль и в каталог `logs/hay_v2_bot/` (в корне репозитория, в git не попадает):

| Файл | Содержимое |
|------|------------|
| `app.log` | Все сообщения уровня `LOG_LEVEL` и выше (ротация 5×5 MB) |
| `errors.log` | Только `ERROR` / `CRITICAL` |
| `vision.log` | Loguru: dogImageAnalyzer / vision |

---

## Команды бота

| Команда | Действие |
|---------|----------|
| `/start` | Приветствие и namespace пользователя |
| `/help` | Справка |
| `/memory` | Релевантные фрагменты памяти и документов |
| `/forget` | Очистить память **и** загруженные документы |

### Текст

Любое сообщение → сохранение в память (кроме команд) → `generation_pipeline` → ответ (и при необходимости фото из dog-тула).

### Файл

Поддерживаемые расширения: `.pdf`, `.docx`, `.doc`, `.pptx`, `.html`, `.htm`, `.md`, `.txt`, `.xlsx`, `.asciidoc`.

Поток:

1. Если `file_size` > **20 МБ** → сразу отказ («Документ больше 20 МБ. Скачиваться не будет.»), `getFile` не вызывается.
2. Скачивание файла ботом → «Файл получен. Запускаю анализ…»
3. `ingestion_pipeline`: Docling → чанки → Pinecone  
   (если Hugging Face/сеть падает на PDF — fallback через `pypdfium2`)
4. «Готово. Я изучил этот файл, теперь можем его обсудить.»
5. Одно предложение-резюме

Дальше можно спрашивать о содержимом файла обычными сообщениями.

> Полоска загрузки в клиенте Telegram (когда вы сами открываете файл) — это не скачивание ботом.

---

## Структура пакета

```
hay_v2_bot/
├── main.py                 # точка входа (+ подготовка сети HF/proxy)
├── README.md               # этот файл
├── ARCHITECTURE.md         # устройство системы
├── components/
│   ├── tools.py            # инструменты генерации (Agent tools)
│   ├── context.py          # контекст + обновление в Pinecone
│   ├── config.py
│   ├── document_store.py
│   ├── meta_enricher.py
│   ├── prompts.py
│   ├── logging_setup.py
│   └── proxy_sanitize.py   # socks4 / HF token / XET для Docling
├── pipelines/              # ingestion, generation, summary
└── bot/                    # TeleBot: assistant + handlers
```

Подробнее о потоках данных и связях модулей — в [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## Тулы агента

| Tool | API | Результат |
|------|-----|-----------|
| `dogFactTool` | [dogapi.dog](https://dogapi.dog) | Случайный факт |
| `dogImageTool` | [dog.ceo](https://dog.ceo) | URL картинки → фото в Telegram |
| `dogImageAnalyzerTool` | dog.ceo + ChatGPT vision | Фото + caption с описанием породы |

---

## Namespaces Pinecone

| Префикс | Содержимое |
|---------|------------|
| `hay_v2_user_{telegram_id}` | Сообщения пользователя (память диалога) |
| `hay_v2_docs_{telegram_id}` | Чанки загруженных файлов |

Не пересекаются с v1 (`hay_user_*`) и базовым ботом (`user_*`).

---

## Отличия от `hay/` (v1)

| | v1 (`hay/`) | v2 (`hay_v2_bot/`) |
|--|-------------|---------------------|
| Архитектура | 2–3 файла | components / pipelines / bot |
| Файлы | нет | Docling + ingestion (+ PDF fallback) |
| Документы в ответах | нет | retriever + контекст в Agent |
| Резюме файла | нет | одно предложение после загрузки |
| Namespace | `hay_user_*` | `hay_v2_user_*` + `hay_v2_docs_*` |

---

## Зависимости

В корневом `requirements.txt` для v2:

- `docling`, `docling-haystack` — разбор документов
- `tiktoken` — чанкинг без токенизатора Hugging Face

Первый запуск Docling может скачать модели парсинга с Hugging Face — это нормально и может занять время. При проблемах с сетью/HF для PDF срабатывает текстовый fallback.
