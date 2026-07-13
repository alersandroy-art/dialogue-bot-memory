# Dialogue Bot Memory

Проект Telegram-чат-бота с **диалоговой долговременной памятью** на базе векторного хранилища [Pinecone](https://www.pinecone.io/) и языковых моделей через [ProxyAPI](https://proxyapi.ru/).

Бот общается с пользователем, сохраняет **только оригинальные текстовые сообщения пользователя** в Pinecone, сравнивает новые фрагменты с уже сохранёнными и использует релевантный контекст при генерации ответа через LLM.

---

## Назначение проекта

Система решает задачу **персистентной памяти** в диалоге:

1. Пользователь отправляет текстовое сообщение.
2. Текст преобразуется в эмбеддинг и сохраняется в Pinecone (namespace `user_{telegram_id}`).
3. Перед записью выполняется проверка косинусного сходства с уже сохранёнными фрагментами **этого** пользователя.
4. Если сходство **низкое** (`< 0.85`) — новая информация, создаётся новый фрагмент памяти.
5. Если сходство **высокое** (`>= 0.85`) — дубликат или вариация: фрагмент **обновляется** (в боте `on_duplicate="update"`).

### Что сохраняется в векторной базе

| Сохраняется | Не сохраняется |
|-------------|----------------|
| Оригинальный текст сообщения пользователя | Ответы бота (`bot_response`) |
| Служебные поля в `metadata` (timestamp, username и т.д.) | Шаблонные фразы («Профиль пользователя Telegram: ...») |
| | Команды `/start`, `/help`, `/memory`, `/forget` |

Память каждого пользователя изолирована в отдельном namespace: `user_{telegram_id}`.

---

## Компоненты проекта

| Файл / папка | Назначение |
|--------------|------------|
| `bot.py` | Telegram-бот (`AssistantBot`) на pyTelegramBotAPI: диалог, LLM, вызовы менеджера памяти |
| `pinecone_manager.py` | Класс `PineconeManager`: Pinecone, эмбеддинги, долговременная память |
| `.env` | Секреты и настройки. **Не коммитить в git** |
| `.env.example` | Шаблон переменных окружения |
| `requirements.txt` | Зависимости Python |
| `.venv/` | Виртуальное окружение |
| `.vscode/settings.json` | Автоактивация `.venv` и настройка Code Runner |

### Модуль `bot.py`

Класс `AssistantBot`:

- принимает сообщения через long polling;
- сохраняет в Pinecone только `user_message` через `save_to_long_term_memory`;
- ищет контекст через `recall_user_memory` (top_k = 5);
- генерирует ответ через OpenAI API (`LLM_MODEL` из `.env`).

**Команды бота:**

| Команда | Описание |
|---------|----------|
| `/start` | Приветствие (в память не пишет) |
| `/help` | Справка по возможностям |
| `/memory` | Показать релевантные фрагменты памяти |
| `/forget` | Полностью очистить память пользователя |

Любое **текстовое сообщение** (не команда) сохраняется в память и обрабатывается с учётом контекста.

### Модуль `pinecone_manager.py`

Класс `PineconeManager` объединяет:

- подключение к Pinecone и OpenAI (через `OPENAI_BASE_URL`);
- создание эмбеддингов (`create_embedding`);
- запись и чтение векторов и документов;
- долговременную память с проверкой дубликатов.

#### Ключевые настройки (в начале файла)

```python
COSINE_SIMILARITY_THRESHOLD = 0.85   # порог косинусного сходства
USER_NAMESPACE_PREFIX = "user_"      # префикс namespace пользователя
```

#### Основные методы для чат-бота

| Метод | Описание |
|-------|----------|
| `save_to_long_term_memory(text, telegram_id, ...)` | Сохранить фрагмент с проверкой на дубликаты |
| `find_similar_memory(text, telegram_id, ...)` | Найти похожий фрагмент памяти |
| `recall_user_memory(query_text, telegram_id, ...)` | Вспомнить релевантные фрагменты по запросу |
| `clear_user_memory(telegram_id)` | Очистить память пользователя |
| `build_user_namespace(telegram_id)` | Получить namespace вида `user_12345` |

#### Методы работы с Pinecone

| Метод | Описание |
|-------|----------|
| `upsert_vector` / `upsert_vectors` | Запись векторов |
| `upsert_document` / `upsert_documents` | Запись документов (текст → эмбеддинг) |
| `query_by_vector` / `query_by_text` | Поиск по вектору или тексту |
| `fetch_vectors` | Получение векторов по ID |
| `delete` / `delete_by_filter` / `delete_all` | Удаление данных |
| `describe_index_stats` | Статистика индекса |
| `update_metadata` | Обновление метаданных вектора |

Точка входа для проверки менеджера:

```bash
python pinecone_manager.py
```

---

## Требования

- Python 3.10+
- Аккаунт [Pinecone](https://www.pinecone.io/) с созданным индексом
- Ключ [ProxyAPI](https://proxyapi.ru/) для OpenAI-совместимого API
- Токен Telegram-бота от [@BotFather](https://t.me/BotFather)

Размерность индекса Pinecone должна соответствовать модели эмбеддингов.  
Для `text-embedding-3-small` — **1536** измерений, метрика **cosine**.

---

## Настройка

### 1. Переход в папку проекта

```bash
cd "Dialogue bot memory"
```

### 2. Виртуальное окружение

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

В Cursor / VS Code окружение активируется автоматически (см. `.vscode/settings.json`).

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Переменные окружения

```bash
copy .env.example .env
```

**Linux / macOS:** `cp .env.example .env`

| Переменная | Описание |
|------------|----------|
| `OPENAI_API_KEY` | Ключ ProxyAPI |
| `OPENAI_BASE_URL` | `https://api.proxyapi.ru/openai/v1` |
| `PINECONE_API_KEY` | Ключ Pinecone |
| `PINECONE_INDEX_NAME` | Имя индекса в Pinecone |
| `LLM_MODEL` | Модель для ответов (например, `gpt-4o-mini`) |
| `EMBEDDING_MODEL` | Модель эмбеддингов (например, `text-embedding-3-small`) |
| `TELEGRAM_BOT_TOKEN` | Токен Telegram-бота |

---

## Запуск

### Telegram-бот

```bash
python bot.py
```

Бот работает в режиме long polling. Остановка: `Ctrl+C`.

### Пример работы с менеджером из кода

```python
from pinecone_manager import PineconeManager

manager = PineconeManager()
telegram_id = 123456789

# Сохранить сообщение пользователя
result = manager.save_to_long_term_memory(
    text="Люблю пиццу с ананасами",
    telegram_id=telegram_id,
    metadata={"type": "user_message", "role": "user"},
    on_duplicate="update",
)
print(result["action"], result["namespace"], result["message"])

# Вспомнить релевантные фрагменты
memories = manager.recall_user_memory(
    query_text="Какую еду любит пользователь?",
    telegram_id=telegram_id,
    top_k=5,
)
for item in memories:
    print(item["score"], item["metadata"].get("text"))
```

### Проверка подключения к Pinecone

```bash
python -c "from pinecone_manager import PineconeManager; m = PineconeManager(); print(m.describe_index_stats())"
```

---

## Структура проекта

```
Dialogue bot memory/
├── bot.py                # Telegram-бот на pyTelegramBotAPI
├── pinecone_manager.py   # Менеджер Pinecone и долговременной памяти
├── requirements.txt      # Зависимости
├── .env.example          # Шаблон настроек
├── .env                  # Локальные секреты (не в git)
├── .gitignore
├── .vscode/
│   └── settings.json     # Автоактивация venv и Code Runner
└── README.md
```

---

## Логика работы бота

```
Текстовое сообщение пользователя
        │
        ├─► save_to_long_term_memory (только user_message)
        │         ├─ create_embedding()
        │         ├─ find_similar_memory() в user_{telegram_id}
        │         ├─ similarity < 0.85 → новый фрагмент
        │         └─ similarity >= 0.85 → update существующего
        │
        ├─► recall_user_memory() → контекст для LLM
        │
        ├─► OpenAI chat.completions (LLM_MODEL)
        │
        └─► Ответ пользователю в Telegram (в память не сохраняется)
```

---

## Зависимости

- `pinecone` — векторное хранилище
- `openai` — эмбеддинги и LLM через ProxyAPI
- `python-dotenv` — загрузка переменных из `.env`
- `pyTelegramBotAPI` — Telegram-бот

---

## Безопасность

- Не публикуйте файл `.env` и не добавляйте его в git.
- Используйте `.env.example` только как шаблон без реальных ключей.
- Каждый пользователь Telegram работает только со своим namespace `user_{telegram_id}`.
