# Архитектура hay_v2_bot

Документ описывает слои, пайплайны и потоки данных модульного Telegram-бота на Haystack.

---

## 1. Цели архитектуры

- **Декомпозиция**: Telegram, Haystack-пайплайны и инфраструктура (Pinecone / OpenAI) разделены по пакетам.
- **Компонентный подход**: переиспользуемые блоки (`PineconeContext`, `DocumentMetaEnricher`, `tools`) не смешаны с хендлерами.
- **Пайплайны как ядро**: индексация файлов — явный Haystack `Pipeline`; генерация ответа — оркестрация retrieval + `Agent`.
- **Изоляция пользователей**: отдельные Pinecone namespace для памяти и документов.
- **Совместимость с v1**: тот же ProxyAPI, те же dog-тулы и правила памяти (в вектор пишется только текст пользователя).

---

## 2. Слои пакета

```text
Telegram (bot/)
        │
        ▼
  pipelines/          ← бизнес-сценарии
        │
        ▼
  components/         ← store, memory, Docling enricher, tools, config
        │
        ▼
  Внешние сервисы: Pinecone · ProxyAPI (OpenAI) · Docling · dog APIs
```

| Слой | Путь | Ответственность |
|------|------|-----------------|
| Entry | `main.py` | Логирование, создание `HaystackV2Bot`, polling |
| Transport | `bot/` | Команды, текст, документы, отправка фото |
| Pipelines | `pipelines/` | Ingestion, generation, summary |
| Components | `components/` | Конфиг, память, документы, метаданные, тулы, промпты |

Зависимости направлены **вниз**: `bot` → `pipelines` → `components`. Обратных импортов из `components` в `bot` нет.

---

## 3. Карта модулей

### 3.1. `components/`

| Модуль | Роль |
|--------|------|
| `config.py` | `.env` → `Settings`, константы (top_k, расширения, префиксы namespace) |
| `document_store.py` | `PineconeStoreFactory`, `memory_namespace` / `docs_namespace` |
| `context.py` | `PineconeContext`: update_memory, build, retrieve, clear, format |
| `tools.py` | `build_generation_tools`: dogFact / dogImage / dogImageAnalyzer |
| `meta_enricher.py` | Haystack-компонент: `filename`, `chunk_index`, `page`, `section` |
| `prompts.py` | System prompt агента и промпт однопредложного резюме |

### 3.2. `pipelines/`

| Модуль | Роль |
|--------|------|
| `ingestion.py` | Docling → enricher → embedder → Pinecone writer |
| `generation.py` | Recall памяти + retrieve документов → Agent |
| `summary.py` | LLM: ровно одно предложение о файле |

### 3.3. `bot/`

| Модуль | Роль |
|--------|------|
| `assistant.py` | `HaystackV2Bot`: сборка зависимостей и хендлеры |
| `handlers.py` | Регистрация хендлеров, скачивание файла, тексты /start и /help, отправка dog-фото |

---

## 4. Потоки данных

### 4.1. Текстовое сообщение

```text
Пользователь (text)
        │
        ├─► PineconeContext.update_memory    (namespace hay_v2_user_*)
        │
        ▼
GenerationPipeline.run
        │
        ├─► PineconeContext.build            (память + документы)
        │
        ▼
Agent (OpenAIChatGenerator + tools.py)
        │
        ├─► system prompt: память + фрагменты документов
        │
        ▼
Ответ в Telegram
  (или send_photo, если сработал dogImage* tool)
```

В Pinecone **не** сохраняются ответы бота и результаты тулов.

### 4.2. Загрузка файла

```text
Пользователь (document)
        │
        ▼
«Файл получен. Запускаю анализ…»   ← после успешного скачивания ботом
        │                              (если file_size > 20 МБ — отказ без getFile)
        ▼
IngestionPipeline  (Haystack Pipeline)
  DoclingConverter (DOC_CHUNKS + HybridChunker/tiktoken)
  или PDF-fallback pypdfium2 при сбое HF/сети
        → enricher (filename, chunk_index, page, …)
        → OpenAIDocumentEmbedder (ProxyAPI)
        → DocumentWriter → Pinecone (hay_v2_docs_*)
        │
        ▼
«Готово. Я изучил этот файл…»
        │
        ▼
SummaryPipeline → одно предложение → Telegram
```

Метаданные чанка (минимум):

- `type=file_chunk`
- `filename`
- `chunk_index`
- `page` — если Docling отдал provenance
- `section` — заголовки из `dl_meta`, если есть
- `telegram_id`

### 4.3. Команды

| Команда | Поведение |
|---------|-----------|
| `/start`, `/help` | Статические/шаблонные ответы |
| `/memory` | `PineconeContext.build` по запросу |
| `/forget` | `PineconeContext.clear_all` |

---

## 5. Пайплайны подробно

### 5.1. Ingestion (Haystack `Pipeline`)

```text
[DoclingConverter] → [DocumentMetaEnricher] → [OpenAIDocumentEmbedder] → [DocumentWriter]
```

- Режим экспорта: `ExportType.DOC_CHUNKS`.
- Чанкер: Docling `HybridChunker` с tokenizer `sentence-transformers/all-MiniLM-L6-v2` (только для лимита длины чанка; эмбеддинги — OpenAI).
- Store создаётся **на пользователя** через `PineconeStoreFactory.get_docs_store(telegram_id)` и передаётся в writer при каждом запуске.
- Вход конвертера: `sources` (с fallback на `paths` для совместимости со старыми версиями пакета).

### 5.2. Generation (оркестратор)

Не один линейный `Pipeline.connect`, а явный сценарий:

1. `PineconeContext.build` (память + документы)  
2. Сборка `SYSTEM_PROMPT`  
3. `Agent.run` с тулами из `tools.build_generation_tools`  

Так сохраняются tool-calling и поведение v1, плюс RAG по файлам.

### 5.3. Summary

- На вход: список чанков после ingestion (или fallback-retrieve по имени файла).
- Берётся ограниченное число первых фрагментов.
- `OpenAIGenerator` + пост-обработка: остаётся **одно** предложение.

---

## 6. Модель данных в Pinecone

Один **индекс** (`PINECONE_INDEX_NAME`), разные **namespace**:

```text
index
 ├── hay_v2_user_12345     # память пользователя 12345
 ├── hay_v2_docs_12345     # файлы пользователя 12345
 ├── hay_v2_user_67890
 └── hay_v2_docs_67890
```

| Поле Document | Память | Файл |
|---------------|--------|------|
| `content` | Текст сообщения | Текст чанка |
| `meta.type` | `user_message` | `file_chunk` |
| `meta.role` | `user` | — |
| `meta.filename` | — | имя файла |
| `meta.chunk_index` | — | порядковый номер |
| `embedding` | OpenAI embedding(content) | то же |

Дедупликация памяти: косинусное сходство ≥ `0.85` → запись пропускается.

---

## 7. Сборка при старте (`HaystackV2Bot`)

```text
load_settings()
    → PineconeStoreFactory
    → PineconeContext
    → OpenAI client (ProxyAPI, trust_env=False)
    → GenerationPipeline (Agent + tools.py)
    → IngestionPipeline
    → SummaryPipeline
    → TeleBot + register_handlers
```

Все вызовы OpenAI/эмбеддингов проходят проверку `assert_proxy_base_url`: пустой URL или `api.openai.com` — ошибка старта/вызова.

---

## 8. Границы ответственности (что куда не класть)

| Не делать | Куда вместо этого |
|-----------|-------------------|
| Парсить PDF в `handlers.py` | `IngestionPipeline` |
| Писать в Pinecone из хендлера напрямую | `PineconeContext.update_memory` / `DocumentWriter` в ingestion |
| Дублировать system prompt в assistant | `components/prompts.py` |
| Смешивать память и файлы в одном namespace | `hay_v2_user_*` vs `hay_v2_docs_*` |
| Вызывать OpenAI без `OPENAI_BASE_URL` | `Settings` + `assert_proxy_base_url` |
| Размазывать тулы по хендлерам | `components/tools.py` → `build_generation_tools` |

---

## 9. Расширение

Типовые точки роста:

1. **Новый тул** — добавить в `components/tools.py` (`build_generation_tools`).
2. **Другой vector store** — заменить реализацию в `document_store.py`, сохранив API `PineconeContext`.
3. **Жёсткий RAG без Agent** — добавить классический Haystack pipeline `embedder → retriever → prompt_builder → llm` рядом с generation и выбирать сценарий в assistant.
4. **Новый формат файла** — расширить `SUPPORTED_EXTENSIONS` в `config.py` (если Docling его поддерживает).

---

## 10. Связанные документы

- [README.md](./README.md) — запуск, команды, env
- Корневой `README.md` репозитория — сравнение v1 / v2 / базового бота
- `docs/04doc_RAG_with_Haystack.md` — идея Docling + indexing/RAG пайплайнов
