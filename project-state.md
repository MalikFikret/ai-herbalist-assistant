# AI Herbalist Assistant — Project State

_Last updated: 2026-05-29_

This document is the living architecture / status brief for the project.
It reflects the code actually on disk after all hardening passes and
subsequent feature additions through Phase 4.

---

## 1. What the project is

An **educational RAG chatbot** about herbal remedies. Users ask wellness
questions (headache, cold, sleep, digestion, etc.) and the app answers
from a local corpus of herbal PDFs. The app is bilingual (English /
Turkish), has per-user login with persistent chat history and
"remember me" cookies, and ships an admin panel for managing the
knowledge base, model selection, and user feedback.

> **Not** a medical device, not a diagnostic tool — see `SECURITY.md` §7.

---

## 2. Tech stack

| Layer              | Choice                                               |
| ------------------ | ---------------------------------------------------- |
| UI                 | Streamlit 1.38+ with `extra-streamlit-components` (cookie manager) |
| Orchestration      | LangGraph 1.x (`langgraph.graph.StateGraph`)         |
| LLM                | Groq (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`), Google Gemini (`gemini-1.5-flash`, `gemini-2.5-flash`), DeepSeek (`deepseek-chat`) — via LangChain |
| Embeddings         | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` via HF (multilingual) |
| Vector store       | Chroma 1.x (persisted to `.chroma_db/`) with incremental manifest-based indexing |
| Document ingest    | `pypdf` + `RecursiveCharacterTextSplitter`           |
| Web search         | Tavily (primary) + DuckDuckGo (fallback), with trusted-domain priority |
| Config / env       | `python-dotenv`, `HA_*` env vars                     |
| Observability      | LangSmith (optional, off by default)                 |
| Tests / CI         | `pytest`, `ruff`, GitHub Actions                     |
| Build              | `setuptools` via `pyproject.toml`                    |
| Runtime image      | `python:3.11-slim` (`Dockerfile`)                    |

Python requirement: **3.10+** (CI matrix covers 3.10 and 3.11).

Key dependencies added since original spec: `langchain-google-genai`,
`langchain-openai`, `langchain-tavily-search`, `extra-streamlit-components`,
`duckduckgo-search`.

---

## 3. Repository layout

```
.
├── src/
│   ├── app.py                    # Top-level entry point (imports and calls run())
│   └── herbalist_assistant/
│       ├── __init__.py            # eager .env load + LangSmith tracing setup
│       ├── config.py              # paths, chunk sizes, model defaults, trusted domains
│       ├── llm/
│       │   ├── __init__.py
│       │   └── groq.py            # ChatGroq factory with env-based API key
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── embeddings.py      # HuggingFace embeddings factory (CUDA auto-detect)
│       │   ├── index_manifest.py  # JSON manifest for incremental PDF indexing
│       │   ├── loaders.py         # PyPDFLoader wrappers with pdf_filename tagging
│       │   ├── splitter.py        # RecursiveCharacterTextSplitter logic
│       │   └── vectorstore.py     # Chroma store: load/build/sync/delete/manifest
│       ├── graph/
│       │   ├── __init__.py
│       │   ├── advanced_graph.py  # THE orchestration graph + Pydantic grade schemas
│       │   ├── extractors.py      # Output parsing and info extraction (pure functions)
│       │   ├── runtime.py         # LLM factories, retriever cache, model routing
│       │   ├── state.py           # AgentState TypedDict + type literals
│       │   └── nodes/             # LangGraph node implementations
│       │       ├── __init__.py
│       │       ├── direct_answer.py   # Non-medical answer generation
│       │       ├── grading.py         # Doc grading, hallucination, relevance checks
│       │       ├── medical_answer.py  # RAG-grounded medical answer generation
│       │       ├── retrieval.py       # Query expansion + vectorstore retrieval
│       │       ├── router.py          # 3-tier routing (regex→regex→LLM)
│       │       └── web_search.py      # 2-stage trusted/open web search + grading
│       ├── db/
│       │   ├── __init__.py
│       │   ├── models.py          # User / ChatSession / ChatMessage / AppSetting ORM
│       │   ├── engine.py          # Engine + session_scope() + init_db()
│       │   ├── repository.py      # High-level CRUD ops used by the UI layer
│       │   └── migration.py       # Idempotent schema bootstrap (calls init_db())
│       └── ui/
│           ├── __init__.py
│           ├── auth.py            # Authentication, registration, session state init
│           ├── botanical_assets.py # Base64-encoded botanical PNG backgrounds (~1.9 MB)
│           ├── components.py      # Reusable Streamlit UI components & widgets
│           ├── cookies.py         # HMAC-signed remember-me cookies + stored credentials
│           ├── i18n.py            # EN / TR string tables (~100+ keys per language)
│           ├── resources.py       # @st.cache_resource factories + reindex/sync
│           ├── state.py           # Session-state glue over repository
│           ├── streamlit_app.py   # Main orchestrator / entrypoint
│           ├── styles.py          # Centralized CSS and JS injection (6700+ lines)
│           ├── pages/             # Sub-pages
│           │   ├── __init__.py
│           │   ├── admin.py       # Admin panel: PDF ops, metrics, feedback log
│           │   ├── chat.py        # Chat page: conversation UI, agent invocation
│           │   ├── login.py       # Full-bleed login / register / reset-password
│           │   └── profile.py     # User health profile editing + about section
│           └── static/            # Static assets (10 PNG, 5 SVG)
│               ├── herbalist_logo.png
│               ├── login_background.png
│               ├── login_card_bg.png
│               ├── login_hero_left.png
│               ├── chat_shell_background.png
│               ├── chat_hero_bottle.png
│               ├── chat_hero_mortar.png
│               ├── chat_corner_br.png
│               ├── chat_corner_tl.png
│               ├── shell_corner_bl.png, shell_corner_br.png, shell_corner_tl.png, shell_corner_tr.png
│               └── shell_bg_corner_bl.svg, shell_bg_corner_br.svg
│                   shell_bg_leaf_br.svg, shell_bg_leaf_tl.svg, shell_bg_leaf_tr.svg
├── scripts/
│   ├── build_chat_shell_background.py  # Copies + blurs chat shell background from assets
│   ├── evaluate_system.py              # Per-language RAG evaluation (Gemini judge)
│   ├── generate_admin_password_hash.py # PBKDF2 admin credential generator
│   ├── visualize_graph.py              # Mermaid → langgraph_diagram.png
│   └── visualize_results.py            # Multi-subplot evaluation dashboard
├── tests/
│   ├── conftest.py                     # sys.path + streamlit / langchain stubs
│   ├── test_db_repository.py           # SQLite backend tests (8 tests)
│   ├── test_graph_extractors.py        # Pure-function extractor tests (16 tests)
│   ├── test_graph_router.py            # Deterministic router safeguards (3 tests)
│   ├── test_query_language_priority.py # Same-language-first query parsing (3 tests)
│   ├── test_state.py                   # UI session-state tests (9 tests)
│   ├── test_ui_helpers.py              # UI helper pure-function tests (9 tests)
│   └── test_vectorstore_indexing.py    # Incremental indexing + manifest tests (4 tests)
├── assets/
│   └── reference/                      # Design reference images
│       ├── chat_herbal_background.png
│       └── chat_mockup_reference.png
├── .github/workflows/ci.yml   # ruff + pytest on push/PR
├── .streamlit/
├── data/                       # herbal PDFs (gitignored content)
├── eval_dataset_en.json        # 5 English evaluation test cases
├── eval_dataset_tr.json        # 5 Turkish evaluation test cases
├── evaluation_reports/         # CSV results + PNG dashboard outputs
├── .chroma_db/                 # persisted vector DB (gitignored)
├── .herbalist.db               # SQLite store (users, chats, messages, settings, feedback)
├── Dockerfile                  # production image
├── .dockerignore
├── pyproject.toml              # canonical dependency + tool config
├── requirements.txt            # mirror of pyproject.toml deps
├── README.md
├── SECURITY.md
├── WORK_LOG.md                 # Risk registry (20 categorized issues)
├── langgraph.json              # LangGraph Studio config
├── langgraph_diagram.png       # Generated graph visualization
└── project-state.md            # ← you are here
```

---

## 4. Runtime architecture

### 4.1 LangGraph agent (`graph/advanced_graph.py`)

```
route_question
  ├─► DIRECT_ANSWER ─► direct_answer_node ─► answer_relevance_node ─► END
  └─► VECTOR_SEARCH ─► expand_and_retrieve_node ─► grade_documents_node
                                              ├─► has_docs ─► generate_medical_answer_node ─► hallucination_grader_node
                                              │                                                ├─► answer_relevance ─► answer_relevance_node ─► END
                                              │                                                └─► retry_web ─► web_search_node ─► generate_medical_answer_node (loop, max 1 retry)
                                              └─► no_docs ─► web_search_node ─► generate_medical_answer_node
```

Key properties:

- **Expanded state + quality gates.** `AgentState` carries
  `candidate_docs`, `selected_docs`, `generation_retries`,
  `hallucination_score`, `answer_relevance_score`,
  `answer_relevance_feedback`, `user_profile`, `ui_language`,
  `web_search_provider`, `web_search_queries`, `direct_answer`,
  `final_answer`, and `answer_source_type`
  (`Literal["local_rag", "web_search", "direct", "unknown"]`).
- **Three-tier routing** (`graph/nodes/router.py`). Four regex patterns
  (`_MEDICAL_SIGNAL_RE`, `_FOLLOWUP_SIGNAL_RE`, `_IDENTITY_SIGNAL_RE`,
  `_SMALLTALK_SIGNAL_RE`) provide instant shortcuts before falling through
  to LLM classification with full chat history. Follow-up questions are
  force-routed to `VECTOR_SEARCH` without an LLM call.
- **Conversational memory + profile-aware retrieval.** `chat_history`
  (last 10 turns) is threaded through router / expansion / generation,
  and query expansion can use `user_profile` (age, allergies, conditions,
  meds) when relevant.
- **Model selector wiring.** LLM factories
  (`_router_llm`, `_expansion_llm`, `_grader_llm`, `_generator_llm`)
  live in `graph/runtime.py` and are `@lru_cache(8)`-d keyed on
  `model_name`. Supported models: `llama-3.1-8b-instant`,
  `llama-3.3-70b-versatile`, `gemini-1.5-flash`, `gemini-2.5-flash`,
  and `deepseek-chat`. The UI passes `st.session_state.selected_model`
  into the graph state.
- **Language-priority query expansion** (`graph/nodes/retrieval.py`).
  LLM generates primary same-language queries (up to 4) and optional
  cross-language fallback queries (up to 3). Both JSON shapes
  (`{primary_queries, fallback_queries}` and legacy `{expanded_queries}`)
  are supported. Each expanded query runs a separate `retriever.invoke()`,
  results are deduped and capped at 12 candidate docs.
- **Numerical CRAG grading** (`graph/nodes/grading.py`). Candidate
  documents are scored 0-100 via structured output (`DocumentGrade`),
  filtered at >55, and top-4 are kept. Grading uses `ThreadPoolExecutor`
  with 4 workers for parallel evaluation. If no document passes, the
  workflow falls back to web search.
- **Two-stage web search** (`graph/nodes/web_search.py`).
  - *Stage 1*: Search only `TRUSTED_HERB_DOMAINS` (5 Turkish herbal
    sites from `config.py`). Tavily uses `include_domains`; DuckDuckGo
    uses `site:` query operators.
  - *Stage 2*: Search the open web only if Stage 1 returns fewer than
    `TRUSTED_DOMAIN_MIN_RESULTS` (2) results.
  - Trusted results are marked with `is_trusted_domain=True` in metadata.
  - Tavily is the primary provider; DuckDuckGo is an automatic fallback
    on exception.
  - Web results are graded in parallel (threshold 40, lower than local
    docs' 55) before being passed to generation.
- **Grounding + intent checks.** After answer generation, the graph runs:
  - `hallucination_grader_node` (`HallucinationGrade`) to verify grounding.
    Lenient: minor LLM additions are allowed.
  - `answer_relevance_node` (`AnswerRelevanceGrade`) to verify the answer
    actually addresses user intent. Very lenient: "no" only for
    catastrophic failures (off-topic, contradictions, ignoring allergies).
    Does NOT overwrite `final_answer` on "no" — preserves the answer.
  - Direct answers also pass through `answer_relevance_node` as a safety
    gate (catches insults/inappropriate content even for greetings).
- **Retryable web-search recovery.** When hallucination grading fails, the
  graph can rewrite search queries and retry web search/generation, bounded
  by `generation_retries < 1` (max 1 retry).
- **Reindex cache-coherency.** `advanced_graph.reset_runtime_caches()`
  clears every `lru_cache` (5 total: 4 LLM factories + retriever).
  `resources.reindex_pdfs()` calls it after rebuilding Chroma so
  retrieval never uses a stale handle.
- **Custom prompt logic.** Each node has its own system prompt:
  - `ROUTER_SYSTEM` — JSON output routing instruction.
  - `EXPANSION_SYSTEM` — language-priority query generation.
  - `GRADER_SYSTEM` — 0-100 relevance scoring with treatment-type
    mismatch penalty.
  - `HALLUCINATION_GRADER_SYSTEM` — lenient grounding check.
  - `ANSWER_RELEVANCE_SYSTEM` — very lenient intent match.
  - `DIRECT_ANSWER_SYSTEM` — 10 rules: language match, greetings,
    identity/creator info, out-of-domain decline, insult handling
    (with special Azra case), health profile awareness.
  - `MEDICAL_ANSWER_SYSTEM` — 9 rules: language match, context-only
    answers, no meta-references, no medical disclaimers (STRICTLY
    FORBIDDEN), conditional health-profile reminder, creator identity,
    calm/practical tone, conversational continuity, insult handling.
  - `FIRST_ATTEMPT_WEB_QUERY_SYSTEM` — 2-3 language-aware web queries.
  - `RETRY_WEB_QUERY_SYSTEM` — rewrite with different phrasing.
- **Structured error handling.** Every LLM call is wrapped in
  `try / except` with `logger.exception`. Failures degrade gracefully:
  router → `VECTOR_SEARCH`, expansion → original question, grading →
  fail-closed document drop, generation → language-aware fallback,
  web-query rewrite → original question.

### 4.2 Incremental PDF indexing (`rag/`)

The `rag/` package supports both full rebuilds and incremental indexing:

- **`index_manifest.py`** — Persisted JSON registry
  (`{persist_dir}/index_manifest.json`) tracking each PDF's `mtime_ns`
  and `chunk_count`. Used to detect new, changed, or orphaned files.
- **`vectorstore.py`** provides:
  - `load_or_build_vectorstore()` — Load persisted Chroma or build from
    scratch + write manifest.
  - `sync_new_and_changed_pdfs()` — Incremental sync: compare on-disk
    PDFs against manifest, re-index only changed/new files, remove
    orphans. Returns stats dict
    `{"indexed": [...], "skipped": [...], "removed_orphans": [...]}`.
  - `remove_pdfs_from_index()` — Delete chunks + manifest entries for
    specific PDFs.
  - `rebuild_manifest_after_full_index()` — Rewrite manifest from actual
    store contents after a full rebuild.
  - `_bootstrap_manifest_if_needed()` — Create manifest from existing
    store if manifest is empty (migration path).
  - `delete_chunks_for_pdf()` — Uses public Chroma API (not `_collection`
    private accessor).
- **`embeddings.py`** — `create_embeddings()` with CUDA auto-detection
  and `normalize_embeddings=True` for better cosine similarity.
- **Tuning** (`config.py`): `CHUNK_SIZE=800`, `CHUNK_OVERLAP=150`,
  `RETRIEVER_K=5`.

### 4.3 Streamlit UI (`ui/streamlit_app.py`)

- Pages: **Login** (full-bleed), **Chat**, **Admin Panel** (admins only),
  **Profile**. The UI is modularized into `ui/pages/` with
  `ui/streamlit_app.py` acting as a lightweight orchestrator.
- **Premium botanical shell.** `styles.py` (6,700+ lines / 299 KB)
  generates all CSS and JS. Background images come from
  `botanical_assets.py` (1.9 MB of base64-encoded PNGs with C2PA/Google
  Generative AI provenance) and `ui/static/` (10 PNG + 5 SVG assets).
  The design includes glassmorphism, custom sidebar panels, chat shell
  backgrounds, corner decorations, and herbal themed cards.
- Bilingual (EN / TR) via `ui/i18n.py` (~100+ string keys per language).
  All UI text uses `get_string(lang, key)`.
- **API key validation.** On startup, `_check_api_keys()` validates
  GROQ/GEMINI/DEEPSEEK/TAVILY keys and shows `st.error` / `st.warning`
  banners. Runs once per session.
- **Agent warm-up.** `_is_agent_warm()` / `_warm_up_agent()` pre-load
  the embedding model and vectorstore on first chat visit to avoid
  cold-start latency.
- **Async, graceful timeout.** Agent invocation uses
  `asyncio.wait_for(agent_graph_app.ainvoke(...), timeout=120)`.
  `TimeoutError` / other exceptions are caught and surfaced as a
  localized user-facing message.
- **Chat history limit.** Last 10 messages sent to agent context (raised
  from original 6).
- **Stale question guard.** `_pending_question_is_stale()` discards
  stale pending questions after browser refresh.
- **Assistant action row** under every answer:
  `[ Copy ] [ Sources (N) ] [ 👍 ] [ 👎 ]`.
  - **Copy** — JS `navigator.clipboard` component with a clean fallback.
  - **Sources (N)** — popover (fallback: expander on older Streamlit)
    with one entry per source. PDFs show `file (p. N)`; URL sources are
    clickable — the schema supports `kind = "pdf" | "url"`.
  - **👍 / 👎** — toggles; persisted to `.herbalist.db` via
    `state.update_message_feedback`.
- **Admin Panel.** Hero + advanced settings expander + HTML metric cards
  (total PDFs, last index, active model) + 3-column ops
  (Upload / Reindex / Delete) + feedback log with filter
  (all / helpful / unhelpful), counts, and expandable entries showing
  the question, the answer, and the sources.
- **Guest flow.** "Continue as Guest" uses anonymous session keys
  (`anon_{hex}`) for DB persistence. Auto-login is suppressed for the
  session.
- **Model + provider selectors.** `_render_advanced_settings_widgets()`
  exposes model selection and web search provider (Tavily / DuckDuckGo)
  in both admin and chat views.

### 4.4 Auth, cookies & storage

- **SQLite + SQLAlchemy.** All persistence lives in `.herbalist.db`
  (override with `HA_DB_PATH`). Four tables:
  - `users(id, username, password_hash, salt, role, health_profile_json,
    active_session_id, created_at)`
  - `chat_sessions(id, user_id, title, created_at, updated_at)`
  - `chat_messages(id, session_id, position, role, content, timestamp,
    sources_json, feedback, feedback_at)`
  - `app_settings(key, value)` — generic key-value store for runtime
    settings (e.g. admin password override).
  Foreign keys are enabled via SQLite `PRAGMA foreign_keys=ON`, with
  `ON DELETE CASCADE` so deleting a chat removes its messages and
  deleting a user removes their chats.
- **Database initialization.** On every startup,
  `herbalist_assistant.db.ensure_database_ready()` creates any missing
  tables. The legacy JSON migration logic has been removed.
- **User auth.** PBKDF2-SHA256 with 100k iterations and a per-user
  16-byte salt (`secrets.token_hex(16)`).
- **Admin auth.** In preference order:
  `HA_ADMIN_PASSWORD_HASH` + `HA_ADMIN_PASSWORD_SALT`,
  `HA_ADMIN_PASSWORD`, runtime `AppSetting` from DB, or the dev default
  `1234` with a loud warning.
- **Remember-me cookies** (`ui/cookies.py`). HMAC-SHA256–signed browser
  cookie (`ha_remember`) with `username|expiry|hmac` format and 30-day
  TTL. Secret from `HA_REMEMBER_SECRET` env var or ephemeral per-process
  secret. `extra-streamlit-components` CookieManager is conditionally
  rendered only when needed.
- **Stored credentials (auto-fill).** XOR-obfuscated password stored in
  `data/.remembered_logins.json` for login form auto-fill UX.
  Functions: `_store_remembered_password`, `_lookup_remembered_password`,
  `_forget_remembered_password`.
- **Conversational memory.** Preserved end-to-end through the DB:
  `repository.load_active_chat(username)` hydrates
  `st.session_state.messages`, and `_collect_chat_history_for_agent()`
  in the UI feeds the last 10 turns into `AgentState.chat_history`.
- **Feedback loop.** 👍 / 👎 persists in `chat_messages.feedback`
  (+ `feedback_at`). The Admin → User Feedback view runs a single
  cross-user `SELECT` via `repository.iter_all_feedback()`.
- **App settings.** `repository.get_app_setting(key)` /
  `set_app_setting(key, value)` provide a generic key-value store
  backed by the `app_settings` table.

---

## 5. Configuration (`config.py`)

| Constant                    | Value                                                  |
| --------------------------- | ------------------------------------------------------ |
| `GROQ_MODEL`                | `llama-3.1-8b-instant`                                 |
| `EMBEDDING_MODEL`           | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| `CHUNK_SIZE`                | 800                                                    |
| `CHUNK_OVERLAP`             | 150                                                    |
| `RETRIEVER_K`               | 5                                                      |
| `LLM_TEMPERATURE`           | 0.2                                                    |
| `TRUSTED_HERB_DOMAINS`      | 5 Turkish herbal sites                                 |
| `TRUSTED_DOMAIN_MIN_RESULTS`| 2                                                      |
| `DATA_DIR`                  | `data/`                                                |
| `CHROMA_DIR`                | `.chroma_db/`                                          |
| `DB_PATH`                   | `.herbalist.db`                                        |

UI-level constants (`auth.py`):
- `AVAILABLE_MODELS` = `["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemini-2.5-flash", "deepseek-chat"]`
- `DEFAULT_MODEL` = `"llama-3.1-8b-instant"`
- `AVAILABLE_WEB_SEARCH_PROVIDERS` = `["Tavily", "DuckDuckGo"]`

Graph-level tuning constants:
- `_PRIMARY_QUERY_LIMIT` = 4, `_FALLBACK_QUERY_LIMIT` = 3
- `_CANDIDATE_DOC_CAP` = 12
- `_MAX_GRADE_DOCS` = 6, `_MAX_KEPT_DOCS` = 4
- `_PASSING_SCORE_THRESHOLD` = 55
- `_GRADER_WORKERS` = 4
- `_MAX_HALLUCINATION_RETRIES` = 1
- `_WEB_SEARCH_RESULT_LIMIT` = 6, `_WEB_GRADE_THRESHOLD` = 40
- `_CHAT_HISTORY_LIMIT` = 10, `_AGENT_TIMEOUT_SEC` = 120

---

## 6. Security posture

See `SECURITY.md` for the full policy. Highlights:

- No API keys in source. `.env` is gitignored; keys rotated on any
  suspected exposure.
- Admin credentials sourced from env; insecure default only used with a
  loud warning in the log.
- All password handling uses PBKDF2-SHA256 with 100k iterations +
  per-credential salt.
- Remember-me cookie is HMAC-signed with a configurable secret
  (`HA_REMEMBER_SECRET`).
- `.dockerignore` keeps `.env`, `.users.json`, `.chat_history.json`,
  `.herbalist.db`, post-migration `*.migrated-backup-*` files,
  `.chroma_db/`, and `data/` out of the built image.
- Server-side logs emit plain text (no emojis) for easier grepping; UI
  keeps its emoji affordances.

All 20 categorized risks (4 high, 11 medium, 5 low) originally identified in `WORK_LOG.md` have been remediated as part of the Phase 5 system hardening.

---

## 7. Quality gates

- **Ruff** (`pyproject.toml` → `[tool.ruff]`, line-length 100, py310)
  clean on `src/` and `tests/`.
- **Pytest** suite (52 test functions across 7 test files) covering:
  - JSON extractors: `_strip_fences`, `_extract_route`,
    `_extract_expanded_queries`, `_extract_score` (16 tests).
  - LLM-output sanitizer: `_sanitize_medical_answer`.
  - Document dedupe: `_dedupe_documents`.
  - Conversational memory formatter: `_format_chat_history`.
  - Password hashing round-trip: `_hash_password` / `_verify_password`.
  - Admin password resolution: `_verify_admin_password` against env
    hash and env plaintext paths.
  - Allergy-alias blocking: `_extract_blocked_herbs`.
  - Source normalization & labels for the UI popover.
  - **DB repository (`tests/test_db_repository.py`):** user create /
    auth, profile round-trip, chat lifecycle, title promotion from the
    first user message, feedback update + aggregation, chat
    ownership enforcement, `start_new_chat` / `delete_chat`,
    `set_active_chat`, stats counters (8 tests).
  - **UI state (`tests/test_state.py`):** session-state snapshotting,
    append → title promotion, feedback round-trip with
    `st.session_state` sync, `iter_all_feedback`, and switching
    between chats (9 tests).
  - **Router safeguards (`tests/test_graph_router.py`):** deterministic
    identity/greeting bypass, LLM-backed medical routing, and lexical
    fallback when the LLM errors (3 tests).
  - **Query language priority (`tests/test_query_language_priority.py`):**
    same-language-first query parsing in retrieval and web-search nodes,
    legacy JSON shape support (3 tests).
  - **Vectorstore indexing (`tests/test_vectorstore_indexing.py`):**
    manifest mtime tracking, PDF loader filename tagging, chunk deletion,
    incremental sync skip/index logic (4 tests).
- **CI**: `.github/workflows/ci.yml` runs ruff + pytest on Python 3.10
  and 3.11 for every push / PR. Heavy third-party libs (chromadb,
  sentence-transformers, etc.) are stubbed inside `tests/conftest.py`
  so CI runs in seconds and stays deterministic.

---

## 8. Evaluation system

- **Per-language evaluation.** `scripts/evaluate_system.py` accepts
  `--lang en|tr` to select the appropriate dataset
  (`eval_dataset_en.json` or `eval_dataset_tr.json`). Each dataset
  contains 5 test cases covering headache, sleep, digestion, immunity,
  and stress topics.
- **Judge model.** Google Gemini 2.0 Flash (`ChatGoogleGenerativeAI`)
  replaces the original ChatGroq judge. Reads `GEMINI_API_KEY` or
  `GOOGLE_API_KEY` env var.
- **Metrics tracked.** Latency, hallucination pass, relevance pass,
  document grade score (based on `selected_docs` presence).
- **Dashboard generation.** `scripts/visualize_results.py` accepts
  `--lang en|tr` and generates a 3-metric quality dashboard
  (Document Pass, Hallucination Pass, Relevance Pass) as
  `technical_dashboard_{lang}_{timestamp}.png`.
- **Results directory.** `evaluation_reports/` contains 16 files
  (CSV results + PNG dashboards) from evaluation runs in both
  languages.

---

## 9. Deployment

### 9.1 Local dev
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in API keys (+ admin creds if deploying)
streamlit run src/herbalist_assistant/ui/streamlit_app.py
```

### 9.2 Production container
```bash
docker build -t herbalist-assistant:latest .
docker run --rm -p 8501:8501 \
    --env-file .env \
    -v "$(pwd)/.chroma_db:/app/.chroma_db" \
    -v "$(pwd)/data:/app/data" \
    herbalist-assistant:latest
```

The image:
- Runs as a non-root user (`herbalist`).
- Exposes port `8501`.
- Has a Streamlit health check on `/_stcore/health`.
- Does **not** ship `.env`, `.users.json`, or `.chat_history.json` (see
  `.dockerignore`). Mount them (or set env vars) at run time.
- Is intentionally separate from the course-provided DevContainer; the
  repo has no `.devcontainer/` of its own.

---

## 10. Known limitations / future work

The most valuable items explicitly deferred for later:

1. **Multi-node store.** The SQLite file works great for a single
   container; for horizontal scale, swap the engine URL to Postgres
   (`postgresql+psycopg://...`) — the repository + models are
   dialect-agnostic. No UI change needed.
2. **Wiring relevance grade to regeneration.** `answer_relevance_node`
   currently records `answer_relevance_score` as a terminal quality signal.
   A future improvement is to route `"no"` into a guided regeneration step.
3. **Retrieval caching** (intentionally deferred per product decision).
4. **URL sources.** Schema and popover already support `kind: "url"`;
   still need an ingestion path that produces URL-kind sources.
5. **`CONTRIBUTING.md` + `LICENSE`.** Recommended once the project
   opens up to outside contributors.
6. **Pre-commit hooks.** A `.pre-commit-config.yaml` wiring ruff
   (lint + format) would close the loop for contributors who don't
   run CI locally.
7. **`answer_source_type` plumbing.** The `AnswerSourceType` literal is
   defined in `state.py` but not yet set by all nodes.

---

## 11. Change log (latest updates)

- **P0.1** Admin password read from env, PBKDF2-supported, warning on
  insecure default.
- **P0.2** `reset_runtime_caches()` clears retriever + LLM caches on
  reindex.
- **P0.3a** Model selector wired end-to-end into the graph state.
- **P0.3b** History page and its i18n strings removed.
- **P0.4** `pyproject.toml` populated with dependencies and tool config;
  `requirements.txt` kept in sync.
- **P0.5** README updated to Python 3.10+ and new `HA_ADMIN_*` env vars.
- **Memory fix** `chat_history` threaded through `AgentState` and into
  every relevant prompt in `advanced_graph.py`.
- **Prompts migration** Custom rules (language, domain, attribution,
  conditional reminder, no disclaimers) moved from the deleted
  `prompts.py` into the advanced graph's system prompts.
- **P1.6** `prompts.py`, `types.py`, `graph/builder.py`, `graph/nodes.py`
  deleted; `resources.get_graph()` / `get_llm()` removed.
- **P1.7** Structured `logging` with per-node `logger.exception` for
  failed LLM calls.
- **P1.13** SQLite + SQLAlchemy backend (`herbalist_assistant.db`)
  replaces `.users.json` + `.chat_history.json`. Schema: `users`,
  `chat_sessions`, `chat_messages`, `app_settings`.
  `ensure_database_ready()` now acts as schema bootstrap only.
- **P2.18** `ThreadPoolExecutor` replaced with `asyncio.wait_for`;
  120-second budget with a graceful localized message.
- **P2.20** 👍 / 👎 feedback per assistant message, persisted, plus
  Admin-panel viewer with filters and counts.
- **P2.21** Sources popover (page-numbered PDFs, URL-ready) replacing
  the plain caption.
- **New UI** Copy button next to feedback / sources.
- **P2.23** Server logs stripped of emojis; UI retains them.
- **Docs** `SECURITY.md` (repo-only), production `Dockerfile` +
  `.dockerignore`, `tests/` + CI, this rewrite of `project-state.md`.
- **Advanced Graph Hardening** Fixed Tavily web-search import, optimized
  LLM cache memory usage, capped document processing in serial CRAG grading,
  and ensured consistent state returns in the web-search node.
- **Modular UI Refactor** `src/herbalist_assistant/ui/streamlit_app.py` was
  broken down into a clean package structure (`auth.py`, `components.py`,
  `styles.py`, `pages/`, etc.) to improve maintainability.
- **Migration Code Cleanup** Legacy JSON-to-SQLite migration logic was removed
  from `db/migration.py` now that the data transition is complete.
- **Graph Quality Loop Upgrade** Added structured graders in
  `advanced_graph.py` (`DocumentGrade`, `HallucinationGrade`,
  `AnswerRelevanceGrade`) and extended the workflow with
  `hallucination_grader_node` + `answer_relevance_node`.
- **Retrieval/Grading Refactor** `expand_and_retrieve_node` now stores
  deduped candidate docs (cap 12); `grade_documents_node` scores each doc
  0-100 and keeps high-confidence docs (threshold 55, top-4).
- **Web Search Retry Strategy** `web_search_node` can rewrite queries on
  retries and aggregate/dedupe results across multiple rewritten queries.
- **Router/State Enhancements** Router initializes `generation_retries`,
  and state schema now includes retry counters, quality scores, profile
  context, and staged document collections.
- **Docs Alignment** README now reflects that runtime auto-migration from
  legacy JSON files is removed; database startup is schema bootstrap only.
- **Model Update** Replaced decommissioned `mixtral-8x7b-32768` with
  `llama-3.3-70b-versatile`; added `gemini-2.5-flash` support.
- **Router Tests** Added `tests/test_graph_router.py` covering deterministic
  routing safeguards and lexical fallback paths (3 tests).
- **RAG Technical Evaluation Upgrade** Per-language evaluation
  (`--lang en|tr`) with Gemini 2.0 Flash judge. Multi-subplot dashboard
  generation tracking Document Pass, Hallucination Pass, and Relevance Pass.
- **Phase-3: Language-aware pipeline.** `ui_language` field threaded
  through state. Router, retrieval expansion, and direct answer nodes
  produce language-aware output. `AnswerSourceType` literal added to state.
- **Phase-4: Trusted-domain web search.** Two-stage search (trusted
  domains first, open web fallback). Web result grading with threshold 40.
  `TRUSTED_HERB_DOMAINS` and `TRUSTED_DOMAIN_MIN_RESULTS` in config.
- **Phase-4: Incremental PDF indexing.** `rag/index_manifest.py` provides
  JSON-based manifest tracking. `vectorstore.sync_new_and_changed_pdfs()`
  enables selective re-indexing; `remove_pdfs_from_index()` for targeted
  deletion.
- **Multilingual embeddings.** Embedding model changed from
  `all-MiniLM-L6-v2` to `paraphrase-multilingual-mpnet-base-v2` for
  better bilingual retrieval.
- **Premium UI shell.** `botanical_assets.py` (1.9 MB base64 PNGs),
  `styles.py` (6,700+ lines CSS/JS), 15 static assets (10 PNG + 5 SVG).
  Full-bleed login page, glassmorphism sidebar, chat shell backgrounds,
  corner decorations, herbal themed cards.
- **Cookie / remember-me system.** `cookies.py` adds HMAC-signed
  browser cookie (`ha_remember`) with 30-day TTL, XOR-obfuscated stored
  credentials for auto-fill, and conditional CookieManager rendering.
- **Query language priority tests.** `tests/test_query_language_priority.py`
  validates same-language-first parsing for retrieval and web-search nodes.
- **Chat shell build script.** `scripts/build_chat_shell_background.py`
  copies and blurs background image from `assets/reference/`.
- **Risk registry.** `WORK_LOG.md` documents 20 categorized risks
  (4 high / 11 medium / 5 low) for future remediation.
- **Tuning adjustments.** CRAG threshold lowered from 70 → 55. Kept docs
  raised from 3 → 4. Candidate cap raised from 10 → 12. Hallucination
  retries reduced from 3 → 1. Chat history limit raised from 6 → 10.
  Web search results raised from 4 → 6.
- **Phase 5 Hardening (Risk Remediation).** Remediated all 20 categorized
  risks from `WORK_LOG.md`, including SQLite WAL concurrency (Risk-5),
  timezone-aware timestamps (Risk-20), minimum password lengths (Risk-12),
  and fallback routing safeguards (Risk-9).
