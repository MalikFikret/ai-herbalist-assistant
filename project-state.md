# AI Herbalist Assistant — Project State

_Last updated: 2026-05-04_

This document is the living architecture / status brief for the project.
It replaces the original "state of the codebase" snapshot and reflects the
code actually on disk after the P0/P1/P2 hardening pass.

---

## 1. What the project is

An **educational RAG chatbot** about herbal remedies. Users ask wellness
questions (headache, cold, sleep, digestion, etc.) and the app answers
from a local corpus of herbal PDFs. The app is bilingual (English /
Turkish), has per-user login with persistent chat history, and ships an
admin panel for managing the knowledge base, model, and user feedback.

> **Not** a medical device, not a diagnostic tool — see `SECURITY.md` §7.

---

## 2. Tech stack

| Layer              | Choice                                               |
| ------------------ | ---------------------------------------------------- |
| UI                 | Streamlit 1.38+                                      |
| Orchestration      | LangGraph 1.x (`langgraph.graph.StateGraph`)         |
| LLM                | Groq (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`), Google Gemini, DeepSeek (via LangChain) |
| Embeddings         | `sentence-transformers/all-MiniLM-L6-v2` via HF      |
| Vector store       | Chroma 1.x (persisted to `.chroma_db/`)              |
| Document ingest    | `pypdf` + `RecursiveCharacterTextSplitter`           |
| Config / env       | `python-dotenv`, `HA_*` env vars                     |
| Observability      | LangSmith (optional, off by default)                 |
| Tests / CI         | `pytest`, `ruff`, GitHub Actions                     |
| Build              | `setuptools` via `pyproject.toml`                    |
| Runtime image      | `python:3.11-slim` (`Dockerfile`)                    |

Python requirement: **3.10+** (CI matrix covers 3.10 and 3.11).

---

## 3. Repository layout

```
.
├── src/herbalist_assistant/
│   ├── __init__.py            # eager .env load + LangSmith banner
│   ├── config.py              # paths, chunk sizes, model defaults
│   ├── llm/
│   │   └── groq.py            # ChatGroq factory with env-based API key
│   ├── rag/
│   │   ├── embeddings.py      # HuggingFace embeddings factory
│   │   ├── loaders.py         # PyPDFLoader wrappers
│   │   ├── splitter.py        # RecursiveCharacterTextSplitter logic
│   │   └── vectorstore.py     # Chroma persistent store
│   ├── graph/
│   │   ├── advanced_graph.py  # THE orchestration graph (router, expand+retrieve, etc)
│   │   ├── extractors.py      # Output parsing and info extraction
│   │   ├── runtime.py         # Execution handling
│   │   ├── state.py           # AgentState definition
│   │   └── nodes/             # LangGraph node implementations
│   │       ├── __init__.py
│   │       ├── direct_answer.py
│   │       ├── grading.py
│   │       ├── medical_answer.py
│   │       ├── retrieval.py
│   │       ├── router.py
│   │       └── web_search.py
│   ├── db/
│   │   ├── models.py          # User / ChatSession / ChatMessage ORM
│   │   ├── engine.py          # Engine + session_scope() + init_db()
│   │   ├── repository.py      # High-level ops used by the UI layer
│   │   └── migration.py       # Idempotent schema bootstrap (calls init_db())
│   └── ui/
│       ├── auth.py            # Authentication logic & credentials
│       ├── components.py      # Reusable Streamlit UI components
│       ├── cookies.py         # Cookie management logic
│       ├── i18n.py            # EN / TR string tables
│       ├── resources.py       # @st.cache_resource factories + reindex
│       ├── state.py           # thin session-state glue over repository
│       ├── streamlit_app.py   # Main orchestrator / entrypoint
│       ├── styles.py          # Centralized CSS and styling definitions
│       ├── pages/             # Sub-pages
│       │   ├── __init__.py
│       │   ├── admin.py
│       │   ├── chat.py
│       │   ├── login.py
│       │   └── profile.py
│       └── static/            # Static assets (images, fonts, etc.)
├── scripts/
│   ├── generate_admin_password_hash.py
│   └── visualize_graph.py
├── tests/
│   ├── conftest.py            # sys.path + streamlit / langchain stubs
│   ├── test_db_repository.py  # SQLite backend tests
│   ├── test_graph_extractors.py
│   ├── test_graph_router.py   # Deterministic router safeguards
│   ├── test_ui_helpers.py
│   └── test_state.py
├── .github/workflows/ci.yml   # ruff + pytest on push/PR
├── .streamlit/
├── data/                       # herbal PDFs (gitignored content)
├── .chroma_db/                 # persisted vector DB (gitignored)
├── .herbalist.db               # SQLite store (users, chats, messages, feedback)
├── Dockerfile                  # production image
├── .dockerignore
├── pyproject.toml              # canonical dependency + tool config
├── requirements.txt            # mirror of pyproject.toml deps
├── README.md
├── SECURITY.md
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
                                              │                                                └─► retry_web ─► web_search_node ─► generate_medical_answer_node (loop, max 3 retries)
                                              └─► no_docs ─► web_search_node ─► generate_medical_answer_node
```

Key properties baked in during the hardening pass:

- **Expanded state + quality gates.** `AgentState` now carries
  `candidate_docs`, `selected_docs`, `generation_retries`,
  `hallucination_score`, `answer_relevance_score`, and `user_profile`.
  This enables a multi-step answer quality loop instead of a single-pass
  RAG answer.
- **Conversational memory + profile-aware retrieval.** `chat_history` is
  threaded through router / expansion / generation, and query expansion
  can use `user_profile` (age, allergies, conditions, meds) when relevant.
- **Model selector wiring.** LLM factories
  (`_router_llm`, `_expansion_llm`, `_grader_llm`, `_generator_llm`)
  live in `graph/runtime.py` and are `@lru_cache`-d keyed on `model_name`.
  Supported models include `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`,
  `gemini-1.5-flash`, and `deepseek-chat`. The UI passes
  `st.session_state.selected_model` into the graph state.
- **Numerical CRAG grading.** Candidate documents are scored 0-100 via
  structured output (`DocumentGrade`), filtered at >70, and top-3 are kept.
  If no document passes, the workflow falls back to web search.
- **Grounding + intent checks.** After answer generation, the graph runs:
  - `hallucination_grader_node` (`HallucinationGrade`) to verify grounding.
  - `answer_relevance_node` (`AnswerRelevanceGrade`) to verify the answer
    actually addresses user intent.
- **Retryable web-search recovery.** When hallucination grading fails, the
  graph can rewrite search queries and retry web search/generation, bounded
  by `generation_retries < 3`.
- **Reindex cache-coherency.** `advanced_graph.reset_runtime_caches()`
  clears every `lru_cache` (including the retriever). `resources.reindex_pdfs()`
  calls it after rebuilding Chroma so retrieval never uses a stale handle.
- **Custom prompt logic (migrated from the deleted `prompts.py`).**
  - Language-mirroring ("respond in the same language as the user").
  - Strict domain scope (herbs, remedies, symptoms, wellness).
  - Creator attribution ("Malik Fikret, Ebru Tuğçe Polat, Melisa Yıldırım,
    Prof. Dr. Ramazan KATIRCI" when asked).
  - Conditional profile reminder when allergies / conditions are present.
  - Explicit prohibition of generic medical disclaimers and
    meta-references ("based on the context…").
- **Structured error handling.** Every LLM call is wrapped in
  `try / except` with `logger.exception`. Failures degrade gracefully:
  router → `VECTOR_SEARCH`, expansion → original question, grading →
  fail-closed document drop, generation → user-friendly fallback,
  web-query rewrite → original question.

### 4.2 Streamlit UI (`ui/streamlit_app.py`)

- Pages: **Chat**, **Admin Panel** (admins only), **Profile**, **Login**. The UI is
  modularized into `ui/pages/` with `ui/streamlit_app.py` acting as a lightweight orchestrator.
  Users reach prior chats from the new Sidebar UI.
- Bilingual (EN / TR) via `ui/i18n.py`. All new strings were added to
  both tables.
- **Async, graceful timeout.** Agent invocation uses
  `asyncio.wait_for(agent_graph_app.ainvoke(...), timeout=120)`.
  `TimeoutError` / other exceptions are caught and surfaced as a
  localized user-facing message — the app no longer freezes.
- **Assistant action row** under every answer:
  `[ Copy ] [ Sources (N) ] [ 👍 ] [ 👎 ]`.
  - **Copy** — JS `navigator.clipboard` component with a clean fallback.
  - **Sources (N)** — popover (fallback: expander on older Streamlit)
    with one entry per source. PDFs show `file (p. N)`; URL sources are
    clickable — the schema is already extensible (`kind = "pdf" | "url"`).
  - **👍 / 👎** — toggles; persisted to `.herbalist.db` via
    `state.update_message_feedback`.
- **Admin Panel → User Feedback.** New section lists every 👍 / 👎
  across all users / chats, newest-first, with filters (all / up / down),
  counts, and expanders showing the question, the answer, and the
  sources of the rated message.
- Removed hardcoded Turkish instruction from the profile context; the
  graph now uses the runtime language from `st.session_state.language`.

### 4.3 Auth & storage

- **SQLite + SQLAlchemy.** All persistence lives in `.herbalist.db`
  (override with `HA_DB_PATH`). Three tables:
  - `users(id, username, password_hash, salt, role, health_profile_json,
    active_session_id, created_at)`
  - `chat_sessions(id, user_id, title, created_at, updated_at)`
  - `chat_messages(id, session_id, position, role, content, timestamp,
    sources_json, feedback, feedback_at)`
  Foreign keys are enabled via SQLite `PRAGMA foreign_keys=ON`, with
  `ON DELETE CASCADE` so deleting a chat removes its messages and
  deleting a user removes their chats.
- **Database initialization.** On every startup,
  `herbalist_assistant.db.ensure_database_ready()` creates any missing
  tables. The legacy JSON migration logic has been removed as the transition
  to SQLite is complete.
- **User auth.** PBKDF2-SHA256 with a per-user 16-byte salt
  (`secrets.token_hex(16)`). `_hash_password` + `_verify_password`
  remain pure Python; only the storage backend changed.
- **Admin auth.** Unchanged — env-var driven, in preference order:
  `HA_ADMIN_PASSWORD_HASH` + `HA_ADMIN_PASSWORD_SALT`,
  `HA_ADMIN_PASSWORD`, or the dev default `1234` with a loud warning.
- **Conversational memory.** Preserved end-to-end through the DB:
  `repository.load_active_chat(username)` hydrates
  `st.session_state.messages`, and `_collect_chat_history_for_agent()`
  in the UI feeds the last six turns into `AgentState.chat_history`.
- **Feedback loop.** 👍 / 👎 persists in `chat_messages.feedback`
  (+ `feedback_at`). The Admin → User Feedback view runs a single
  cross-user `SELECT` via `repository.iter_all_feedback()` instead of
  scanning a JSON file.

---

## 5. Security posture

See `SECURITY.md` for the full policy. Highlights:

- No API keys in source. `.env` is gitignored; keys rotated on any
  suspected exposure.
- Admin credentials sourced from env; insecure default only used with a
  loud warning in the log.
- All password handling uses PBKDF2-SHA256 + per-credential salt.
- `.dockerignore` keeps `.env`, `.users.json`, `.chat_history.json`,
  `.herbalist.db`, post-migration `*.migrated-backup-*` files,
  `.chroma_db/`, and `data/` out of the built image.
- Server-side logs emit plain text (no emojis) for easier grepping; UI
  keeps its emoji affordances.

---

## 6. Quality gates

- **Ruff** (`pyproject.toml` → `[tool.ruff]`) clean on `src/` and
  `tests/`.
- **Pytest** suite (53 tests) covering:
  - JSON extractors: `_strip_fences`, `_extract_route`,
    `_extract_expanded_queries`, `_extract_score`.
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
    ownership enforcement, `start_new_chat` / `delete_chat`
    `set_active_chat`, stats counters, and the idempotent
    JSON → SQLite migration (including the post-import rename to
    `*.migrated-backup-*`).
  - **UI state (`tests/test_state.py`):** session-state snapshotting,
    append → title promotion, feedback round-trip with
    `st.session_state` sync, `iter_all_feedback`, and switching
    between chats.
  - **Router safeguards (`tests/test_graph_router.py`):** deterministic
    identity/greeting bypass, LLM-backed medical routing, and lexical
    fallback when the LLM errors.
- **CI**: `.github/workflows/ci.yml` runs ruff + pytest on Python 3.10
  and 3.11 for every push / PR. Heavy third-party libs (chromadb,
  sentence-transformers, etc.) are stubbed inside `tests/conftest.py`
  so CI runs in seconds and stays deterministic.

---

## 7. Deployment

### 7.1 Local dev
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in API keys (+ admin creds if deploying)
streamlit run src/herbalist_assistant/ui/streamlit_app.py
```

### 7.2 Production container
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

## 8. Known limitations / future work

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

---

## 9. Change log (latest updates)

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
- **P1.13** SQLite + SQLAlchemy backend (`herbalist_assistant.db`)
  replaces `.users.json` + `.chat_history.json`. Schema: `users`,
  `chat_sessions`, `chat_messages`. `ensure_database_ready()` now acts
  as schema bootstrap only (idempotent `init_db()`); historical one-time
  JSON migration logic has been removed from runtime startup. Existing
  chat memory and the Admin panel's feedback viewer are fully DB-backed.
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
  deduped top-10 `candidate_docs`; `grade_documents_node` scores each doc
  0-100 and keeps high-confidence `selected_docs`.
- **Web Search Retry Strategy** `web_search_node` can rewrite queries on
  retries and aggregate/dedupe results across multiple rewritten queries.
- **Router/State Enhancements** Router initializes `generation_retries`,
  and state schema now includes retry counters, quality scores, profile
  context, and staged document collections.
- **Docs Alignment** README now reflects that runtime auto-migration from
  legacy JSON files is removed; database startup is schema bootstrap only.
- **Model Update** Replaced decommissioned `mixtral-8x7b-32768` with
  `llama-3.3-70b-versatile` across all LLM configuration code.
- **Router Tests** Added `tests/test_graph_router.py` covering deterministic
  routing safeguards and lexical fallback paths (3 tests).
