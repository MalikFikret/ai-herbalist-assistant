import asyncio
import hashlib
import html as _html
import json
import logging
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import herbalist_assistant  # eager .env load before LangChain/LangSmith imports

import streamlit as st
import streamlit.components.v1 as components

from herbalist_assistant import config, log_langsmith_status
from herbalist_assistant.db import ensure_database_ready
from herbalist_assistant.db import repository as db_repository
from herbalist_assistant.graph.advanced_graph import app as agent_graph_app

log_langsmith_status()

# Bring the SQLite schema up-to-date and run the one-time JSON -> SQLite
# migration the first time the process starts. Both steps are idempotent,
# so subsequent Streamlit reruns are near-instant.
ensure_database_ready()

# The relative imports below intentionally follow log_langsmith_status() so
# the LangSmith banner is emitted before the rest of the UI layer pulls in
# LangChain modules. Ruff flags this as E402, which we silence on purpose.
from .i18n import get_string  # noqa: E402
from .resources import reindex_pdfs  # noqa: E402
from .state import (  # noqa: E402
    append_message,
    delete_chat,
    get_chat_messages,
    get_user_chat_summaries,
    init_session_state,
    iter_all_feedback,
    set_active_chat,
    start_new_chat,
    update_message_feedback,
)

_logger = logging.getLogger("herbalist_assistant.ui")

# Admin credentials are sourced from environment variables so they are NOT
# committed in source. Supported variables (first match wins):
#   HA_ADMIN_USERNAME          - custom admin username (default: "admin")
#   HA_ADMIN_PASSWORD_HASH +
#   HA_ADMIN_PASSWORD_SALT     - PBKDF2-SHA256 hash (hex) and salt (hex)
#                                -- most secure, recommended for production
#   HA_ADMIN_PASSWORD          - plaintext fallback, hashed in-memory only
#
# If NONE of these are set, we fall back to the legacy dev default ("1234")
# and log a loud warning. Never leave this unset in a deployment.
ADMIN_USERNAME = os.environ.get("HA_ADMIN_USERNAME", "admin").strip() or "admin"
_ADMIN_PASSWORD_HASH_ENV = os.environ.get("HA_ADMIN_PASSWORD_HASH", "").strip()
_ADMIN_PASSWORD_SALT_ENV = os.environ.get("HA_ADMIN_PASSWORD_SALT", "").strip()
_ADMIN_PASSWORD_ENV = os.environ.get("HA_ADMIN_PASSWORD", "")
_ADMIN_DEFAULT_PASSWORD = "1234"
_ADMIN_DEFAULT_WARNING_EMITTED = False
_ADMIN_USERNAME_SETTING_KEY = "admin_username"
_ADMIN_PASSWORD_HASH_SETTING_KEY = "admin_password_hash"
_ADMIN_PASSWORD_SALT_SETTING_KEY = "admin_password_salt"

AVAILABLE_MODELS: List[str] = [
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemini-1.5-flash",
    "deepseek-chat",
]
DEFAULT_MODEL = "llama-3.1-8b-instant"
AVAILABLE_WEB_SEARCH_PROVIDERS: List[str] = ["Tavily", "DuckDuckGo"]
DEFAULT_WEB_SEARCH_PROVIDER = "DuckDuckGo"
# Maximum number of past messages we forward to the agent graph for memory.
# Matches _MAX_HISTORY_MESSAGES in advanced_graph.py.
_CHAT_HISTORY_LIMIT = 6

ALLERGY_HERB_ALIASES: Dict[str, List[str]] = {
    "papatya/chamomile": ["papatya", "chamomile", "camomile", "kamille"],
    "zencefil/ginger": ["zencefil", "ginger"],
    "nane/mint": ["nane", "mint", "peppermint"],
    "lavanta/lavender": ["lavanta", "lavender"],
    "ekinazya/echinacea": ["ekinazya", "echinacea"],
    "rezene/fennel": ["rezene", "fennel"],
    "isirgan/nettle": ["isirgan", "ısırgan", "nettle"],
}


_AGENT_TIMEOUT_SEC = 120


def _invoke_agent_with_timeout(
    payload: Dict[str, Any],
    timeout_sec: int = _AGENT_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """Run ``agent_graph_app.ainvoke`` under an asyncio timeout.

    Uses the compiled graph's native async entrypoint, wrapped in
    ``asyncio.wait_for`` so the UI never blocks forever. On timeout we
    raise ``TimeoutError`` so the caller can render a user-friendly message.
    """

    async def _runner() -> Dict[str, Any]:
        return await asyncio.wait_for(
            agent_graph_app.ainvoke(payload),
            timeout=timeout_sec,
        )

    try:
        return asyncio.run(_runner())
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Agent response exceeded {timeout_sec}s limit"
        ) from exc


def _truncate_chat_title(title: str, max_len: int = 34) -> str:
    clean = (title or "New Chat").strip()
    return clean if len(clean) <= max_len else f"{clean[: max_len - 3]}..."


def _sync_conversations_to_session(username: str) -> None:
    summaries = get_user_chat_summaries(username)
    st.session_state.conversations = [
        {
            "id": item.get("id", ""),
            "title": item.get("title", "New Chat"),
            "messages": get_chat_messages(username, item.get("id", "")),
        }
        for item in summaries
        if item.get("id")
    ]


def _build_profile_context(profile: Dict[str, str]) -> str:
    allergies = profile.get("allergies", "").strip()
    conditions = profile.get("conditions", "").strip()
    if not allergies and not conditions:
        return ""

    blocked_herbs = _extract_blocked_herbs(allergies)
    blocked_text = ", ".join(blocked_herbs) if blocked_herbs else "None"

    context_lines = [
        "User health profile:",
        f"- Name: {profile.get('name', '').strip() or 'Not provided'}",
        f"- Age: {profile.get('age', '').strip() or 'Not provided'}",
        f"- Gender: {profile.get('gender', '').strip() or 'Not provided'}",
        f"- Allergies: {allergies or 'None specified'}",
        f"- Current conditions: {conditions or 'None specified'}",
        f"- Strict blocked herbs (do not recommend): {blocked_text}",
        "",
        "Safety instructions for this answer:",
        "- Consider allergy and condition risks before suggesting herbs.",
        "- If a herb matches user allergy, NEVER recommend it.",
        "- Avoid potentially harmful options.",
        "- Provide safer alternatives when risk exists.",
        "",
        "Response format (mandatory):",
        "1) Recommendations list",
        "Keep answer concise, non-repetitive, and natural Turkish.",
    ]
    return "\n".join(context_lines)


def _extract_blocked_herbs(allergies_text: str) -> List[str]:
    text = (allergies_text or "").lower()
    blocked: List[str] = []

    for herb_name, aliases in ALLERGY_HERB_ALIASES.items():
        if any(alias in text for alias in aliases):
            blocked.append(herb_name)

    # Also include free-form allergy entries as blocked terms.
    extra_terms = re.split(r"[,;/\n]+", text)
    for term in extra_terms:
        clean = term.strip()
        if len(clean) >= 3 and clean not in blocked:
            blocked.append(clean)

    return blocked


def _collect_chat_history_for_agent() -> List[Dict[str, str]]:
    """Return the last few chat turns, excluding the current (just-appended) user msg.

    The advanced graph uses this for conversational memory so follow-up
    questions ("how do I prepare it?") keep their referent.
    """
    messages = st.session_state.get("messages", []) or []
    # The current user message has already been appended by the caller, so
    # drop it -- it lives in the `question` state key, not in history.
    prior = messages[:-1] if messages else []
    trimmed = prior[-_CHAT_HISTORY_LIMIT:]
    return [
        {
            "role": msg.get("role", ""),
            "content": msg.get("content", ""),
        }
        for msg in trimmed
        if msg.get("role") in ("user", "assistant") and msg.get("content")
    ]


def _generate_ai_response(user_input: str, profile: Dict[str, str]) -> tuple[str, List[Dict[str, Any]]]:
    """Generate answer from advanced LangGraph app, with graceful fallback.

    Returns (answer_text, sources) where ``sources`` is a list of structured
    dicts (``{"kind": "pdf", "file": ..., "page": ...}`` today; kept flexible
    so web URLs etc. can be added later without changing the call-site).
    """
    lang = st.session_state.get("language", "en")
    profile_context = _build_profile_context(profile)
    question_payload = user_input if not profile_context else f"{user_input}\n\n{profile_context}"
    chat_history = _collect_chat_history_for_agent()
    model_name = (
        st.session_state.get("selected_model")
        or DEFAULT_MODEL
    )
    web_search_provider = (
        st.session_state.get("web_search_provider")
        or DEFAULT_WEB_SEARCH_PROVIDER
    )

    try:
        with st.status(get_string(lang, "agent_status_thinking"), expanded=True) as status:
            status.write(get_string(lang, "agent_status_routing"))
            status.write(get_string(lang, "agent_status_searching"))
            status.write(get_string(lang, "agent_status_grading"))
            status.write(get_string(lang, "agent_status_generating"))
            final_state: Dict[str, Any] = _invoke_agent_with_timeout(
                {
                    "question": question_payload,
                    "chat_history": chat_history,
                    "model_name": model_name,
                    "web_search_provider": web_search_provider,
                },
                timeout_sec=_AGENT_TIMEOUT_SEC,
            )
            status.update(
                label=get_string(lang, "agent_status_done"),
                state="complete",
                expanded=False,
            )

        answer = final_state.get("final_answer", "I'm sorry, I could not generate an answer.")
        docs = final_state.get("documents", []) or []
        sources = _extract_sources_from_docs(docs)
        answer = _dedupe_answer_lines(answer)
        return answer, sources
    except TimeoutError:
        _logger.warning("Agent graph timed out after %ss for question=%r", _AGENT_TIMEOUT_SEC, user_input)
        return get_string(lang, "agent_timeout_msg"), []
    except Exception:
        _logger.exception("Agent graph invocation failed for question=%r", user_input)
        fallback = get_string(lang, "agent_error_msg")
        if profile_context:
            fallback += " " + get_string(lang, "agent_error_profile_hint")
        return fallback, []


def _extract_sources_from_docs(docs: List[Any]) -> List[Dict[str, Any]]:
    """Turn LangChain Documents into structured, UI-friendly source entries.

    Current shape:
        {"kind": "pdf", "file": "herbs.pdf", "page": 7}

    Leaves room for future kinds (e.g. ``{"kind": "url", "url": "https://..."}``)
    without changing the public return type.
    """
    structured: List[Dict[str, Any]] = []
    seen: set[tuple] = set()
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        url = str(meta.get("url", "")).strip()
        if url:
            title = str(meta.get("title", "")).strip() or str(
                meta.get("source", "Web Search")
            ).strip()
            key = ("url", url)
            if key in seen:
                continue
            seen.add(key)
            structured.append({"kind": "url", "url": url, "title": title})
            continue
        source = str(meta.get("source", "")).strip()
        if not source:
            continue
        file_name = Path(source).name
        page_raw = meta.get("page")
        try:
            page_num = int(page_raw) + 1 if page_raw is not None else None
        except (TypeError, ValueError):
            page_num = None
        key = (file_name, page_num)
        if key in seen:
            continue
        seen.add(key)
        structured.append({"kind": "pdf", "file": file_name, "page": page_num})
    return structured


def _dedupe_answer_lines(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    seen = set()
    result = []
    for line in lines:
        key = line.strip().lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        result.append(line)
    return "\n".join(result).strip()


def _inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ha-bg: var(--st-background-color, #ffffff);
            --ha-bg-2: var(--st-secondary-background-color, #f7faf8);
            --ha-text: var(--st-text-color, rgba(49, 51, 63, 1));
            --ha-border: var(--st-border-color, rgba(120, 120, 120, 0.22));
        }
        html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
            font-size: 16px !important;
            zoom: 1 !important;
            transform: none !important;
        }
        .main .block-container {
            width: 100%;
            max-width: 1560px;
            padding-bottom: 2rem;
        }
        [data-testid="stAppViewContainer"] {
            background: var(--ha-bg) !important;
        }
        .ha-auth-layout {
            --auth-panel-height: 640px;
            max-width: 1240px;
            margin: 0.75rem auto 0 auto;
        }
        .ha-auth-layout [data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }
        /* Style the right column's vertical block as the auth card. */
        .ha-auth-layout [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) > div[data-testid="stVerticalBlock"] {
            background: var(--ha-bg-2) !important;
            border: 1px solid var(--ha-border) !important;
            border-radius: 20px;
            padding: 1.3rem 1.3rem 1.1rem 1.3rem;
            box-shadow: 0 10px 28px rgba(22, 61, 39, 0.08);
            min-height: var(--auth-panel-height, 640px);
            box-sizing: border-box;
        }
        .ha-auth-hero {
            height: 100%;
            min-height: var(--auth-panel-height, 640px);
            box-sizing: border-box;
            border-radius: 20px;
            padding: 2rem 2rem 1.6rem 2rem;
            background: radial-gradient(circle at 18% 20%, #234835 0%, #1b3528 38%, #12251d 100%);
            color: #eaf5ef;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .ha-auth-hero-brand {
            font-size: 1rem;
            letter-spacing: 0.02em;
            opacity: 0.92;
        }
        .ha-auth-hero-title {
            margin-top: 1.4rem;
            font-size: clamp(2rem, 2.2vw, 2.9rem);
            line-height: 1.22;
            font-weight: 700;
            max-width: 18ch;
        }
        .ha-auth-hero-sub {
            margin-top: 0.85rem;
            font-size: 1rem;
            line-height: 1.45;
            color: #cfe6d8;
            max-width: 34ch;
        }
        .ha-auth-hero-foot {
            font-size: 0.85rem;
            color: #a8ceb8;
        }
        .ha-auth-wrap {
            max-width: 100%;
            margin: 0;
        }
        .ha-auth-card {
            display: contents;
        }
        .ha-auth-logo {
            width: 44px;
            height: 44px;
            margin: 0 auto 0.45rem auto;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #e8f5ec;
            color: #1f7a4d;
            font-size: 1.35rem;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(120, 120, 120, 0.16);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.1rem;
        }
        [data-testid="stSidebar"] .ha-sidebar-title {
            font-size: 1rem;
            font-weight: 600;
            color: var(--ha-text);
            margin: 0.25rem 0 0.5rem 0;
            opacity: 0.85;
        }
        [data-testid="stSidebar"] .ha-sidebar-subtitle {
            font-size: 0.85rem;
            color: var(--ha-text);
            opacity: 0.65;
            margin-bottom: 0.45rem;
        }
        [data-testid="stSidebar"] .ha-sidebar-selected {
            background: rgba(120, 120, 120, 0.16);
            border-radius: 10px;
            padding: 0.48rem 0.62rem;
            margin-bottom: 0.35rem;
            font-size: 0.96rem;
            font-weight: 500;
            color: var(--ha-text);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 10px;
            padding: 0.42rem 0.58rem;
            margin-bottom: 0.1rem;
            transition: background-color 0.15s ease;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(120, 120, 120, 0.12);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(120, 120, 120, 0.16);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none;
        }
        .ha-auth-title {
            text-align: center;
            font-size: 1.7rem;
            margin-bottom: 0.2rem;
            font-weight: 700;
            color: var(--ha-text);
        }
        .ha-auth-subtitle {
            text-align: center;
            margin-bottom: 1rem;
            color: var(--ha-text);
            opacity: 0.9;
        }
        .ha-auth-switch {
            margin-top: 0.45rem;
            text-align: center;
            font-size: 0.86rem;
            color: var(--ha-text);
        }
        .ha-auth-switch strong {
            color: #25784f;
            font-weight: 600;
        }
        .ha-auth-right-head {
            text-align: center;
            font-size: 1.9rem;
            font-weight: 700;
            color: var(--ha-text);
            margin-top: 0.1rem;
            margin-bottom: 0.65rem;
        }
        [data-testid="stRadio"] label p {
            font-size: 1.05rem !important;
        }
        /* Make labels visible against white card bg */
        label, [data-testid="stMarkdownContainer"] p {
            color: var(--ha-text) !important;
        }
        [data-testid="stForm"] div[data-baseweb="input"],
        [data-testid="stForm"] div[data-baseweb="select"] {
            border-radius: 12px !important;
            border: 1px solid var(--ha-border) !important;
            background: var(--ha-bg-2) !important;
            min-height: 56px;
            transition: all 0.15s ease;
            overflow: hidden;
            color: var(--ha-text) !important;
        }
        [data-testid="stForm"] input,
        [data-testid="stForm"] textarea,
        [data-testid="stForm"] [contenteditable="true"] {
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            caret-color: var(--ha-text) !important;
        }
        [data-testid="stForm"] div[data-baseweb="input"]:focus-within,
        [data-testid="stForm"] div[data-baseweb="select"]:focus-within {
            border-color: #5ca87d !important;
            box-shadow: 0 0 0 3px rgba(92, 168, 125, 0.16) !important;
        }
        [data-testid="stForm"] div[data-baseweb="input"] > div,
        [data-testid="stForm"] div[data-baseweb="select"] > div {
            background-color: transparent !important;
            border: none !important;
        }
        [data-testid="stForm"] button[kind="primary"] {
            border-radius: 12px !important;
            min-height: 44px;
            background: linear-gradient(180deg, #2f9d68 0%, #2a8a5d 100%) !important;
            border: 1px solid #2a8a5d !important;
            transition: transform 0.05s ease, filter 0.15s ease;
        }
        [data-testid="stFormSubmitButton"] > button {
            border-radius: 12px !important;
            min-height: 56px !important;
            background: linear-gradient(180deg, #2f9d68 0%, #2a8a5d 100%) !important;
            border: 1px solid #2a8a5d !important;
            color: #ffffff !important;
        }
        [data-testid="stFormSubmitButton"] > button:hover {
            filter: brightness(1.05) !important;
        }
        button[kind="primary"] {
            background: linear-gradient(180deg, #2f9d68 0%, #2a8a5d 100%) !important;
            border: 1px solid #2a8a5d !important;
        }
        a {
            color: #2a8a5d !important;
        }
        .ha-forgot-link > div > button {
            background: transparent !important;
            border: none !important;
            color: #2a8a5d !important;
            padding: 0 !important;
            min-height: auto !important;
            font-size: 0.9rem !important;
            justify-content: flex-start !important;
            box-shadow: none !important;
        }
        .ha-forgot-link > div > button:hover {
            color: #1e6644 !important;
            text-decoration: underline !important;
        }
        [data-testid="stForm"] button[kind="primary"]:hover {
            filter: brightness(1.05);
        }
        [data-testid="stForm"] button[kind="primary"]:active {
            transform: translateY(1px);
        }
        @media (max-width: 980px) {
            .ha-auth-hero,
            .ha-auth-layout [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) > div[data-testid="stVerticalBlock"] {
                border-radius: 16px;
                min-height: auto;
            }
            .ha-auth-hero {
                margin-bottom: 0.8rem;
            }
            .ha-auth-hero-title {
                font-size: 1.6rem;
            }
        }
        .ha-section-title {
            margin-top: 0.1rem;
            margin-bottom: 0.2rem;
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text-color);
        }
        .ha-section-subtitle {
            margin-bottom: 1rem;
            color: var(--text-color);
            opacity: 0.68;
            font-size: 0.95rem;
        }
        #MainMenu, [data-testid="stToolbarActions"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _force_sidebar_collapsed_on_load() -> None:
    """
    Streamlit persists sidebar open/closed state in browser localStorage.
    This forces the sidebar to start collapsed again on refresh/re-run.
    """
    components.html(
        """
        <script>
          (function () {
            try {
              const flagKey = "ha_sidebar_collapsed_reset";
              if (window.sessionStorage && window.sessionStorage.getItem(flagKey) === "1") {
                return;
              }
              if (window.sessionStorage) {
                window.sessionStorage.setItem(flagKey, "1");
              }
              for (const k of Object.keys(window.localStorage || {})) {
                if (k.toLowerCase().includes("sidebar")) {
                  window.localStorage.removeItem(k);
                }
              }
              window.location.reload();
            } catch (e) {
              // If anything fails, fall back to normal Streamlit behavior.
            }
          })();
        </script>
        """,
        height=0,
    )


def _init_auth_state() -> None:
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "is_logged_in" not in st.session_state:
        st.session_state.is_logged_in = False
    if "role" not in st.session_state:
        st.session_state.role = "user"
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL
    elif st.session_state.selected_model not in AVAILABLE_MODELS:
        st.session_state.selected_model = DEFAULT_MODEL
    if "web_search_provider" not in st.session_state:
        st.session_state.web_search_provider = DEFAULT_WEB_SEARCH_PROVIDER
    elif st.session_state.web_search_provider not in AVAILABLE_WEB_SEARCH_PROVIDERS:
        st.session_state.web_search_provider = DEFAULT_WEB_SEARCH_PROVIDER
    if "last_index_time" not in st.session_state:
        st.session_state.last_index_time = None
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Chat"
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {
            "name": "",
            "age": "",
            "gender": "",
            "allergies": "",
            "conditions": "",
        }


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()


def _verify_password(password: str, password_hash: str, salt: str) -> bool:
    return _hash_password(password, salt) == password_hash


def _admin_credentials_locked_by_env() -> bool:
    return bool(
        _ADMIN_PASSWORD_ENV
        or (_ADMIN_PASSWORD_HASH_ENV and _ADMIN_PASSWORD_SALT_ENV)
    )


def _load_runtime_admin_credentials() -> Dict[str, str]:
    username = db_repository.get_app_setting(_ADMIN_USERNAME_SETTING_KEY) or ADMIN_USERNAME
    password_hash = db_repository.get_app_setting(_ADMIN_PASSWORD_HASH_SETTING_KEY) or ""
    salt = db_repository.get_app_setting(_ADMIN_PASSWORD_SALT_SETTING_KEY) or ""
    return {
        "username": username.strip() or ADMIN_USERNAME,
        "password_hash": password_hash.strip(),
        "salt": salt.strip(),
    }


def _register_user(username: str, password: str, confirm_password: str) -> str:
    username = username.strip()
    if not username or not password or not confirm_password:
        return "All fields are required."
    if password != confirm_password:
        return "Passwords do not match."
    admin_username = _load_runtime_admin_credentials()["username"]
    if username == admin_username:
        return "This username is reserved."

    if db_repository.user_exists(username):
        return "User already exists."

    salt = secrets.token_hex(16)
    created = db_repository.create_user(
        username=username,
        password_hash=_hash_password(password, salt),
        salt=salt,
        role="user",
    )
    if not created:
        return "User already exists."
    return "Account created successfully. You can now log in."


def _reset_user_password(username: str, new_password: str, confirm_password: str) -> str:
    username = username.strip()
    if not username or not new_password or not confirm_password:
        return "All fields are required."
    if new_password != confirm_password:
        return "Passwords do not match."
    if len(new_password) < 4:
        return "Password must be at least 4 characters."
    admin_username = _load_runtime_admin_credentials()["username"]
    if username == admin_username:
        return "Admin password reset is disabled in this screen."

    salt = secrets.token_hex(16)
    updated = db_repository.reset_user_password(
        username=username,
        password_hash=_hash_password(new_password, salt),
        salt=salt,
    )
    if not updated:
        return "User not found."
    return "Password reset successful. You can now log in."


def _verify_admin_password(password: str) -> bool:
    """Verify the admin password against the most secure source available.

    Order of precedence:
      1. HA_ADMIN_PASSWORD_HASH + HA_ADMIN_PASSWORD_SALT (PBKDF2-SHA256).
      2. HA_ADMIN_PASSWORD (plaintext in env, compared with constant time).
      3. Hardcoded dev default "1234" (with a WARNING log).
    """
    global _ADMIN_DEFAULT_WARNING_EMITTED

    if _ADMIN_PASSWORD_HASH_ENV and _ADMIN_PASSWORD_SALT_ENV:
        expected = _ADMIN_PASSWORD_HASH_ENV
        actual = _hash_password(password, _ADMIN_PASSWORD_SALT_ENV)
        return secrets.compare_digest(expected, actual)

    if _ADMIN_PASSWORD_ENV:
        return secrets.compare_digest(_ADMIN_PASSWORD_ENV, password)

    runtime_creds = _load_runtime_admin_credentials()
    runtime_hash = runtime_creds.get("password_hash", "")
    runtime_salt = runtime_creds.get("salt", "")
    if runtime_hash and runtime_salt:
        actual = _hash_password(password, runtime_salt)
        return secrets.compare_digest(runtime_hash, actual)

    if not _ADMIN_DEFAULT_WARNING_EMITTED:
        _logger.warning(
            "Admin is using the insecure DEFAULT password. "
            "Set HA_ADMIN_PASSWORD (or HA_ADMIN_PASSWORD_HASH + "
            "HA_ADMIN_PASSWORD_SALT) in your environment before deploying."
        )
        _ADMIN_DEFAULT_WARNING_EMITTED = True
    return secrets.compare_digest(_ADMIN_DEFAULT_PASSWORD, password)


def _authenticate_user(username: str, password: str) -> Dict[str, str]:
    username = username.strip()
    if not username or not password:
        return {"status": "error", "message": "Username and password are required."}

    admin_username = _load_runtime_admin_credentials()["username"]
    if username == admin_username:
        if _verify_admin_password(password):
            return {"status": "ok", "role": "admin"}
        return {"status": "error", "message": "Wrong password."}

    record = db_repository.get_user_auth(username)
    if not record:
        return {"status": "error", "message": "User not found."}

    password_hash = record.get("password_hash", "")
    salt = record.get("salt", "")
    if not password_hash or not salt or not _verify_password(password, password_hash, salt):
        return {"status": "error", "message": "Wrong password."}

    return {"status": "ok", "role": record.get("role", "user")}


def _render_header() -> None:
    lang = st.session_state.get("language", "en")
    st.markdown(
        f'''
        <style>
        [data-testid="stHeader"]::after {{
            content: "{get_string(lang, "app_title")}";
            position: absolute;
            left: 3.8rem;
            top: 50%;
            transform: translateY(-50%);
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--ha-text, #31333f);
            pointer-events: none;
        }}
        .block-container {{
            padding-top: 3.5rem !important;
        }}
        </style>
        ''',
        unsafe_allow_html=True,
    )


def _list_pdf_files() -> List[Path]:
    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    return sorted(data_dir.glob("*.pdf"))


def _get_last_index_time() -> str:
    if st.session_state.last_index_time:
        return st.session_state.last_index_time

    chroma_db = Path(config.CHROMA_DIR) / "chroma.sqlite3"
    if chroma_db.exists():
        return datetime.fromtimestamp(chroma_db.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return "Not indexed yet"


def _require_admin() -> None:
    if st.session_state.role != "admin":
        st.error("Unauthorized access")
        st.stop()


def _logout() -> None:
    username = st.session_state.get("username", "").strip()
    role = st.session_state.get("role", "user")
    if username and role == "user":
        # Ensure next login starts in a fresh chat for the user.
        start_new_chat(username)
    st.session_state.clear()
    st.rerun()


def _render_auth_screen() -> None:
    lang = st.session_state.get("language", "en")
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "Login"

    st.markdown('<div class="ha-auth-layout">', unsafe_allow_html=True)
    left_col, right_col = st.columns([1, 1], gap="large")

    hero_foot = get_string(lang, "hero_foot")
    hero_foot_html = f'<div class="ha-auth-hero-foot">{hero_foot}</div>' if hero_foot else ""

    with left_col:
        st.markdown(
            f'''
            <div class="ha-auth-hero">
                <div>
                    <div class="ha-auth-hero-brand">{get_string(lang, "hero_brand")}</div>
                    <div class="ha-auth-hero-title">{get_string(lang, "hero_title")}</div>
                    <div class="ha-auth-hero-sub">{get_string(lang, "hero_sub")}</div>
                </div>
                {hero_foot_html}
            </div>
            ''',
            unsafe_allow_html=True,
        )

    with right_col:
        langs = {"en": "English", "tr": "Türkçe"}
        new_lang = st.selectbox(
            "Language / Dil",
            options=list(langs.keys()),
            format_func=lambda x: langs[x],
            index=list(langs.keys()).index(lang),
            key="language_selector_login",
            label_visibility="collapsed",
        )
        if new_lang != lang:
            st.session_state.language = new_lang
            st.rerun()

        st.markdown('<div class="ha-auth-logo">🌿</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ha-auth-right-head">{get_string(lang, "get_started")}</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.auth_mode not in {"Login", "Register", "Reset Password"}:
            st.session_state.auth_mode = "Login"

        auth_mode = st.session_state.auth_mode
        if auth_mode == "Reset Password":
            st.markdown(
                f'<div class="ha-auth-subtitle">{get_string(lang, "reset_subtitle")}</div>',
                unsafe_allow_html=True,
            )
            with st.form("reset_password_form", clear_on_submit=True):
                reset_username = st.text_input(get_string(lang, "username"), key="reset_username")
                new_password = st.text_input(
                    get_string(lang, "new_password"),
                    type="password",
                    key="reset_new_password",
                )
                confirm_new_password = st.text_input(
                    get_string(lang, "confirm_new_password"),
                    type="password",
                    key="reset_confirm_new_password",
                )
                submit_reset = st.form_submit_button(
                    get_string(lang, "reset_pwd_btn"),
                    type="primary",
                    use_container_width=True,
                )

            if submit_reset:
                result = _reset_user_password(reset_username, new_password, confirm_new_password)
                if result.startswith("Password reset successful"):
                    st.success(result)
                    st.session_state.auth_mode = "Login"
                    st.rerun()
                else:
                    st.error(result)

            if st.button(get_string(lang, "back_to_login"), use_container_width=True):
                st.session_state.auth_mode = "Login"
                st.rerun()
        else:
            st.markdown(
                f'<div class="ha-auth-subtitle">{get_string(lang, "auth_subtitle")}</div>',
                unsafe_allow_html=True,
            )
            tab_login, tab_register = st.tabs(
                [get_string(lang, "login_btn"), get_string(lang, "create_account")]
            )

            with tab_login:
                with st.form("login_form", clear_on_submit=False):
                    username = st.text_input(get_string(lang, "username"), key="login_username")
                    password = st.text_input(
                        get_string(lang, "password"),
                        type="password",
                        key="login_password",
                    )
                    submit_login = st.form_submit_button(
                        get_string(lang, "login_btn"),
                        type="primary",
                        use_container_width=True,
                    )

                st.markdown('<div class="ha-forgot-link">', unsafe_allow_html=True)
                if st.button(get_string(lang, "forgot_pwd"), use_container_width=False):
                    st.session_state.auth_mode = "Reset Password"
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            with tab_register:
                with st.form("register_form", clear_on_submit=True):
                    reg_username = st.text_input(get_string(lang, "username"), key="register_username")
                    reg_password = st.text_input(
                        get_string(lang, "password"),
                        type="password",
                        key="register_password",
                    )
                    reg_confirm = st.text_input(
                        get_string(lang, "confirm_password"),
                        type="password",
                        key="register_confirm_password",
                    )
                    submit_register = st.form_submit_button(
                        get_string(lang, "create_account"),
                        type="primary",
                        use_container_width=True,
                    )

            if submit_login:
                auth = _authenticate_user(username, password)
                if auth.get("status") != "ok":
                    st.error(auth.get("message", "Login failed."))
                else:
                    st.session_state.is_logged_in = True
                    st.session_state.username = username.strip()
                    st.session_state.role = auth.get("role", "user")
                    st.session_state.user_profile = _get_user_profile(st.session_state.username)
                    if st.session_state.role == "user":
                        start_new_chat(st.session_state.username)
                        _sync_conversations_to_session(st.session_state.username)
                    st.session_state.active_page = (
                        "Admin Panel" if st.session_state.role == "admin" else "Chat"
                    )
                    st.success("Login successful.")
                    st.rerun()

            if submit_register:
                result = _register_user(reg_username, reg_password, reg_confirm)
                if result.startswith("Account created"):
                    st.success(result)
                else:
                    st.error(result)

    st.markdown("</div>", unsafe_allow_html=True)

def _render_chat_page() -> None:
    lang = st.session_state.get("language", "en")
    username = st.session_state.username
    init_session_state(username)
    _sync_conversations_to_session(username)
    chats = get_user_chat_summaries(username)
    active_chat_id = st.session_state.get("active_chat_id", "")
    active_chat_title = next(
        (chat.get("title", get_string(lang, "new_chat")) for chat in chats if chat.get("id") == active_chat_id),
        get_string(lang, "new_chat"),
    )

    st.markdown(
        f'<div class="ha-section-title">{_truncate_chat_title(active_chat_title, 80)}</div>',
        unsafe_allow_html=True,
    )


    user_input = st.chat_input(get_string(lang, "chat_input_placeholder"))
    if not user_input and st.session_state.get("pending_prompt"):
        user_input = st.session_state.pop("pending_prompt")

    has_user_msg = any(m["role"] == "user" for m in st.session_state.messages)
    will_have_user_msg = has_user_msg or bool(user_input)

    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "assistant":
            if "Hello! I am your 🌿 AI Herbalist" in msg["content"] or "Merhaba! Ben sizin 🌿" in msg["content"]:
                continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_assistant_action_row(
                    lang=lang,
                    username=username,
                    chat_id=active_chat_id,
                    message_index=idx,
                    message=msg,
                )

    if not will_have_user_msg:
        with st.chat_message("assistant"):
            st.markdown(get_string(lang, "bot_greeting"))
            
        st.markdown(get_string(lang, "suggested_questions"))
        prompt_options = get_string(lang, "suggested_prompts")
        columns = st.columns(2)
        for i, question in enumerate(prompt_options):
            with columns[i % 2]:
                if st.button(question, key=f"suggested_{i}", use_container_width=True):
                    st.session_state.pending_prompt = question
                    st.rerun()

    if not user_input:
        return

    append_message(role="user", content=user_input, username=username)
    with st.chat_message("user"):
        st.markdown(user_input)

    answer, sources = _generate_ai_response(user_input, st.session_state.get("user_profile", {}))
    assistant_index = append_message(
        role="assistant",
        content=answer,
        username=username,
        sources=sources,
    )
    # Re-fetch in case append_message re-synced session state with extras (feedback=None).
    assistant_msg = (
        st.session_state.messages[assistant_index]
        if 0 <= assistant_index < len(st.session_state.messages)
        else {"content": answer, "sources": sources, "feedback": None}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
        _render_assistant_action_row(
            lang=lang,
            username=username,
            chat_id=st.session_state.get("active_chat_id", ""),
            message_index=assistant_index,
            message=assistant_msg,
        )


def _normalize_sources(sources: List[Any]) -> List[Dict[str, Any]]:
    """Accept both the legacy string shape and the new dict shape.

    Returns a list of dicts ready for the Sources popover. The schema is
    intentionally kept flexible so new ``kind`` values (e.g. ``"url"``) can
    be added without breaking existing chat history.
    """
    normalized: List[Dict[str, Any]] = []
    for src in sources or []:
        if isinstance(src, dict):
            if not src.get("kind"):
                src = {"kind": "pdf", **src}
            normalized.append(src)
        elif isinstance(src, str) and src.strip():
            normalized.append({"kind": "pdf", "file": src.strip(), "page": None})
    return normalized


def _source_entry_label(src: Dict[str, Any]) -> str:
    kind = str(src.get("kind") or "pdf").lower()
    if kind == "url":
        title = str(src.get("title") or src.get("url") or "Link").strip()
        return title
    file_name = str(src.get("file") or "unknown")
    page = src.get("page")
    return f"{file_name} (p. {page})" if page is not None else file_name


def _render_sources_popover(
    *,
    lang: str,
    sources: List[Any],
    message_index: int,
) -> None:
    """Render the Sources popover with one entry per source, keyed for uniqueness."""
    normalized = _normalize_sources(sources)
    if not normalized:
        return

    label_template = get_string(lang, "sources_btn")
    if not isinstance(label_template, str):
        label_template = "Sources ({count})"
    popover_label = label_template.format(count=len(normalized))

    try:
        container = st.popover(popover_label, use_container_width=False)
    except AttributeError:
        # Older Streamlit: fall back to an expander so nothing breaks.
        container = st.expander(popover_label, expanded=False)

    with container:
        for idx, src in enumerate(normalized, start=1):
            kind = str(src.get("kind") or "pdf").lower()
            if kind == "url" and src.get("url"):
                label = str(src.get("title") or src["url"]).strip() or str(src["url"])
                st.markdown(f"{idx}. [{label}]({src['url']})")
            else:
                st.markdown(f"{idx}. **{_source_entry_label(src)}**")


def _render_copy_button(
    *,
    text: str,
    key: str,
    label: str,
) -> None:
    """Render a small HTML+JS clipboard button.

    Streamlit has no native copy button, and ``st.code`` with its built-in
    copy icon would re-render the full answer as monospace. Instead we
    embed a tiny JS component sized to match the rest of the action row.
    """
    safe_text = json.dumps(text, ensure_ascii=False)
    safe_label = _html.escape(label)
    confirm_text = _html.escape(_copy_confirm_text())
    components.html(
        f"""
        <div style=\"display:flex;\">
          <button id=\"{key}\"
            style=\"
              border: 1px solid rgba(120,120,120,0.35);
              background: var(--background-color, #ffffff);
              color: var(--text-color, #222);
              border-radius: 8px;
              padding: 4px 10px;
              font-size: 0.85rem;
              cursor: pointer;
              display: inline-flex;
              align-items: center;
              gap: 6px;
            \">
            <span>{safe_label}</span>
          </button>
          <script>
            (function() {{
              const btn = document.getElementById(\"{key}\");
              if (!btn) return;
              btn.addEventListener(\"click\", async function () {{
                try {{
                  await navigator.clipboard.writeText({safe_text});
                }} catch (err) {{
                  const ta = document.createElement(\"textarea\");
                  ta.value = {safe_text};
                  document.body.appendChild(ta);
                  ta.select();
                  try {{ document.execCommand(\"copy\"); }} catch (e) {{}}
                  document.body.removeChild(ta);
                }}
                const previous = btn.innerText;
                btn.innerText = \"{confirm_text}\";
                setTimeout(function () {{ btn.innerText = previous; }}, 1200);
              }});
            }})();
          </script>
        </div>
        """,
        height=40,
    )


def _copy_confirm_text() -> str:
    lang = st.session_state.get("language", "en")
    return get_string(lang, "copy_done")


def _render_feedback_controls(
    *,
    lang: str,
    username: str,
    chat_id: str,
    message_index: int,
    current: str | None,
) -> None:
    """Render the 👍 / 👎 pair. Clicking toggles (click-again clears)."""
    up_label = "👍" if current != "up" else "✅ 👍"
    down_label = "👎" if current != "down" else "✅ 👎"
    base_key = f"fb_{chat_id}_{message_index}"

    col_up, col_down = st.columns([1, 1])
    with col_up:
        if st.button(
            up_label,
            key=f"{base_key}_up",
            help=get_string(lang, "feedback_up_help"),
            use_container_width=True,
        ):
            new_value: str | None = None if current == "up" else "up"
            if update_message_feedback(
                username=username,
                chat_id=chat_id,
                message_index=message_index,
                feedback=new_value,
            ):
                st.toast(get_string(lang, "feedback_saved"))
                st.rerun()
    with col_down:
        if st.button(
            down_label,
            key=f"{base_key}_down",
            help=get_string(lang, "feedback_down_help"),
            use_container_width=True,
        ):
            new_value = None if current == "down" else "down"
            if update_message_feedback(
                username=username,
                chat_id=chat_id,
                message_index=message_index,
                feedback=new_value,
            ):
                st.toast(get_string(lang, "feedback_saved"))
                st.rerun()


def _render_assistant_action_row(
    *,
    lang: str,
    username: str,
    chat_id: str,
    message_index: int,
    message: Dict[str, Any],
) -> None:
    """Render the [Copy] [Sources] [👍] [👎] row under an assistant message."""
    if not chat_id:
        return

    content = str(message.get("content", ""))
    sources = message.get("sources", []) or []
    current_feedback = message.get("feedback")

    has_sources = bool(_normalize_sources(sources))
    # 4 compact button-width columns, then a flexible spacer so the buttons
    # stay tight on wide screens.
    col_copy, col_src, col_fb, spacer = st.columns([1.2, 1.6, 1.6, 5])

    with col_copy:
        _render_copy_button(
            text=content,
            key=f"copy_{chat_id}_{message_index}",
            label=get_string(lang, "copy_btn"),
        )

    with col_src:
        if has_sources:
            _render_sources_popover(
                lang=lang,
                sources=sources,
                message_index=message_index,
            )
        else:
            st.caption(get_string(lang, "no_sources"))

    with col_fb:
        _render_feedback_controls(
            lang=lang,
            username=username,
            chat_id=chat_id,
            message_index=message_index,
            current=current_feedback,
        )

    del spacer  # reserved column, intentionally unused


def _render_admin_panel() -> None:
    lang = st.session_state.get("language", "en")
    _require_admin()
    st.markdown(f'<div class="ha-section-title">{get_string(lang, "admin_dashboard")}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ha-section-subtitle">{get_string(lang, "admin_subtitle")}</div>',
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(get_string(lang, "total_pdfs"), len(_list_pdf_files()))
    with metric_col2:
        st.metric(get_string(lang, "last_index_time"), _get_last_index_time())
    with metric_col3:
        st.metric(get_string(lang, "active_model"), st.session_state.selected_model)

    pdf_files = _list_pdf_files()

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown(get_string(lang, "pdf_upload"))
            uploaded = st.file_uploader(
                get_string(lang, "upload_btn"),
                type=["pdf"],
                accept_multiple_files=True,
                key="admin_pdf_uploader",
            )
            if uploaded:
                data_dir = Path(config.DATA_DIR)
                for file in uploaded:
                    target = data_dir / file.name
                    with target.open("wb") as f:
                        f.write(file.getbuffer())
                st.success(f"{len(uploaded)} PDF file(s) uploaded.")

        with st.container(border=True):
            st.markdown(get_string(lang, "reindex"))
            st.caption(get_string(lang, "reindex_desc"))
            if st.button(get_string(lang, "reindex_btn"), type="primary", use_container_width=True):
                with st.spinner("Rebuilding database from PDFs, please wait..."):
                    reindex_pdfs()
                st.session_state.last_index_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success(get_string(lang, "db_rebuilt"))

    with col2:
        with st.container(border=True):
            st.markdown(get_string(lang, "pdf_delete"))
            if pdf_files:
                selected_delete = st.multiselect(
                    get_string(lang, "delete_select"),
                    options=[p.name for p in pdf_files],
                    key="admin_pdf_delete_select",
                )
                if st.button(get_string(lang, "delete_btn"), use_container_width=True):
                    if not selected_delete:
                        st.warning(get_string(lang, "select_pdf_warn"))
                    else:
                        data_dir = Path(config.DATA_DIR)
                        deleted_count = 0
                        for filename in selected_delete:
                            target = data_dir / filename
                            if target.exists():
                                target.unlink()
                                deleted_count += 1
                        st.success(f"Deleted {deleted_count} PDF file(s).")
                        st.rerun()
            else:
                st.info(get_string(lang, "no_pdf"))

    st.markdown(
        f'<div class="ha-section-title" style="margin-top:1.2rem;">'
        f'{get_string(lang, "feedback_log_title")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="ha-section-subtitle">{get_string(lang, "feedback_log_desc")}</div>',
        unsafe_allow_html=True,
    )
    _render_admin_feedback_log(lang=lang)


def _render_admin_feedback_log(*, lang: str) -> None:
    """Flat, newest-first list of 👍/👎 signals across every user and chat."""
    entries = iter_all_feedback()
    if not entries:
        st.info(get_string(lang, "feedback_log_empty"))
        return

    ups = sum(1 for e in entries if e.get("feedback") == "up")
    downs = sum(1 for e in entries if e.get("feedback") == "down")

    filter_label = get_string(lang, "feedback_log_filter")
    filter_options = {
        "all": get_string(lang, "feedback_log_filter_all"),
        "up": get_string(lang, "feedback_log_filter_up"),
        "down": get_string(lang, "feedback_log_filter_down"),
    }
    selected_filter = st.radio(
        filter_label,
        options=list(filter_options.keys()),
        format_func=lambda k: filter_options[k],
        horizontal=True,
        key="admin_feedback_filter",
    )

    total_col, up_col, down_col = st.columns(3)
    with total_col:
        st.metric(get_string(lang, "feedback_log_total"), len(entries))
    with up_col:
        st.metric("👍", ups)
    with down_col:
        st.metric("👎", downs)

    filtered = (
        entries
        if selected_filter == "all"
        else [e for e in entries if e.get("feedback") == selected_filter]
    )

    if not filtered:
        st.info(get_string(lang, "feedback_log_empty"))
        return

    for entry in filtered:
        icon = "👍" if entry.get("feedback") == "up" else "👎"
        title = (
            f"{icon}  "
            f"{entry.get('username', '?')}"
            f" · {_truncate_chat_title(entry.get('chat_title', ''), 40)}"
            f" · {entry.get('feedback_at', '') or entry.get('timestamp', '')}"
        )
        with st.expander(title, expanded=False):
            question = entry.get("question") or ""
            if question:
                st.markdown(f"**{get_string(lang, 'feedback_log_question')}**")
                st.markdown(question)
            st.markdown(f"**{get_string(lang, 'feedback_log_answer')}**")
            st.markdown(entry.get("answer", ""))
            sources = entry.get("sources", []) or []
            normalized_sources = _normalize_sources(sources)
            if normalized_sources:
                st.caption(
                    get_string(lang, "feedback_log_sources")
                    + ": "
                    + ", ".join(_source_entry_label(s) for s in normalized_sources)
                )


def _render_about_page() -> None:
    lang = st.session_state.get("language", "en")
    st.markdown(f'<div class="ha-section-title">{get_string(lang, "about")}</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(get_string(lang, "about_desc"))


def _get_user_profile(username: str) -> Dict[str, str]:
    return db_repository.get_user_profile(username)


def _save_user_profile(username: str, profile: Dict[str, str]) -> bool:
    return db_repository.save_user_profile(username, profile)


def _render_profile_page() -> None:
    lang = st.session_state.get("language", "en")
    st.markdown(f'<div class="ha-section-title">{get_string(lang, "profile")}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ha-section-subtitle">{get_string(lang, "profile_desc")}</div>',
        unsafe_allow_html=True,
    )

    username = st.session_state.username
    if st.session_state.role == "admin":
        st.info("Admin profile editing is disabled in this view.")
        return

    current = _get_user_profile(username)
    st.session_state.user_profile = current
    with st.container(border=True):
        with st.form("profile_form", clear_on_submit=False):
            name = st.text_input(get_string(lang, "name"), value=current["name"])
            age = st.text_input(get_string(lang, "age"), value=current["age"])
            gender_options = get_string(lang, "gender_opts")
            
            # Map the current gender to its index in English/Turkish if exists, default to 0
            try:
                g_idx = gender_options.index(current["gender"])
            except ValueError:
                g_idx = 0

            gender = st.selectbox(
                get_string(lang, "gender"),
                options=gender_options,
                index=g_idx,
            )
            allergies = st.text_area(
                get_string(lang, "allergies"),
                value=current["allergies"],
                placeholder=get_string(lang, "allergies_placeholder"),
                height=90,
            )
            conditions = st.text_area(
                get_string(lang, "conditions"),
                value=current["conditions"],
                placeholder=get_string(lang, "conditions_placeholder"),
                height=120,
            )
            save_profile = st.form_submit_button(get_string(lang, "save_profile"), type="primary", use_container_width=True)

        if save_profile:
            if not name.strip():
                st.error(get_string(lang, "name_required"))
            else:
                is_saved = _save_user_profile(
                    username,
                    {
                        "name": name,
                        "age": age,
                        "gender": gender,
                        "allergies": allergies,
                        "conditions": conditions,
                    },
                )
                if is_saved:
                    st.session_state.user_profile = _get_user_profile(username)
                    st.success(get_string(lang, "profile_saved"))
                else:
                    st.error(get_string(lang, "profile_save_err"))


def run() -> None:
    st.set_page_config(
        page_title="AI Herbalist Assistant",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_global_styles()
    _init_auth_state()

    lang = st.session_state.get("language", "en")

    if not st.session_state.is_logged_in:
        _render_auth_screen()
        return
        
    # The rest of your run() code remains exactly as it is below this line...

    if st.session_state.role == "user":
        st.session_state.user_profile = _get_user_profile(st.session_state.username)

    if st.session_state.role != "admin":
        _render_header()

    with st.sidebar:
        st.markdown(get_string(lang, "sidebar_nav"))
        st.caption(f"**{st.session_state.username}**")

        with st.expander("Advanced Settings", expanded=False):
            selected_model = st.selectbox(
                "LLM Model",
                options=AVAILABLE_MODELS,
                index=AVAILABLE_MODELS.index(st.session_state.selected_model)
                if st.session_state.selected_model in AVAILABLE_MODELS
                else AVAILABLE_MODELS.index(DEFAULT_MODEL),
                key="sidebar_selected_model",
            )
            st.session_state.selected_model = selected_model
            web_search_provider = st.radio(
                "Web Search Provider",
                options=AVAILABLE_WEB_SEARCH_PROVIDERS,
                index=AVAILABLE_WEB_SEARCH_PROVIDERS.index(
                    st.session_state.web_search_provider
                )
                if st.session_state.web_search_provider in AVAILABLE_WEB_SEARCH_PROVIDERS
                else AVAILABLE_WEB_SEARCH_PROVIDERS.index(DEFAULT_WEB_SEARCH_PROVIDER),
                key="sidebar_web_search_provider",
            )
            st.session_state.web_search_provider = web_search_provider

        if st.session_state.role == "admin":
            sections = ["Admin Panel", "Chat"]
            current_index = sections.index(st.session_state.active_page) if st.session_state.active_page in sections else 0
            selected_section = st.radio(
                "Select section",
                sections,
                index=current_index,
                label_visibility="collapsed",
            )
            st.session_state.active_page = selected_section
        else:
            sections = ["Chat", "Profile"]
            current_index = sections.index(st.session_state.active_page) if st.session_state.active_page in sections else 0
            selected_section = st.radio(
                "Select section",
                sections,
                index=current_index,
                label_visibility="collapsed",
            )
            st.session_state.active_page = selected_section
            if st.button(get_string(lang, "new_chat"), use_container_width=True, type="primary"):
                start_new_chat(st.session_state.username)
                _sync_conversations_to_session(st.session_state.username)
                st.session_state.active_page = "Chat"
                st.rerun()

            if selected_section == "Chat":
                st.markdown(f'<div class="ha-sidebar-title">{get_string(lang, "your_chats")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ha-sidebar-subtitle">{get_string(lang, "recent_conversations")}</div>', unsafe_allow_html=True)
                user_chats = get_user_chat_summaries(st.session_state.username)
                active_chat_id = st.session_state.get("active_chat_id", "")
                
                chats_desc = []
                seen_chat_ids = set()
                for chat in reversed(user_chats):
                    chat_id = chat.get("id", "")
                    if not chat_id or chat_id in seen_chat_ids:
                        continue
                    if int(chat.get("message_count", 0) or 0) <= 1:
                        continue
                    if chat.get("title", get_string(lang, "new_chat")).strip().lower() == get_string(lang, "new_chat").lower() or chat.get("title", "New Chat").strip().lower() == "new chat":
                        continue
                    seen_chat_ids.add(chat_id)
                    chats_desc.append(chat)
                chat_ids = [chat.get("id", "") for chat in chats_desc if chat.get("id")]
                title_map = {chat.get("id", ""): chat.get("title", get_string(lang, "new_chat")) for chat in chats_desc}
                if chat_ids:
                    current_chat = active_chat_id if active_chat_id in chat_ids else chat_ids[0]
                    previous_sidebar_chat = st.session_state.get("sidebar_selected_chat_id")
                    selected_chat_id = st.radio(
                        "Conversation list",
                        options=chat_ids,
                        index=chat_ids.index(current_chat),
                        format_func=lambda cid: _truncate_chat_title(title_map.get(cid, get_string(lang, "new_chat"))),
                        label_visibility="collapsed",
                    )
                    st.session_state.sidebar_selected_chat_id = selected_chat_id
                    if previous_sidebar_chat is not None and selected_chat_id != previous_sidebar_chat:
                        if set_active_chat(st.session_state.username, selected_chat_id):
                            _sync_conversations_to_session(st.session_state.username)
                            st.session_state.active_page = "Chat"
                            st.rerun()
                    if st.button(get_string(lang, "delete_chat"), use_container_width=True):
                        if delete_chat(st.session_state.username, selected_chat_id):
                            _sync_conversations_to_session(st.session_state.username)
                            st.session_state.active_page = "Chat"
                            st.success(get_string(lang, "chat_deleted"))
                            st.rerun()
                        else:
                            st.error(get_string(lang, "chat_delete_err"))
                else:
                    st.session_state.sidebar_selected_chat_id = None
                    st.caption(get_string(lang, "no_past_chats"))
        st.divider()
        
        langs = {"en": "English", "tr": "Türkçe"}
        new_lang_sidebar = st.selectbox(
            "Language / Dil",
            options=list(langs.keys()),
            format_func=lambda x: langs[x],
            index=list(langs.keys()).index(lang),
            key="language_selector_sidebar",
            label_visibility="collapsed",
        )
        if new_lang_sidebar != lang:
            st.session_state.language = new_lang_sidebar
            st.rerun()

        if st.button(get_string(lang, "logout"), use_container_width=True):
            _logout()

    if selected_section == "Admin Panel":
        _render_admin_panel()
    elif selected_section == "Chat":
        _render_chat_page()
    elif selected_section == "Profile":
        _render_profile_page()
