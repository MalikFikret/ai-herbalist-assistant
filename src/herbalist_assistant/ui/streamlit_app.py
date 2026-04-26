import asyncio
import base64
import hashlib
import hmac
import html as _html
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Login hero: bundled logo (replaces "Welcome" heading when present).
_AUTH_LOGO_PATH = Path(__file__).resolve().parent / "static" / "herbalist_logo.png"
# Login background: soft botanical scene shown only on guest Login page.
_AUTH_BG_PATH = Path(__file__).resolve().parent / "static" / "login_background.png"


@st.cache_data(show_spinner=False)
def _auth_hero_logo_data_uri() -> str:
    if not _AUTH_LOGO_PATH.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(
        _AUTH_LOGO_PATH.read_bytes()
    ).decode("ascii")


@st.cache_data(show_spinner=False)
def _auth_background_data_uri() -> str:
    if not _AUTH_BG_PATH.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(
        _AUTH_BG_PATH.read_bytes()
    ).decode("ascii")


# ----- "Remember me" cookie support --------------------------------------
# We persist a HMAC-signed (username, expiry) tuple in a browser cookie.
# Verification happens server-side using HA_REMEMBER_SECRET (or a derived
# fallback) so a tampered cookie cannot impersonate another user.
_REMEMBER_COOKIE_NAME = "ha_remember"
_REMEMBER_TTL_DAYS = 30


def _remember_secret() -> bytes:
    """Stable per-deployment secret used to sign remember tokens.

    Production deployments should set ``HA_REMEMBER_SECRET``. For local dev we
    fall back to a hash of the admin password material so the secret stays
    consistent across reloads of the same process and machine.
    """
    env_secret = os.environ.get("HA_REMEMBER_SECRET", "").strip()
    if env_secret:
        return env_secret.encode("utf-8")
    seed = (
        ADMIN_USERNAME
        + "|"
        + (_ADMIN_PASSWORD_HASH_ENV or _ADMIN_PASSWORD_ENV or _ADMIN_DEFAULT_PASSWORD)
    )
    return hashlib.sha256(seed.encode("utf-8")).digest()


def _make_remember_token(username: str, ttl_days: int = _REMEMBER_TTL_DAYS) -> str:
    expires = int(time.time()) + ttl_days * 86400
    payload = f"{username}|{expires}"
    sig = hmac.new(_remember_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}|{sig}"


def _verify_remember_token(token: str) -> Optional[str]:
    if not token or token.count("|") != 2:
        return None
    try:
        username, expires_str, sig = token.split("|")
    except ValueError:
        return None
    payload = f"{username}|{expires_str}"
    expected = hmac.new(_remember_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        if int(expires_str) < int(time.time()):
            return None
    except ValueError:
        return None
    if not username:
        return None
    return username


# ----- Stored credentials (auto-fill password by username) ----------------
# Dosya: ``data/.remembered_logins.json``. İçinde her kullanıcı adı için XOR-
# obfuscate edilmiş şifre saklanır. Bu PBKDF2 hash'in YERİNE değil, login
# formundaki "isim yazınca şifre kendiliğinden gelsin" UX akışı için.
_REMEMBERED_LOGINS_FILENAME = ".remembered_logins.json"


def _remembered_logins_path() -> Path:
    p = Path(config.DATA_DIR) / _REMEMBERED_LOGINS_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _xor_keystream(secret: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(
            hmac.new(
                secret,
                nonce + counter.to_bytes(4, "big"),
                hashlib.sha256,
            ).digest()
        )
        counter += 1
    return bytes(out[:length])


def _obscure_password(plain: str) -> str:
    """Hafif obfuscation. Casual disk inspection'a karşı koruma; gerçek
    güvenlik PBKDF2 hash tarafında zaten var."""
    if not plain:
        return ""
    nonce = secrets.token_bytes(16)
    plain_b = plain.encode("utf-8")
    keystream = _xor_keystream(_remember_secret(), nonce, len(plain_b))
    cipher = bytes(p ^ k for p, k in zip(plain_b, keystream))
    return base64.urlsafe_b64encode(nonce + cipher).decode("ascii")


def _unobscure_password(token: str) -> Optional[str]:
    if not token:
        return None
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
    except Exception:
        return None
    if len(raw) < 16:
        return None
    nonce, cipher = raw[:16], raw[16:]
    keystream = _xor_keystream(_remember_secret(), nonce, len(cipher))
    try:
        return bytes(c ^ k for c, k in zip(cipher, keystream)).decode("utf-8")
    except Exception:
        return None


def _load_remembered_logins() -> dict:
    p = _remembered_logins_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_remembered_logins(data: dict) -> None:
    p = _remembered_logins_path()
    try:
        p.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        _logger.debug("Saving remembered logins failed", exc_info=True)


def _store_remembered_password(username: str, password: str) -> None:
    """Kayıt veya şifre değişimi sonrası username -> şifre eşlemesini saklar."""
    uname = (username or "").strip().lower()
    if not uname or not password:
        return
    data = _load_remembered_logins()
    data[uname] = _obscure_password(password)
    _save_remembered_logins(data)


def _lookup_remembered_password(username: str) -> Optional[str]:
    uname = (username or "").strip().lower()
    if not uname:
        return None
    data = _load_remembered_logins()
    token = data.get(uname)
    if not token:
        return None
    return _unobscure_password(str(token))


def _forget_remembered_password(username: str) -> None:
    uname = (username or "").strip().lower()
    if not uname:
        return
    data = _load_remembered_logins()
    if data.pop(uname, None) is not None:
        _save_remembered_logins(data)


try:
    import extra_streamlit_components as _stx  # type: ignore

    _COOKIES_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _stx = None
    _COOKIES_AVAILABLE = False
    _logger.warning(
        "extra-streamlit-components not installed; 'Remember me' on login will be disabled. "
        "Run: pip install extra-streamlit-components"
    )


_COOKIE_MGR_SESSION_KEY = "_ha_cookie_mgr"


def _initialize_cookie_manager() -> None:
    """Instantiate the CookieManager once per Streamlit script run.

    ``CookieManager(key="ha_cookies")`` registers a Streamlit component; calling
    it more than once in the same script run raises ``StreamlitDuplicateElementKey``.
    We call this helper exactly once at the top of :func:`run` and then read the
    cached instance via :func:`_get_cookie_manager` from anywhere else.
    """
    if not _COOKIES_AVAILABLE:
        st.session_state[_COOKIE_MGR_SESSION_KEY] = None
        return
    try:
        st.session_state[_COOKIE_MGR_SESSION_KEY] = _stx.CookieManager(
            key="ha_cookies"
        )
    except Exception:
        _logger.debug("CookieManager init failed", exc_info=True)
        st.session_state[_COOKIE_MGR_SESSION_KEY] = None


def _get_cookie_manager():
    """Return the CookieManager initialized for this run (or None)."""
    return st.session_state.get(_COOKIE_MGR_SESSION_KEY)

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


def _inject_guest_login_fullbleed_styles() -> None:
    """Guest Login only: remove sidebar column and expand main to full width (Streamlit 1.38+)."""
    bg_uri = _auth_background_data_uri()
    if bg_uri:
        st.markdown(
            f"""
            <style>
            /* Login kutusunun arkaplanı: botanik desen + üzerinde hafif beyaz overlay
               (form okunabilirliği için) — sayfa arkaplanı sade gri/beyaz kalır. */
            .st-key-ha_auth_card {{
                background-color: #ffffff !important;
                background-image:
                    linear-gradient(rgba(255, 255, 255, 0.82), rgba(255, 255, 255, 0.82)),
                    url("{bg_uri}") !important;
                background-size: cover, cover !important;
                background-position: center, center !important;
                background-repeat: no-repeat, no-repeat !important;
                border: 1px solid rgba(0, 0, 0, 0.06) !important;
                border-radius: 18px !important;
                box-shadow: 0 12px 40px rgba(40, 55, 45, 0.10) !important;
            }}
            /* Kart içindeki iç bloklar şeffaf ki desen görünsün */
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"],
            .st-key-ha_auth_card [data-testid="column"] > [data-testid="stVerticalBlock"],
            .st-key-ha_auth_card .ha-lux-welcome,
            .st-key-ha_auth_card .ha-lux-welcome__inner,
            .st-key-ha_auth_form_card,
            .st-key-ha_auth_form_card [data-testid="stVerticalBlockBorderWrapper"],
            .st-key-ha_auth_form_card [data-testid="stVerticalBlock"] {{
                background: transparent !important;
            }}
            /* Karta bindirilen ince halka deseni iptal — sade sadece botanik görünsün. */
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"]::before,
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"]::after {{
                content: none !important;
                display: none !important;
                background: none !important;
                background-image: none !important;
            }}
            /* Sol sütundaki dekoratif sarmaşık/yaprak süslemeleri de kapat */
            .st-key-ha_auth_card .ha-lux-welcome__vine,
            .st-key-ha_auth_card .ha-lux-welcome__ghost-leaf,
            .st-key-ha_auth_card .ha-lux-welcome__rule,
            .st-key-ha_auth_card .ha-lux-botanical,
            .st-key-ha_auth_card .ha-lux-botanical__accent,
            .st-key-ha_auth_card .ha-lux-botanical__photo,
            .st-key-ha_auth_card .ha-lux-pot {{
                display: none !important;
                background: none !important;
                background-image: none !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <style>
        /* Sidebar column + all chrome (1.51: stExpandSidebarButton, stSidebarCollapseButton) */
        [data-testid="stSidebar"],
        [data-testid="stSidebarHeader"],
        [data-testid="stSidebarContent"],
        [data-testid="stSidebarUserContent"],
        [data-testid="stSidebarNav"],
        [data-testid="stSidebarNavItems"] {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            flex: 0 0 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            border: none !important;
            transform: none !important;
            pointer-events: none !important;
        }
        [data-testid="collapsedControl"],
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarNavViewButton"] {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
            pointer-events: none !important;
        }
        /* Flex row: main consumes full width */
        [data-testid="stAppViewContainer"] {
            display: flex !important;
            flex-direction: row !important;
            margin-left: 0 !important;
            padding-left: 0 !important;
            width: 100% !important;
            max-width: 100vw !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stMain"],
        [data-testid="stAppViewContainer"] section.main {
            flex: 1 1 100% !important;
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            margin-left: 0 !important;
        }
        /* Header brand: no gap for hidden sidebar menu */
        [data-testid="stHeader"]::after {
            left: 1rem !important;
        }
        /* Guest login: normal document scroll (tall form / small viewports) */
        html, body {
            overflow-y: auto !important;
            overflow-x: hidden !important;
            min-height: 100dvh !important;
        }
        [data-testid="stApp"] {
            min-height: 100dvh !important;
            overflow: visible !important;
        }
        [data-testid="stAppViewContainer"] {
            min-height: calc(100dvh - 3.5rem) !important;
            overflow: visible !important;
            align-items: stretch !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stMain"],
        [data-testid="stAppViewContainer"] section.main {
            overflow: visible !important;
            min-height: 0 !important;
            height: auto !important;
            max-height: none !important;
            display: flex !important;
            flex-direction: column !important;
        }
        [data-testid="stMainBlockContainer"] {
            flex: 1 1 auto !important;
            min-height: 0 !important;
            max-height: none !important;
            overflow: visible !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            align-items: center !important;
            padding: 0.5rem 0 1.5rem 0 !important;
        }
        section.main .block-container {
            flex: 0 0 auto !important;
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
            box-sizing: border-box !important;
            width: 100% !important;
        }
        .main .block-container:has(.st-key-ha_auth_shell) {
            min-height: 0 !important;
            height: auto !important;
            max-height: none !important;
            padding-top: 0 !important;
            padding-bottom: 0.5rem !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            align-items: center !important;
        }
        .st-key-ha_auth_shell {
            width: 100% !important;
            max-width: 1020px !important;
            flex: 0 0 auto !important;
            margin: 0 auto !important;
            overflow: visible !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            justify-content: flex-start !important;
            gap: 0.2rem !important;
        }
        .st-key-ha_auth_shell [data-testid="element-container"]:has(button[data-testid="baseButton-tertiary"]) {
            margin-bottom: 0 !important;
        }
        /* "Misafir olarak devam et" butonu: login kartının üstünde sağa
           hizalı, belirgin pill stilinde — kullanıcı login olmadan misafir
           Chat'e dönebilsin diye her zaman görünür kalır. SADECE bu key
           hedefleniyor; Forgot Password tertiary butonu etkilenmiyor. */
        .st-key-ha_auth_back_chat {
            display: flex !important;
            justify-content: flex-end !important;
            margin-bottom: 0.5rem !important;
            margin-top: 0.1rem !important;
        }
        .st-key-ha_auth_back_chat button {
            background: rgba(255, 255, 255, 0.92) !important;
            border: 1px solid rgba(0, 0, 0, 0.10) !important;
            border-radius: 999px !important;
            padding: 0.4rem 0.95rem !important;
            min-height: 2.1rem !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            color: var(--ha-text, #1f2937) !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06) !important;
            text-decoration: none !important;
        }
        .st-key-ha_auth_back_chat button:hover {
            background: #ffffff !important;
            border-color: rgba(0, 0, 0, 0.22) !important;
            color: var(--ha-text, #1f2937) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_card,
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"] {
            flex: 0 1 auto !important;
            min-height: 0 !important;
            max-height: none !important;
            overflow: visible !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome {
            min-height: 0 !important;
            max-height: none !important;
            padding: 1.2rem 1.25rem 1rem 1.25rem !important;
            justify-content: center !important;
            align-items: center !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome h1 {
            font-size: clamp(2.05rem, 3.7vw, 2.75rem) !important;
            margin: 0 0 0.5rem 0 !important;
            text-align: center !important;
            width: 100% !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__logo {
            margin: 2.75rem auto !important;
            max-width: min(100%, 560px) !important;
            width: 100% !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__logo img {
            display: block !important;
            width: 100% !important;
            max-width: 560px !important;
            height: auto !important;
            margin: 0 auto !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__rule {
            margin: 0 auto 0.6rem auto !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__lead {
            font-size: 1.12rem !important;
            line-height: 1.58 !important;
            text-align: center !important;
            max-width: 40ch !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stMarkdownContainer"] {
            text-align: center !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stMarkdown"] {
            width: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            > div[data-testid="stVerticalBlock"]
            > [data-testid="element-container"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="element-container"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stElementContainer"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome h1 {
            text-align: center !important;
        }
        [data-testid="stMarkdown"] .ha-lux-welcome h1,
        [data-testid="stMarkdown"] .ha-lux-welcome p.ha-lux-welcome__lead,
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome h1,
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome p.ha-lux-welcome__lead {
            text-align: center !important;
        }
        .st-key-ha_auth_shell .ha-lux-botanical {
            display: none !important;
        }
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) > div[data-testid="stVerticalBlock"] {
            padding: 2.5rem 2.5rem 3rem 2.5rem !important;
            min-height: 0 !important;
            max-height: none !important;
            overflow: visible !important;
            justify-content: flex-start !important;
        }
        .st-key-ha_auth_shell .ha-lux-form-title {
            font-size: 1.35rem !important;
            margin-top: 0 !important;
        }
        .st-key-ha_auth_shell .ha-lux-form-sub {
            font-size: 0.82rem !important;
            margin-bottom: 0.35rem !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"] [data-testid="element-container"] {
            margin-bottom: 0.4rem !important;
        }
        @media (min-width: 901px) {
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(1) > div[data-testid="stVerticalBlock"] {
                height: 100% !important;
                min-height: 100% !important;
                display: flex !important;
                flex-direction: column !important;
                justify-content: center !important;
                align-items: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome {
                flex: 0 1 auto !important;
                min-height: 0 !important;
                width: 100% !important;
                max-width: 100% !important;
                justify-content: center !important;
                align-items: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome__inner {
                align-items: center !important;
                text-align: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome:has(.ha-lux-welcome__logo) .ha-lux-welcome__inner {
                min-height: min(42vh, 24rem) !important;
                justify-content: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome h1 {
                text-align: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome__rule {
                margin-left: auto !important;
                margin-right: auto !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome__lead {
                text-align: center !important;
            }
        }
        @media (max-width: 900px) {
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1)
                > div[data-testid="stVerticalBlock"] {
                justify-content: center !important;
                align-items: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome {
                min-height: 0 !important;
                max-height: none !important;
                justify-content: center !important;
                align-items: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome__inner {
                align-items: center !important;
                text-align: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome h1,
            .st-key-ha_auth_shell .ha-lux-welcome__lead {
                text-align: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome__rule {
                margin-left: auto !important;
                margin-right: auto !important;
            }
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) > div[data-testid="stVerticalBlock"] {
                min-height: 0 !important;
                max-height: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
        @import url("https://fonts.googleapis.com/css2?family=Lora:wght@600;700&family=Inter:wght@400;500;600&display=swap");
        :root {
            /* Soft, minimalist (ChatGPT-vari) palette: nötr beyaz/gri tonlar */
            --ha-bg: #ffffff;
            --ha-bg-2: #f7f7f8;
            --ha-sidebar-bg: #f9f9f9;
            --ha-text: #1f1f1f;
            --ha-text-soft: #6b7280;
            --ha-border: rgba(0, 0, 0, 0.08);
            --ha-surface: #ffffff;
            --ha-chat-ink: #1a1a1a;
            /* Çok hafif sıcak vurgu (herbal kimliğe minik bir gönderme) */
            --ha-accent-soft: #5b6e63;
            --ha-shell-1: #ffffff;
            --ha-shell-2: #ffffff;
            --ha-shell-3: #ffffff;
            --ha-shell-4: #ffffff;
            --ha-card-edge: rgba(0, 0, 0, 0.06);
            --ha-card-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            --ha-btn-max-w: min(100%, 15rem);
            /* Noise dokusu sade görünüm için kapatıldı (boş url, opaklık 0) */
            --ha-noise-tile: none;
        }
        /* App-wide: compact buttons; width follows label (not full column) */
        button[data-testid^="baseButton"] {
            min-height: 2.05rem !important;
            padding-top: 0.2rem !important;
            padding-bottom: 0.2rem !important;
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
            font-size: 0.84rem !important;
            width: auto !important;
            max-width: var(--ha-btn-max-w) !important;
            box-sizing: border-box !important;
        }
        [data-testid="element-container"]:has(button[data-testid^="baseButton"]) {
            width: fit-content !important;
            max-width: 100% !important;
        }
        /* Logged-in UI: nötr beyaz buton dolguları (ChatGPT-vari sade) */
        button[data-testid^="baseButton"] {
            background-color: #ffffff !important;
            background-image: none !important;
            border: 1px solid var(--ha-border) !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04) !important;
        }
        button[data-testid^="baseButton"] p,
        button[data-testid^="baseButton"] span {
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
        }
        button[data-testid^="baseButton"]:hover {
            background: #f5f5f5 !important;
            border-color: rgba(0, 0, 0, 0.12) !important;
        }
        div[role="radiogroup"] label {
            padding-top: 0.28rem !important;
            padding-bottom: 0.28rem !important;
            line-height: 1.2 !important;
        }
        [data-testid="stCheckbox"] label {
            line-height: 1.2 !important;
            padding-top: 0.18rem !important;
            padding-bottom: 0.18rem !important;
        }
        html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
            font-size: 16px !important;
            zoom: 1 !important;
            transform: none !important;
        }
        .main .block-container {
            width: 100%;
            max-width: 1560px;
            padding-top: 0.35rem !important;
            padding-bottom: 1.35rem !important;
            position: relative;
            z-index: 1;
        }
        [data-testid="stAppViewContainer"] {
            background: var(--ha-bg) !important;
        }
        html,
        body,
        #root {
            background: var(--ha-bg) !important;
        }
        [data-testid="stApp"] {
            background: var(--ha-bg) !important;
        }
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            background: transparent !important;
        }
        footer,
        [data-testid="stFooter"] {
            background: var(--ha-bg) !important;
        }
        section.main {
            position: relative;
            z-index: 0;
            background: transparent !important;
        }
        /* Sade görünüm: arka plan dokusu (film noise) tamamen kapatıldı */
        [data-testid="stAppViewContainer"] section.main::before {
            content: none !important;
            display: none !important;
        }
        .main .block-container:has(.st-key-ha_auth_shell) {
            max-width: 1120px;
            padding-top: 1.5rem !important;
            padding-bottom: 2.5rem !important;
            min-height: 0;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .st-key-ha_auth_shell {
            --ha-lux-ink: #4B5940;
            --ha-lux-moss: #708260;
            --ha-lux-cream: #F7F8F3;
            /* soft minimum height only; form can grow and the page scrolls */
            --ha-lux-card-h: min(32rem, 60vh);
            width: 100%;
            max-width: 1020px;
            margin: 0 auto;
            font-family: "Inter", system-ui, -apple-system, sans-serif;
            color: var(--ha-lux-ink);
        }
        .st-key-ha_auth_shell .st-key-ha_auth_card {
            width: 100%;
        }
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"] {
            position: relative;
            isolation: isolate;
            align-items: stretch !important;
            gap: 0 !important;
            background: #fbfbf9 !important;
            border-radius: 22px !important;
            overflow: hidden;
            box-shadow:
                0 24px 48px rgba(75, 89, 64, 0.1),
                0 4px 14px rgba(75, 89, 64, 0.06) !important;
            border: 1px solid rgba(112, 130, 96, 0.2) !important;
        }
        /* İnce halka deseni: tüm kart (sol + sağ) üstünde, sütun renkleri üzerine rgba ile bırakılır */
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"]::before {
            content: "";
            position: absolute;
            inset: 0;
            z-index: 0;
            border-radius: 22px;
            pointer-events: none;
            background-image:
                radial-gradient(
                    circle 200px at 6% 4%,
                    transparent 0%,
                    transparent 86%,
                    rgba(100, 112, 98, 0.1) 86%,
                    rgba(100, 112, 98, 0.1) 90.5%,
                    transparent 90.5%
                ),
                radial-gradient(
                    circle 240px at 96% 2%,
                    transparent 0%,
                    transparent 87%,
                    rgba(100, 112, 98, 0.08) 87%,
                    rgba(100, 112, 98, 0.08) 91%,
                    transparent 91%
                ),
                radial-gradient(
                    circle 300px at 50% 36%,
                    transparent 0%,
                    transparent 89%,
                    rgba(100, 112, 98, 0.065) 89%,
                    rgba(100, 112, 98, 0.065) 92.5%,
                    transparent 92.5%
                ),
                radial-gradient(
                    circle 180px at 2% 48%,
                    transparent 0%,
                    transparent 84%,
                    rgba(100, 112, 98, 0.07) 84%,
                    rgba(100, 112, 98, 0.07) 88%,
                    transparent 88%
                ),
                radial-gradient(
                    circle 220px at 100% 42%,
                    transparent 0%,
                    transparent 86%,
                    rgba(100, 112, 98, 0.075) 86%,
                    rgba(100, 112, 98, 0.075) 90%,
                    transparent 90%
                ),
                radial-gradient(
                    circle 160px at 20% 72%,
                    transparent 0%,
                    transparent 83%,
                    rgba(100, 112, 98, 0.06) 83%,
                    rgba(100, 112, 98, 0.06) 87%,
                    transparent 87%
                ),
                radial-gradient(
                    circle 280px at 80% 78%,
                    transparent 0%,
                    transparent 90%,
                    rgba(100, 112, 98, 0.055) 90%,
                    rgba(100, 112, 98, 0.055) 93.5%,
                    transparent 93.5%
                ),
                radial-gradient(
                    circle 130px at 12% 92%,
                    transparent 0%,
                    transparent 80%,
                    rgba(100, 112, 98, 0.065) 80%,
                    rgba(100, 112, 98, 0.065) 85%,
                    transparent 85%
                ),
                radial-gradient(
                    circle 200px at 55% 100%,
                    transparent 0%,
                    transparent 86%,
                    rgba(100, 112, 98, 0.05) 86%,
                    rgba(100, 112, 98, 0.05) 90%,
                    transparent 90%
                ) !important;
            background-repeat: no-repeat;
            background-size: 100% 100%;
        }
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            min-width: 0 !important;
            position: relative;
            z-index: 1;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1) {
            flex: 0.9 1 0% !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(2) {
            flex: 1.1 1 0% !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            > div[data-testid="stVerticalBlock"] {
            position: relative;
            padding: 0 !important;
            margin: 0 !important;
            /* Yarı saydam: kart genelindeki halka deseni sol panelde de görünsün */
            background-color: rgba(240, 242, 237, 0.78) !important;
            background-image:
                radial-gradient(ellipse 125% 90% at 88% 8%, rgba(180, 195, 172, 0.34) 0%, transparent 58%),
                radial-gradient(ellipse 100% 80% at -5% 78%, rgba(200, 208, 194, 0.3) 0%, transparent 55%),
                radial-gradient(ellipse 70% 55% at 40% 95%, rgba(186, 198, 178, 0.18) 0%, transparent 50%),
                linear-gradient(
                    168deg,
                    rgba(243, 244, 241, 0.92) 0%,
                    rgba(240, 242, 237, 0.88) 40%,
                    rgba(230, 232, 226, 0.9) 100%
                ) !important;
            background-repeat: no-repeat !important;
            gap: 0 !important;
        }
        @media (min-width: 901px) {
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            > div[data-testid="stVerticalBlock"] {
            height: 100% !important;
            min-height: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
        }
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(2)
            > div[data-testid="stVerticalBlock"] {
            position: relative;
            /* Açık zemin: ince halkalar üst seviyede (stHorizontalBlock::before) */
            background: rgba(251, 251, 249, 0.82) !important;
            border: none !important;
            padding: 2.5rem 2.5rem 3rem 2.5rem !important;
            min-height: var(--ha-lux-card-h);
            box-sizing: border-box;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            align-items: stretch !important;
        }
        /* Form / üst bar iç sarmalayıcıları şeffaf: yuvarlak desen tüm kutu alanında görünsün */
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(2)
            [data-testid="stVerticalBlock"]
            [data-testid="stVerticalBlock"] {
            background: transparent !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(2)
            [data-testid="stVerticalBlockBorderWrapper"] {
            background: transparent !important;
        }
        /* —— Welcome column (markdown); gradient is on the column stVerticalBlock —— */
        .st-key-ha_auth_shell .ha-lux-welcome {
            position: relative;
            min-height: var(--ha-lux-card-h);
            box-sizing: border-box;
            padding: 2.2rem 2.05rem 2rem 2.05rem;
            display: flex;
            flex-direction: column;
            background: transparent;
            flex: 1 1 auto;
            width: 100%;
            overflow: hidden;
            justify-content: center;
            align-items: center;
        }
        @media (min-width: 901px) {
        .st-key-ha_auth_shell .ha-lux-welcome {
            min-height: 0;
            flex: 0 1 auto;
            align-self: center;
        }
        .st-key-ha_auth_shell .ha-lux-welcome:has(.ha-lux-welcome__logo) .ha-lux-welcome__inner {
            min-height: min(42vh, 24rem) !important;
            justify-content: center !important;
        }
        }
        .st-key-ha_auth_shell .ha-lux-welcome__inner {
            position: relative;
            z-index: 2;
            flex: 0 1 auto;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            width: 100%;
            max-width: 100%;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__logo {
            margin: 2.75rem auto;
            max-width: min(100%, 560px);
            width: 100%;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__logo img {
            display: block;
            width: 100%;
            max-width: 560px;
            height: auto;
            margin: 0 auto;
            object-fit: contain;
            object-position: center center;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__vine {
            position: absolute;
            top: -24px;
            right: -36px;
            width: 200px;
            height: 200px;
            border: 2px solid rgba(112, 130, 96, 0.14);
            border-radius: 52% 38% 62% 48%;
            pointer-events: none;
            z-index: 0;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__ghost-leaf {
            position: absolute;
            bottom: 12%;
            left: -28px;
            width: 110px;
            height: 110px;
            background: rgba(112, 130, 96, 0.09);
            border-radius: 10% 85% 35% 80%;
            transform: rotate(-18deg);
            pointer-events: none;
            z-index: 0;
        }
        .st-key-ha_auth_shell .ha-lux-welcome h1 {
            font-family: "Lora", Georgia, "Times New Roman", serif;
            font-size: clamp(2.1rem, 3.5vw, 2.8rem);
            font-weight: 700;
            color: var(--ha-lux-ink);
            margin: 0 0 0.9rem 0;
            line-height: 1.18;
            letter-spacing: -0.025em;
            text-align: center;
            width: 100%;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__rule {
            width: 52px;
            height: 3px;
            border-radius: 2px;
            margin: 0 auto 1.05rem auto;
            background: linear-gradient(90deg, var(--ha-lux-moss), rgba(112, 130, 96, 0.35));
        }
        .st-key-ha_auth_shell .ha-lux-welcome__lead {
            margin: 0;
            max-width: 40ch;
            font-size: 1.12rem;
            line-height: 1.65;
            color: #5a6652;
            font-weight: 400;
            text-align: center;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stMarkdownContainer"] {
            text-align: center !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stMarkdown"] {
            width: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            > div[data-testid="stVerticalBlock"]
            > [data-testid="element-container"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="element-container"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stElementContainer"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome h1,
        .st-key-ha_auth_shell p.ha-lux-welcome__lead {
            text-align: center !important;
        }
        /* Streamlit [data-testid="stMarkdown"] wrapper can force start alignment; override on welcome only */
        [data-testid="stMarkdown"] .ha-lux-welcome,
        [data-testid="stMarkdown"] .ha-lux-welcome__inner,
        [data-testid="stMarkdown"] .ha-lux-welcome h1,
        [data-testid="stMarkdown"] .ha-lux-welcome p.ha-lux-welcome__lead,
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome,
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome h1,
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome p.ha-lux-welcome__lead {
            text-align: center !important;
        }
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome h1 {
            font-size: clamp(2.1rem, 3.5vw, 2.8rem) !important;
        }
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome p.ha-lux-welcome__lead {
            font-size: 1.12rem !important;
        }
        .st-key-ha_auth_shell .ha-lux-botanical {
            display: none !important;
        }
        .st-key-ha_auth_shell .ha-lux-botanical__accent {
            position: absolute;
            right: -0.5rem;
            bottom: -0.25rem;
            width: 140px;
            height: 140px;
            background: radial-gradient(
                circle at 35% 35%,
                rgba(112, 130, 96, 0.22),
                transparent 68%
            );
            border-radius: 50%;
            pointer-events: none;
        }
        .st-key-ha_auth_shell .ha-lux-botanical__photo,
        .st-key-ha_auth_shell .ha-lux-pot {
            display: none;
        }
        /* —— Auth language switcher (compact, top-right) —— */
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header {
            width: 100%;
            margin: 0 0 0.1rem 0;
            padding: 0;
            display: flex !important;
            justify-content: flex-end !important;
            align-items: flex-start !important;
            opacity: 0.92;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stVerticalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: flex-end !important;
            flex-wrap: wrap !important;
            gap: 0.25rem !important;
            width: auto !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stElementContainer"] {
            flex: 0 0 auto !important;
            width: auto !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] {
            width: auto !important;
            max-width: 100% !important;
            margin: 0 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] [data-baseweb="button-group"],
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] {
            gap: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            background: rgba(247, 248, 243, 0.65) !important;
            border-radius: 999px !important;
            border: 1px solid rgba(75, 89, 64, 0.07) !important;
            box-shadow: none !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] button[role="radio"],
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] button[role="radio"] {
            border-radius: 999px !important;
            min-height: 20px !important;
            min-width: 0 !important;
            padding: 0.1rem 0.4rem !important;
            font-size: 0.62rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.01em !important;
            line-height: 1.1 !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            background: transparent !important;
            background-image: none !important;
            color: #4b5940 !important;
            transition:
                background-color 0.16s ease,
                color 0.16s ease,
                box-shadow 0.16s ease !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] button[role="radio"]:hover,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] button[role="radio"]:hover {
            background: rgba(75, 89, 64, 0.06) !important;
            color: #4a5640 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"],
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] button[role="radio"][aria-checked="true"] {
            background: rgba(75, 89, 64, 0.2) !important;
            background-image: none !important;
            color: #3d4a35 !important;
            box-shadow: none !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"]:hover,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] button[role="radio"][aria-checked="true"]:hover {
            background: rgba(75, 89, 64, 0.26) !important;
            color: #323c2c !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] button[role="radio"][aria-checked="false"] p,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] button[role="radio"][aria-checked="false"] span,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] button[role="radio"][aria-checked="false"] p,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] button[role="radio"][aria-checked="false"] span {
            color: #6a7562 !important;
            -webkit-text-fill-color: #6a7562 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"] p,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"] span,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] button[role="radio"][aria-checked="true"] p,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] button[role="radio"][aria-checked="true"] span {
            color: #3d4a35 !important;
            -webkit-text-fill-color: #3d4a35 !important;
            font-weight: 600 !important;
        }
        /* Top: EN/TR sağda, hemen altında Login | Create Account (mockup ile aynı sıra) */
        .st-key-ha_auth_shell .st-key-ha_auth_top_bar {
            width: 100%;
            margin: 0 0 0.15rem 0;
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            gap: 0 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_top_bar .st-key-ha_auth_lang_header {
            align-self: flex-end !important;
            width: auto !important;
            max-width: 100% !important;
            margin: 0 0 0.4rem 0 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_top_bar .st-key-ha_auth_tab_row {
            width: 100%;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_top_bar .st-key-ha_auth_tab_row div[role="radiogroup"] {
            justify-content: center !important;
            margin: 0 0 1.05rem 0 !important;
        }
        /* Login / Create Account tabs — scoped so language control is not styled as a pill track */
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] {
            justify-content: center !important;
            gap: 0.5rem !important;
            margin: 0.55rem 0 1.05rem 0 !important;
            width: 100%;
            padding: 0.2rem !important;
            background: rgba(247, 248, 243, 0.85) !important;
            border-radius: 999px !important;
            border: 1px solid rgba(112, 130, 96, 0.14) !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label {
            border-radius: 999px !important;
            padding: 0.34rem 1.2rem !important;
            font-size: 0.84rem !important;
            font-weight: 500 !important;
            border: 1px solid transparent !important;
            background: transparent !important;
            margin: 0 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label p {
            font-size: 0.84rem !important;
            text-align: center !important;
            margin: 0 !important;
            color: var(--ha-lux-ink) !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label:has(input:checked) {
            background: var(--ha-lux-ink) !important;
            border-color: var(--ha-lux-ink) !important;
            box-shadow: 0 2px 8px rgba(75, 89, 64, 0.18) !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label:has(input:checked) p {
            color: #ffffff !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }
        /* —— Form stack —— */
        .st-key-ha_auth_shell .ha-lux-form-title {
            font-family: "Lora", Georgia, serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--ha-lux-ink);
            text-align: center;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.02em;
        }
        .st-key-ha_auth_shell .ha-lux-form-sub {
            text-align: center;
            font-size: 0.92rem;
            line-height: 1.5;
            color: #5f6b56;
            margin: 0 auto 1.35rem auto;
            max-width: 38ch;
        }
        .st-key-ha_auth_form_card [data-testid="stVerticalBlock"] {
            gap: 0.75rem !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"] div[data-baseweb="input"],
        .st-key-ha_auth_shell [data-testid="stForm"] div[data-baseweb="select"] {
            border-radius: 14px !important;
            min-height: 56px !important;
            background: rgba(247, 248, 243, 0.95) !important;
            border: 1px solid rgba(112, 130, 96, 0.2) !important;
            box-shadow: none !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"] [data-baseweb="input"] input {
            padding: 0.65rem 1.1rem !important;
            line-height: 1.45 !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"] input::placeholder,
        .st-key-ha_auth_shell [data-testid="stForm"] input::-webkit-input-placeholder {
            color: #7a8578 !important;
            opacity: 1 !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"] div[data-baseweb="input"]:focus-within {
            border-color: var(--ha-lux-moss) !important;
            box-shadow: 0 0 0 3px rgba(112, 130, 96, 0.2) !important;
        }
        .st-key-ha_lux_remember_row [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-ha_lux_remember_row [data-testid="stVerticalBlock"] {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        .st-key-ha_lux_remember_row [data-testid="stHorizontalBlock"] {
            align-items: center !important;
            margin-top: 0.1rem !important;
            margin-bottom: 0.45rem !important;
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            gap: 0.35rem !important;
        }
        .st-key-ha_lux_remember_row [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            display: flex !important;
            align-items: center !important;
        }
        .st-key-ha_lux_remember_row [data-testid="column"]:first-child [data-testid="stCheckbox"] {
            display: flex !important;
            align-items: center !important;
        }
        .st-key-ha_lux_remember_row [data-testid="column"]:first-child label {
            margin-bottom: 0 !important;
        }
        .st-key-ha_lux_remember_row [data-testid="column"]:last-child {
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
        }
        .st-key-ha_lux_remember_row [data-testid="stFormSubmitButton"] {
            flex: 0 0 auto !important;
            margin-bottom: 0 !important;
        }
        .st-key-ha_lux_remember_row [data-testid="stFormSubmitButton"] [data-testid="stMarkdownContainer"] p {
            margin: 0 !important;
            line-height: 1.3 !important;
        }
        /* Forgot password butonu: form'un DIŞINDA, Login butonunun hemen
           altında ortalı tertiary link olarak. Form içinde tek submit
           (Login) bulunduğundan Enter doğrudan Login'i tetikler. */
        .st-key-ha_lux_forgot_row {
            margin-top: 0.35rem !important;
            display: flex !important;
            justify-content: center !important;
        }
        .st-key-ha_lux_forgot_row [data-testid="stElementContainer"] {
            width: auto !important;
            margin: 0 !important;
        }
        .st-key-ha_lux_forgot_row [data-testid="stButton"] {
            display: inline-flex !important;
            justify-content: center !important;
        }
        .st-key-ha_lux_forgot_row button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0.25rem 0.5rem !important;
            min-height: 1.8rem !important;
            font-size: 0.88rem !important;
            color: var(--ha-text-soft, #6b7280) !important;
            text-decoration: underline;
            text-underline-offset: 3px;
            text-decoration-color: rgba(107, 114, 128, 0.35);
        }
        .st-key-ha_lux_forgot_row button:hover {
            background: transparent !important;
            color: var(--ha-text, #1f1f1f) !important;
            text-decoration-color: currentColor;
        }
        /* Primary submit — smaller pill, centered; does not touch card edges (column padding) */
        .st-key-ha_auth_shell .st-key-ha_auth_primary_submit {
            display: flex !important;
            justify-content: center !important;
            width: 100% !important;
            margin-top: 0.25rem !important;
            margin-bottom: 0.75rem !important;
            padding: 0 0.75rem 0.5rem 0.75rem !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_primary_submit [data-testid="element-container"] {
            max-width: 180px !important;
            width: 100% !important;
            margin-left: auto !important;
            margin-right: auto !important;
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_primary_submit [data-testid="stFormSubmitButton"] {
            margin-bottom: 0 !important;
            width: 100% !important;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"],
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"] {
            border-radius: 12px !important;
            min-height: 32px !important;
            font-weight: 600 !important;
            font-size: 0.8rem !important;
            letter-spacing: 0.02em !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background-color: #6d7f62 !important;
            background-image: linear-gradient(180deg, #95a88a 0%, #6d7f62 100%) !important;
            border: 1px solid #5d6e54 !important;
            box-shadow:
                0 3px 10px rgba(75, 89, 64, 0.14),
                inset 0 1px 0 rgba(255, 255, 255, 0.18) !important;
            transition:
                box-shadow 0.2s ease,
                transform 0.2s ease,
                filter 0.2s ease !important;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] *,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"] * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-weight: inherit !important;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"]:hover,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
            filter: brightness(1.05) saturate(1.03) !important;
            box-shadow:
                0 6px 16px rgba(75, 89, 64, 0.2),
                0 2px 5px rgba(75, 89, 64, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.22) !important;
            transform: translateY(-1px) !important;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"]:active,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"]:active {
            transform: translateY(0) !important;
            filter: brightness(0.98) !important;
            box-shadow:
                0 2px 8px rgba(75, 89, 64, 0.14),
                inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"]:has(.st-key-ha_auth_primary_submit) [data-testid="stVerticalBlock"] > div:last-of-type [data-testid="element-container"] {
            margin-bottom: 0 !important;
        }
        /* Forgot password — link style (tertiary + secondary fallback) */
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-secondary"],
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="secondary"],
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-tertiary"],
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="tertiary"] {
            border-radius: 0 !important;
            min-height: auto !important;
            min-width: 0 !important;
            padding: 0.2rem 0 !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
            color: var(--ha-lux-moss) !important;
            background: transparent !important;
            background-image: none !important;
            border: none !important;
            box-shadow: none !important;
            text-decoration: underline !important;
            text-underline-offset: 3px;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-secondary"]:hover,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="secondary"]:hover,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-tertiary"]:hover,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="tertiary"]:hover {
            color: var(--ha-lux-ink) !important;
            background: transparent !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"] [data-testid="element-container"] {
            margin-bottom: 0.65rem;
        }
        .st-key-ha_auth_shell [data-testid="stAlert"],
        .st-key-ha_auth_shell [data-baseweb="notification"] {
            width: 100% !important;
            max-width: 420px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            border-radius: 14px !important;
        }
        .st-key-ha_auth_shell [data-testid="column"]:nth-of-type(2) [data-testid="stVerticalBlock"] [data-testid="stAlert"] {
            margin-top: 0.45rem !important;
            margin-bottom: 0.35rem !important;
        }
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-ft-strong {
            color: var(--ha-lux-moss);
            font-weight: 600;
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
        [data-testid="stSidebar"] {
            position: relative;
            z-index: 0;
            border-right: 1px solid var(--ha-border) !important;
            background: var(--ha-sidebar-bg) !important;
        }
        /* Sidebar üzerindeki noise dokusu sade görünüm için kapatıldı */
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"]::before {
            content: none !important;
            display: none !important;
        }
        [data-testid="stSidebar"] > * {
            position: relative;
            z-index: 1;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 0.65rem !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }
        /* Sidebar: position: relative ki içindeki absolute konumlu login bloğu
           en alta yapışabilsin. Block-container'a ise login bloğunun altta
           kalacağı kadar bottom padding veriyoruz, böylece üstteki içerik
           login bloğu altında kaybolmasın. */
        [data-testid="stSidebar"] {
            position: relative !important;
        }
        [data-testid="stSidebar"] .block-container:has(.st-key-ha_sidebar_login_row) {
            padding-bottom: 11.5rem !important;
        }
        /* Login satırı (açıklama + buton) sidebar'ın gerçek diptaki kenarına
           absolute olarak yapıştırılır. Bu sayede flex zincirine bağımlı değil. */
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(> .st-key-ha_sidebar_login_row),
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.st-key-ha_sidebar_login_row) {
            position: absolute !important;
            left: 0.85rem !important;
            right: 0.85rem !important;
            bottom: 0.6rem !important;
            width: auto !important;
            margin: 0 !important;
            padding-top: 0.85rem !important;
            padding-bottom: 0 !important;
            border-top: 1px solid var(--ha-border) !important;
            background: var(--ha-sidebar-bg) !important;
            z-index: 5 !important;
        }
        .st-key-ha_sidebar_login_row {
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            border: none !important;
        }
        /* Login kartı: başlık + açıklama (sade, kutusuz) */
        .ha-sidebar-login-card {
            margin: 0 0 0.55rem 0 !important;
            padding: 0 !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .ha-sidebar-login-title {
            margin: 0 0 0.22rem 0 !important;
            padding: 0 !important;
            font-family: "Inter", system-ui, -apple-system, sans-serif !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            line-height: 1.3 !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
        }
        /* Açıklama metni: küçük, açık gri, sade (kutusuz) */
        .ha-sidebar-login-hint {
            margin: 0 !important;
            padding: 0 !important;
            font-family: "Inter", system-ui, -apple-system, sans-serif !important;
            font-size: 0.72rem !important;
            font-weight: 400 !important;
            line-height: 1.4 !important;
            color: var(--ha-text-soft) !important;
            -webkit-text-fill-color: var(--ha-text-soft) !important;
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
        }
        /* Premium header: nav label + user card */
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(.ha-sidebar-header) {
            margin-bottom: 0.15rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(.ha-sidebar-header) p {
            margin: 0 !important;
            color: inherit !important;
        }
        .ha-sidebar-header {
            margin: 0 0 0.75rem 0;
        }
        .ha-sidebar-header__eyebrow {
            display: flex !important;
            align-items: center;
            gap: 0.45rem;
            margin: 0 0 0.65rem 0 !important;
            padding: 0 !important;
            font-family: "Inter", system-ui, -apple-system, sans-serif !important;
            font-size: 0.62rem !important;
            font-weight: 650 !important;
            letter-spacing: 0.2em !important;
            text-transform: uppercase !important;
            color: var(--ha-text-soft) !important;
            -webkit-text-fill-color: var(--ha-text-soft) !important;
        }
        .ha-sidebar-header__eyebrow--tr {
            text-transform: none !important;
            letter-spacing: 0.14em !important;
            font-size: 0.68rem !important;
        }
        .ha-sidebar-header__eyebrow::before {
            content: "";
            flex-shrink: 0;
            width: 1.1rem;
            height: 2px;
            border-radius: 2px;
            background: linear-gradient(90deg, rgba(0, 0, 0, 0.35), rgba(0, 0, 0, 0.05));
        }
        .ha-sidebar-header__user-card {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.5rem 0.65rem;
            background: #ffffff;
            border: 1px solid var(--ha-border);
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .ha-sidebar-header__avatar {
            width: 2.1rem;
            height: 2.1rem;
            border-radius: 10px;
            flex-shrink: 0;
            display: flex !important;
            align-items: center;
            justify-content: center;
            font-family: "Inter", system-ui, sans-serif !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            line-height: 1 !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background: linear-gradient(145deg, #4a4a4a 0%, #1f1f1f 92%);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }
        .ha-sidebar-header__user-meta {
            min-width: 0;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 0.12rem;
        }
        .ha-sidebar-header__hint {
            font-family: "Inter", system-ui, sans-serif !important;
            font-size: 0.62rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
            color: var(--ha-text-soft) !important;
            -webkit-text-fill-color: var(--ha-text-soft) !important;
        }
        .ha-sidebar-header__name {
            font-family: "Inter", system-ui, sans-serif !important;
            font-size: 0.94rem !important;
            font-weight: 650 !important;
            letter-spacing: -0.02em !important;
            line-height: 1.25 !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            word-break: break-word;
        }
        [data-testid="stSidebar"] div[data-testid="stExpander"] {
            border: 1px solid var(--ha-card-edge) !important;
            border-radius: 12px !important;
            overflow: hidden;
            background: var(--ha-surface) !important;
            margin-bottom: 0.65rem !important;
            box-shadow: var(--ha-card-shadow);
        }
        [data-testid="stSidebar"] div[data-testid="stExpander"] details {
            border: none !important;
            background: transparent !important;
        }
        [data-testid="stSidebar"] hr {
            margin: 1.15rem 0 0.75rem 0 !important;
            border: none !important;
            border-top: 1px solid var(--ha-border) !important;
        }
        [data-testid="stSidebar"] button[kind="primary"] {
            background: #ffffff !important;
            background-image: none !important;
            border: 1px solid var(--ha-border) !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            border-radius: 10px !important;
            min-height: 2.2rem !important;
            font-size: 0.85rem !important;
            font-weight: 550 !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04) !important;
        }
        [data-testid="stSidebar"] button[kind="primary"] p,
        [data-testid="stSidebar"] button[kind="primary"] span {
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
        }
        [data-testid="stSidebar"] button[kind="primary"]:hover {
            background: #f3f3f4 !important;
            filter: none !important;
            border-color: rgba(0, 0, 0, 0.12) !important;
        }
        [data-testid="stSidebar"] button[kind="secondary"] {
            min-height: 2.05rem !important;
            font-size: 0.83rem !important;
            border-radius: 9px !important;
            border: 1px solid var(--ha-border) !important;
            background: #ffffff !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
        }
        [data-testid="stSidebar"] .ha-sidebar-title {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--ha-text-soft);
            margin: 0.65rem 0 0.28rem 0;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            opacity: 0.85;
        }
        [data-testid="stSidebar"] .ha-sidebar-subtitle {
            font-size: 0.74rem;
            color: var(--ha-text-soft);
            opacity: 0.85;
            margin-bottom: 0.32rem;
            font-weight: 500;
        }
        [data-testid="stSidebar"] .ha-sidebar-selected {
            background: rgba(0, 0, 0, 0.04);
            border: 1px solid transparent;
            border-radius: 9px;
            padding: 0.38rem 0.52rem;
            margin-bottom: 0.32rem;
            font-size: 0.86rem;
            font-weight: 520;
            color: var(--ha-text);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            box-shadow: none;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 9px;
            padding: 0.34rem 0.55rem;
            margin-bottom: 0.08rem;
            transition: background-color 0.15s ease;
            font-size: 0.9rem !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(0, 0, 0, 0.04);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(0, 0, 0, 0.06);
            border: 1px solid transparent !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none;
        }
        /* ====== ChatGPT tarzı sabit chat düzeni ======
           - body/html: 100vh, overflow hidden (sayfa scroll kapalı)
           - stApp / stAppViewContainer: Streamlit'in kendi flex/height hesabına
             güveniriz; sadece overflow'u kapatırız (yoksa bizim verdiğimiz 100vh
             header'ın altında taşmaya yol açıyordu).
           - Tek scrollable alan: stMainBlockContainer (mesajlar)
           - Composer: position: fixed; viewport'a göre konumlanır, sidebar açıkken
             ana içerik alanına hizalı gözükür. */
        html:has(.st-key-ha_chat_composer_row),
        body:has(.st-key-ha_chat_composer_row) {
            height: 100vh !important;
            max-height: 100vh !important;
            overflow: hidden !important;
        }
        [data-testid="stApp"]:has(.st-key-ha_chat_composer_row),
        [data-testid="stAppViewContainer"]:has(.st-key-ha_chat_composer_row) {
            overflow: hidden !important;
        }
        /* stMain: taşma kesilir, içerik scroll alanı içinde kalsın. */
        [data-testid="stAppViewContainer"]:has(.st-key-ha_chat_composer_row) [data-testid="stMain"] {
            overflow: hidden !important;
            position: relative !important;
        }
        /* stMainBlockContainer: TEK scroll alanı (mesajlar). Composer'ın altta
           kapladığı alan kadar bottom padding ekleriz, son mesaj gizlenmesin. */
        [data-testid="stAppViewContainer"]:has(.st-key-ha_chat_composer_row) [data-testid="stMainBlockContainer"] {
            height: 100% !important;
            max-height: 100% !important;
            min-height: 0 !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding-bottom: calc(7.5rem + env(safe-area-inset-bottom, 0px)) !important;
            scroll-behavior: smooth !important;
        }
        /* İçerideki vertical block'ların yüksekliği serbest kalsın. */
        [data-testid="stAppViewContainer"]:has(.st-key-ha_chat_composer_row) [data-testid="stMain"] [data-testid="stVerticalBlock"] {
            overflow: visible !important;
        }
        section.main:has(.st-key-ha_chat_composer_row) [data-testid="stChatMessage"],
        [data-testid="stMain"]:has(.st-key-ha_chat_composer_row) [data-testid="stChatMessage"] {
            margin-bottom: 0.3rem !important;
        }
        section.main:has(.st-key-ha_chat_composer_row) [class*="st-key-ha_assistant_actions_row"],
        [data-testid="stMain"]:has(.st-key-ha_chat_composer_row) [class*="st-key-ha_assistant_actions_row"] {
            margin-bottom: 0.05rem !important;
        }
        /* Agent thinking + welcome empty slot'ları composer'ı yukarı itmesin */
        [data-testid="stMain"]:has(.st-key-ha_chat_composer_row) [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(
            + [data-testid="stElementContainer"] .st-key-ha_chat_composer_row
        ),
        [data-testid="stMain"]:has(.st-key-ha_chat_composer_row) [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(
            + [data-testid="stElementContainer"]:has(
                + [data-testid="stElementContainer"] .st-key-ha_chat_composer_row
            )
        ) {
            margin: 0 !important;
            padding: 0 !important;
            min-height: 0 !important;
        }
        /* Chat composer: HER ZAMAN ekranın altında sabit (viewport bottom).
           Sidebar açıkken ana içerik alanına hizalanması için aşağıdaki JS,
           stMain'in soluna ve genişliğine göre --ha-main-left ve --ha-main-width
           CSS değişkenlerini günceller. Değişkenler yoksa viewport tamamına yayılır. */
        .st-key-ha_chat_composer_row {
            position: fixed !important;
            bottom: 0 !important;
            top: auto !important;
            left: var(--ha-main-left, 0) !important;
            right: auto !important;
            width: var(--ha-main-width, 100vw) !important;
            max-width: none !important;
            z-index: 999 !important;
            margin: 0 !important;
            padding: 0 !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            border-radius: 0 !important;
            pointer-events: none; /* boş kenarlar tıklanabilir olmasın */
        }
        /* Composer'ın iç sütun bloğunu ortalayıp kart görünümü ona veriyoruz.
           Streamlit ara sarmalayıcıları (stVerticalBlockBorderWrapper,
           stVerticalBlock, stElementContainer) varsayılan olarak şeffaf bırakılır. */
        .st-key-ha_chat_composer_row [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-ha_chat_composer_row > div,
        .st-key-ha_chat_composer_row [data-testid="stVerticalBlock"],
        .st-key-ha_chat_composer_row [data-testid="stElementContainer"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stHorizontalBlock"] {
            pointer-events: auto;
            max-width: 48rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
            margin-bottom: max(0.6rem, env(safe-area-inset-bottom, 0px)) !important;
            padding: 0.5rem 0.65rem !important;
            background: #ffffff !important;
            border: 1px solid var(--ha-card-edge) !important;
            border-radius: 14px !important;
            box-shadow:
                0 -4px 14px rgba(40, 55, 45, 0.06),
                var(--ha-card-shadow) !important;
        }
        @media (max-width: 720px) {
            .st-key-ha_chat_composer_row [data-testid="stHorizontalBlock"] {
                margin-left: 0.5rem !important;
                margin-right: 0.5rem !important;
                margin-bottom: max(0.5rem, env(safe-area-inset-bottom, 0px)) !important;
            }
        }
        .st-key-ha_chat_composer_row [data-testid="stHorizontalBlock"] {
            align-items: flex-end !important;
            gap: 0.35rem !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {
            flex: 0 0 2.5rem !important;
            width: 2.5rem !important;
            min-width: 2.5rem !important;
            max-width: 2.5rem !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stPopover"] button {
            min-height: 2.35rem !important;
            width: 100% !important;
            max-width: none !important;
            padding: 0.15rem !important;
            border-radius: 11px !important;
            border: 1px solid rgba(72, 92, 78, 0.2) !important;
            background: #ffffff !important;
            color: #4d564e !important;
            -webkit-text-fill-color: #4d564e !important;
            box-shadow: 0 1px 2px rgba(40, 55, 45, 0.05) !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stPopover"] button:hover {
            background: #f9fbf9 !important;
            border-color: rgba(72, 92, 78, 0.28) !important;
            color: #3d433d !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stPopover"] button [data-testid="stMarkdownContainer"] {
            display: none !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stChatInput"] > div {
            border-radius: 14px !important;
            border-color: var(--ha-border) !important;
            background: var(--ha-surface) !important;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04) !important;
        }
        /* Chat bubbles: ChatGPT tarzı sade — asistan baloncuğu çerçevesiz,
           kullanıcı baloncuğu hafif gri pill. */
        section.main [data-testid="stChatMessage"] {
            background: transparent !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.45rem 0.65rem !important;
            margin-bottom: 0.45rem !important;
            box-shadow: none !important;
        }
        /* Kullanıcı mesajı (sağ avatarlı): yumuşak gri pill */
        section.main [data-testid="stChatMessage"]:has(
            [data-testid="stChatMessageAvatarUser"]
        ),
        section.main [data-testid="stChatMessage"].st-emotion-cache-user,
        section.main [data-testid="stChatMessage"]:has(
            [data-testid="chatAvatarIcon-user"]
        ) {
            background: #f4f4f5 !important;
            border-radius: 14px !important;
            padding: 0.6rem 0.85rem !important;
        }
        /* Admin: modern control panel (scoped to ha_admin_panel / ha_admin_feedback) */
        .st-key-ha_admin_panel .ha-admin-hero,
        .st-key-ha_admin_feedback .ha-admin-feedback-head {
            margin-bottom: 0.35rem;
        }
        .ha-admin-hero__title {
            font-size: 1.42rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #2a3028;
            margin: 0 0 0.35rem 0;
            line-height: 1.2;
        }
        .ha-admin-hero__sub {
            font-size: 0.92rem;
            color: #5a6456;
            margin: 0;
            line-height: 1.45;
        }
        .ha-admin-metric-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.35rem 0 1.1rem 0;
        }
        .ha-admin-metric-row--tight {
            margin-top: 0.5rem;
            margin-bottom: 0.85rem;
        }
        @media (max-width: 900px) {
            .ha-admin-metric-row {
                grid-template-columns: 1fr;
            }
        }
        .ha-admin-metric-card {
            background: #ffffff;
            border: 1px solid var(--ha-card-edge);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            box-shadow: var(--ha-card-shadow);
        }
        .ha-admin-metric-card__label {
            font-size: 0.68rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6a7562;
            margin-bottom: 0.35rem;
        }
        .ha-admin-metric-card__value {
            font-size: 1.28rem;
            font-weight: 700;
            color: #2d352b;
            line-height: 1.25;
            word-break: break-word;
        }
        .ha-admin-metric-card__value--sm {
            font-size: 0.95rem;
            font-weight: 600;
        }
        .ha-admin-section-h {
            font-size: 0.85rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #5a6456;
            margin: 0.25rem 0 0.65rem 0;
        }
        .st-key-ha_admin_panel .ha-admin-op-title {
            font-size: 0.95rem;
            font-weight: 650;
            color: #2d352b;
            margin: 0 0 0.5rem 0;
        }
        .st-key-ha_admin_panel [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px !important;
            background: #ffffff !important;
            box-shadow: var(--ha-card-shadow) !important;
            border: 1px solid var(--ha-card-edge) !important;
        }
        .st-key-ha_admin_feedback .ha-admin-feedback-head__title {
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #2a3028;
            margin: 0 0 0.2rem 0;
        }
        .st-key-ha_admin_feedback .ha-admin-feedback-head__sub {
            font-size: 0.88rem;
            color: #5a6456;
            margin: 0 0 0.75rem 0;
            line-height: 1.45;
        }
        .st-key-ha_admin_feedback .ha-admin-feedback-toolbar {
            background: #ffffff;
            border: 1px solid var(--ha-card-edge);
            border-radius: 12px;
            padding: 0.55rem 0.75rem 0.65rem 0.75rem;
            margin-bottom: 0.75rem;
            box-shadow: var(--ha-card-shadow);
        }
        .st-key-ha_admin_feedback .ha-admin-feedback-toolbar__label {
            font-size: 0.72rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #6a7562;
            margin: 0 0 0.35rem 0;
        }
        .st-key-ha_admin_feedback .ha-admin-glance-label {
            font-size: 0.72rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #6a7562;
            margin: 0.15rem 0 0.45rem 0;
        }
        .st-key-ha_admin_feedback div[data-testid="stExpander"] {
            margin-bottom: 0.55rem !important;
        }
        .st-key-ha_admin_feedback [data-testid="stExpander"] summary {
            font-weight: 500 !important;
            font-size: 0.88rem !important;
        }
        .ha-admin-entry-meta {
            font-size: 0.78rem;
            color: #6a7562;
            margin: 0 0 0.5rem 0;
        }
        /* Main: expanders (e.g. admin, profile blocks) as distinct sections */
        section.main div[data-testid="stExpander"] {
            border: 1px solid var(--ha-card-edge) !important;
            border-radius: 12px !important;
            background: var(--ha-surface) !important;
            box-shadow: var(--ha-card-shadow) !important;
            margin-bottom: 0.85rem !important;
        }
        section.main [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        section.main [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
            color: var(--ha-chat-ink) !important;
            line-height: 1.55 !important;
        }
        /* Suggested prompts: ChatGPT-vari sade, hafif kenarlı chip butonlar */
        .st-key-ha_suggested_prompts {
            margin-top: 0.5rem !important;
            margin-bottom: 0.15rem !important;
            max-width: 48rem;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        .st-key-ha_suggested_prompts [data-testid="stHorizontalBlock"] {
            gap: 0.55rem !important;
        }
        .st-key-ha_suggested_prompts button[data-testid^="baseButton"] {
            min-height: 2.25rem !important;
            font-size: 0.85rem !important;
            font-weight: 450 !important;
            line-height: 1.4 !important;
            border-radius: 12px !important;
            border: 1px solid var(--ha-border) !important;
            background: #ffffff !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            box-shadow: none !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }
        .st-key-ha_suggested_prompts button[data-testid^="baseButton"]:hover {
            background: #f7f7f8 !important;
            border-color: rgba(0, 0, 0, 0.12) !important;
            color: var(--ha-text) !important;
        }
        /* Assistant bubbles: compact copy + sources + merged 👍/👎 */
        [class*="st-key-ha_assistant_actions_row"] [data-testid="stHorizontalBlock"] {
            align-items: center !important;
            flex-wrap: nowrap !important;
        }
        [class*="st-key-ha_assistant_copy_cell"] {
            overflow: visible !important;
        }
        [class*="st-key-ha_assistant_copy_cell"] iframe {
            height: 32px !important;
            min-height: 32px !important;
            max-height: 32px !important;
        }
        [class*="st-key-ha_assistant_sources_cell"] [data-testid="stPopover"] button {
            min-height: 1.72rem !important;
            padding: 0.14rem 0.55rem !important;
            font-size: 0.78rem !important;
            line-height: 1.25 !important;
            white-space: nowrap !important;
        }
        [class*="st-key-ha_assistant_sources_cell"] [data-testid="stCaptionContainer"] p {
            font-size: 0.72rem !important;
            margin: 0 !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] [data-testid="stHorizontalBlock"] {
            gap: 0 !important;
            align-items: stretch !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 0 !important;
            min-width: 2.55rem !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] button[data-testid^="baseButton"] {
            min-height: 1.65rem !important;
            padding: 0.08rem 0.32rem !important;
            font-size: 1.05rem !important;
            line-height: 1.15 !important;
            border-radius: 0 !important;
            border: 1px solid rgba(120, 120, 120, 0.32) !important;
            background: #ffffff !important;
            width: 100% !important;
            max-width: none !important;
        }
        [class*="st-key-ha_assistant_feedback_group"]
            [data-testid="stHorizontalBlock"]
            > div[data-testid="column"]:first-child
            button[data-testid^="baseButton"] {
            border-radius: 7px 0 0 7px !important;
        }
        [class*="st-key-ha_assistant_feedback_group"]
            [data-testid="stHorizontalBlock"]
            > div[data-testid="column"]:last-child
            button[data-testid^="baseButton"] {
            border-radius: 0 7px 7px 0 !important;
            border-left: none !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] button[data-testid^="baseButton"]:hover {
            background: #f9fbf9 !important;
        }
        /* Profile: full-width settings control */
        .st-key-ha_profile_adv {
            margin-bottom: 0.35rem;
        }
        .st-key-ha_profile_adv [data-testid="stPopover"] button {
            border-radius: 10px !important;
            border: 1px solid rgba(72, 92, 78, 0.2) !important;
            background: #ffffff !important;
            font-weight: 550 !important;
            color: #3a4534 !important;
            -webkit-text-fill-color: #3a4534 !important;
            box-shadow: 0 1px 2px rgba(40, 55, 45, 0.05) !important;
        }
        .st-key-ha_profile_adv [data-testid="stPopover"] button:hover {
            background: #f9fbf9 !important;
            border-color: rgba(72, 92, 78, 0.28) !important;
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
            color: #5a7d6e;
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
        /* Login primary: Streamlit puts label in stMarkdownContainer; global rule above must not win. */
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] p,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"] p,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] span,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"] span,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"]:has(button[data-testid="baseButton-primary"]) [data-testid="stMarkdownContainer"] p,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"]:has(button[kind="primary"]) [data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            opacity: 1 !important;
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
            border-color: rgba(109, 139, 124, 0.55) !important;
            box-shadow: 0 0 0 2px rgba(109, 139, 124, 0.12) !important;
        }
        [data-testid="stForm"] div[data-baseweb="input"] > div,
        [data-testid="stForm"] div[data-baseweb="select"] > div {
            background-color: transparent !important;
            border: none !important;
        }
        /* Forms (logged-in): primary = nötr beyaz buton sade çerçevesi */
        [data-testid="stForm"] button[kind="primary"] {
            border-radius: 10px !important;
            min-height: 2.2rem !important;
            font-size: 0.85rem !important;
            background: #ffffff !important;
            background-image: none !important;
            border: 1px solid var(--ha-border) !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            transition: transform 0.05s ease, filter 0.15s ease;
        }
        [data-testid="stFormSubmitButton"] button[kind="primary"],
        [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] {
            border-radius: 10px !important;
            min-height: 2.35rem !important;
            font-size: 0.85rem !important;
            background: #ffffff !important;
            background-image: none !important;
            border: 1px solid var(--ha-border) !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
        }
        [data-testid="stFormSubmitButton"] button[kind="primary"] p,
        [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] p,
        [data-testid="stFormSubmitButton"] button[kind="primary"] span,
        [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] span {
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
        }
        [data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
        [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"]:hover {
            background: #f5f5f5 !important;
            filter: none !important;
            border-color: rgba(0, 0, 0, 0.12) !important;
        }
        button[kind="primary"] {
            background: #ffffff !important;
            background-image: none !important;
            border: 1px solid var(--ha-border) !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
        }
        a {
            color: #2563eb !important;
        }
        .ha-forgot-link > div > button {
            background: transparent !important;
            border: none !important;
            color: var(--ha-text-soft) !important;
            padding: 0 !important;
            min-height: auto !important;
            font-size: 0.9rem !important;
            justify-content: flex-start !important;
            box-shadow: none !important;
        }
        .ha-forgot-link > div > button:hover {
            color: var(--ha-text) !important;
            text-decoration: underline !important;
        }
        [data-testid="stForm"] button[kind="primary"]:hover {
            background: #f9fbf9 !important;
            filter: none !important;
        }
        [data-testid="stForm"] button[kind="primary"]:active {
            transform: translateY(1px);
        }
        @media (max-width: 900px) {
            .main .block-container:has(.st-key-ha_auth_shell) {
                min-height: auto;
                justify-content: flex-start;
            }
            .st-key-ha_auth_shell {
                --ha-lux-card-h: auto;
            }
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                border-radius: 18px !important;
            }
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"]::before {
                border-radius: 18px;
            }
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1),
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(2) {
                flex: 1 1 auto !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome {
                min-height: 240px;
                padding: 1.75rem 1.35rem 1.25rem 1.35rem;
                justify-content: center;
                align-items: center;
            }
            .st-key-ha_auth_shell .ha-lux-welcome__inner {
                align-items: center;
                text-align: center;
            }
            .st-key-ha_auth_shell .ha-lux-welcome h1,
            .st-key-ha_auth_shell .ha-lux-welcome__lead {
                text-align: center;
            }
            .st-key-ha_auth_shell .ha-lux-welcome__rule {
                margin-left: auto;
                margin-right: auto;
            }
            .ha-lux-botanical {
                padding-top: 1.25rem;
            }
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) > div[data-testid="stVerticalBlock"] {
                min-height: auto;
                padding: 1.75rem 1.5rem 2.5rem 1.5rem !important;
            }
            .st-key-ha_auth_form_card [data-testid="stVerticalBlock"] {
                padding: 0.5rem 0.4rem 1.1rem 0.4rem !important;
            }
            .st-key-ha_auth_shell .st-key-ha_auth_lang_header,
            .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stVerticalBlock"] {
                justify-content: flex-end !important;
            }
        }
        .ha-section-title {
            margin-top: 0.05rem;
            margin-bottom: 0.15rem;
            font-size: 1.18rem;
            font-weight: 650;
            color: var(--ha-chat-ink);
            letter-spacing: -0.02em;
        }
        /* Chat page: space between conversation title and composer (gear + input) */
        .ha-section-title.ha-chat-page-title {
            margin-top: 0.05rem !important;
            margin-bottom: 0.55rem !important;
            padding-bottom: 0.5rem !important;
            line-height: 1.25 !important;
            border-bottom: 1px solid rgba(72, 92, 78, 0.11) !important;
        }
        .ha-chat-welcome-line {
            margin: -0.2rem 0 0.28rem 0;
            font-size: 1.08rem;
            font-weight: 650;
            color: #3d4540 !important;
            letter-spacing: -0.01em;
            line-height: 1.3;
        }
        /* Section #### headings in main (incl. suggested questions): calmer scale */
        section.main [data-testid="stMarkdownContainer"] h4 {
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            color: #4d544d !important;
            margin: 0.45rem 0 0.32rem 0 !important;
            letter-spacing: 0.02em;
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
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] p.ha-lux-footer {
            color: #6a7168 !important;
        }
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-form-title,
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-form-sub {
            color: inherit;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_chat_layout_script() -> None:
    """Sidebar açıkken/kapalıyken composer'ın stMain ile hizalı kalması için
    iki CSS değişkenini günceller: --ha-main-left, --ha-main-width.

    Composer 'position: fixed' kullanır; CSS değişkenleri sayesinde stMain'in
    gerçek bounding rect'ine göre konumlanır."""
    components.html(
        """
        <script>
          (function () {
            try {
              const doc = window.parent && window.parent.document
                ? window.parent.document
                : document;
              if (!doc) return;
              const root = doc.documentElement;

              function update() {
                try {
                  const main = doc.querySelector('[data-testid="stMain"]');
                  if (!main) return;
                  const rect = main.getBoundingClientRect();
                  root.style.setProperty('--ha-main-left', rect.left + 'px');
                  root.style.setProperty('--ha-main-width', rect.width + 'px');
                } catch (e) {}
              }

              update();
              const win = window.parent || window;
              win.addEventListener('resize', update, { passive: true });

              // Sidebar / main boyut değişimlerini izle
              try {
                const targets = [
                  doc.querySelector('[data-testid="stMain"]'),
                  doc.querySelector('[data-testid="stSidebar"]'),
                  doc.querySelector('[data-testid="stAppViewContainer"]'),
                ].filter(Boolean);
                if (window.ResizeObserver && targets.length) {
                  const ro = new ResizeObserver(update);
                  targets.forEach((t) => ro.observe(t));
                }
              } catch (e) {}

              // Streamlit re-render sonrası DOM yenilenmesi için emniyet kemeri
              setInterval(update, 600);
            } catch (e) {}
          })();
        </script>
        """,
        height=0,
        width=0,
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


def _normalize_active_page() -> None:
    """Keep ``active_page`` valid for the current auth role."""
    page = st.session_state.get("active_page", "Chat")
    if st.session_state.get("is_logged_in"):
        if st.session_state.get("role") == "admin":
            allowed = {"Admin Panel", "Chat"}
        else:
            allowed = {"Chat", "Profile"}
    else:
        allowed = {"Chat", "Login", "Profile"}
    if page not in allowed:
        st.session_state.active_page = "Chat"


def _section_nav_label(lang: str, section_id: str) -> str:
    return {
        "Chat": get_string(lang, "nav_chat"),
        "Profile": get_string(lang, "profile"),
        "Login": get_string(lang, "sidebar_login"),
        "Admin Panel": get_string(lang, "admin_dashboard"),
    }.get(section_id, section_id)


def _on_guest_top_nav() -> None:
    """Sync ``active_page`` with Chat/Profile radio (Login is a separate control)."""
    pick = st.session_state.get("ha_nav_guest_top")
    if pick in ("Chat", "Profile"):
        st.session_state["active_page"] = pick


def _on_login_username_change() -> None:
    """Login formu UX: kullanıcı adı yazılıp commit edildiğinde (Tab/Enter),
    daha önce bu makinada kayıt olmuş ya da giriş yapmış kullanıcı için
    hatırlanan şifre ``login_password`` session_state alanına yazılır. Bir
    sonraki rerun'da password input bu değerle pre-fill olarak render edilir.

    Eşleşen kayıt yoksa, önceki auto-fill artığı bırakmamak için password
    alanı temizlenir — böylece farklı bir hesaba geçildiğinde yanlış şifre
    formda kalmaz.
    """
    user = str(st.session_state.get("login_username", "")).strip()
    if not user:
        return
    pwd = _lookup_remembered_password(user)
    last_user = st.session_state.get("_ha_autofill_last_user")
    if pwd:
        st.session_state["login_password"] = pwd
        st.session_state["_ha_autofill_last_user"] = user
    elif last_user is not None and last_user != user:
        # Önceki user için autofill yapılmıştı, bu user için yok → temizle.
        st.session_state["login_password"] = ""
        st.session_state["_ha_autofill_last_user"] = user


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
    # Eğer kullanıcı daha önce Remember me ile şifresini hatırlatmışsa,
    # eski (artık geçersiz) entry'yi temizliyoruz. Yeni şifreyi otomatik
    # KAYDETMİYORUZ — bu davranış login formundaki Remember me'ye bağlı.
    _forget_remembered_password(username)
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
    raw_title = get_string(lang, "app_title")
    title = raw_title if isinstance(raw_title, str) else "AI Herbalist Assistant"
    safe_title = (
        title.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .strip()
    )
    st.markdown(
        f'''
        <style>
        [data-testid="stHeader"] {{
            background: linear-gradient(180deg, #ffffff 0%, #f9fbf9 100%) !important;
            border-bottom: 1px solid rgba(72, 92, 78, 0.14) !important;
            box-shadow: 0 2px 12px rgba(40, 55, 45, 0.05) !important;
            min-height: 3.45rem !important;
            position: relative !important;
            z-index: 1000010 !important;
        }}
        [data-testid="stHeader"]::after {{
            content: "{safe_title}";
            position: absolute;
            left: clamp(3.5rem, 12vw, 4.25rem);
            top: 50%;
            transform: translateY(-50%);
            z-index: 2;
            font-size: 1.06rem;
            font-weight: 650;
            letter-spacing: -0.03em;
            color: #2d352b !important;
            pointer-events: none;
            font-family: "Inter", system-ui, -apple-system, sans-serif !important;
            line-height: 1.2;
        }}
        /* Deploy / header actions: calmer chrome */
        [data-testid="stHeader"] a {{
            color: #5a6b5e !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            opacity: 0.92;
        }}
        [data-testid="stHeader"] a:hover {{
            color: #3d4a40 !important;
            opacity: 1;
        }}
        /* Main column: keep content snug under header */
        section.main .block-container {{
            padding-top: 0.5rem !important;
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
    cookie_mgr = _get_cookie_manager()
    cookie_deleted = False
    if cookie_mgr is not None:
        try:
            cookie_mgr.delete(cookie=_REMEMBER_COOKIE_NAME, key="ha_cookie_logout_del")
            cookie_deleted = True
        except Exception:
            _logger.debug("Remember-me cookie delete failed", exc_info=True)
    st.session_state.clear()
    # Logout sonrası Misafir Chat yerine doğrudan Login ekranı açılsın.
    st.session_state["active_page"] = "Login"
    st.session_state["auth_mode"] = "Login"
    # Bu session için auto-login'i kapat: cookie henüz tarayıcıdan silinmemiş
    # olsa bile kullanıcı çıkış yapmayı seçtiği için tekrar oturum açılmasın.
    st.session_state["ha_remember_consumed"] = True
    # Cookie delete komutunun WebSocket üzerinden tarayıcıya iletilmesi için
    # kısa süre bekle (Login set/delete'le aynı race condition'ı önler).
    if cookie_deleted:
        time.sleep(0.4)
    st.rerun()


def _try_auto_login_from_cookie() -> None:
    """If a valid HMAC-signed remember cookie is present, log the user in.

    Runs once per Streamlit session: after a successful auto-login we set
    ``ha_remember_consumed`` so subsequent reruns skip the cookie roundtrip.
    """
    if st.session_state.get("is_logged_in"):
        return
    if st.session_state.get("ha_remember_consumed"):
        return
    cookie_mgr = _get_cookie_manager()
    if cookie_mgr is None:
        return
    try:
        token = cookie_mgr.get(cookie=_REMEMBER_COOKIE_NAME)
    except Exception:
        _logger.debug("Remember-me cookie read failed", exc_info=True)
        return
    # The cookie manager returns None on the first run before the JS round-trip
    # completes; mark consumed only when we actually saw a token (or the cookie
    # is conclusively absent on a later rerun).
    if not token:
        return
    username = _verify_remember_token(str(token))
    if not username:
        try:
            cookie_mgr.delete(cookie=_REMEMBER_COOKIE_NAME, key="ha_cookie_invalid_clear")
        except Exception:
            pass
        st.session_state["ha_remember_consumed"] = True
        return
    profile = _get_user_profile(username)
    if not profile:
        try:
            cookie_mgr.delete(cookie=_REMEMBER_COOKIE_NAME, key="ha_cookie_unknown_clear")
        except Exception:
            pass
        st.session_state["ha_remember_consumed"] = True
        return
    st.session_state.is_logged_in = True
    st.session_state.username = username
    st.session_state.role = (
        "admin" if username == ADMIN_USERNAME else "user"
    )
    st.session_state.user_profile = profile
    if st.session_state.role == "user":
        start_new_chat(username)
        _sync_conversations_to_session(username)
    st.session_state.active_page = (
        "Admin Panel" if st.session_state.role == "admin" else "Chat"
    )
    st.session_state["ha_remember_consumed"] = True
    st.rerun()


def _render_sidebar_user_header(lang: str, username: str) -> None:
    """Premium sidebar top: nav label + user card (markup/CSS only)."""
    uname = (username or "").strip()
    initial = _html.escape((uname[:1] or "?").upper())
    safe_user = _html.escape(uname or "—")
    nav_lbl = _html.escape(get_string(lang, "sidebar_nav"))
    hint = _html.escape(get_string(lang, "signed_in_as"))
    eyebrow_mod = " ha-sidebar-header__eyebrow--tr" if lang == "tr" else ""
    st.markdown(
        f"""
<div class="ha-sidebar-header">
  <p class="ha-sidebar-header__eyebrow{eyebrow_mod}">{nav_lbl}</p>
  <div class="ha-sidebar-header__user-card">
    <div class="ha-sidebar-header__avatar" aria-hidden="true">{initial}</div>
    <div class="ha-sidebar-header__user-meta">
      <span class="ha-sidebar-header__hint">{hint}</span>
      <span class="ha-sidebar-header__name">{safe_user}</span>
    </div>
  </div>
</div>
""".strip(),
        unsafe_allow_html=True,
    )


def _render_sidebar_guest_header(lang: str) -> None:
    """Sidebar top when not signed in."""
    guest = _html.escape(get_string(lang, "guest_display_name"))
    nav_lbl = _html.escape(get_string(lang, "sidebar_nav"))
    hint = _html.escape(get_string(lang, "signed_in_as"))
    eyebrow_mod = " ha-sidebar-header__eyebrow--tr" if lang == "tr" else ""
    st.markdown(
        f"""
<div class="ha-sidebar-header">
  <p class="ha-sidebar-header__eyebrow{eyebrow_mod}">{nav_lbl}</p>
  <div class="ha-sidebar-header__user-card">
    <div class="ha-sidebar-header__avatar" aria-hidden="true">?</div>
    <div class="ha-sidebar-header__user-meta">
      <span class="ha-sidebar-header__hint">{hint}</span>
      <span class="ha-sidebar-header__name">{guest}</span>
    </div>
  </div>
</div>
""".strip(),
        unsafe_allow_html=True,
    )


def _render_auth_language_switch(lang: str) -> None:
    """Compact EN/TR control (auth top bar: rendered above tabs, CSS aligns right)."""
    with st.container(key="ha_auth_lang_header"):
        new_lang = st.segmented_control(
            get_string(lang, "auth_lang_label"),
            options=["en", "tr"],
            selection_mode="single",
            format_func=lambda x: get_string(
                lang,
                "auth_lang_option_english" if x == "en" else "auth_lang_option_turkish",
            ),
            default=lang,
            key="auth_lang_segmented",
            label_visibility="collapsed",
            width="content",
        )
    picked = new_lang if new_lang is not None else lang
    if picked != lang:
        st.session_state.language = picked
        st.rerun()


def _render_auth_screen() -> None:
    lang = st.session_state.get("language", "en")
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "Login"

    with st.container(key="ha_auth_shell"):
        if not st.session_state.is_logged_in:
            if st.button(
                get_string(lang, "auth_back_to_chat"),
                key="ha_auth_back_chat",
                type="tertiary",
            ):
                st.session_state.active_page = "Chat"
                # Keep nav radio in sync (sidebar was skipped on Login; widget state could stay "Login").
                st.session_state["ha_nav_guest_top"] = "Chat"
                # Misafir mod açıkça seçildi: cookie hala tarayıcıda olsa
                # bile bu session boyunca auto-login'i devreye sokma.
                st.session_state["ha_remember_consumed"] = True
                st.session_state["ha_guest_explicit"] = True
                st.rerun()
        with st.container(key="ha_auth_card"):
            left_col, right_col = st.columns([0.9, 1.1], gap="small")

            w_title = _html.escape(get_string(lang, "auth_lux_welcome_title"))
            w_lead = _html.escape(get_string(lang, "auth_lux_welcome_lead"))

            with left_col:
                st.markdown(
                    f"""
                    <div class="ha-lux-welcome ha-lux-welcome--text">
                        <div class="ha-lux-welcome__inner">
                            <h1 class="ha-lux-welcome__title">{w_title}</h1>
                            <p class="ha-lux-welcome__lead">{w_lead}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right_col:
                if st.session_state.auth_mode not in {"Login", "Register", "Reset Password"}:
                    st.session_state.auth_mode = "Login"

                auth_mode = st.session_state.auth_mode

                with st.container(key="ha_auth_form_card"):
                    if auth_mode == "Reset Password":
                        with st.container(key="ha_auth_top_bar"):
                            _render_auth_language_switch(lang)
                        st.markdown(
                            f'<div class="ha-lux-form-title">{_html.escape(get_string(lang, "forgot_pwd"))}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="ha-lux-form-sub">{_html.escape(get_string(lang, "reset_subtitle"))}</div>',
                            unsafe_allow_html=True,
                        )
                        with st.form("reset_password_form", clear_on_submit=True, border=False):
                            reset_username = st.text_input(
                                get_string(lang, "username"),
                                key="reset_username",
                                label_visibility="collapsed",
                                placeholder=get_string(lang, "auth_lux_email_ph"),
                            )
                            new_password = st.text_input(
                                get_string(lang, "new_password"),
                                type="password",
                                key="reset_new_password",
                                label_visibility="collapsed",
                                placeholder=get_string(lang, "auth_lux_password_ph"),
                            )
                            confirm_new_password = st.text_input(
                                get_string(lang, "confirm_new_password"),
                                type="password",
                                key="reset_confirm_new_password",
                                label_visibility="collapsed",
                                placeholder=get_string(lang, "auth_lux_password_ph"),
                            )
                            with st.container(key="ha_auth_primary_submit"):
                                submit_reset = st.form_submit_button(
                                    get_string(lang, "reset_pwd_btn"),
                                    type="primary",
                                    use_container_width=True,
                                )

                        if submit_reset:
                            result = _reset_user_password(
                                reset_username, new_password, confirm_new_password
                            )
                            if result.startswith("Password reset successful"):
                                st.success(result)
                                st.session_state.auth_mode = "Login"
                                st.rerun()
                            else:
                                st.error(result)

                        if st.button(
                            get_string(lang, "back_to_login"),
                            use_container_width=True,
                            type="secondary",
                        ):
                            st.session_state.auth_mode = "Login"
                            st.rerun()
                        return

                    submit_login = False
                    submit_register = False
                    submit_forgot = False

                    with st.container(key="ha_auth_top_bar"):
                        _render_auth_language_switch(lang)
                        with st.container(key="ha_auth_tab_row"):
                            auth_lux_tab = st.radio(
                                "auth_lux_tab_ui",
                                options=["login", "register"],
                                format_func=lambda v: (
                                    get_string(lang, "login_btn")
                                    if v == "login"
                                    else get_string(lang, "create_account")
                                ),
                                horizontal=True,
                                label_visibility="collapsed",
                                key="auth_lux_tab_ui",
                            )

                    if auth_lux_tab == "login":
                        st.markdown(
                            f'<div class="ha-lux-form-title">{_html.escape(get_string(lang, "auth_lux_login_title"))}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="ha-lux-form-sub">{_html.escape(get_string(lang, "auth_lux_login_sub"))}</div>',
                            unsafe_allow_html=True,
                        )
                        # Username form'un DIŞINDA: on_change burada gerçek
                        # zamanlı tetiklenir → kullanıcı adı yazılıp Tab/Enter
                        # ile commit edilince hatırlanan şifre otomatik dolar.
                        with st.container(key="ha_lux_username_outside"):
                            st.text_input(
                                get_string(lang, "username"),
                                key="login_username",
                                label_visibility="collapsed",
                                placeholder=get_string(lang, "auth_lux_email_ph"),
                                on_change=_on_login_username_change,
                            )

                        with st.form("login_form", clear_on_submit=False, border=False):
                            st.text_input(
                                get_string(lang, "password"),
                                type="password",
                                key="login_password",
                                label_visibility="collapsed",
                                placeholder=get_string(lang, "auth_lux_password_ph"),
                            )
                            with st.container(key="ha_lux_remember_row"):
                                st.checkbox(
                                    get_string(lang, "auth_remember_me"),
                                    key="auth_remember_me",
                                )
                            # Form içinde TEK submit button: Login. Bu sayede
                            # password içinde Enter'a basıldığında doğrudan
                            # Login submit edilir.
                            with st.container(key="ha_auth_primary_submit"):
                                submit_login = st.form_submit_button(
                                    get_string(lang, "login_btn"),
                                    type="primary",
                                    use_container_width=False,
                                )

                        # Forgot password: form'un DIŞINDA, sıradan bir buton.
                        # Form içinde ekstra submit_button olmadığı için Enter
                        # sadece Login'i tetikler.
                        with st.container(key="ha_lux_forgot_row"):
                            submit_forgot = st.button(
                                get_string(lang, "forgot_pwd"),
                                type="tertiary",
                                key="login_forgot_outside",
                            )
                    else:
                        st.markdown(
                            f'<div class="ha-lux-form-title">{_html.escape(get_string(lang, "auth_lux_register_title"))}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="ha-lux-form-sub">{_html.escape(get_string(lang, "auth_lux_register_sub"))}</div>',
                            unsafe_allow_html=True,
                        )
                        with st.form("register_form", clear_on_submit=True, border=False):
                            st.text_input(
                                get_string(lang, "username"),
                                key="register_username",
                                label_visibility="collapsed",
                                placeholder=get_string(lang, "auth_lux_email_ph"),
                            )
                            st.text_input(
                                get_string(lang, "password"),
                                type="password",
                                key="register_password",
                                label_visibility="collapsed",
                                placeholder=get_string(lang, "auth_lux_password_ph"),
                            )
                            st.text_input(
                                get_string(lang, "confirm_password"),
                                type="password",
                                key="register_confirm_password",
                                label_visibility="collapsed",
                                placeholder=get_string(lang, "auth_lux_password_ph"),
                            )
                            with st.container(key="ha_auth_primary_submit"):
                                submit_register = st.form_submit_button(
                                    get_string(lang, "create_account"),
                                    type="primary",
                                    use_container_width=False,
                                )

                    if submit_forgot:
                        st.session_state.auth_mode = "Reset Password"
                        st.rerun()

                    if submit_login:
                        login_user = str(st.session_state.get("login_username", "")).strip()
                        login_pass = str(st.session_state.get("login_password", ""))
                        auth = _authenticate_user(login_user, login_pass)
                        if auth.get("status") != "ok":
                            st.error(auth.get("message", "Login failed."))
                        else:
                            st.session_state.is_logged_in = True
                            st.session_state.username = login_user
                            st.session_state.role = auth.get("role", "user")
                            st.session_state.user_profile = _get_user_profile(
                                st.session_state.username
                            )
                            if st.session_state.role == "user":
                                start_new_chat(st.session_state.username)
                                _sync_conversations_to_session(st.session_state.username)
                            st.session_state.active_page = (
                                "Admin Panel" if st.session_state.role == "admin" else "Chat"
                            )
                            st.session_state.pop("ha_anon_chat_key", None)
                            remember_checked = bool(
                                st.session_state.get("auth_remember_me", False)
                            )
                            # Auto-fill (isim yazınca şifrenin gelmesi) ve
                            # cookie tabanlı auto-login YALNIZCA kullanıcı
                            # "Remember me"yi açıkça işaretlediyse aktif olur.
                            # Aksi halde önceden hatırlanan kayıt da silinir.
                            if st.session_state.role == "user":
                                if remember_checked:
                                    _store_remembered_password(login_user, login_pass)
                                else:
                                    _forget_remembered_password(login_user)
                            # Persist (or clear) the remember-me cookie so the
                            # next visit can auto-login the same browser.
                            cookie_mgr = _get_cookie_manager()
                            wrote_cookie = False
                            if cookie_mgr is not None:
                                try:
                                    if remember_checked:
                                        token = _make_remember_token(login_user)
                                        cookie_mgr.set(
                                            cookie=_REMEMBER_COOKIE_NAME,
                                            val=token,
                                            expires_at=datetime.utcnow()
                                            + timedelta(days=_REMEMBER_TTL_DAYS),
                                            key="ha_cookie_login_set",
                                        )
                                        wrote_cookie = True
                                    else:
                                        cookie_mgr.delete(
                                            cookie=_REMEMBER_COOKIE_NAME,
                                            key="ha_cookie_login_clear",
                                        )
                                        wrote_cookie = True
                                except Exception:
                                    _logger.debug(
                                        "Remember-me cookie write failed",
                                        exc_info=True,
                                    )
                            st.success("Login successful.")
                            # Streamlit'in cookie set/delete komutunu tarayıcıya
                            # iletmesi için kısa bir yumuşatma. ``st.rerun()``
                            # mevcut run'ı kestiğinde queued mesajların flush
                            # olmasına imkan veriyor.
                            if wrote_cookie:
                                time.sleep(0.4)
                            st.rerun()

                    if submit_register:
                        reg_u = str(st.session_state.get("register_username", "")).strip()
                        reg_p = str(st.session_state.get("register_password", ""))
                        reg_c = str(st.session_state.get("register_confirm_password", ""))
                        result = _register_user(reg_u, reg_p, reg_c)
                        if result.startswith("Account created"):
                            st.success(result)
                        else:
                            st.error(result)


def _render_advanced_settings_widgets(*, model_key: str, web_key: str) -> None:
    """Model + web search controls; syncs ``st.session_state`` each run."""
    selected_model = st.selectbox(
        "LLM Model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.selected_model)
        if st.session_state.selected_model in AVAILABLE_MODELS
        else AVAILABLE_MODELS.index(DEFAULT_MODEL),
        key=model_key,
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
        key=web_key,
    )
    st.session_state.web_search_provider = web_search_provider


def _welcome_display_name(username: str) -> str:
    """Prefer profile name when set; otherwise fall back to login username."""
    prof = st.session_state.get("user_profile") or {}
    display = str(prof.get("name") or "").strip()
    if display:
        return display
    return str(username or "").strip()


def _chat_owner_key() -> str:
    """DB username key for chat persistence: real user when logged in, else per-session anon."""
    if st.session_state.get("is_logged_in"):
        return str(st.session_state.get("username") or "").strip()
    if "ha_anon_chat_key" not in st.session_state:
        st.session_state.ha_anon_chat_key = f"anon_{secrets.token_hex(8)}"
    return str(st.session_state.ha_anon_chat_key)


def _render_chat_page() -> None:
    lang = st.session_state.get("language", "en")
    logged_in = bool(st.session_state.get("is_logged_in"))
    username = _chat_owner_key()
    init_session_state(username)
    _sync_conversations_to_session(username)
    chats = get_user_chat_summaries(username)
    active_chat_id = st.session_state.get("active_chat_id", "")
    active_chat_title = next(
        (chat.get("title", get_string(lang, "new_chat")) for chat in chats if chat.get("id") == active_chat_id),
        get_string(lang, "new_chat"),
    )

    if logged_in:
        _welcome_line = get_string(lang, "chat_welcome_user").format(
            name=_html.escape(_welcome_display_name(st.session_state.get("username", "")))
        )
    else:
        _welcome_line = _html.escape(get_string(lang, "chat_welcome_guest"))
    st.markdown(
        f'<div class="ha-chat-welcome-line">{_welcome_line}</div>',
        unsafe_allow_html=True,
    )
    if not logged_in:
        st.markdown(
            f'<div class="ha-section-subtitle" style="margin-top:0.15rem;margin-bottom:0.65rem;">'
            f'{get_string(lang, "chat_guest_banner")}</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="ha-section-title ha-chat-page-title">{_truncate_chat_title(active_chat_title, 80)}</div>',
        unsafe_allow_html=True,
    )

    has_user_msg = any(m["role"] == "user" for m in st.session_state.messages)

    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "assistant":
            if "Hello! I am your AI Herbalist Assistant" in msg["content"] or (
                "Merhaba! Ben sizin AI Bitki Uzmanı Asistanınızım" in msg["content"]
            ):
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
                    with_feedback=logged_in,
                )

    # Reserve space between messages and composer for ``st.status`` while the
    # agent runs (filled only when ``_agent_pending_question`` is set).
    _agent_thinking_slot = st.empty()
    # Empty-state greeting sits above the gear + chat input.
    _welcome_above_input = st.empty()

    with st.container(key="ha_chat_composer_row"):
        _gear_col, _chat_col = st.columns([1, 18], gap="small")
        with _gear_col:
            with st.popover(
                "\u2007",
                icon=":material/tune:",
                type="tertiary",
                help=get_string(lang, "advanced_settings_help"),
                use_container_width=True,
            ):
                _render_advanced_settings_widgets(
                    model_key="main_adv_model", web_key="main_adv_web"
                )
        with _chat_col:
            user_input = st.chat_input(
                get_string(lang, "chat_input_placeholder"),
                key="ha_main_chat_input",
            )
    if not user_input and st.session_state.get("pending_prompt"):
        user_input = st.session_state.pop("pending_prompt")

    if not has_user_msg and not user_input:
        with _welcome_above_input.container():
            with st.chat_message("assistant"):
                st.markdown(get_string(lang, "bot_greeting"))

            st.markdown(get_string(lang, "suggested_questions"))
            prompt_options = get_string(lang, "suggested_prompts")
            with st.container(key="ha_suggested_prompts"):
                columns = st.columns(2, gap="small")
                for i, question in enumerate(prompt_options):
                    with columns[i % 2]:
                        if st.button(
                            question,
                            key=f"suggested_{i}",
                            use_container_width=True,
                            type="secondary",
                        ):
                            st.session_state.pending_prompt = question
                            st.rerun()

    # Run the agent after the composer exists so input + settings stay on-screen;
    # keep ``st.status`` in the slot above the composer.
    _pending_agent_q = st.session_state.get("_agent_pending_question")
    if _pending_agent_q is not None:
        with _agent_thinking_slot.container():
            try:
                answer, sources = _generate_ai_response(
                    str(_pending_agent_q),
                    st.session_state.get("user_profile", {}),
                )
                append_message(
                    role="assistant",
                    content=answer,
                    username=username,
                    sources=sources,
                )
            finally:
                st.session_state.pop("_agent_pending_question", None)
        st.rerun()

    if not user_input:
        return

    append_message(role="user", content=user_input, username=username)
    st.session_state["_agent_pending_question"] = user_input
    st.rerun()


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
            type=\"button\"
            style=\"
              border: 1px solid rgba(120,120,120,0.32);
              background: var(--background-color, #ffffff);
              color: var(--text-color, #222);
              border-radius: 7px;
              padding: 3px 8px;
              font-size: 0.74rem;
              font-weight: 500;
              cursor: pointer;
              display: inline-flex;
              align-items: center;
              gap: 4px;
              line-height: 1.2;
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
        height=32,
        width=132,
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
    """Render helpful / unhelpful pair. Clicking toggles (click-again clears)."""
    up_label = (
        get_string(lang, "feedback_btn_helpful_selected")
        if current == "up"
        else get_string(lang, "feedback_btn_helpful")
    )
    down_label = (
        get_string(lang, "feedback_btn_unhelpful_selected")
        if current == "down"
        else get_string(lang, "feedback_btn_unhelpful")
    )
    base_key = f"fb_{chat_id}_{message_index}"

    with st.container(key=f"ha_assistant_feedback_group_{message_index}"):
        col_up, col_down = st.columns(2, gap="small", vertical_alignment="center")
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
    with_feedback: bool = True,
) -> None:
    """Render the Copy / Sources / feedback row under an assistant message."""
    if not chat_id:
        return

    content = str(message.get("content", ""))
    sources = message.get("sources", []) or []
    current_feedback = message.get("feedback")

    has_sources = bool(_normalize_sources(sources))
    with st.container(key=f"ha_assistant_actions_row_{message_index}"):
        if with_feedback:
            col_copy, col_src, col_fb, spacer = st.columns(
                [1.15, 2.35, 1.05, 5.5], vertical_alignment="center"
            )
        else:
            col_copy, col_src, spacer = st.columns(
                [1.15, 2.35, 6.5], vertical_alignment="center"
            )
            col_fb = None

        with col_copy:
            with st.container(key=f"ha_assistant_copy_cell_{message_index}"):
                _render_copy_button(
                    text=content,
                    key=f"copy_{chat_id}_{message_index}",
                    label=get_string(lang, "copy_btn"),
                )

        with col_src:
            with st.container(key=f"ha_assistant_sources_cell_{message_index}"):
                if has_sources:
                    _render_sources_popover(
                        lang=lang,
                        sources=sources,
                        message_index=message_index,
                    )
                else:
                    st.caption(get_string(lang, "no_sources"))

        if with_feedback and col_fb is not None:
            with col_fb:
                _render_feedback_controls(
                    lang=lang,
                    username=username,
                    chat_id=chat_id,
                    message_index=message_index,
                    current=current_feedback,
                )

        del spacer  # reserved column, intentionally unused


def _html_admin_top_metrics(
    lang: str,
    *,
    total_pdfs: int,
    last_index: str,
    model: str,
) -> str:
    """Top admin metric row (HTML cards; values escaped)."""
    l1 = _html.escape(get_string(lang, "total_pdfs"))
    l2 = _html.escape(get_string(lang, "last_index_time"))
    l3 = _html.escape(get_string(lang, "active_model"))
    v2 = _html.escape(str(last_index))
    v3 = _html.escape(str(model))
    return (
        f'<div class="ha-admin-metric-row" role="group">'
        f'<div class="ha-admin-metric-card">'
        f'<div class="ha-admin-metric-card__label">{l1}</div>'
        f'<div class="ha-admin-metric-card__value">{int(total_pdfs)}</div>'
        f"</div>"
        f'<div class="ha-admin-metric-card">'
        f'<div class="ha-admin-metric-card__label">{l2}</div>'
        f'<div class="ha-admin-metric-card__value ha-admin-metric-card__value--sm">{v2}</div>'
        f"</div>"
        f'<div class="ha-admin-metric-card">'
        f'<div class="ha-admin-metric-card__label">{l3}</div>'
        f'<div class="ha-admin-metric-card__value ha-admin-metric-card__value--sm">{v3}</div>'
        f"</div>"
        f"</div>"
    )


def _html_admin_feedback_metrics(lang: str, *, total: int, ups: int, downs: int) -> str:
    """Feedback summary metric row."""
    l1 = _html.escape(get_string(lang, "feedback_log_total"))
    l2 = _html.escape(get_string(lang, "feedback_metric_helpful"))
    l3 = _html.escape(get_string(lang, "feedback_metric_unhelpful"))
    return (
        f'<div class="ha-admin-metric-row ha-admin-metric-row--tight" role="group">'
        f'<div class="ha-admin-metric-card">'
        f'<div class="ha-admin-metric-card__label">{l1}</div>'
        f'<div class="ha-admin-metric-card__value">{int(total)}</div>'
        f"</div>"
        f'<div class="ha-admin-metric-card">'
        f'<div class="ha-admin-metric-card__label">{l2}</div>'
        f'<div class="ha-admin-metric-card__value">{int(ups)}</div>'
        f"</div>"
        f'<div class="ha-admin-metric-card">'
        f'<div class="ha-admin-metric-card__label">{l3}</div>'
        f'<div class="ha-admin-metric-card__value">{int(downs)}</div>'
        f"</div>"
        f"</div>"
    )


def _render_admin_panel() -> None:
    lang = st.session_state.get("language", "en")
    _require_admin()
    pdf_files = _list_pdf_files()
    with st.container(key="ha_admin_panel"):
        st.markdown(
            f'<header class="ha-admin-hero">'
            f'<h1 class="ha-admin-hero__title">{_html.escape(get_string(lang, "admin_dashboard"))}</h1>'
            f'<p class="ha-admin-hero__sub">{_html.escape(get_string(lang, "admin_subtitle"))}</p>'
            f"</header>",
            unsafe_allow_html=True,
        )

        with st.expander(get_string(lang, "advanced_settings"), expanded=False):
            _render_advanced_settings_widgets(
                model_key="admin_adv_model", web_key="admin_adv_web"
            )

        st.markdown(
            _html_admin_top_metrics(
                lang,
                total_pdfs=len(pdf_files),
                last_index=_get_last_index_time(),
                model=str(st.session_state.selected_model),
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<h2 class="ha-admin-section-h">{_html.escape(get_string(lang, "admin_op_section"))}</h2>',
            unsafe_allow_html=True,
        )

        col_u, col_r, col_d = st.columns(3)
        with col_u:
            with st.container(border=True):
                st.markdown(
                    f'<p class="ha-admin-op-title">{_html.escape(get_string(lang, "admin_card_upload"))}</p>',
                    unsafe_allow_html=True,
                )
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

        with col_r:
            with st.container(border=True):
                st.markdown(
                    f'<p class="ha-admin-op-title">{_html.escape(get_string(lang, "admin_card_reindex"))}</p>',
                    unsafe_allow_html=True,
                )
                st.caption(get_string(lang, "reindex_desc"))
                if st.button(
                    get_string(lang, "reindex_btn"),
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner("Rebuilding database from PDFs, please wait..."):
                        reindex_pdfs()
                    st.session_state.last_index_time = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    st.success(get_string(lang, "db_rebuilt"))

        with col_d:
            with st.container(border=True):
                st.markdown(
                    f'<p class="ha-admin-op-title">{_html.escape(get_string(lang, "admin_card_delete"))}</p>',
                    unsafe_allow_html=True,
                )
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

        _render_admin_feedback_log(lang=lang)


def _render_admin_feedback_log(*, lang: str) -> None:
    """Flat, newest-first list of feedback signals across every user and chat."""
    with st.container(key="ha_admin_feedback"):
        st.markdown(
            f'<header class="ha-admin-feedback-head" style="margin-top:1.1rem;">'
            f'<p class="ha-admin-feedback-head__title">{_html.escape(get_string(lang, "feedback_log_title"))}</p>'
            f'<p class="ha-admin-feedback-head__sub">{_html.escape(get_string(lang, "feedback_log_desc"))}</p>'
            f"</header>",
            unsafe_allow_html=True,
        )

        entries = iter_all_feedback()
        if not entries:
            st.info(get_string(lang, "feedback_log_empty"))
            return

        ups = sum(1 for e in entries if e.get("feedback") == "up")
        downs = sum(1 for e in entries if e.get("feedback") == "down")

        st.markdown(
            f'<p class="ha-admin-glance-label">{_html.escape(get_string(lang, "feedback_at_a_glance"))}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            _html_admin_feedback_metrics(lang, total=len(entries), ups=ups, downs=downs),
            unsafe_allow_html=True,
        )

        filter_options = {
            "all": get_string(lang, "feedback_log_filter_all"),
            "up": get_string(lang, "feedback_log_filter_up"),
            "down": get_string(lang, "feedback_log_filter_down"),
        }
        with st.container(border=True):
            selected_filter = st.radio(
                get_string(lang, "feedback_log_filter"),
                options=list(filter_options.keys()),
                format_func=lambda k: filter_options[k],
                horizontal=True,
                key="admin_feedback_filter",
            )

        filtered = (
            entries
            if selected_filter == "all"
            else [e for e in entries if e.get("feedback") == selected_filter]
        )

        if not filtered:
            st.info(get_string(lang, "feedback_log_empty"))
            return

        for entry in filtered:
            icon = (
                get_string(lang, "feedback_expander_helpful")
                if entry.get("feedback") == "up"
                else get_string(lang, "feedback_expander_unhelpful")
            )
            user = str(entry.get("username", "?"))
            chat_part = _truncate_chat_title(entry.get("chat_title", ""), 48)
            ts = str(entry.get("feedback_at", "") or entry.get("timestamp", ""))
            title = f"{icon}  {user}  ·  {chat_part}"
            with st.expander(title, expanded=False):
                st.markdown(
                    f'<p class="ha-admin-entry-meta">{_html.escape(ts)}</p>',
                    unsafe_allow_html=True,
                )
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

    if not st.session_state.get("is_logged_in"):
        st.info(get_string(lang, "profile_login_prompt"))
        return

    username = st.session_state.username
    if st.session_state.role == "admin":
        st.info("Admin profile editing is disabled in this view.")
        return

    with st.container(key="ha_profile_adv"):
        with st.popover(
            get_string(lang, "advanced_settings"),
            help=get_string(lang, "advanced_settings_help"),
            icon=":material/tune:",
        ):
            _render_advanced_settings_widgets(
                model_key="main_adv_model", web_key="main_adv_web"
            )

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
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_global_styles()
    _inject_chat_layout_script()
    _init_auth_state()
    _initialize_cookie_manager()
    _try_auto_login_from_cookie()
    _normalize_active_page()

    # ``guest_on_login``: hide sidebar on full-bleed auth; Login is a bottom sidebar button, not
    # the top Chat/Profile radio.
    guest_on_login = False
    if not st.session_state.get("is_logged_in") and st.session_state.get("active_page") == "Login":
        guest_on_login = True

    lang = st.session_state.get("language", "en")

    if st.session_state.is_logged_in and st.session_state.role == "user":
        st.session_state.user_profile = _get_user_profile(st.session_state.username)

    _render_header()
    if guest_on_login:
        # After header styles so sidebar is gone and brand offset overrides header CSS.
        _inject_guest_login_fullbleed_styles()

    if not guest_on_login:
        with st.sidebar:
            if st.session_state.is_logged_in:
                _render_sidebar_user_header(lang, str(st.session_state.get("username", "")))
            else:
                _render_sidebar_guest_header(lang)

            if st.session_state.is_logged_in and st.session_state.role == "admin":
                sections = ["Admin Panel", "Chat"]
                current_index = sections.index(st.session_state.active_page) if st.session_state.active_page in sections else 0
                selected_section = st.radio(
                    "Select section",
                    sections,
                    index=current_index,
                    format_func=lambda s: _section_nav_label(lang, s),
                    label_visibility="collapsed",
                    key="ha_nav_admin",
                )
                st.session_state.active_page = selected_section
            elif st.session_state.is_logged_in:
                sections = ["Chat", "Profile"]
                current_index = sections.index(st.session_state.active_page) if st.session_state.active_page in sections else 0
                selected_section = st.radio(
                    "Select section",
                    sections,
                    index=current_index,
                    format_func=lambda s: _section_nav_label(lang, s),
                    label_visibility="collapsed",
                    key="ha_nav_user",
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
            else:
                _guest_top = ("Chat", "Profile")
                if st.session_state.active_page in _guest_top:
                    _gidx = _guest_top.index(st.session_state.active_page)
                else:
                    # On Login: keep last Chat/Profile in the radio (key ``ha_nav_guest_top``)
                    _prev = st.session_state.get("ha_nav_guest_top", "Chat")
                    _gidx = _guest_top.index(_prev) if _prev in _guest_top else 0
                st.radio(
                    "App section",
                    list(_guest_top),
                    index=_gidx,
                    format_func=lambda s: _section_nav_label(lang, s),
                    label_visibility="collapsed",
                    key="ha_nav_guest_top",
                    on_change=_on_guest_top_nav,
                )

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

            if not st.session_state.is_logged_in:
                with st.container(key="ha_sidebar_login_row"):
                    st.markdown(
                        (
                            '<div class="ha-sidebar-login-card">'
                            f'<p class="ha-sidebar-login-title">{_html.escape(get_string(lang, "sidebar_login_title"))}</p>'
                            f'<p class="ha-sidebar-login-hint">{_html.escape(get_string(lang, "sidebar_login_hint"))}</p>'
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        _section_nav_label(lang, "Login"),
                        use_container_width=True,
                        type="primary" if st.session_state.get("active_page") == "Login" else "secondary",
                        key="ha_sidebar_login_btn",
                    ):
                        st.session_state.active_page = "Login"
                        st.rerun()

            if st.session_state.is_logged_in:
                if st.button(get_string(lang, "logout"), use_container_width=True):
                    _logout()

    selected_section = st.session_state.get("active_page", "Chat")

    if selected_section == "Admin Panel":
        _render_admin_panel()
    elif selected_section == "Chat":
        _render_chat_page()
    elif selected_section == "Profile":
        _render_profile_page()
    elif selected_section == "Login":
        _render_auth_screen()
