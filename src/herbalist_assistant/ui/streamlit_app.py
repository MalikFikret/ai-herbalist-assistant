import hashlib
import json
import re
import secrets
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components

from herbalist_assistant import config
from herbalist_assistant.types import HerbalistState

from .i18n import get_string
from .resources import get_graph, reindex_pdfs
from .state import (
    append_message,
    delete_chat,
    get_chat_messages,
    get_user_chat_summaries,
    init_session_state,
    set_active_chat,
    start_new_chat,
)

USERS_FILE = Path(".users.json")
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "1234"
AVAILABLE_MODELS: List[str] = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
]

ALLERGY_HERB_ALIASES: Dict[str, List[str]] = {
    "papatya/chamomile": ["papatya", "chamomile", "camomile", "kamille"],
    "zencefil/ginger": ["zencefil", "ginger"],
    "nane/mint": ["nane", "mint", "peppermint"],
    "lavanta/lavender": ["lavanta", "lavender"],
    "ekinazya/echinacea": ["ekinazya", "echinacea"],
    "rezene/fennel": ["rezene", "fennel"],
    "isirgan/nettle": ["isirgan", "ısırgan", "nettle"],
}


def _run_with_timeout(func, *args, timeout_sec: int = 90):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        return future.result(timeout=timeout_sec)


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


def _generate_ai_response(user_input: str, profile: Dict[str, str]) -> tuple[str, List[str]]:
    """Generate answer from graph, fallback to simple local response."""
    profile_context = _build_profile_context(profile)
    question_payload = user_input if not profile_context else f"{user_input}\n\n{profile_context}"

    try:
        with st.spinner("Preparing retrieval and model..."):
            graph = _run_with_timeout(
                get_graph,
                st.session_state.selected_model,
                timeout_sec=90,
            )
        with st.spinner("Thinking..."):
            initial_state: HerbalistState = {
                "question": question_payload,
                "context": "",
                "answer": "",
                "sources": [],
            }
            final_state: Dict[str, Any] = _run_with_timeout(
                graph.invoke,
                initial_state,
                timeout_sec=90,
            )
        answer = final_state.get("answer", "I'm sorry, I could not generate an answer.")
        sources = final_state.get("sources", [])
        answer = _dedupe_answer_lines(answer)
        return answer, sources
    except Exception:
        # Simple fallback to keep chat UX responsive if model call fails.
        fallback = (
            "I could not reach the model right now, but I received your question: "
            f"'{user_input}'. Please try again in a moment."
        )
        if profile_context:
            fallback += " I will prioritize your allergy and condition information."
        return fallback, []


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
        .main .block-container {
            max-width: 1100px;
            padding-bottom: 2rem;
        }
        [data-testid="stAppViewContainer"] {
            background: var(--ha-bg) !important;
        }
        .ha-auth-layout {
            --auth-panel-height: 560px;
            max-width: 1020px;
            margin: 0.65rem auto 0 auto;
        }
        [data-testid="stHorizontalBlock"]:has(.ha-auth-hero) {
            align-items: stretch !important;
            column-gap: 12px !important;
        }
        [data-testid="stHorizontalBlock"]:has(.ha-auth-hero) > [data-testid="column"] {
            display: flex;
            align-items: stretch;
        }
        [data-testid="stHorizontalBlock"]:has(.ha-auth-hero) > [data-testid="column"] > div {
            width: 100%;
        }
        .ha-auth-hero {
            height: 100%;
            min-height: var(--auth-panel-height, 560px);
            box-sizing: border-box;
            border-radius: 20px 0 0 20px;
            padding: 1.8rem 1.8rem 1.45rem 1.8rem;
            background: radial-gradient(circle at 18% 20%, #234835 0%, #1b3528 38%, #12251d 100%);
            color: #eaf5ef;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .ha-auth-hero-brand {
            font-size: 0.9rem;
            letter-spacing: 0.02em;
            opacity: 0.92;
        }
        .ha-auth-hero-title {
            margin-top: 1.4rem;
            font-size: 2rem;
            line-height: 1.22;
            font-weight: 700;
            max-width: 18ch;
        }
        .ha-auth-hero-sub {
            margin-top: 0.85rem;
            font-size: 0.95rem;
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
            background: var(--ha-bg-2) !important;
            border: 1px solid var(--ha-border) !important;
            border-radius: 0 20px 20px 0;
            padding: 1.2rem 1.1rem 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(22, 61, 39, 0.08);
            min-height: 520px;
            max-width: 460px;
            margin: 0 auto;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ha-auth-logo) {
            background: var(--ha-bg-2) !important;
            border: 1px solid rgba(92, 168, 125, 0.25) !important;
            border-radius: 8px 20px 20px 8px !important;
            box-shadow: 0 10px 28px rgba(22, 61, 39, 0.08) !important;
            min-height: var(--auth-panel-height, 560px);
            box-sizing: border-box;
            max-width: none;
            width: 100%;
            margin: 0 auto !important;
            padding: 1.05rem 1.1rem 1rem 1.1rem !important;
            overflow: hidden;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ha-auth-logo) [data-testid="stVerticalBlock"] {
            min-height: calc(var(--auth-panel-height, 560px) - 2.1rem);
            max-width: 450px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 0.15rem;
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
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--ha-text);
            margin-top: 0.1rem;
            margin-bottom: 0.45rem;
        }
        [data-testid="stRadio"] label p {
            font-size: 0.95rem !important;
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
            min-height: 46px;
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
            min-height: 46px !important;
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
            .ha-auth-card,
            [data-testid="stVerticalBlockBorderWrapper"]:has(.ha-auth-logo) {
                border-radius: 16px;
                min-height: auto;
            }
            [data-testid="stVerticalBlockBorderWrapper"]:has(.ha-auth-logo) [data-testid="stVerticalBlock"] {
                min-height: auto;
                max-width: none;
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
        st.session_state.selected_model = config.GROQ_MODEL
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


def _load_users() -> Dict[str, Dict[str, str]]:
    if not USERS_FILE.exists():
        return {}
    try:
        with USERS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: Dict[str, Dict[str, str]]) -> None:
    with USERS_FILE.open("w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()


def _verify_password(password: str, password_hash: str, salt: str) -> bool:
    return _hash_password(password, salt) == password_hash


def _register_user(username: str, password: str, confirm_password: str) -> str:
    username = username.strip()
    if not username or not password or not confirm_password:
        return "All fields are required."
    if password != confirm_password:
        return "Passwords do not match."
    if username == ADMIN_USERNAME:
        return "This username is reserved."

    users = _load_users()
    if username in users:
        return "User already exists."

    salt = secrets.token_hex(16)
    users[username] = {
        "password_hash": _hash_password(password, salt),
        "salt": salt,
        "role": "user",
        "profile": {
            "name": "",
            "age": "",
            "gender": "",
            "allergies": "",
            "conditions": "",
        },
    }
    _save_users(users)
    return "Account created successfully. You can now log in."


def _reset_user_password(username: str, new_password: str, confirm_password: str) -> str:
    username = username.strip()
    if not username or not new_password or not confirm_password:
        return "All fields are required."
    if new_password != confirm_password:
        return "Passwords do not match."
    if len(new_password) < 4:
        return "Password must be at least 4 characters."
    if username == ADMIN_USERNAME:
        return "Admin password reset is disabled in this screen."

    users = _load_users()
    if username not in users:
        return "User not found."

    salt = secrets.token_hex(16)
    users[username]["password_hash"] = _hash_password(new_password, salt)
    users[username]["salt"] = salt
    _save_users(users)
    return "Password reset successful. You can now log in."


def _authenticate_user(username: str, password: str) -> Dict[str, str]:
    username = username.strip()
    if not username or not password:
        return {"status": "error", "message": "Username and password are required."}

    if username == ADMIN_USERNAME:
        if password == ADMIN_PASSWORD:
            return {"status": "ok", "role": "admin"}
        return {"status": "error", "message": "Wrong password."}

    users = _load_users()
    record = users.get(username)
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

    left_col, right_col = st.columns([1.03, 1], gap="small")
    hero_foot = get_string(lang, "hero_foot")
    hero_foot_html = (
        f'<div class="ha-auth-hero-foot">{hero_foot}</div>' if hero_foot else ""
    )

    with left_col:
        st.markdown(
            f'''
            <div class="ha-auth-hero">
                <div>
                    <div class="ha-auth-hero-brand">{get_string(lang, "hero_brand")}</div>
                    <div class="ha-auth-hero-title">{get_string(lang, "hero_title")}</div>
                    <div class="ha-auth-hero-sub">
                        {get_string(lang, "hero_sub")}
                    </div>
                </div>
                {hero_foot_html}
            </div>
            ''',
            unsafe_allow_html=True,
        )

    with right_col:
        with st.container(border=True):
            st.markdown('<div class="ha-auth-logo">🌿</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ha-auth-right-head">{get_string(lang, "get_started")}</div>', unsafe_allow_html=True)
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
                tab_login, tab_register = st.tabs([get_string(lang, "login_btn"), get_string(lang, "create_account")])
                
                with tab_login:
                    with st.form("login_form", clear_on_submit=False):
                        username = st.text_input(get_string(lang, "username"), key="login_username")
                        password = st.text_input(get_string(lang, "password"), type="password", key="login_password")
                        submit_login = st.form_submit_button(get_string(lang, "login_btn"), type="primary", use_container_width=True)
                    
                    st.markdown('<div class="ha-forgot-link">', unsafe_allow_html=True)
                    if st.button(get_string(lang, "forgot_pwd"), use_container_width=False):
                        st.session_state.auth_mode = "Reset Password"
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

                with tab_register:
                    with st.form("register_form", clear_on_submit=True):
                        reg_username = st.text_input(get_string(lang, "username"), key="register_username")
                        reg_password = st.text_input(get_string(lang, "password"), type="password", key="register_password")
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
                        st.session_state.active_page = "Admin Panel" if st.session_state.role == "admin" else "Chat"
                        st.success("Login successful.")
                        st.rerun()
                        
                if submit_register:
                    result = _register_user(reg_username, reg_password, reg_confirm)
                    if result.startswith("Account created"):
                        st.success(result)
                    else:
                        st.error(result)


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

    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            if "Hello! I am your 🌿 AI Herbalist" in msg["content"] or "Merhaba! Ben sizin 🌿" in msg["content"]:
                continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            sources = msg.get("sources", [])
            if msg["role"] == "assistant" and sources:
                st.caption(f"Sources: {', '.join(sources)}")

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
    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            st.caption(f"Sources: {', '.join(sources)}")
    append_message(role="assistant", content=answer, username=username, sources=sources)


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

        with st.container(border=True):
            st.markdown(get_string(lang, "model_selection"))
            selected_model = st.selectbox(
                get_string(lang, "choose_model"),
                options=AVAILABLE_MODELS,
                index=AVAILABLE_MODELS.index(st.session_state.selected_model)
                if st.session_state.selected_model in AVAILABLE_MODELS
                else 0,
                key="admin_model_select",
            )
            st.session_state.selected_model = selected_model


def _render_history_page() -> None:
    lang = st.session_state.get("language", "en")
    username = st.session_state.username
    init_session_state(username)
    st.markdown(f'<div class="ha-section-title">{get_string(lang, "history")}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ha-section-subtitle">{get_string(lang, "history_desc")}</div>',
        unsafe_allow_html=True,
    )

    chats = get_user_chat_summaries(username)
    if not chats:
        st.info(get_string(lang, "no_history"))
        return

    for idx, chat in enumerate(reversed(chats), start=1):
        title = chat.get("title", get_string(lang, "new_chat"))
        updated_at = chat.get("updated_at", "-")
        chat_id = chat.get("id", "")
        with st.expander(f"{idx}. {title}  ({updated_at})", expanded=idx == 1):
            col_info, col_action = st.columns([4, 1])
            with col_info:
                st.caption(
                    f"Messages: {chat.get('message_count', 0)} | Created: {chat.get('created_at', '-')}"
                )
            with col_action:
                if st.button(get_string(lang, "open_btn"), key=f"open_chat_{chat_id}", use_container_width=True):
                    if set_active_chat(username, chat_id):
                        st.session_state.active_page = "Chat"
                        st.rerun()

            messages = get_chat_messages(username, chat_id)
            turns = []
            i = 0
            while i < len(messages):
                msg = messages[i]
                if msg.get("role") != "user":
                    i += 1
                    continue
                user_msg = msg
                assistant_msg = None
                if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                    assistant_msg = messages[i + 1]
                    i += 2
                else:
                    i += 1
                turns.append((user_msg, assistant_msg))

            if not turns:
                st.caption("No user messages yet in this chat.")
            for user_msg, assistant_msg in turns:
                st.markdown(f"**User:** {user_msg.get('content', '')}")
                if assistant_msg:
                    st.markdown(f"**Assistant:** {assistant_msg.get('content', '')}")
                    sources = assistant_msg.get("sources", [])
                    if sources:
                        st.markdown(f"**Sources:** {', '.join(sources)}")
                st.divider()


def _render_about_page() -> None:
    lang = st.session_state.get("language", "en")
    st.markdown(f'<div class="ha-section-title">{get_string(lang, "about")}</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(get_string(lang, "about_desc"))


def _get_user_profile(username: str) -> Dict[str, str]:
    users = _load_users()
    record = users.get(username, {})
    profile = record.get("profile", {}) if isinstance(record, dict) else {}
    return {
        "name": profile.get("name", profile.get("full_name", "")),
        "age": profile.get("age", ""),
        "gender": profile.get("gender", ""),
        "allergies": profile.get("allergies", ""),
        "conditions": profile.get("conditions", profile.get("notes", "")),
    }


def _save_user_profile(username: str, profile: Dict[str, str]) -> bool:
    users = _load_users()
    if username not in users:
        return False

    users[username]["profile"] = {
        "name": profile.get("name", "").strip(),
        "age": profile.get("age", "").strip(),
        "gender": profile.get("gender", "").strip(),
        "allergies": profile.get("allergies", "").strip(),
        "conditions": profile.get("conditions", "").strip(),
    }
    _save_users(users)
    return True


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
        initial_sidebar_state="expanded",
    )
    _inject_global_styles()
    _init_auth_state()

    lang = st.session_state.get("language", "en")

    if not st.session_state.is_logged_in:
        # Language selector: visible on the login screen.
        top_left, top_right = st.columns([8, 2], gap="small")
        with top_right:
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

        _render_auth_screen()
        return

    if st.session_state.role == "user":
        st.session_state.user_profile = _get_user_profile(st.session_state.username)

    if st.session_state.role != "admin":
        _render_header()

    with st.sidebar:
        st.markdown(get_string(lang, "sidebar_nav"))
        st.caption(f"**{st.session_state.username}**")

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
