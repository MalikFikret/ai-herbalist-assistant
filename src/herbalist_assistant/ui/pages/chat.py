"""Chat page: conversation UI, agent invocation, and message handling."""

import asyncio
import html as _html
import logging
import secrets
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st



from ..components import (
    _normalize_sources,
    _render_advanced_settings_widgets,
    _render_assistant_action_row,
)
from ..i18n import get_string
from ..state import (
    append_message,
    get_chat_messages,
    get_user_chat_summaries,
    init_session_state,
)

_logger = logging.getLogger("herbalist_assistant.ui.pages.chat")

# Maximum number of past messages we forward to the agent graph for memory.
# Matches _MAX_HISTORY_MESSAGES in graph/extractors.py.
_CHAT_HISTORY_LIMIT = 6

_AGENT_TIMEOUT_SEC = 120


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


def _invoke_agent_with_timeout(
    payload: Dict[str, Any],
    timeout_sec: int = _AGENT_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """Run ``agent_graph_app.ainvoke`` under an asyncio timeout.

    Uses the compiled graph's native async entrypoint, wrapped in
    ``asyncio.wait_for`` so the UI never blocks forever. On timeout we
    raise ``TimeoutError`` so the caller can render a user-friendly message.
    """

    # Lazy import to avoid triggering heavy LangChain module loads at
    # import time (breaks test environments without full deps).
    from herbalist_assistant.graph.advanced_graph import app as agent_graph_app

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


def _generate_ai_response(user_input: str, profile: Dict[str, str]) -> tuple[str, List[Dict[str, Any]]]:
    """Generate answer from advanced LangGraph app, with graceful fallback.

    Returns (answer_text, sources) where ``sources`` is a list of structured
    dicts (``{"kind": "pdf", "file": ..., "page": ...}`` today; kept flexible
    so web URLs etc. can be added later without changing the call-site).
    """
    from ..auth import DEFAULT_MODEL, DEFAULT_WEB_SEARCH_PROVIDER

    lang = st.session_state.get("language", "en")
    normalized_profile = profile if isinstance(profile, dict) else {}
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
                    "question": user_input,
                    "user_profile": normalized_profile,
                    "chat_history": chat_history,
                    "model_name": model_name,
                    "web_search_provider": web_search_provider,
                    "ui_language": lang,
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
        if normalized_profile:
            fallback += " " + get_string(lang, "agent_error_profile_hint")
        return fallback, []


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

    has_user_msg = any(m["role"] == "user" for m in st.session_state.messages)

    if not has_user_msg:
        if logged_in:
            _welcome_line = get_string(lang, "chat_welcome_user").format(
                name=_html.escape(_welcome_display_name(st.session_state.get("username", "")))
            )
        else:
            _welcome_line = "Welcome" if lang == "en" else "Hoş Geldiniz"

        _subtitle = (
            "Ask me about herbs, traditional remedies, and general wellness support."
            if lang == "en"
            else "Bana şifalı bitkiler, geleneksel tedaviler ve genel sağlık desteği hakkında sorular sorabilirsiniz."
        )

        if not logged_in:
            banner_text = get_string(lang, "chat_guest_banner")
            if isinstance(banner_text, str):
                banner_text = banner_text.replace("Login", "<b>Login</b>")
            st.markdown(
                f'<div class="ha-chat-hero__guest-banner">ℹ️  {banner_text}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="ha-chat-hero">
                <div class="ha-chat-hero__logo">
                    <svg viewBox="0 0 100 100" width="80" height="80" style="filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.1));">
                        <circle cx="50" cy="50" r="48" fill="url(#grad)" stroke="rgba(47, 79, 79, 0.1)" stroke-width="1.5"/>
                        <defs>
                            <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" style="stop-color:#FCFBFA;stop-opacity:1" />
                                <stop offset="100%" style="stop-color:#E8E4DB;stop-opacity:1" />
                            </linearGradient>
                        </defs>
                        <!-- Leaf V silhouette -->
                        <path d="M 50 80 C 40 60, 25 45, 30 30 C 35 20, 45 25, 50 40 C 55 25, 65 20, 70 30 C 75 45, 60 60, 50 80 Z" fill="#2F4F4F" opacity="0.85"/>
                        <path d="M 50 80 C 42 65, 30 55, 30 40 C 30 30, 40 30, 50 50" fill="none" stroke="#FAF9F6" stroke-width="1.5"/>
                        <path d="M 50 80 C 58 65, 70 55, 70 40 C 70 30, 60 30, 50 50" fill="none" stroke="#FAF9F6" stroke-width="1.5"/>
                        <path d="M 50 50 L 50 80" fill="none" stroke="#FAF9F6" stroke-width="2"/>
                    </svg>
                </div>
                <h1 class="ha-chat-hero__title">{_welcome_line}</h1>
                <p class="ha-chat-hero__subtitle">{_subtitle}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
