"""Reusable UI widgets and shared rendering helpers.

Contains the header bar, sidebar headers, copy/sources/feedback controls,
advanced settings popover, and navigation helpers used across pages.
"""

import html as _html
import json
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components

from .auth import (
    AVAILABLE_MODELS,
    AVAILABLE_WEB_SEARCH_PROVIDERS,
    DEFAULT_MODEL,
    DEFAULT_WEB_SEARCH_PROVIDER,
)
from .i18n import get_string
from .state import update_message_feedback


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


def _copy_confirm_text() -> str:
    lang = st.session_state.get("language", "en")
    return get_string(lang, "copy_done")


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
        <div style=\\"display:flex;\\">
          <button id=\\"{key}\\"
            type=\\"button\\"
            style=\\"
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
            \\">
            <span>{safe_label}</span>
          </button>
          <script>
            (function() {{
              const btn = document.getElementById(\\"{key}\\");
              if (!btn) return;
              btn.addEventListener(\\"click\\", async function () {{
                try {{
                  await navigator.clipboard.writeText({safe_text});
                }} catch (err) {{
                  const ta = document.createElement(\\"textarea\\");
                  ta.value = {safe_text};
                  document.body.appendChild(ta);
                  ta.select();
                  try {{ document.execCommand(\\"copy\\"); }} catch (e) {{}}
                  document.body.removeChild(ta);
                }}
                const previous = btn.innerText;
                btn.innerText = \\"{confirm_text}\\";
                setTimeout(function () {{ btn.innerText = previous; }}, 1200);
              }});
            }})();
          </script>
        </div>
        """,
        height=32,
        width=132,
    )


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
