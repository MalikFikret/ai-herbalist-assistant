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
from .styles import _auth_hero_logo_data_uri


def _brand_logo_html(
    *,
    wrapper_class: str,
    lang: str,
    fallback_eyebrow_class: str = "",
) -> str:
    """Brand logo image (login + chat hero). Falls back to leaf/sparkle eyebrow when PNG is absent."""
    uri = _auth_hero_logo_data_uri()
    if uri:
        alt = _html.escape(str(get_string(lang, "app_title")))
        return (
            f'<div class="{wrapper_class}">'
            f'<img src="{uri}" alt="{alt}" loading="lazy" decoding="async" />'
            f"</div>"
        )
    if fallback_eyebrow_class:
        return (
            f'<p class="{fallback_eyebrow_class}" aria-hidden="true">'
            f'<span class="{fallback_eyebrow_class}__leaf">🌿</span>'
            f'<span class="{fallback_eyebrow_class}__sparkle">✦</span>'
            f'<span class="{fallback_eyebrow_class}__sparkle">✦</span>'
            f"</p>"
        )
    return ""


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


def _render_header(*, in_sidebar: bool = False) -> None:
    lang = st.session_state.get("language", "en")
    raw_title = get_string(lang, "app_title")
    title = raw_title if isinstance(raw_title, str) else "AI Herbalist Assistant"
    safe_title = (
        title.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .strip()
    )
    target = st.sidebar if in_sidebar else st
    target.markdown(
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
        section.main:has(.st-key-ha_chat_composer_row) .block-container {{
            padding-top: 0 !important;
            margin-top: 0 !important;
        }}
        section.main:has(.st-key-ha_profile_page) .block-container {{
            padding-top: 0 !important;
            margin-top: 0 !important;
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
  <div class="ha-sidebar-header__eyebrow{eyebrow_mod}" role="heading" aria-level="2">{nav_lbl}</div>
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
  <div class="ha-sidebar-header__eyebrow{eyebrow_mod}" role="heading" aria-level="2">{nav_lbl}</div>
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
    """Auth language dropdown (EN / TR); changes app language on selection."""
    options = ["en", "tr"]
    current = lang if lang in options else "en"

    def _lang_label(code: str) -> str:
        key = "auth_lang_option_english" if code == "en" else "auth_lang_option_turkish"
        return str(get_string(current, key))

    with st.container(key="ha_auth_lang_header"):
        picked = st.selectbox(
            get_string(current, "auth_lang_label"),
            options=options,
            index=options.index(current),
            format_func=_lang_label,
            key="auth_lang_select",
            label_visibility="collapsed",
        )
    if picked and picked != current:
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
        <div class=\\"ha-assistant-copy-wrap\\" style=\\"display:flex;align-items:center;\\">
          <button id=\\"{key}\\" type=\\"button\\" class=\\"ha-assistant-copy-btn\\"
            title=\\"{safe_label}\\" aria-label=\\"{safe_label}\\">
            <svg class=\\"ha-assistant-copy-btn__icon\\" width=\\"18\\" height=\\"18\\"
              viewBox=\\"0 0 24 24\\" fill=\\"none\\" aria-hidden=\\"true\\">
              <rect width=\\"14\\" height=\\"14\\" x=\\"8\\" y=\\"8\\" rx=\\"2\\" ry=\\"2\\"></rect>
              <path d=\\"M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2\\"></path>
            </svg>
          </button>
          <style>
            .ha-assistant-copy-btn {{
              border: 1px solid rgba(92, 111, 94, 0.16);
              background: rgba(255, 255, 255, 0.78);
              color: #4a5248;
              border-radius: 10px;
              padding: 0;
              width: 1.85rem;
              height: 1.85rem;
              min-width: 1.85rem;
              min-height: 1.85rem;
              cursor: pointer;
              display: inline-flex;
              align-items: center;
              justify-content: center;
              box-shadow: 0 1px 3px rgba(60, 78, 58, 0.04);
              transition: background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
            }}
            .ha-assistant-copy-btn__icon {{
              stroke: #4a5248;
              stroke-width: 1.75;
              stroke-linecap: round;
              stroke-linejoin: round;
              fill: none;
              display: block;
            }}
            .ha-assistant-copy-btn:hover {{
              background: #ffffff;
              border-color: rgba(92, 111, 94, 0.26);
              box-shadow: 0 2px 8px rgba(60, 78, 58, 0.08);
            }}
            .ha-assistant-copy-btn:active {{
              transform: translateY(1px);
            }}
          </style>
          <script>
            (function() {{
              const btn = document.getElementById(\\"{key}\\");
              if (!btn) return;
              const defaultTitle = \\"{safe_label}\\";
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
                btn.title = \\"{confirm_text}\\";
                setTimeout(function () {{ btn.title = defaultTitle; }}, 1200);
              }});
            }})();
          </script>
        </div>
        """,
        height=36,
        width=44,
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
    base_key = f"fb_{chat_id}_{message_index}"

    with st.container(key=f"ha_assistant_feedback_group_{message_index}"):
        col_up, col_down = st.columns(2, gap="small", vertical_alignment="center")
        with col_up:
            if st.button(
                "",
                key=f"{base_key}_up",
                icon=":material/thumb_up:",
                type="primary" if current == "up" else "secondary",
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
                "",
                key=f"{base_key}_down",
                icon=":material/thumb_down:",
                type="primary" if current == "down" else "secondary",
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
            if has_sources:
                col_copy, col_src, col_fb, spacer = st.columns(
                    [0.72, 1.65, 0.72, 6.6], vertical_alignment="center"
                )
            else:
                col_copy, col_fb, spacer = st.columns(
                    [0.72, 0.72, 8.5], vertical_alignment="center"
                )
                col_src = None
        else:
            if has_sources:
                col_copy, col_src, spacer = st.columns(
                    [0.72, 1.65, 7.3], vertical_alignment="center"
                )
            else:
                col_copy, spacer = st.columns([0.72, 9.2], vertical_alignment="center")
                col_src = None
            col_fb = None

        with col_copy:
            with st.container(key=f"ha_assistant_copy_cell_{message_index}"):
                _render_copy_button(
                    text=content,
                    key=f"copy_{chat_id}_{message_index}",
                    label=get_string(lang, "copy_btn"),
                )

        if col_src is not None and has_sources:
            with col_src:
                with st.container(key=f"ha_assistant_sources_cell_{message_index}"):
                    _render_sources_popover(
                        lang=lang,
                        sources=sources,
                        message_index=message_index,
                    )

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
