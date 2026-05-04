"""Streamlit application orchestrator.

This module is the entry point for the Streamlit UI. It wires together
the page modules, sidebar navigation, and global initialization. All
heavy logic (auth, styles, pages) lives in dedicated submodules.
"""

import herbalist_assistant  # eager .env load before LangChain/LangSmith imports

import streamlit as st

from herbalist_assistant import log_langsmith_status
from herbalist_assistant.db import ensure_database_ready, repository as db_repository

log_langsmith_status()

# Bring the SQLite schema up-to-date. Idempotent, so subsequent
# Streamlit reruns are near-instant.
ensure_database_ready()

# The relative imports below intentionally follow log_langsmith_status() so
# the LangSmith banner is emitted before the rest of the UI layer pulls in
# LangChain modules. Ruff flags this as E402, which we silence on purpose.
from .auth import (  # noqa: E402
    _init_auth_state,
    _logout,
    _normalize_active_page,
    _try_auto_login_from_cookie,
)
from .components import (  # noqa: E402
    _on_guest_top_nav,
    _render_header,
    _render_sidebar_guest_header,
    _render_sidebar_user_header,
    _section_nav_label,
)
from .cookies import _initialize_cookie_manager  # noqa: E402
from .i18n import get_string  # noqa: E402
from .pages.admin import _render_admin_panel  # noqa: E402
from .pages.chat import (  # noqa: E402
    _render_chat_page,
    _sync_conversations_to_session,
    _truncate_chat_title,
)
from .pages.login import _render_auth_screen  # noqa: E402
from .pages.profile import _get_user_profile, _render_profile_page  # noqa: E402
from .state import (  # noqa: E402
    delete_chat,
    get_user_chat_summaries,
    set_active_chat,
    start_new_chat,
)
from .styles import (  # noqa: E402
    _inject_chat_layout_script,
    _inject_global_styles,
    _inject_guest_login_fullbleed_styles,
)

# ---------------------------------------------------------------------------
# Backward-compatible re-exports so existing tests that import from
# ``herbalist_assistant.ui.streamlit_app`` continue to work unchanged.
# ---------------------------------------------------------------------------
from .auth import (  # noqa: E402, F811
    ADMIN_USERNAME,
    AVAILABLE_MODELS,
    AVAILABLE_WEB_SEARCH_PROVIDERS,
    DEFAULT_MODEL,
    DEFAULT_WEB_SEARCH_PROVIDER,
    _ADMIN_DEFAULT_PASSWORD,
    _ADMIN_PASSWORD_ENV,
    _ADMIN_PASSWORD_HASH_ENV,
    _ADMIN_PASSWORD_SALT_ENV,
    _authenticate_user,
    _hash_password,
    _register_user,
    _reset_user_password,
    _verify_admin_password,
    _verify_password,
)
from .components import (  # noqa: E402, F811
    _normalize_sources,
    _render_advanced_settings_widgets,
    _render_assistant_action_row,
    _source_entry_label,
)
from .pages.chat import (  # noqa: E402, F811
    _extract_sources_from_docs,
)


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
                if st.button(
                    get_string(lang, "new_chat"),
                    use_container_width=True,
                    type="primary",
                    icon=":material/edit_square:",
                    key="ha_sidebar_new_chat",
                ):
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
                        if chat_id != active_chat_id:
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
                        if st.button(
                            get_string(lang, "delete_chat"),
                            use_container_width=True,
                            icon=":material/delete:",
                            key="ha_sidebar_delete_chat",
                        ):
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
                import html as _html

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
                if st.button(
                    get_string(lang, "logout"),
                    use_container_width=True,
                    icon=":material/logout:",
                    key="ha_sidebar_logout",
                ):
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
