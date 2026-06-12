"""Full-bleed login/registration screen."""

import html as _html
import time
from datetime import datetime, timedelta

import streamlit as st

from ..auth import (
    _authenticate_user,
    _register_user,
    _reset_user_password,
)
from ..components import _render_auth_language_switch
from ..cookies import (
    _REMEMBER_COOKIE_NAME,
    _REMEMBER_TTL_DAYS,
    _forget_remembered_password,
    _get_cookie_manager,
    _make_remember_token,
    _on_login_username_change,
    _store_remembered_password,
)
from ..i18n import get_string
from ..pages.chat import _sync_conversations_to_session
from ..pages.profile import _get_user_profile
from ..state import start_new_chat


def _render_auth_screen() -> None:
    lang = st.session_state.get("language", "en")
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "Login"

    with st.container(key="ha_auth_shell"):
        with st.container(key="ha_auth_card"):
            if not st.session_state.is_logged_in:
                with st.container(key="ha_auth_back_chat_wrap"):
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
            left_col, right_col = st.columns([1, 1], gap="small")

            w_panel = _html.escape(get_string(lang, "auth_lux_welcome_title"))
            w_tagline = _html.escape(get_string(lang, "auth_lux_panel_tagline"))
            with left_col:
                st.markdown(
                    f"""
                    <div class="ha-lux-welcome ha-lux-welcome--panel">
                        <div class="ha-lux-welcome--panel__inner">
                            <h1 class="ha-lux-welcome--panel__title">{w_panel}</h1>
                            <p class="ha-lux-welcome--panel__tagline">{w_tagline}</p>
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

                    with st.container(key="ha_auth_form_body"):
                        if auth_lux_tab == "login":
                            st.markdown(
                                f'<div class="ha-lux-form-title">{_html.escape(get_string(lang, "auth_lux_login_title"))}</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<div class="ha-lux-form-sub">{_html.escape(get_string(lang, "auth_lux_login_sub"))}</div>',
                                unsafe_allow_html=True,
                            )
                            with st.container(key="ha_lux_remember_forgot_bar"):
                                with st.form("login_form", clear_on_submit=False, border=False):
                                    st.text_input(
                                        get_string(lang, "username"),
                                        key="login_username",
                                        label_visibility="collapsed",
                                        placeholder=get_string(lang, "auth_lux_email_ph"),
                                    )
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
                                    with st.container(key="ha_auth_primary_submit"):
                                        submit_login = st.form_submit_button(
                                            get_string(lang, "login_btn"),
                                            type="primary",
                                            use_container_width=True,
                                        )
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
                                        use_container_width=True,
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
                                    import logging

                                    logging.getLogger("herbalist_assistant.ui.pages.login").debug(
                                        "Remember-me cookie write failed",
                                        exc_info=True,
                                    )
                            # Streamlit'in cookie set/delete komutunu tarayıcıya
                            # iletmesi için kısa bir yumuşatma. ``st.rerun()``
                            # mevcut run'ı kestiğinde queued mesajların flush
                            # olmasına imkan veriyor.
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
