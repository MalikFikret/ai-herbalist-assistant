"""Profile page: user health profile editing and about section."""

import html as _html
from typing import Dict

import streamlit as st

from herbalist_assistant.db import repository as db_repository

from ..i18n import get_string


def _get_user_profile(username: str) -> Dict[str, str]:
    return db_repository.get_user_profile(username)


def _save_user_profile(username: str, profile: Dict[str, str]) -> bool:
    return db_repository.save_user_profile(username, profile)


def _render_about_page() -> None:
    lang = st.session_state.get("language", "en")
    st.markdown(f'<div class="ha-section-title">{get_string(lang, "about")}</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(get_string(lang, "about_desc"))


def _render_profile_page() -> None:
    lang = st.session_state.get("language", "en")
    with st.container(key="ha_profile_center"):
        _render_profile_page_content(lang)


def _render_profile_page_content(lang: str) -> None:
    title = _html.escape(get_string(lang, "profile"))
    desc = _html.escape(get_string(lang, "profile_desc"))

    with st.container(key="ha_profile_page"):
        st.markdown(
            f"""
            <header class="ha-profile-header">
              <h1 class="ha-profile-header__title">{title}</h1>
              <p class="ha-profile-header__desc">
                <span class="ha-profile-header__leaf" aria-hidden="true">🌿</span>
                {desc}
              </p>
            </header>
            """,
            unsafe_allow_html=True,
        )

        if not st.session_state.get("is_logged_in"):
            st.info(get_string(lang, "profile_login_prompt"))
            return

        username = st.session_state.username
        if st.session_state.role == "admin":
            st.info("Admin profile editing is disabled in this view.")
            return

        current = _get_user_profile(username)
        st.session_state.user_profile = current

        with st.container(key="ha_profile_card"):
            with st.form("profile_form", clear_on_submit=False):
                with st.container(key="ha_profile_row_name_age"):
                    col_name, col_age = st.columns([3, 2], gap="medium")
                    with col_name:
                        name = st.text_input(
                            get_string(lang, "name"),
                            value=current.get("name", ""),
                            icon=":material/person:",
                        )
                    with col_age:
                        age = st.text_input(
                            get_string(lang, "age"),
                            value=current.get("age", ""),
                            icon=":material/calendar_today:",
                        )

                gender_options = get_string(lang, "gender_opts")
                try:
                    g_idx = gender_options.index(current.get("gender", ""))
                except ValueError:
                    g_idx = 0

                with st.container(key="ha_profile_row_gender"):
                    col_gender, _col_spacer = st.columns([1, 1], gap="medium")
                    with col_gender:
                        gender = st.selectbox(
                            get_string(lang, "gender"),
                            options=gender_options,
                            index=g_idx,
                        )

                with st.container(key="ha_profile_row_allergies"):
                    allergies = st.text_area(
                        get_string(lang, "allergies"),
                        value=current.get("allergies", ""),
                        placeholder=get_string(lang, "allergies_placeholder"),
                        height=80,
                    )
                with st.container(key="ha_profile_row_conditions"):
                    conditions = st.text_area(
                        get_string(lang, "conditions"),
                        value=current.get("conditions", ""),
                        placeholder=get_string(lang, "conditions_placeholder"),
                        height=80,
                    )

                with st.container(key="ha_profile_save_row"):
                    save_profile = st.form_submit_button(
                        get_string(lang, "save_profile"),
                        type="primary",
                        icon=":material/eco:",
                        use_container_width=False,
                    )

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
                        st.rerun()
                    else:
                        st.error(get_string(lang, "profile_save_err"))
