"""Profile page: user health profile editing and about section."""

from typing import Dict

import streamlit as st

from herbalist_assistant.db import repository as db_repository

from ..components import _render_advanced_settings_widgets
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
