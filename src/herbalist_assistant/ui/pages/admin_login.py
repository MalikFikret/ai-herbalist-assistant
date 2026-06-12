"""Admin login screen."""

import html as _html
import streamlit as st

from herbalist_assistant.ui.auth import _verify_admin_password, _init_auth_state
from herbalist_assistant.ui.pages.admin import _render_admin_panel


def _render_admin_login_screen() -> None:
    # First command must be set_page_config to ensure full-width layout
    st.set_page_config(page_title="Admin Login", layout="wide", initial_sidebar_state="collapsed")
    
    _init_auth_state()
    if st.session_state.get("admin_is_logged_in"):
        _render_admin_panel()
        return

    st.markdown(
        """
        <style>
        /* Hide default Streamlit header and sidebar */
        header, [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], footer { 
            display: none !important; 
        }

        /* Premium Card Styling */
        [data-testid="stForm"] {
            background-color: #FFFFFF !important;
            border-radius: 24px !important;
            box-shadow: 0 25px 50px -12px rgba(15, 23, 42, 0.15), 0 0 0 1px rgba(15, 23, 42, 0.05) !important;
            padding: 2.5rem 3rem !important;
            border: none !important;
        }

        /* Icon Container */
        .admin-icon-wrapper {
            display: flex;
            justify-content: center;
            margin-bottom: 1.5rem;
        }
        
        .admin-icon {
            background: linear-gradient(135deg, #3e4e35 0%, #5c6f5e 100%);
            width: 64px;
            height: 64px;
            border-radius: 18px;
            display: flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 10px 25px -5px rgba(62, 78, 53, 0.3);
        }

        /* Typography */
        .admin-login-title {
            text-align: center;
            font-size: 32px;
            font-family: 'Inter', -apple-system, sans-serif;
            font-weight: 800;
            margin-bottom: 8px;
            color: #3e4e35;
            letter-spacing: -0.02em;
        }

        .admin-login-subtitle {
            text-align: center;
            font-size: 15px;
            font-family: 'Inter', -apple-system, sans-serif;
            color: #5c6f5e;
            margin-bottom: 24px;
            font-weight: 500;
        }
        
        /* Input and Spacing Refinements */
        [data-testid="stTextInput"] {
            margin-bottom: 12px !important;
        }
        
        [data-testid="stTextInput"] input:focus {
            border-color: #7a9170 !important;
            box-shadow: 0 0 0 1px rgba(122, 145, 112, 0.28), 0 0 22px rgba(122, 145, 112, 0.14) !important;
        }
        
        [data-testid="stCheckbox"] {
            margin-bottom: 24px !important;
            accent-color: #5c6f5e !important;
        }
        
        [data-testid="stFormSubmitButton"] > button {
            border-radius: 12px !important;
            padding: 0.75rem 1rem !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            background: linear-gradient(135deg, #3e4e35 0%, #5c6f5e 100%) !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(62, 78, 53, 0.2) !important;
            transition: all 0.2s ease !important;
            color: #FFFFFF !important;
        }
        
        [data-testid="stFormSubmitButton"] > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 16px rgba(62, 78, 53, 0.3) !important;
            color: #FFFFFF !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Push the columns down slightly to simulate vertical centering
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col2:
        with st.form("admin_login_form", clear_on_submit=False, border=False):
            st.markdown(
                """
                <div class="admin-icon-wrapper">
                    <div class="admin-icon">
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                            <path d="M12 8v4"/>
                            <circle cx="12" cy="14" r="1"/>
                        </svg>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="admin-login-title">{_html.escape("Admin Login")}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="admin-login-subtitle">{_html.escape("Secure Area - Administrators Only")}</div>',
                unsafe_allow_html=True,
            )

            st.text_input(
                "Password",
                type="password",
                key="admin_login_password",
                label_visibility="collapsed",
                placeholder="Enter admin password",
            )

            st.checkbox(
                "Remember me",
                key="admin_remember_me",
            )

            submit_login = st.form_submit_button(
                "Login",
                type="primary",
                use_container_width=True,
            )

    if submit_login:
        password = st.session_state.get("admin_login_password", "")
        try:
            if _verify_admin_password(password):
                st.session_state.admin_is_logged_in = True
                st.rerun()
            else:
                st.error("Invalid admin password.")
        except RuntimeError as e:
            st.error(str(e))
