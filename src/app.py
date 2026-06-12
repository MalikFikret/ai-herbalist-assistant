import streamlit as st

from herbalist_assistant.ui.streamlit_app import run as run_main
from herbalist_assistant.ui.pages.admin_login import _render_admin_login_screen

pg = st.navigation(
    [
        st.Page(run_main, title="Home", default=True),
        st.Page(_render_admin_login_screen, title="Admin Login", url_path="admin"),
    ],
    position="hidden"
)

pg.run()
# Force Streamlit to reload

