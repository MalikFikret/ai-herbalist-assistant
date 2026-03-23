from typing import Any, Dict

import streamlit as st

from herbalist_assistant.types import HerbalistState

from .resources import get_graph, reindex_pdfs
from .state import append_message, init_session_state


def run() -> None:
    st.set_page_config(
        page_title="AI Herbalist Assistant",
        page_icon="🌿",
        initial_sidebar_state="collapsed",
    )
    st.title("🌿 AI Herbalist Assistant")
    st.caption(
        "Ask questions about herbal medicine and natural remedies based on your local PDF library. "
        "This tool is for educational purposes only and does not provide medical advice."
    )

    with st.sidebar:
        st.subheader("Library")
        if st.button("Re-index PDFs", type="primary"):
            with st.spinner("Rebuilding database from PDFs, please wait..."):
                reindex_pdfs()
            st.success("Database rebuilt successfully!")

    init_session_state()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about herbs and remedies...")
    if not user_input:
        return

    append_message(role="user", content=user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        graph = get_graph()
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                initial_state: HerbalistState = {
                    "question": user_input,
                    "context": "",
                    "answer": "",
                }
                final_state: Dict[str, Any] = graph.invoke(initial_state)
                answer = final_state.get("answer", "I'm sorry, I could not generate an answer.")
                st.markdown(answer)
        append_message(role="assistant", content=answer)
    except RuntimeError as e:
        error_msg = f"Configuration error: {e}"
        with st.chat_message("assistant"):
            st.error(error_msg)
        append_message(role="assistant", content=error_msg)

