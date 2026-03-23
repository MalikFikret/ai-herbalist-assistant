import streamlit as st


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your 🌿 AI Herbalist Assistant. "
                    "Ask me about herbs, traditional remedies, and general wellness support. "
                    "I base my answers on the herbal PDFs you provide in the local `data/` folder.\n\n"
                    "_Note: This is for educational purposes only and not medical advice._"
                ),
            }
        ]


def append_message(*, role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})

