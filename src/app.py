import os
from pathlib import Path
from typing import TypedDict, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, END


class HerbalistState(TypedDict):
    question: str
    context: str
    answer: str


@st.cache_resource(show_spinner=False)
def load_environment() -> None:
    """Load environment variables from .env once per process."""
    load_dotenv()


@st.cache_resource(show_spinner=True)
def get_embeddings():
    """Create and cache the HuggingFace embedding model."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)


def _load_pdfs_from_data_dir(data_dir: Path):
    """Load all PDF documents from the given data directory."""
    if not data_dir.exists():
        return []

    pdf_paths = sorted(data_dir.glob("*.pdf"))
    docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    return docs


@st.cache_resource(show_spinner=True)
def get_vectorstore():
    """Load or build a persistent Chroma vectorstore from local PDFs."""
    load_environment()

    persist_dir = ".chroma_db"
    embeddings = get_embeddings()

    data_dir = Path("data")
    # If the persist directory exists and is non-empty, load existing DB
    if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        return vectorstore

    # Otherwise, build from PDFs (if any)
    docs = _load_pdfs_from_data_dir(data_dir)
    if not docs:
        # Create an empty store to avoid hard failure; retrieval will just return nothing.
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        return vectorstore

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


@st.cache_resource(show_spinner=False)
def get_llm():
    """Instantiate the Groq Llama 3 model."""
    load_environment()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Please add it to your .env file or environment."
        )

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=api_key,
    )
    return llm


def build_prompt(question: str, context: str) -> str:
    """Construct a simple RAG-style prompt for the herbalist assistant."""
    system_instructions = (
        "You are an AI herbalist assistant specializing in herbal medicine and natural remedies. "
        "Use only the provided context from herbal medicine books to answer the question. "
        "If the context does not contain enough information, say you are not sure and suggest "
        "consulting a qualified healthcare professional.\n\n"
        "Always include a brief disclaimer that this is not medical advice and that users should "
        "consult a doctor or licensed healthcare provider before trying any remedy.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer as a concise, friendly herbalist assistant:"
    )
    return system_instructions


def create_graph():
    """Create and compile the LangGraph StateGraph for RAG."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()

    def retrieval_node(state: HerbalistState) -> HerbalistState:
        question = state.get("question", "")
        if not question:
            return state

        docs = retriever.invoke(question)
        if not docs:
            context = ""
        else:
            context = "\n\n".join(d.page_content for d in docs)
        new_state: HerbalistState = {
            "question": question,
            "context": context,
            "answer": state.get("answer", ""),
        }
        return new_state

    def generation_node(state: HerbalistState) -> HerbalistState:
        question = state.get("question", "")
        context = state.get("context", "")
        prompt = build_prompt(question, context)

        response = llm.invoke(prompt)
        # ChatGroq returns an AIMessage; get the content attribute if present
        answer = getattr(response, "content", str(response))

        new_state: HerbalistState = {
            "question": question,
            "context": context,
            "answer": answer,
        }
        return new_state

    graph = StateGraph(HerbalistState)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("generation", generation_node)

    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "generation")
    graph.add_edge("generation", END)

    app_graph = graph.compile()
    return app_graph


@st.cache_resource(show_spinner=True)
def get_graph():
    """Cached accessor for the compiled LangGraph app."""
    return create_graph()


def init_session_state():
    """Initialize Streamlit session_state for chat history."""
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


def render_chat_ui():
    """Render the Streamlit chat interface and handle interactions."""
    st.set_page_config(page_title="AI Herbalist Assistant", page_icon="🌿")
    st.title("🌿 AI Herbalist Assistant")
    st.caption(
        "Ask questions about herbal medicine and natural remedies based on your local PDF library. "
        "This tool is for educational purposes only and does not provide medical advice."
    )

    init_session_state()

    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box for new question
    user_input = st.chat_input("Ask about herbs and remedies...")
    if not user_input:
        return

    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the LangGraph RAG pipeline
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
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except RuntimeError as e:
        # Likely missing API key or similar configuration error
        error_msg = f"Configuration error: {e}"
        with st.chat_message("assistant"):
            st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})


def main():
    render_chat_ui()


if __name__ == "__main__":
    main()

