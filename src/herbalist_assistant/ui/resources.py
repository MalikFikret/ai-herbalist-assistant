import streamlit as st
from dotenv import load_dotenv

from herbalist_assistant import config
from herbalist_assistant.graph.builder import build_graph
from herbalist_assistant.llm.groq import create_groq_llm, get_groq_api_key
from herbalist_assistant.prompts import build_prompt
from herbalist_assistant.rag.embeddings import create_embeddings
from herbalist_assistant.rag.vectorstore import load_or_build_vectorstore, make_retriever
import shutil


@st.cache_resource(show_spinner=False)
def load_environment() -> None:
    load_dotenv()


@st.cache_resource(show_spinner=True)
def get_embeddings():
    return create_embeddings(config.EMBEDDING_MODEL)


@st.cache_resource(show_spinner=True)
def get_vectorstore():
    load_environment()
    embeddings = get_embeddings()
    return load_or_build_vectorstore(
        data_dir=config.DATA_DIR,
        persist_dir=config.CHROMA_DIR,
        embeddings=embeddings,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    load_environment()
    api_key = get_groq_api_key()
    return create_groq_llm(
        api_key=api_key,
        model_name=config.GROQ_MODEL,
        temperature=config.LLM_TEMPERATURE,
    )


@st.cache_resource(show_spinner=True)
def get_graph():
    vectorstore = get_vectorstore()
    retriever = make_retriever(vectorstore, k=config.RETRIEVER_K)
    llm = get_llm()
    return build_graph(retriever=retriever, llm=llm, prompt_fn=build_prompt)


def reindex_pdfs() -> None:
    """Eagerly rebuild the local Chroma index from PDFs in data/."""
    # Clear cached objects first so nothing holds open handles.
    get_graph.clear()
    get_vectorstore.clear()
    get_embeddings.clear()

    # Remove persisted Chroma DB on disk.
    chroma_dir = config.CHROMA_DIR
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)

    # Eagerly rebuild the vectorstore (and thus the on-disk DB) now.
    _ = get_vectorstore()

