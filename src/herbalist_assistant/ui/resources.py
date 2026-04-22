"""Streamlit-scoped cached resources for embeddings, vectorstore, and reindex.

The live chat path is served by ``herbalist_assistant.graph.advanced_graph.app``,
which carries its own module-level ``lru_cache`` for the retriever and LLMs.
This module keeps the Streamlit-side caches (so the admin UI can show metrics
without re-running expensive setup) and exposes a single ``reindex_pdfs`` that
invalidates BOTH cache layers.
"""

from __future__ import annotations

import shutil

import streamlit as st
from dotenv import load_dotenv

from herbalist_assistant import config
from herbalist_assistant.rag.embeddings import create_embeddings
from herbalist_assistant.rag.vectorstore import load_or_build_vectorstore


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


def reindex_pdfs() -> None:
    """Eagerly rebuild the local Chroma index from PDFs in ``data/``.

    Invalidates:
      - Streamlit ``@st.cache_resource`` caches (embeddings, vectorstore).
      - The advanced-graph's module-level ``lru_cache`` caches (retriever,
        LLMs) so the live chat path picks up the fresh index immediately
        without a process restart.
    """
    # Clear Streamlit caches first so nothing holds an open DB handle.
    get_vectorstore.clear()
    get_embeddings.clear()

    # Clear the advanced-graph's own caches -- otherwise the chat path would
    # keep hitting a stale retriever until the Streamlit process restarts.
    from herbalist_assistant.graph.advanced_graph import reset_runtime_caches

    reset_runtime_caches()

    # Remove persisted Chroma DB on disk.
    chroma_dir = config.CHROMA_DIR
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)

    # Eagerly rebuild the vectorstore (and thus the on-disk DB) now.
    _ = get_vectorstore()
