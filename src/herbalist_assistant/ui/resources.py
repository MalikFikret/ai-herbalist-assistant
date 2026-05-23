"""Streamlit-scoped cached resources for embeddings, vectorstore, and reindex.

The live chat path is served by ``herbalist_assistant.graph.advanced_graph.app``,
which carries its own module-level ``lru_cache`` for the retriever and LLMs.
This module keeps the Streamlit-side caches (so the admin UI can show metrics
without re-running expensive setup) and exposes reindex helpers that invalidate
BOTH cache layers.
"""

from __future__ import annotations

import shutil
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from herbalist_assistant import config
from herbalist_assistant.rag.embeddings import create_embeddings
from herbalist_assistant.rag.vectorstore import (
    load_or_build_vectorstore,
    rebuild_manifest_after_full_index,
    remove_pdfs_from_index,
    sync_new_and_changed_pdfs,
)


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


def _invalidate_index_caches() -> None:
    get_vectorstore.clear()
    get_embeddings.clear()

    from herbalist_assistant.graph.advanced_graph import reset_runtime_caches

    reset_runtime_caches()


def reindex_pdfs() -> None:
    """Eagerly rebuild the local Chroma index from PDFs in ``data/``.

    Invalidates Streamlit and graph runtime caches, deletes the persisted DB,
    and rebuilds from all PDFs on disk.
    """
    _invalidate_index_caches()

    chroma_dir = config.CHROMA_DIR
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)

    load_environment()
    embeddings = create_embeddings(config.EMBEDDING_MODEL)
    load_or_build_vectorstore(
        data_dir=config.DATA_DIR,
        persist_dir=config.CHROMA_DIR,
        embeddings=embeddings,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    rebuild_manifest_after_full_index(
        data_dir=config.DATA_DIR,
        persist_dir=config.CHROMA_DIR,
        embeddings=embeddings,
    )
    _ = get_vectorstore()


def index_new_pdfs() -> dict[str, Any]:
    """Embed only PDFs that are new or changed since the last index run."""
    _invalidate_index_caches()

    load_environment()
    embeddings = create_embeddings(config.EMBEDDING_MODEL)
    stats = sync_new_and_changed_pdfs(
        data_dir=config.DATA_DIR,
        persist_dir=config.CHROMA_DIR,
        embeddings=embeddings,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    _ = get_vectorstore()
    return stats


def delete_pdfs_from_index(filenames: list[str]) -> None:
    """Remove embedded chunks for deleted PDFs and refresh caches."""
    if not filenames:
        return

    _invalidate_index_caches()

    load_environment()
    embeddings = create_embeddings(config.EMBEDDING_MODEL)
    remove_pdfs_from_index(
        filenames=filenames,
        persist_dir=config.CHROMA_DIR,
        embeddings=embeddings,
    )
    _ = get_vectorstore()
