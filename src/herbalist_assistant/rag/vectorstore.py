from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma

from herbalist_assistant.rag import index_manifest
from herbalist_assistant.rag.loaders import (
    PDF_FILENAME_METADATA,
    load_pdf_document,
    load_pdf_documents,
)
from herbalist_assistant.rag.splitter import split_documents

PDF_FILENAME_METADATA_KEY = PDF_FILENAME_METADATA


def _persist_dir_ready(persist_dir: Path) -> bool:
    return persist_dir.exists() and any(persist_dir.iterdir())


def open_vectorstore(*, persist_dir: Path, embeddings) -> Chroma:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


def load_or_build_vectorstore(
    *,
    data_dir: Path,
    persist_dir: Path,
    embeddings,
    chunk_size: int,
    chunk_overlap: int,
):
    """
    Load a persisted Chroma DB if it exists, else build from PDFs in data_dir.

    Returns a Chroma vectorstore (possibly empty if no PDFs exist).
    """
    persist_dir = Path(persist_dir)
    manifest_file = index_manifest.manifest_path(persist_dir)

    if _persist_dir_ready(persist_dir):
        return open_vectorstore(persist_dir=persist_dir, embeddings=embeddings)

    docs = load_pdf_documents(data_dir)
    if not docs:
        store = open_vectorstore(persist_dir=persist_dir, embeddings=embeddings)
        index_manifest.save_manifest(manifest_file, index_manifest.load_manifest(manifest_file))
        return store

    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    _write_manifest_for_data_dir(
        manifest_file,
        data_dir=data_dir,
        chunk_counts=_chunk_counts_by_filename(chunks),
    )
    return store


def delete_chunks_for_pdf(vectorstore: Chroma, filename: str) -> None:
    """Remove all embedded chunks belonging to ``filename``."""
    # FIX (Risk-19): use public API instead of vectorstore._collection (private).
    # Get IDs first, then delete — Chroma.delete() only accepts IDs, not where.
    result = vectorstore.get(where={PDF_FILENAME_METADATA_KEY: filename}, include=[])
    ids_to_delete = result.get("ids") or []
    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)

    # Legacy rows may only have LangChain's ``source`` path metadata.
    try:
        result = vectorstore.get(include=["metadatas"])
    except Exception:
        return
    ids_to_drop: list[str] = []
    all_ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    for doc_id, meta in zip(all_ids, metadatas, strict=False):
        if not meta:
            continue
        if meta.get(PDF_FILENAME_METADATA_KEY) == filename:
            ids_to_drop.append(doc_id)
            continue
        source = meta.get("source")
        if source and Path(str(source)).name == filename:
            ids_to_drop.append(doc_id)
    if ids_to_drop:
        vectorstore.delete(ids=ids_to_drop)


def _chunk_counts_by_filename(chunks) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        name = str(chunk.metadata.get(PDF_FILENAME_METADATA_KEY, "") or "")
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _write_manifest_for_data_dir(
    manifest_file: Path,
    *,
    data_dir: Path,
    chunk_counts: dict[str, int],
) -> None:
    manifest = index_manifest.load_manifest(manifest_file)
    files: dict[str, Any] = {}
    if data_dir.exists():
        for pdf_path in sorted(data_dir.glob("*.pdf")):
            count = chunk_counts.get(pdf_path.name, 0)
            files[pdf_path.name] = index_manifest.manifest_entry_for_file(
                pdf_path, chunk_count=count
            )
    manifest["files"] = files
    index_manifest.save_manifest(manifest_file, manifest)


def _remove_manifest_entries(manifest_file: Path, filenames: list[str]) -> None:
    manifest = index_manifest.load_manifest(manifest_file)
    files = manifest.setdefault("files", {})
    for name in filenames:
        files.pop(name, None)
    index_manifest.save_manifest(manifest_file, manifest)


def _index_single_pdf(
    vectorstore: Chroma,
    *,
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    delete_chunks_for_pdf(vectorstore, pdf_path.name)
    docs = load_pdf_document(pdf_path)
    if not docs:
        return 0
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return 0
    vectorstore.add_documents(chunks)
    return len(chunks)


def _discover_indexed_filenames(vectorstore: Chroma) -> set[str]:
    """Infer indexed PDF basenames from stored chunk metadata."""
    try:
        # FIX (Risk-19): use public API instead of vectorstore._collection (private).
        result = vectorstore.get(include=["metadatas"])
    except Exception:
        return set()
    metadatas = result.get("metadatas") or []
    names: set[str] = set()
    for meta in metadatas:
        if not meta:
            continue
        if PDF_FILENAME_METADATA_KEY in meta:
            names.add(str(meta[PDF_FILENAME_METADATA_KEY]))
            continue
        source = meta.get("source")
        if source:
            names.add(Path(str(source)).name)
    return names


def _bootstrap_manifest_if_needed(
    manifest_file: Path,
    *,
    data_dir: Path,
    vectorstore: Chroma,
) -> dict[str, Any]:
    manifest = index_manifest.load_manifest(manifest_file)
    if manifest.get("files"):
        return manifest

    indexed = _discover_indexed_filenames(vectorstore)
    files: dict[str, Any] = {}
    for name in sorted(indexed):
        pdf_path = data_dir / name
        if pdf_path.exists():
            files[name] = index_manifest.manifest_entry_for_file(pdf_path, chunk_count=0)
    manifest["files"] = files
    if files:
        index_manifest.save_manifest(manifest_file, manifest)
    return manifest


def sync_new_and_changed_pdfs(
    *,
    data_dir: Path,
    persist_dir: Path,
    embeddings,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    """Embed only PDFs that are new or modified since the last index run."""
    data_dir = Path(data_dir)
    persist_dir = Path(persist_dir)
    manifest_file = index_manifest.manifest_path(persist_dir)

    vectorstore = open_vectorstore(persist_dir=persist_dir, embeddings=embeddings)
    manifest = _bootstrap_manifest_if_needed(
        manifest_file, data_dir=data_dir, vectorstore=vectorstore
    )

    stats = {"indexed": [], "skipped": [], "removed_orphans": []}

    on_disk = {p.name: p for p in sorted(data_dir.glob("*.pdf"))} if data_dir.exists() else {}
    manifest_files = set(manifest.get("files", {}).keys())

    for orphan in sorted(manifest_files - set(on_disk.keys())):
        delete_chunks_for_pdf(vectorstore, orphan)
        stats["removed_orphans"].append(orphan)

    if stats["removed_orphans"]:
        _remove_manifest_entries(manifest_file, stats["removed_orphans"])
        manifest = index_manifest.load_manifest(manifest_file)

    for name, pdf_path in on_disk.items():
        if index_manifest.is_file_unchanged(manifest, pdf_path):
            stats["skipped"].append(name)
            continue

        chunk_count = _index_single_pdf(
            vectorstore,
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        manifest.setdefault("files", {})[name] = index_manifest.manifest_entry_for_file(
            pdf_path, chunk_count=chunk_count
        )
        stats["indexed"].append(name)

    index_manifest.save_manifest(manifest_file, manifest)
    return stats


def remove_pdfs_from_index(
    *,
    filenames: list[str],
    persist_dir: Path,
    embeddings,
) -> None:
    """Delete embedded chunks and manifest entries for the given PDF names."""
    if not filenames:
        return

    persist_dir = Path(persist_dir)
    if not _persist_dir_ready(persist_dir):
        _remove_manifest_entries(index_manifest.manifest_path(persist_dir), filenames)
        return

    vectorstore = open_vectorstore(persist_dir=persist_dir, embeddings=embeddings)
    for name in filenames:
        delete_chunks_for_pdf(vectorstore, name)
    _remove_manifest_entries(index_manifest.manifest_path(persist_dir), filenames)


def rebuild_manifest_after_full_index(
    *,
    data_dir: Path,
    persist_dir: Path,
    embeddings,
) -> None:
    """Rewrite the manifest to match PDFs currently on disk after a full rebuild."""
    persist_dir = Path(persist_dir)
    manifest_file = index_manifest.manifest_path(persist_dir)
    if not _persist_dir_ready(persist_dir):
        index_manifest.save_manifest(manifest_file, index_manifest.load_manifest(manifest_file))
        return

    vectorstore = open_vectorstore(persist_dir=persist_dir, embeddings=embeddings)
    counts: dict[str, int] = {}
    try:
        # FIX (Risk-19): use public API instead of vectorstore._collection (private).
        result = vectorstore.get(include=["metadatas"])
        for meta in result.get("metadatas") or []:
            if not meta:
                continue
            name = meta.get(PDF_FILENAME_METADATA_KEY) or Path(
                str(meta.get("source", ""))
            ).name
            if name:
                counts[str(name)] = counts.get(str(name), 0) + 1
    except Exception:
        counts = {}

    _write_manifest_for_data_dir(manifest_file, data_dir=data_dir, chunk_counts=counts)


def make_retriever(vectorstore: Chroma, *, k: int):
    return vectorstore.as_retriever(search_kwargs={"k": k})