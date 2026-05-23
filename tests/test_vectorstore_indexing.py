"""Tests for incremental PDF indexing and manifest tracking."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from herbalist_assistant.rag import index_manifest
from herbalist_assistant.rag.loaders import PDF_FILENAME_METADATA, load_pdf_document
from herbalist_assistant.rag.vectorstore import (
    delete_chunks_for_pdf,
    sync_new_and_changed_pdfs,
)


def test_manifest_tracks_mtime(tmp_path: Path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 minimal")
    manifest_file = tmp_path / "index_manifest.json"

    data = index_manifest.load_manifest(manifest_file)
    data["files"]["a.pdf"] = index_manifest.manifest_entry_for_file(pdf, chunk_count=3)
    index_manifest.save_manifest(manifest_file, data)

    reloaded = index_manifest.load_manifest(manifest_file)
    assert index_manifest.is_file_unchanged(reloaded, pdf)

    reloaded["files"]["a.pdf"]["mtime_ns"] = 0
    assert not index_manifest.is_file_unchanged(reloaded, pdf)


def test_load_pdf_document_tags_filename(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "herbs.pdf"
    pdf.write_bytes(b"x")

    class FakeDoc:
        metadata = {}

    monkeypatch.setattr(
        "herbalist_assistant.rag.loaders.PyPDFLoader",
        lambda _path: MagicMock(load=lambda: [FakeDoc()]),
    )

    docs = load_pdf_document(pdf)
    assert docs[0].metadata[PDF_FILENAME_METADATA] == "herbs.pdf"


def test_delete_chunks_for_pdf_uses_metadata_key():
    store = MagicMock()
    delete_chunks_for_pdf(store, "herbs.pdf")
    store._collection.delete.assert_called_once_with(where={"pdf_filename": "herbs.pdf"})


def test_sync_skips_unchanged_and_indexes_new(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    persist_dir = tmp_path / "chroma"
    data_dir.mkdir()
    persist_dir.mkdir()

    new_pdf = data_dir / "new.pdf"
    new_pdf.write_bytes(b"n")
    old_pdf = data_dir / "old.pdf"
    old_pdf.write_bytes(b"o")

    manifest_file = index_manifest.manifest_path(persist_dir)
    manifest = index_manifest.load_manifest(manifest_file)
    manifest["files"]["old.pdf"] = index_manifest.manifest_entry_for_file(old_pdf, chunk_count=1)
    index_manifest.save_manifest(manifest_file, manifest)

    store = MagicMock()
    indexed: list[str] = []

    def fake_index(vectorstore, *, pdf_path, chunk_size, chunk_overlap):
        indexed.append(pdf_path.name)
        return 2

    monkeypatch.setattr(
        "herbalist_assistant.rag.vectorstore.open_vectorstore",
        lambda **_: store,
    )
    monkeypatch.setattr(
        "herbalist_assistant.rag.vectorstore._index_single_pdf",
        fake_index,
    )

    stats = sync_new_and_changed_pdfs(
        data_dir=data_dir,
        persist_dir=persist_dir,
        embeddings=MagicMock(),
        chunk_size=100,
        chunk_overlap=10,
    )

    assert stats["indexed"] == ["new.pdf"]
    assert stats["skipped"] == ["old.pdf"]
    assert indexed == ["new.pdf"]

    updated = index_manifest.load_manifest(manifest_file)
    assert "new.pdf" in updated["files"]
    assert index_manifest.is_file_unchanged(updated, old_pdf)
