from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


PDF_FILENAME_METADATA = "pdf_filename"


def _tag_pdf_filename(documents, *, filename: str) -> None:
    for doc in documents:
        doc.metadata[PDF_FILENAME_METADATA] = filename


def load_pdf_document(pdf_path: Path):
    """Load a single PDF and tag each page with ``pdf_filename`` metadata."""
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    _tag_pdf_filename(documents, filename=pdf_path.name)
    return documents


def load_pdf_documents(data_dir: Path):
    """Load all PDF documents from the given directory."""
    if not data_dir.exists():
        return []

    documents = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        documents.extend(load_pdf_document(pdf_path))
    return documents

