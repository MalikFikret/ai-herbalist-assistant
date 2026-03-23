from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdf_documents(data_dir: Path):
    """Load all PDF documents from the given directory."""
    if not data_dir.exists():
        return []

    pdf_paths = sorted(data_dir.glob("*.pdf"))
    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())
    return documents

