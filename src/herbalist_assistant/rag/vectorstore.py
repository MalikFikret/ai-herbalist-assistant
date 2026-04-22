from pathlib import Path

from langchain_chroma import Chroma

from .loaders import load_pdf_documents
from .splitter import split_documents


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

    if persist_dir.exists() and any(persist_dir.iterdir()):
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

    docs = load_pdf_documents(data_dir)
    if not docs:
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )


def make_retriever(vectorstore: Chroma, *, k: int):
    return vectorstore.as_retriever(search_kwargs={"k": k})

