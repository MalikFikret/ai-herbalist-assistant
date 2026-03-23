from pathlib import Path

GROQ_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DATA_DIR = Path("data")
CHROMA_DIR = Path(".chroma_db")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 4

LLM_TEMPERATURE = 0.2

