from pathlib import Path

GROQ_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DATA_DIR = Path("data")
CHROMA_DIR = Path(".chroma_db")

# Persistence. SQLite file at repo root.
DB_PATH = Path(".herbalist.db")

CHUNK_SIZE = 550
CHUNK_OVERLAP = 100
RETRIEVER_K = 3

LLM_TEMPERATURE = 0.2

