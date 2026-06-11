"""Application-wide configuration constants.

FIXES applied (v2):
  - CHUNK_SIZE:    550  → 800   (less context splitting = more complete passages)
  - CHUNK_OVERLAP: 100  → 150   (better continuity across chunk boundaries)
  - RETRIEVER_K:   3    → 5     (more candidate docs per query before grading)
"""

from pathlib import Path

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Embedding ─────────────────────────────────────────────────────────────────
# Fix #9: switched from English-only MiniLM to multilingual mpnet for better
# Turkish retrieval quality. Supports 50+ languages including Turkish.
# ⚠️ Changing this model requires deleting .chroma_db/ and rebuilding the vectorstore.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# ── Storage paths ─────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
CHROMA_DIR = Path(".chroma_db")
DB_PATH = Path(".herbalist.db")  # SQLite file at repo root

# ── RAG chunking ──────────────────────────────────────────────────────────────
# FIX: Larger chunks preserve more context per passage (was 550).
CHUNK_SIZE = 800

# FIX: More overlap reduces information loss at chunk boundaries (was 100).
CHUNK_OVERLAP = 150

# FIX: Retrieve more candidates so the grader has better material to work with
#      (was 3). With query expansion (3 queries × 5 = 15 raw, deduped to ~10)
#      the grader still caps at 6, but the input pool is richer.
RETRIEVER_K = 5

# ── Generation ────────────────────────────────────────────────────────────────
LLM_TEMPERATURE = 0.2

# ── Trusted domain search ─────────────────────────────────────────────────────
# These Turkish herbal sites are searched FIRST before the open web.
# If they return enough results, the open-web search is skipped entirely.
TRUSTED_HERB_DOMAINS = [
    "medikalakademi.com.tr",   # academic medical portal
    "tuketicisagligi.com.tr",  # Abdi İbrahim pharmaceutical
    "probitki.com",            # herbal specialist
    "sifasokagi.com",          # herbal remedies (may block bots)
    "gelenekseltedavi.com",    # traditional medicine (may block bots)
]

# Minimum number of trusted-domain results to skip the open-web fallback.
TRUSTED_DOMAIN_MIN_RESULTS = 1