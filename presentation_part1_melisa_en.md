# 🌿 RAG System Presentation - Part 1
**Presenter:** Melisa Yıldırım (Data Pipeline and Testing)

> [!NOTE]
> This section focuses on the **Indexing Pipeline**, which runs when the database is being built, and the **Query Pipeline**, which initiates the search when a user asks a question.

---

## Axis 1: Indexing Pipeline

> This pipeline runs **once** when building the database, or when new PDF files are added/modified.

### 1. Loading PDF Files

**File:** [loaders.py](file:///d:/aiherbal/src/herbalist_assistant/rag/loaders.py)

- Uses `PyPDFLoader` from the LangChain library to load each PDF file.
- Each **page** becomes a separate `Document`.
- The PDF filename is added as metadata (`pdf_filename`) to each document.
- The `load_pdf_documents()` function iterates through all PDFs in the `data/` folder in alphabetical order.

**Current Number of Books:** 17 PDF books — a mix of Turkish, Arabic, and English herbal medicine books.

---

### 2. Document Chunking

**File:** [splitter.py](file:///d:/aiherbal/src/herbalist_assistant/rag/splitter.py)

- Uses `RecursiveCharacterTextSplitter`.
- **Chunk Size:** `800` characters
- **Chunk Overlap:** `150` characters

**Settings from:** [config.py](file:///d:/aiherbal/src/herbalist_assistant/config.py)

> [!TIP]
> The overlap ensures that information located at the boundary of two chunks is not lost.

---

### 3. Embedding Model

**File:** [embeddings.py](file:///d:/aiherbal/src/herbalist_assistant/rag/embeddings.py)

```text
Model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

- **Multilingual** model — supports Turkish, Arabic, and English.
- Runs on GPU if available, otherwise on CPU.
- Vectors are **normalized** (`normalize_embeddings=True`) to improve cosine similarity search accuracy.

---

### 4. Vector Store

**File:** [vectorstore.py](file:///d:/aiherbal/src/herbalist_assistant/rag/vectorstore.py)

- Uses **ChromaDB** as the local vector database.
- Path: `.chroma_db/` folder in the project root.

**How `load_or_build_vectorstore()` works:**
1. If the database **exists** → opens it directly.
2. If **not exists** → loads all PDFs → chunks them → builds ChromaDB from scratch.

**Incremental Sync `sync_new_and_changed_pdfs()`:**
- Detects and indexes **new** or **modified** files.
- Deletes chunks for files that have been **removed** from the disk.
- Uses `mtime_ns` (last modification time) to determine if a file has changed.

---

### 5. Index Manifest

**File:** [index_manifest.py](file:///d:/aiherbal/src/herbalist_assistant/rag/index_manifest.py)

A JSON file at `.chroma_db/index_manifest.json` tracking indexed files:

```json
{
  "version": 1,
  "files": {
    "herbs.pdf": {"mtime_ns": 1234567890, "chunk_count": 42}
  }
}
```

- Stores the last modification time and chunk count for each file.
- **Current Total:** ~30,774 chunks stored in ChromaDB.

---

## Axis 2: Query Pipeline

> This pipeline runs **every time** the user asks a medical question about herbs.

### 1. Query Expansion

**File:** [retrieval.py](file:///d:/aiherbal/src/herbalist_assistant/graph/nodes/retrieval.py)

The system **does not search with a single query!** Instead, it does the following:

1. Uses an LLM (temperature 0.4) to generate **up to 7 different queries**:
   - **4 Primary Queries** — in the same language as the user's question
   - **3 Fallback Queries** — in other languages (e.g., English scientific terms)
2. Runs **each query** through the retriever with `k=5`.
3. **Removes duplicates** using SHA256 hashing.
4. **Limits to a maximum of 12 candidate documents**.

**Theoretically:** 7 queries × 5 documents = 35 raw documents → after deduplication ≈ 10-12 unique documents.

---

### 2. Retrieval Strategy

- **Type:** Similarity Search — the default in ChromaDB.
- **Number of results per query:** `k = 5`
- The Retriever is cached via `@lru_cache(maxsize=1)`.

### 3. Is there Reranking?

**There is no dedicated reranker model.** Instead, the system uses **LLM-based CRAG grading** as an intelligent parallel filter for the documents (Explained in Part 2).
