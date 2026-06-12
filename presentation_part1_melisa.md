# 🌿 RAG Sistemi Sunumu - Bölüm 1
**Sunan:** Melisa Yıldırım (Veri Boru Hattı ve Testler)

> [!NOTE]
> Bu bölüm, veritabanı oluşturulurken çalışan **İndeksleme Hattı** ve kullanıcı soru sorduğunda aramayı başlatan **Sorgulama Hattı** üzerine odaklanır.

---

## Eksen 1: İndeksleme Hattı (Indexing Pipeline)

> Bu bölüm veritabanı oluşturulurken **bir kez** veya yeni PDF dosyaları eklenip değiştirildiğinde çalışır.

### 1. PDF Dosyalarının Yüklenmesi

**Dosya:** [loaders.py](file:///d:/aiherbal/src/herbalist_assistant/rag/loaders.py)

- LangChain kütüphanesinden `PyPDFLoader` kullanarak her PDF dosyasını yükler.
- Her **sayfa** ayrı bir `Document` olur.
- Her belgeye PDF dosya adı metadata olarak (`pdf_filename`) eklenir.
- `load_pdf_documents()` fonksiyonu `data/` klasöründeki tüm PDF dosyalarını alfabetik sırayla tarar.

**Mevcut kitap sayısı:** 17 PDF kitap — Türkçe, Arapça ve İngilizce bitkisel tıp kitaplarının karışımı.

---

### 2. Belgelerin Parçalara Bölünmesi (Chunking)

**Dosya:** [splitter.py](file:///d:/aiherbal/src/herbalist_assistant/rag/splitter.py)

- `RecursiveCharacterTextSplitter` kullanır.
- **Parça boyutu:** `800` karakter
- **Parçalar arası örtüşme:** `150` karakter

**Ayarlar:** [config.py](file:///d:/aiherbal/src/herbalist_assistant/config.py)

> [!TIP]
> Örtüşme (overlap), iki parçanın sınırında kalan bilgilerin kaybolmamasını sağlar.

---

### 3. Gömme Modeli (Embedding Model)

**Dosya:** [embeddings.py](file:///d:/aiherbal/src/herbalist_assistant/rag/embeddings.py)

```text
Model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

- **Çok dilli** model — Türkçe, Arapça ve İngilizce destekler.
- GPU varsa GPU'da, yoksa CPU'da çalışır.
- Kosinüs benzerliğini iyileştirmek için vektörler **normalleştirilir** (`normalize_embeddings=True`).

---

### 4. Vektör Veritabanı (Vector Store)

**Dosya:** [vectorstore.py](file:///d:/aiherbal/src/herbalist_assistant/rag/vectorstore.py)

- Yerel vektör veritabanı olarak **ChromaDB** kullanır.
- Yol: proje kökündeki `.chroma_db/` klasörü.

**`load_or_build_vectorstore()` nasıl çalışır:**
1. Veritabanı **mevcutsa** → doğrudan açar.
2. **Mevcut değilse** → tüm PDF'leri yükler → parçalar → sıfırdan ChromaDB oluşturur.

**Artımlı senkronizasyon `sync_new_and_changed_pdfs()`:**
- **Yeni** veya **değiştirilmiş** dosyaları tespit edip indeksler.
- Diskten **silinmiş** dosyaların parçalarını (chunk) siler.
- Dosyanın değişip değişmediğini belirlemek için `mtime_ns` (son değişiklik zamanı) kullanır.

---

### 5. İndeks Manifesti (Index Manifest)

**Dosya:** [index_manifest.py](file:///d:/aiherbal/src/herbalist_assistant/rag/index_manifest.py)

`.chroma_db/index_manifest.json` dosyası indekslenen dosyaları takip eder:

```json
{
  "version": 1,
  "files": {
    "herbs.pdf": {"mtime_ns": 1234567890, "chunk_count": 42}
  }
}
```

- Her dosya için son değişiklik zamanı ve parça sayısını saklar.
- **Mevcut toplam:** ChromaDB'de ~30.774 parça (chunk) depolanmıştır.

---

## Eksen 2: Sorgulama Hattı (Query Pipeline)

> Bu bölüm kullanıcı bitkisel tıp hakkında her soru sorduğunda çalışır.

### 1. Sorgu Genişletme (Query Expansion)

**Dosya:** [retrieval.py](file:///d:/aiherbal/src/herbalist_assistant/graph/nodes/retrieval.py)

Sistem **tek bir sorguyla aramaz!** Şunları yapar:

1. LLM (sıcaklık 0.4) kullanarak **7'ye kadar farklı sorgu** üretir:
   - **4 birincil sorgu** (primary) — sorunun diliyle aynı
   - **3 yedek sorgu** (fallback) — diğer dillerde (ör. İngilizce bilimsel terimler)
2. Her sorguyu retriever'da çalıştırır (`k=5`).
3. SHA256 hash ile **tekrarları kaldırır**.
4. **Maksimum 12 aday belge** sınırı uygular.

**Teorik olarak:** 7 sorgu × 5 belge = 35 ham belge → tekrar kaldırma sonrası ≈ 10-12 benzersiz belge.

---

### 2. Geri Getirme Stratejisi

- **Tür:** Similarity Search (benzerlik araması) — ChromaDB varsayılanı.
- **Sorgu başına sonuç sayısı:** `k = 5`
- Retriever `@lru_cache(maxsize=1)` ile önbelleğe alınır.

### 3. Yeniden Sıralama (Reranking) Var mı?

**Özel bir reranker modeli yoktur.** Bunun yerine sistem, belgeler için akıllı bir paralel filtre olarak **LLM tabanlı CRAG değerlendirmesi** kullanır (Bölüm 2'de açıklanacaktır).
