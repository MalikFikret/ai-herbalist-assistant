# 🌿 AI Herbalist Assistant — Sunum Bölümleri (Taslak)

> [!NOTE]
> Bu belge, RAG sisteminin sunumunu ekip üyeleri arasında 3 mantıksal bölüme ayırmak için hazırlanmıştır. Her bölüm, veri akışını takip edecek şekilde düzenlenmiştir ve ekip üyelerinin projede üstlendikleri rollere göre önerilmiştir.

---

## 🧑‍💻 Bölüm 1: Veri ve Geri Getirme Altyapısı (The Foundation)
**Önerilen Konuşmacı:** Melisa Yıldırım *(Veri Boru Hattı ve Testler)*

**Bu bölümün amacı:** PDF kitaplarının nasıl akıllı bir veritabanına dönüştüğünü ve sistemin bilgileri nasıl arayıp bulduğunu anlatmaktır.

**Alt Başlıklar ve Odak Noktaları:**
1. **İndeksleme Temelleri (Indexing):** 
   - 17 farklı kitabın (Türkçe, Arapça, İngilizce) `PyPDFLoader` ile yüklenmesi.
   - Metinlerin anlamını kaybetmemesi için parçalara ayrılması (Chunking: 800 karakter, 150 örtüşme/overlap).
   - `paraphrase-multilingual-mpnet-base-v2` kullanılarak çok dilli vektör (embedding) oluşturulması.
2. **Akıllı Veritabanı Yönetimi:** 
   - ChromaDB kullanımı.
   - `index_manifest.json` dosyası sayesinde tüm veritabanını sıfırdan kurmak yerine sadece değişen dosyaların (artımlı/incremental) indekslenmesi mantığı.
3. **Gelişmiş Arama Stratejisi (Retrieval):** 
   - *Sorgu Genişletme (Query Expansion):* Sistemin tek bir soruyu neden LLM kullanarak 7 farklı arama sorgusuna (4 ana, 3 yedek dil) dönüştürdüğünün ve bu sayede arama hassasiyetinin nasıl arttığının açıklanması.

---

## 🧠 Bölüm 2: Karar Mekanizması ve Kalite Kontrol (The Brain & CRAG)
**Önerilen Konuşmacı:** Malik Fikret *(Teknik Lider ve Yapay Zeka Entegrasyonu)*

**Bu bölümün amacı:** Agentic RAG mimarisinin "beynini", düğümlerin (nodes) birbirine nasıl bağlandığını ve sistemin bilgileri nasıl puanladığını anlatmaktır.

**Alt Başlıklar ve Odak Noktaları:**
1. **LangGraph Mimarisi:** 
   - Akış şemasının genel mantığı ve düğümlerin bir orkestra gibi nasıl çalıştığı.
2. **Akıllı Yönlendirme (Router):** 
   - Sistemin bir sorunun "tıbbi" mi yoksa "genel sohbet" mi olduğunu 3 katmanlı filtreyle (Hızlı Regex → Regex → LLM) nasıl hızlıca anladığı.
3. **Belge Puanlama (Document Grading - CRAG):** 
   - RAG sistemlerindeki en önemli kalite kapısı: Getirilen belgelerin paralel olarak (aynı anda) nasıl okunduğu, 0'dan 100'e kadar nasıl puanlandığı ve sadece eşiği (65) geçen en iyi 3 belgenin nasıl seçildiği.
4. **Web Araması Yedeklemesi (Web Search Fallback):** 
   - Kitaplarda yeterli bilgi bulunmadığında sistemin önce 5 güvenilir sitede, sonuç bulamazsa açık web'de nasıl arama yaptığı.
5. **LLM Konfigürasyonları:** 
   - Yönlendirme ve değerlendirme için neden 0.0 sıcaklık (netlik), sorgu üretimi için neden 0.4 sıcaklık (yaratıcılık) kullanıldığının teknik açıklaması.

---

## 🛡️ Bölüm 3: Yanıt Üretimi, Güvenlik ve Kullanıcı Deneyimi (Generation & Safety)
**Önerilen Konuşmacı:** Ebru Tuğçe Polat *(Kullanıcı Arayüzü ve Sistem Tasarimi)*

**Bu bölümün amacı:** Nihai yanıtın kullanıcıya ulaşmadan önce nasıl kişiselleştirildiğini ve en önemlisi tıbbi olarak nasıl güvende tutulduğunu anlatmaktır.

**Alt Başlıklar ve Odak Noktaları:**
1. **Yanıtın Şekillenmesi (Medical Answer):** 
   - Sadece belgelerin değil, aynı zamanda **Kullanıcı Sağlık Profilinin** (Alerjiler, kullanılan ilaçlar, yaş) ve sohbet geçmişinin yanıt üretimine nasıl doğrudan etki ettiği.
   - Doğrudan yanıtların (kimlik soruları, selamlama) nasıl basitçe yönetildiği.
2. **Güvenlik Kapısı 1: Halüsinasyon Kontrolü:** 
   - Yapay zekanın en büyük sorunu olan "uydurma" (halüsinasyon) probleminin nasıl çözüldüğü. Yanıtın gerçekten belgelere dayanıp dayanmadığının son kontrolü.
3. **Güvenlik Kapısı 2: İlgi Kontrolü (Relevance Check):** 
   - Yanıtın gerçekten sorulan soruyu cevaplayıp cevaplamadığının ve profildeki kısıtlamalara uyup uymadığının kontrolü.
4. **Dinamik Özür Mesajı Mekanizması:** 
   - Sistem bir halüsinasyon veya ilgisiz yanıt yakaladığında, kullanıcıya yanlış bilgi vermek yerine, soruyu sorduğu dilde nazik bir yönlendirme mesajının (`_generate_polite_fallback`) nasıl otomatik üretildiği.

---

> Lütfen bu taslağı inceleyin. Eğer içerik dağılımını ve odak noktalarını beğenirseniz, bir sonraki aşamada her bir konuşmacı için (sunumda kullanabilecekleri daha teknik ve derinlemesine detaylar içeren) 3 ayrı ve detaylı belge oluşturabiliriz.
