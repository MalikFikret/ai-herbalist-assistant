# 🌿 RAG Sistemi Sunumu - Bölüm 3
**Sunan:** Ebru Tuğçe Polat (Kullanıcı Arayüzü ve Sistem Tasarımı)

> [!NOTE]
> Bu bölüm, sistemin yanıtı ürettiği son aşamalara ve kullanıcı güvenliğini sağlamak için kullanılan **Güvenlik Kapılarına (Safety Gates)** odaklanır.

---

## Eksen 3: Tam LangGraph Akışı (Devamı)

### Düğüm 2: Doğrudan Yanıt (Direct Answer)

**Dosya:** [direct_answer.py](file:///d:/aiherbal/src/herbalist_assistant/graph/nodes/direct_answer.py)

- Selamlamalar, kimlik soruları ve genel sohbet için.
- Yalnızca kullanıcı adını kullanır (tam sağlık profilini değil).
- Arayüz dilini (Türkçe/İngilizce) dikkate alır.

---

### Düğüm 6: Tıbbi Yanıt Üretme

**Dosya:** [medical_answer.py](file:///d:/aiherbal/src/herbalist_assistant/graph/nodes/medical_answer.py)

1. Kabul edilen belgeleri toplar (yerel RAG'den veya web aramasından).
2. Belgelerden **bilgi bağlamı** oluşturur (kaynak, sayfa, URL, içerik).
3. **Kullanıcı bağlamı** oluşturur (ad, yaş, cinsiyet, alerjiler, sağlık durumları + güvenlik gereksinimleri).
4. Konuşma sürekliliği için sohbet geçmişini ekler.
5. Sıcaklık `0.2` ile LLM'i çağırır.
6. **Yanıt temizleme** — "Doktorunuza danışın", "based on the context" gibi ifadeleri kaldırır.

---

### Düğüm 7: Halüsinasyon Kontrolü (Hallucination Check)

**Dosya:** [grading.py](file:///d:/aiherbal/src/herbalist_assistant/graph/nodes/grading.py)

- Kontrol: Yanıt **gerçekten** getirilen belgelere dayalı mı?
- **Maksimum yeniden deneme:** `1`

| Durum | İşlem |
|-------|-------|
| Destekli ✅ | → Yanıt ilgi kontrolüne (Düğüm 8) |
| Halüsinasyon ❌ + deneme < 1 | → Web araması ile yeniden deneme |
| Halüsinasyon ❌ + denemeler tükendi | → **Yanıt, `_generate_polite_fallback()` ile dinamik özür mesajıyla değiştirilir** |

> [!IMPORTANT]
> **Güvenlik güncellemesi:** Son güncellemeden sonra, doğrulanmamış yanıtlar artık kullanıcıya iletilmez. Bunun yerine, sorunun diliyle aynı dilde üretilmiş bir özür mesajıyla değiştirilir.

---

### Düğüm 8: Yanıt İlgi Kontrolü (Answer Relevance)

**Dosya:** [grading.py](file:///d:/aiherbal/src/herbalist_assistant/graph/nodes/grading.py)

- Kontrol: Yanıt **gerçekten** kullanıcının sorusunu karşılıyor mu?

**Red kriterleri (güncellenmiş):**
- Yanıt, sorunun özünü belirli bitkisel/tıbbi içerikle ele almıyor.
- Yanıt, ilgili bilgi içermeyen genel dolgu malzemesi.
- Yanıt, açık bir kullanıcı güvenlik kısıtlamasını (alerji, sağlık durumu) görmezden geliyor — yalnızca soruda kısıtlamalar belirtilmişse.
- Yanıt tamamen konu dışı veya taleple çelişiyor.

| Durum | İşlem |
|-------|-------|
| İlgili ✅ | → `SON` — Yanıt kullanıcıya ulaşır |
| İlgisiz ❌ | → **Yanıt, `_generate_polite_fallback()` ile dinamik özür mesajıyla değiştirilir** → `SON` |

> [!IMPORTANT]
> **Güvenlik güncellemesi:** Son güncellemeden sonra, uygun olmayan yanıtlar artık kullanıcıya iletilmez. Bunun yerine, sorunun konusunu belirten ve kullanıcıyı bir uzmana yönlendiren, sorunun diliyle aynı dilde üretilmiş bir özür mesajıyla değiştirilir.

---

### Dinamik Özür Mesajı Mekanizması

**Fonksiyon:** `_generate_polite_fallback(question, reason, model_name)`

**Dosya:** [grading.py](file:///d:/aiherbal/src/herbalist_assistant/graph/nodes/grading.py)

- Kullanıcının **sorusuyla aynı dilde** (arayüz dili değil) nazik bir mesaj üretmek için `_generator_llm` kullanır.
- Orijinal sorunun konusundan bahseder.
- Kullanıcıya veritabanının bu konuda yeterli bilgiye sahip olmadığını ve sürekli geliştirildiğini bildirir.
- LLM çağrısının kendisi başarısız olursa sabit bir İngilizce yedek mesajı vardır.
