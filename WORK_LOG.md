# WORK_LOG — AI Herbalist Assistant
> هذا الملف هو الجسر بين المحادثات. في كل محادثة جديدة قل:
> **"اقرأ `pipeline_report.md` و `WORK_LOG.md` ثم أخبرني بالخطوة التالية"**

---

## ✅ مكتمل

| # | المشكلة | الملف المعدَّل | التفاصيل |
|---|---------|--------------|---------|
| Risk-1 | Default admin password `"1234"` | `auth.py` | حُذف `_ADMIN_DEFAULT_PASSWORD` تماماً، أي غياب لكلمة المرور يرفع `RuntimeError` واضح |
| Risk-2 | Cookie secret يرجع لـ GROQ_API_KEY | `cookies.py` | استُبدل بـ `_EPHEMERAL_SECRET` عشوائي عند كل startup — الـ import من `auth` حُذف كلياً |
| Risk-3 | نتائج Web Search بدون grading | `web_search.py` | أُضيفت `_grade_web_docs()` تُشغَّل قبل كل return، threshold=40، fallback للأصل إذا كل شيء فشل |
| Risk-7 | Embeddings على CPU فقط | `embeddings.py` | أُضيفت `_best_device()` تكتشف CUDA تلقائياً وتستخدم GPU إذا توفّر |
| Risk-9 | Router fallback يختار DIRECT_ANSWER | `router.py` | `_fallback_route` تختار الآن VECTOR_SEARCH افتراضياً، DIRECT_ANSWER فقط للـ greetings الواضحة |
| Risk-18 | `_ADMIN_DEFAULT_WARNING_EMITTED` يتيم | `auth.py` | حُلَّ تلقائياً مع Risk-1 — المتغير والـ `global` statement حُذفا |
| Risk-19 | Chroma `_collection` private API | `vectorstore.py` | 3 استخدامات استُبدلت بـ `vectorstore.get()` و `vectorstore.delete()` الرسميتين |

---

## 🔄 قيد العمل (الخطوة الحالية)
_لا يوجد — كل المقرر منجز._

---

## ⏭️ متجاهل (قرار واعٍ)

| # | المشكلة | السبب |
|---|---------|-------|
| Risk-4 | TAVILY crash عند غياب المفتاح | جميع المفاتيح موجودة في `.env` |
| Risk-5 | SQLite bottleneck | مستخدم واحد حالياً، لا داعي |
| Risk-6 | تغيير Embedding يتطلب حذف `.chroma_db` | المستخدم يتعامل معه يدوياً |
| Risk-8 | لا rate limiting على Login | لا داعي حالياً |
| Risk-10 | Trusted domains تحجب bots | المواقع مراجعة يدوياً في محادثة سابقة |
| Risk-11 | DuckDuckGo rate limiting صامت | لا داعي حالياً |
| Risk-12 | لا حد أدنى لكلمة المرور | لا حاجة لهذا القيد |
| Risk-13 | HA_ADMIN_PASSWORD plaintext | لا مشكلة في البيئة الحالية |
| Risk-14 | Hallucination مستنفد → إجابة تمر | لا رسائل تزعج المستخدم |
| Risk-15 | Agent timeout 120s | تم تغييره يدوياً إلى 300s |
| Risk-16 | LangSmith probe في كل cold start | لا داعي للتعديل |
| Risk-17 | Sanitizer يحذف "consult a doctor" | مقصود — التطبيق لا يُظهر تحذيرات طبية |
| Risk-20 | Timestamps بدون timezone | server وحيد، لا أثر عملي |

---

## 🗂️ الملفات المعدّلة

| الملف | المسار الكامل |
|-------|--------------|
| `auth.py` | `src/herbalist_assistant/ui/auth.py` |
| `cookies.py` | `src/herbalist_assistant/ui/cookies.py` |
| `web_search.py` | `src/herbalist_assistant/graph/nodes/web_search.py` |
| `embeddings.py` | `src/herbalist_assistant/rag/embeddings.py` |
| `router.py` | `src/herbalist_assistant/graph/nodes/router.py` |
| `vectorstore.py` | `src/herbalist_assistant/rag/vectorstore.py` |

---

## 📋 ملاحظات مهمة

- بعد استبدال `embeddings.py` → احذف `.chroma_db/` وأعد بناء الـ index (GPU vs CPU يختلفان عددياً)
- `generate_admin_password_hash.py` موجود في `scripts/` لتوليد كلمة مرور admin آمنة
- `HA_REMEMBER_SECRET` في `.env` يجعل جلسات "تذكرني" دائمة عبر restarts — بدونه تنتهي عند كل restart