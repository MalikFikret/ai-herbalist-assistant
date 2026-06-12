# 🌿 RAG System Presentation - Part 3
**Presenter:** Ebru Tuğçe Polat (UI and System Design)

> [!NOTE]
> This section focuses on the final stages where the system generates the answer and the **Safety Gates** used to ensure user safety.

---

## Axis 3: Full LangGraph Flow (Continued)

### Node 2: Direct Answer

**File:** [direct_answer.py](file:///d:/aiherbal/src/herbalist_assistant/graph/nodes/direct_answer.py)

- For greetings, identity questions, and general small talk.
- Uses only the user's name (not the full health profile).
- Respects the UI language (Turkish/English/Arabic).

---

### Node 6: Generate Medical Answer

**File:** [medical_answer.py](file:///d:/aiherbal/src/herbalist_assistant/graph/nodes/medical_answer.py)

1. Collects the accepted documents (from local RAG or web search).
2. Builds the **information context** from the documents (Source, Page, URL, Content).
3. Builds the **user context** (Name, Age, Gender, Allergies, Health Conditions + Safety Requirements).
4. Adds the chat history for conversational continuity.
5. Calls the LLM with a temperature of `0.2`.
6. **Answer Sanitization** — Removes phrases like "Consult your doctor", "based on the context", "doktora danışın".

---

### Node 7: Hallucination Check

**File:** [grading.py](file:///d:/aiherbal/src/herbalist_assistant/graph/nodes/grading.py)

- Checks: Is the answer **actually grounded** in the retrieved documents?
- **Max retries:** `1`

| Status | Action |
|--------|--------|
| Grounded ✅ | → Answer Relevance Check (Node 8) |
| Hallucination ❌ + retries < 1 | → Retry via Web Search |
| Hallucination ❌ + out of retries | → **Replace answer with a dynamic apology message** via `_generate_polite_fallback()` |

> [!IMPORTANT]
> **Safety Update:** Following the latest update, ungrounded answers are no longer passed to the user. Instead, they are replaced with an apology message generated in the same language as the question.

---

### Node 8: Answer Relevance Check

**File:** [grading.py](file:///d:/aiherbal/src/herbalist_assistant/graph/nodes/grading.py)

- Checks: Does the answer **actually address** what the user asked?

**Rejection Criteria (Updated):**
- The answer does not address the core of the question with specific herbal/medical content.
- The answer is general filler with no relevant information.
- The answer ignores an explicit user safety constraint (allergy, health condition) — only if constraints are mentioned in the question.
- The answer is completely off-topic or contradicts the request.

| Status | Action |
|--------|--------|
| Relevant ✅ | → `END` — Answer reaches the user |
| Irrelevant ❌ | → **Replace answer with a dynamic apology message** via `_generate_polite_fallback()` → `END` |

> [!IMPORTANT]
> **Safety Update:** Following the latest update, inappropriate answers are no longer passed to the user. Instead, they are replaced with an apology message (generated in the same language as the question) that mentions the topic and directs the user to consult a professional.

---

### Dynamic Apology Message Mechanism

**Function:** `_generate_polite_fallback(question, reason, model_name)`

**File:** [grading.py](file:///d:/aiherbal/src/herbalist_assistant/graph/nodes/grading.py)

- Uses the `_generator_llm` to generate a polite message **in the same language as the user's question** (not the UI language).
- Mentions the original topic of the question.
- Informs the user that the database currently does not contain sufficient information and is still under development.
- Has a hardcoded English fallback message in case the LLM call itself fails.
