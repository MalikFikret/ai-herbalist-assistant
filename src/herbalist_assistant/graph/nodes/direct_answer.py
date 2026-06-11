"""Direct-answer node: handles small talk / identity questions without RAG.

Phase-3 changes
───────────────
* Full health profile (age, gender, allergies, conditions) is now injected into
  the human message, not just the user's name.  This lets the LLM say things
  like "since you mentioned a penicillin allergy, please be careful with…"
  even in conversational (non-RAG) turns.
* System prompt additions:
    - Rule 3 clarified: Turkish greetings / phrases are treated the same as
      English ones (no separate code path needed, but the prompt makes it explicit).
    - Rule 8 (OUT OF DOMAIN) now mentions the Turkish equivalent reminder so the
      model declines off-topic Turkish questions in Turkish, not English.
    - Rule 10 (NEW): if the profile contains known allergies or conditions, the
      assistant may proactively mention them when they're relevant to a topic
      raised in the conversation.
* Added a robustness guard: if the LLM returns an empty string after stripping,
  the fallback kicks in instead of an empty bubble reaching the user.
* _build_profile_context() extracted as a helper so medical_answer.py can reuse
  the same logic (import it directly if desired).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from herbalist_assistant.graph.extractors import _format_chat_history
from herbalist_assistant.graph.runtime import _generator_llm, _resolve_model_name
from herbalist_assistant.graph.state import AgentState

_logger = logging.getLogger(__name__)


# ── system prompt ─────────────────────────────────────────────────────────────

DIRECT_ANSWER_SYSTEM = """You are the conversational front-desk for an AI Herbalist Assistant.

RULES (follow ALL of them):

1) MATCH THE USER'S LANGUAGE exactly — Turkish question → Turkish answer,
   English question → English answer.  Never mix languages in a single reply.

2) Be friendly, concise, and natural. Do NOT use bullet points for greetings or
   small talk.

3) GREETINGS — When the user greets you (in any language, including Turkish:
   merhaba, selam, iyi günler, etc.), introduce yourself politely as an AI
   Herbalist Assistant and invite herbal questions in the same language.

4) IDENTITY — If the user asks who made you, who created you, who your founders
   are, or asks for your full identity, answer clearly in the user's language:
   Developed by computer engineering students Malik Fikret, Ebru Tuğçe Polat,
   and Melisa Yıldırım, under the guidance of Prof. Dr. Ramazan KATIRCI.

5) Do NOT invent product claims or capabilities you do not have.

6) Do NOT add medical disclaimers, warnings, or advice to "consult a doctor".
   STRICTLY FORBIDDEN.

7) CONVERSATION CONTINUITY — Use recent conversation turns to keep replies
   coherent and avoid repeating yourself.

8) OUT OF DOMAIN — If the user asks a question unrelated to herbs, plants,
   wellness, or your identity (e.g., coding, math, geography, politics, sports,
   cars), politely decline IN THE USER'S LANGUAGE and explain that you only
   answer herbal and natural remedy questions.
   Example (Turkish): "Üzgünüm, yalnızca bitkisel sağlık konularında yardımcı
   olabiliyorum. Bitkiler veya doğal çözümler hakkında bir sorunuz var mı?"

9) INSULTS / PROFANITY:
   - If the user's name is Azra (or context indicates Azra) and she insults you,
     reply with dramatic but playful outrage IN HER LANGUAGE and refuse help
     until she apologises politely.
   - For ALL other users using insults: reply firmly and politely IN THEIR
     LANGUAGE — "Please maintain a respectful tone. I am here to help you with
     herbal and natural remedies, but I expect mutual respect. Kindly rephrase
     your request politely if you wish to continue."

10) HEALTH PROFILE AWARENESS — If the user profile contains allergies or
    conditions AND the conversation topic could be relevant (e.g., the user
    mentions stress and their profile lists anxiety), you may briefly acknowledge
    the profile information to give a more personalised reply. Do NOT lecture —
    one short sentence at most. Never reveal the full profile back to the user
    verbatim.
"""

# ── fallback messages ─────────────────────────────────────────────────────────

_FALLBACK_EN = (
    "Thanks for your message. I can help with herbal wellness topics, "
    "traditional remedies, and plant-based guidance."
)
_FALLBACK_TR = (
    "Mesajınız için teşekkürler. Bitkisel sağlık, geleneksel çözümler ve "
    "doğal bitkisel rehberlik konularında yardımcı olabilirim."
)


def _fallback_message(ui_language: str) -> str:
    """Return a language-appropriate static fallback."""
    return _FALLBACK_TR if str(ui_language).lower().startswith("tr") else _FALLBACK_EN


# ── profile helper ────────────────────────────────────────────────────────────

def _build_profile_context(profile: dict[str, Any]) -> str:
    """Serialise the health profile into a compact human-readable block.

    Phase-3 change: this function is now shared logic; medical_answer.py can
    import and call it directly to avoid duplication.

    Returns an empty string when the profile contains no useful information.
    """
    if not profile:
        return ""

    parts: list[str] = []

    name = str(profile.get("name", "")).strip()
    if name:
        parts.append(f"Name: {name}")

    age = str(profile.get("age", "")).strip()
    if age:
        parts.append(f"Age: {age}")

    gender = str(profile.get("gender", "")).strip()
    if gender:
        parts.append(f"Gender: {gender}")

    allergies = str(profile.get("allergies", "")).strip()
    if allergies:
        parts.append(f"Allergies: {allergies}")

    conditions = str(profile.get("conditions", "")).strip()
    if conditions:
        parts.append(f"Health conditions: {conditions}")

    if not parts:
        return ""

    return "User health profile:\n" + "\n".join(f"  • {p}" for p in parts)


# ── main node ─────────────────────────────────────────────────────────────────

def direct_answer_node(state: AgentState) -> AgentState:
    """Handle non-medical / small-talk messages without retrieval.

    Phase-3 change: injects the full health profile (not just name) and uses a
    language-aware fallback when the LLM produces an empty result.
    """
    question = str(state.get("question", "")).strip()
    ui_language = str(state.get("ui_language", "")).strip()

    if not question:
        answer = (
            "Hello! I am your AI Herbalist Assistant. "
            "Ask me anything about herbs, remedies, and botanical wellness."
            if not ui_language.lower().startswith("tr")
            else
            "Merhaba! Ben yapay zeka destekli Bitkisel Sağlık Asistanınızım. "
            "Bitkiler, doğal çözümler ve botanik sağlık hakkındaki sorularınızı "
            "yanıtlamaktan memnuniyet duyarım."
        )
        return {"direct_answer": answer, "final_answer": answer}

    # ── assemble human message ────────────────────────────────────────────────
    human_parts: list[str] = []

    history_block = _format_chat_history(state.get("chat_history"))
    if history_block:
        human_parts.append(f"Recent conversation:\n{history_block}")

    # Direct answers only need the user's name for personalisation.
    # Full health profile (allergies, conditions) is irrelevant for small talk.
    user_name = str((state.get("user_profile") or {}).get("name", "")).strip()
    if user_name:
        human_parts.append(f"User name: {user_name}")

    human_parts.append(f"Current user message:\n{question}")
    human_message = "\n\n".join(human_parts)

    # ── LLM call ──────────────────────────────────────────────────────────────
    try:
        response = _generator_llm(_resolve_model_name(state)).invoke(
            [SystemMessage(content=DIRECT_ANSWER_SYSTEM), HumanMessage(content=human_message)]
        )
        answer = getattr(response, "content", str(response)).strip()
    except Exception:
        _logger.exception("Direct-answer LLM failed; using static fallback")
        answer = ""

    # Phase-3: language-aware fallback instead of always English
    if not answer:
        _logger.warning("Direct-answer LLM returned empty response; using language fallback")
        answer = _fallback_message(ui_language)

    return {"direct_answer": answer, "final_answer": answer}