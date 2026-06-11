"""Router node: classify the user message as medical (RAG) vs small-talk.

FIX (v2):
  - Expanded _MEDICAL_SIGNAL_RE with more Turkish, English and common
    medical/herb terms that were missing and causing misclassification.
  - Added _FOLLOWUP_SIGNAL_RE to catch "how do I prepare it?" style
    follow-up questions and route them to VECTOR_SEARCH via regex
    (before hitting the LLM), improving speed and reliability.
  - Expanded _SMALLTALK_SIGNAL_RE with more Turkish/English phrases.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from herbalist_assistant.graph.extractors import _extract_route, _format_chat_history
from herbalist_assistant.graph.runtime import _resolve_model_name, _router_llm
from herbalist_assistant.graph.state import AgentState

_logger = logging.getLogger(__name__)


# ── Regex signal patterns ─────────────────────────────────────────────────────

_MEDICAL_SIGNAL_RE = re.compile(
    r"\b("
    # ── English — herbs & preparations ───────────────────────────────────────
    r"herb|herbal|plant|remedy|tea|infusion|tincture|decoction|poultice|"
    r"dose|dosage|recipe|preparation|brew|steep|boil|extract|supplement|"
    r"botanical|medicinal|natural remedy|home remedy|"
    # ── English — symptoms & conditions ──────────────────────────────────────
    r"symptom|pain|ache|headache|migraine|stomach|digestion|nausea|"
    r"cold|flu|fever|cough|sore throat|inflammation|infection|"
    r"sleep|insomnia|stress|anxiety|depression|fatigue|tired|"
    r"allergy|rash|skin|blood pressure|diabetes|cholesterol|"
    r"pregnan|pregnancy|period|menstrual|hormone|immune|"
    r"weight|detox|liver|kidney|joint|arthritis|"
    # ── Turkish — herbs & preparations ───────────────────────────────────────
    r"bitki|bitkisel|sifa|şifa|cay|çay|demleme|demle|kaynat|"
    r"tarif|doz|miktar|hazirla|hazırla|nasil yapilir|nasıl yapılır|"
    r"ilaç|ilac|dogal|doğal|"
    # ── Turkish — symptoms & conditions ──────────────────────────────────────
    r"agri|ağrı|bas agri|baş ağrı|bas agrisi|baş ağrısı|"
    r"mide|mide bulantisi|mide bulantısı|hazim|hazım|"
    r"soguk alginligi|soğuk algınlığı|nezle|grip|ates|ateş|"
    r"oksuruk|öksürük|bogaz|boğaz|"
    r"uyku|uykusuzluk|stres|kaygi|kaygı|"
    r"alerji|hamile|hamilelik|"
    r"yara|sisme|şişme|enfeksiyon|bagisiklik|bağışıklık|"
    r"tansiyon|seker|şeker|kolesterol|eklem|"
    # ── Common herb names (EN + TR) ───────────────────────────────────────────
    r"chamomile|papatya|peppermint|nane|ginger|zencefil|"
    r"turmeric|zerdeçal|zerdecal|lavender|lavanta|"
    r"echinacea|ekinezya|valerian|kediotu|elderberry|"
    r"rosemary|biberiye|thyme|kekik|sage|adacayi|adaçayı|"
    r"green tea|yesil cay|yeşil çay|black seed|corek otu|çörek otu"
    r")\b",
    re.IGNORECASE,
)

# FIX: Follow-up questions that lack medical keywords but clearly reference
#      an earlier herbal discussion.  Route these to VECTOR_SEARCH directly
#      so short follow-ups like "how do I prepare it?" are not lost.
_FOLLOWUP_SIGNAL_RE = re.compile(
    r"\b("
    r"how (do i|to|can i|should i)|"
    r"prepare (it|this|that)|"
    r"what (is the |are the )?(dose|dosage|amount|quantity)|"
    r"is it safe|can i (use|take|drink|eat)|"
    r"side effect|benefit|"
    r"nasil (kullanirim|kullanırım|hazirlarim|hazırlarım|"
    r"yapilir|yapılır|icilir|içilir)|"
    r"ne kadar|kac (kez|defa|bardak)|kaç (kez|defa|bardak)|"
    r"zarari var mi|zararı var mı|faydasi|faydası|etkisi|yan etki"
    r")\b",
    re.IGNORECASE,
)

_IDENTITY_SIGNAL_RE = re.compile(
    r"\b("
    r"who made you|who created you|who built you|who are you|your founders|"
    r"what are you|tell me about yourself|"
    r"kim yapti|kim yaptı|kim olusturdu|kim oluşturdu|seni kim|"
    r"sen kimsin|yaraticin|yaratıcın|beni kim yapti|beni kim yaptı"
    r")\b",
    re.IGNORECASE,
)

_SMALLTALK_SIGNAL_RE = re.compile(
    r"\b("
    r"hi|hello|hey|thanks|thank you|good morning|good evening|good night|"
    r"good afternoon|how are you|nice to meet|bye|goodbye|see you|"
    r"merhaba|selam|günaydın|gunaydin|iyi geceler|iyi aksamlar|iyi akşamlar|"
    r"tesekkur|teşekkür|sag ol|sağ ol|nasılsın|nasilsin|"
    r"görüşürüz|gorururuz|hoşça kal|hosca kal"
    r")\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────


ROUTER_SYSTEM = """You are a query router for an herbal health assistant application.

Classify the user's message into exactly one route:
- VECTOR_SEARCH: The user asks about herbs, plants, natural remedies, recipes,
  cures, herbal preparations, symptoms, body pain, headache, stomachache, cold,
  sleep, stress, digestion, or any wellness topic that could be answered from a
  herbal knowledge corpus. This INCLUDES follow-up questions that refer to
  earlier herbal topics (e.g. "how do I prepare it?", "what is the dose?",
  "is it safe during pregnancy?", "what are its benefits?").
- DIRECT_ANSWER: Small talk, greetings, thanks, meta-conversation about the
  app, or identity questions such as "who made you?".

IMPORTANT PRIORITY:
- If the current user message is an explicit greeting, thanks, app meta-chat,
  or identity question ("who made you?", "who created you?"), ALWAYS choose
  DIRECT_ANSWER even if previous turns discussed herbal topics.
- If the message is short and vague (e.g. "how do I use it?", "is it safe?"),
  look at the recent conversation: if any herb or remedy was discussed, choose
  VECTOR_SEARCH.

When uncertain, prefer VECTOR_SEARCH as soon as any herb, remedy, symptom, or
wellness signal is present.

Return only JSON:
{"route": "VECTOR_SEARCH"} or {"route": "DIRECT_ANSWER"}"""


def _has_medical_signal(text: str) -> bool:
    return bool(_MEDICAL_SIGNAL_RE.search(text))


def _has_followup_signal(text: str) -> bool:
    """Short follow-up question (dose, safety, preparation) with no greet/identity."""
    if _IDENTITY_SIGNAL_RE.search(text) or _SMALLTALK_SIGNAL_RE.search(text):
        return False
    return bool(_FOLLOWUP_SIGNAL_RE.search(text))


def _should_force_direct(question: str) -> bool:
    """Return True only when the message is clearly small-talk or identity with
    zero medical signal.  Everything else goes through the LLM classifier."""
    text = (question or "").strip()
    if not text:
        return True
    # Any medical keyword → never force direct
    if _has_medical_signal(text):
        return False
    # Explicit greeting / thanks / identity with NO medical signal → direct
    return bool(_IDENTITY_SIGNAL_RE.search(text) or _SMALLTALK_SIGNAL_RE.search(text))


def _should_force_medical(question: str) -> bool:
    """Return True for clear follow-up questions so we skip the LLM call."""
    return _has_followup_signal(question)


def _fallback_route(question: str) -> Literal["VECTOR_SEARCH", "DIRECT_ANSWER"]:
    # FIX (Risk-9): default to VECTOR_SEARCH on LLM failure.
    # Only force DIRECT_ANSWER for clear greetings/identity questions.
    # Routing a non-medical question through RAG is harmless; routing a
    # medical question through DIRECT_ANSWER skips retrieval entirely.
    return "DIRECT_ANSWER" if _should_force_direct(question) else "VECTOR_SEARCH"


def route_question(state: AgentState) -> AgentState:
    """Route user question to either medical retrieval flow or direct chat flow."""
    question = str(state.get("question", "")).strip()
    if not question:
        return {"question": "", "is_medical": False, "generation_retries": 0}

    # ── Fast paths (no LLM needed) ────────────────────────────────────────────
    if _should_force_direct(question):
        _logger.info("Router: DIRECT (regex shortcut — greeting/identity)")
        return {"question": question, "is_medical": False, "generation_retries": 0}

    if _should_force_medical(question):
        _logger.info("Router: VECTOR_SEARCH (regex shortcut — follow-up question)")
        return {"question": question, "is_medical": True, "generation_retries": 0}
    # ─────────────────────────────────────────────────────────────────────────

    # ── LLM classification (with chat history context) ────────────────────────
    history_block = _format_chat_history(state.get("chat_history"))
    human_parts: list[str] = []
    if history_block:
        human_parts.append(f"Recent conversation:\n{history_block}")
    human_parts.append(f"Current user message:\n{question}")
    human_message = "\n\n".join(human_parts)

    try:
        response = _router_llm(_resolve_model_name(state)).invoke(
            [SystemMessage(content=ROUTER_SYSTEM), HumanMessage(content=human_message)]
        )
        route = _extract_route(getattr(response, "content", str(response)))
        _logger.info("Router: %s (LLM)", route)
    except Exception:
        route = _fallback_route(question)
        _logger.exception("Router LLM failed; lexical fallback route=%s", route)
    # ─────────────────────────────────────────────────────────────────────────

    return {
        "question": question,
        "is_medical": route == "VECTOR_SEARCH",
        "generation_retries": 0,
    }


def _route_from_router(state: AgentState) -> Literal["medical", "direct"]:
    return "medical" if bool(state.get("is_medical")) else "direct"