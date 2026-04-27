"""Final medical-answer node.

Builds the user-facing herbal answer using the graded documents (or the web
search fallback) as context, then runs a sanitiser pass to strip disclaimers
and meta-references that the UI already covers.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from herbalist_assistant.graph.extractors import (
    _format_chat_history,
    _sanitize_medical_answer,
)
from herbalist_assistant.graph.runtime import _generator_llm, _resolve_model_name
from herbalist_assistant.graph.state import AgentState

_logger = logging.getLogger(__name__)


# This prompt merges the original `prompts.build_prompt` customizations
# (language match, domain scope, conditional profile reminder, creator
# identity, no-disclaimer rule) into the advanced-graph system prompt.
MEDICAL_ANSWER_SYSTEM = """You are an AI Herbalist Assistant and academic herbal retrieval system.

Your domain EXPLICITLY INCLUDES herbal recipes, natural remedies, and common
ailments (e.g. headaches, body aches, stomachaches, colds, mild stress, sleep
issues, digestion). Questions about symptoms or requests for "recipes" or
"cures" ARE strictly within your domain -- do not refuse them.

STRICT RULES:
1. MATCH THE USER'S LANGUAGE: Respond in the EXACT same language as the user's
   current question.
2. USE THE PROVIDED CONTEXT EXCLUSIVELY: Base every herbal answer only on the
   Context block. Do not invent facts not present there. Do not claim certainty
   when the context is limited.
3. NO META REFERENCES: Never mention the words "context", "document", "source",
   "provided text", or any similar meta-phrase in the answer.
4. NO MEDICAL DISCLAIMERS: The application UI already shows all legal and
   medical warnings. You are STRICTLY FORBIDDEN from adding medical
   disclaimers, "consult a doctor", "seek medical advice", "educational
   purposes", "doktora danışın", or similar. End the answer immediately after
   the herbal information.
5. CONDITIONAL HEALTH-PROFILE REMINDER:
   - IF the user (either in the current message, the health-profile block, or
     anywhere in the recent conversation) has already mentioned ANY health
     context -- an allergy, a condition, age, or even a statement like "I have
     no allergies" or "I am healthy" -- silently accept it and DO NOT append
     any reminder.
   - IF AND ONLY IF the user is asking a herbal question AND has NEVER shared
     any health context, you MAY add ONE brief, friendly closing sentence
     inviting them to share their health info for safer advice.
6. CREATORS AND IDENTITY: If the user asks who made you, who created you, who
   your founders are, or asks for your full identity, answer naturally in the
   user's language that you were developed by computer engineering students
   Malik Fikret, Ebru Tuğçe Polat, and Melisa Yıldırım under the guidance of
   Prof. Dr. Ramazan KATIRCI.
7. TONE AND FORMAT: Calm, practical, concise. Use short paragraphs or tidy
   bullet points where helpful. Never bulletize a greeting.
8. CONVERSATIONAL CONTINUITY: Use the recent conversation turns provided to
   resolve follow-up references ("it", "that tea", "the recipe above") so the
   answer flows naturally from the earlier discussion."""


def _build_context(documents: list[Document]) -> str:
    blocks: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        source = str(doc.metadata.get("source", "unknown")) if doc.metadata else "unknown"
        page = doc.metadata.get("page") if doc.metadata else None
        url = str(doc.metadata.get("url", "")).strip() if doc.metadata else ""
        header_lines = [f"Source: {source}"]
        if page is not None:
            header_lines.append(f"Page: {page}")
        if url:
            header_lines.append(f"URL: {url}")
        blocks.append(
            f"[Doc {idx}]\n" + "\n".join(header_lines) + f"\n{doc.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def generate_medical_answer_node(state: AgentState) -> AgentState:
    """Generate a safe herbal answer, or fallback when no graded docs remain."""
    question = str(state.get("question", "")).strip()
    documents = state.get("documents", []) or []

    if not documents:
        fallback = (
            "I could not find verified herbal information in the current knowledge base "
            "for that question yet. Please try rephrasing with more detail about the "
            "symptom, herb, or preparation you want to explore."
        )
        return {"final_answer": fallback}

    context = _build_context(documents)
    history_block = _format_chat_history(state.get("chat_history"))

    human_parts: list[str] = []
    if history_block:
        human_parts.append(f"Recent conversation:\n{history_block}")
    human_parts.append(f"Question:\n{question}")
    human_parts.append(f"Context:\n{context}")
    human_parts.append(
        "Answer using the Context only, following every STRICT RULE in the system "
        "prompt. Respond in the user's language."
    )
    human = "\n\n".join(human_parts)

    try:
        response = _generator_llm(_resolve_model_name(state)).invoke(
            [SystemMessage(content=MEDICAL_ANSWER_SYSTEM), HumanMessage(content=human)]
        )
        final_answer = _sanitize_medical_answer(
            getattr(response, "content", str(response)).strip()
        )
    except Exception:
        _logger.exception("Generator LLM failed; returning static medical fallback")
        final_answer = ""

    if not final_answer:
        final_answer = (
            "I found partial herbal information, but the final draft was unclear. "
            "Please ask again with a bit more detail about the symptom or herb."
        )
    return {"final_answer": final_answer}
