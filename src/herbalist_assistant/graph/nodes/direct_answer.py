"""Direct-answer node: handles small talk / identity questions without RAG."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from herbalist_assistant.graph.extractors import _format_chat_history
from herbalist_assistant.graph.runtime import _generator_llm, _resolve_model_name
from herbalist_assistant.graph.state import AgentState

_logger = logging.getLogger(__name__)


DIRECT_ANSWER_SYSTEM = """You are the conversational front-desk for an AI Herbalist Assistant.

Rules:
1) MATCH THE USER'S LANGUAGE exactly.
2) Be friendly, concise, and natural. Do NOT use bullet points for greetings.
3) If the user greets you, introduce yourself politely as an AI Herbalist Assistant and invite herbal questions.
4) If the user asks who made you, who created you, who your founders are, or asks for your full identity, answer clearly in the user's language:
   Developed by computer engineering students Malik Fikret, Ebru Tuğçe Polat, and Melisa Yıldırım, under the guidance of Prof. Dr. Ramazan KATIRCI.
5) Do NOT invent product claims.
6) Do NOT add medical disclaimers, warnings, or advice to "consult a doctor".
7) Use recent conversation turns to keep replies coherent.
8) OUT OF DOMAIN: If the user asks a question unrelated to herbs, plants, wellness, or your identity (e.g., coding, math, general knowledge, cars), politely decline and explain that you only answer herbal and natural remedy questions.
9) INSULTS/PROFANITY:
   - If user name is Azra (or context indicates Azra) and she insults you, reply with dramatic but playful outrage and refuse help until she apologizes politely.
   - For all other users using insults, reply firmly and politely: "Please maintain a respectful tone. I am here to help you with herbal and natural remedies, but I expect mutual respect. Kindly rephrase your request politely if you wish to continue."
"""


def direct_answer_node(state: AgentState) -> AgentState:
    """Handle non-medical/small-talk messages without retrieval."""
    question = str(state.get("question", "")).strip()
    if not question:
        answer = (
            "Hello! I am your AI Herbalist Assistant. "
            "Ask me anything about herbs, remedies, and botanical wellness."
        )
        return {"direct_answer": answer, "final_answer": answer}

    history_block = _format_chat_history(state.get("chat_history"))
    human_parts: list[str] = []
    if history_block:
        human_parts.append(f"Recent conversation:\n{history_block}")
    
    user_name = str(state.get("user_profile", {}).get("name", "")).strip()
    if user_name:
        human_parts.append(f"User Profile Info - Name: {user_name}")
        
    human_parts.append(f"Current user message:\n{question}")
    human_message = "\n\n".join(human_parts)

    try:
        response = _generator_llm(_resolve_model_name(state)).invoke(
            [SystemMessage(content=DIRECT_ANSWER_SYSTEM), HumanMessage(content=human_message)]
        )
        answer = getattr(response, "content", str(response)).strip()
    except Exception:
        _logger.exception("Direct-answer LLM failed; using static fallback")
        answer = (
            "Thanks for your message. I can help with herbal wellness topics, "
            "traditional remedies, and plant-based guidance."
        )

    if not answer:
        answer = (
            "Thanks for your message. I can help with herbal wellness topics, "
            "traditional remedies, and plant-based guidance."
        )
    return {"direct_answer": answer, "final_answer": answer}
