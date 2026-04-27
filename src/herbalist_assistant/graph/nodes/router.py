"""Router node: classify the user message as medical (RAG) vs small-talk."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from herbalist_assistant.graph.extractors import _extract_route, _format_chat_history
from herbalist_assistant.graph.runtime import _resolve_model_name, _router_llm
from herbalist_assistant.graph.state import AgentState

_logger = logging.getLogger(__name__)


ROUTER_SYSTEM = """You are a query router for an herbal health assistant application.

Classify the user's message into exactly one route:
- VECTOR_SEARCH: The user asks about herbs, plants, natural remedies, recipes,
  cures, herbal preparations, symptoms, body pain, headache, stomachache, cold,
  sleep, stress, digestion, or any wellness topic that could be answered from a
  herbal knowledge corpus. This INCLUDES follow-up questions that refer to
  earlier herbal topics (e.g. "how do I prepare it?", "what is the dose?",
  "is it safe during pregnancy?").
- DIRECT_ANSWER: Small talk, greetings, thanks, meta-conversation about the
  app, or identity questions such as "who made you?".

When uncertain, prefer VECTOR_SEARCH as soon as any herb, remedy, symptom, or
wellness signal is present. Use the RECENT CONVERSATION below to resolve vague
follow-ups ("it", "this", "that tea") back to the earlier herbal subject.

Return only JSON:
{"route": "VECTOR_SEARCH"} or {"route": "DIRECT_ANSWER"}"""


def route_question(state: AgentState) -> AgentState:
    """Route user question to either medical retrieval flow or direct chat flow."""
    question = str(state.get("question", "")).strip()
    if not question:
        return {"question": "", "is_medical": False}

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
    except Exception:
        _logger.exception("Router LLM failed; defaulting to VECTOR_SEARCH")
        route = "VECTOR_SEARCH"

    return {"question": question, "is_medical": route == "VECTOR_SEARCH"}


def _route_from_router(state: AgentState) -> Literal["medical", "direct"]:
    return "medical" if bool(state.get("is_medical")) else "direct"
