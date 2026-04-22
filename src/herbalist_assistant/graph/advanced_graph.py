"""Phase 4: full LangGraph integration for the agentic RAG workflow.

This graph stitches together:
1) Router node (medical vs direct conversation),
2) Query expansion + retriever node (context-aware, uses chat history),
3) CRAG document grading node,
4) Final response generation node(s).

It also carries `chat_history`, `model_name`, and `web_search_provider` in the
state, so follow-up questions preserve conversational context while the sidebar
can switch models/search providers at invoke time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from functools import lru_cache
from typing import Literal, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

# Importing the package eagerly loads .env (see herbalist_assistant/__init__.py),
# which must happen before any langchain/langsmith import for tracing to work.
import herbalist_assistant  # noqa: F401
from herbalist_assistant import config
from herbalist_assistant.llm.groq import create_groq_llm, get_groq_api_key

_logger = logging.getLogger("herbalist_assistant.graph")

RouteLiteral = Literal["VECTOR_SEARCH", "DIRECT_ANSWER"]
ScoreLiteral = Literal["yes", "no"]

# How many past messages to include when we format chat history for the LLMs.
# 6 = roughly the last 3 user/assistant turns; enough for follow-ups without
# blowing up the prompt.
_MAX_HISTORY_MESSAGES = 6
_GROQ_MODELS = {"llama-3.1-8b-instant", "mixtral-8x7b-32768"}
_DEFAULT_WEB_SEARCH_PROVIDER = "DuckDuckGo"
_WEB_SEARCH_RESULT_LIMIT = 4


class AgentState(TypedDict, total=False):
    """State shared across all nodes in the advanced LangGraph workflow."""

    question: str
    expanded_queries: list[str]
    documents: list[Document]
    is_medical: bool
    direct_answer: str
    final_answer: str
    # Recent chat turns as [{"role": "user"|"assistant", "content": str}, ...].
    # The CURRENT user question is NOT in this list; it lives in `question`.
    chat_history: list[dict]
    # Runtime model id chosen by the sidebar settings.
    model_name: str
    # Runtime web-search backend chosen by the sidebar settings.
    web_search_provider: str


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

EXPANSION_SYSTEM = """You generate retrieval rewrites for a herbal health assistant.

If the user's current question is a follow-up that contains pronouns or vague
references (it / this / that / the tea / the herb), USE the recent conversation
turns to resolve the referent before rewriting. For example, if the user asked
about chamomile earlier and now asks "how do I prepare it?", rewrite the three
queries around chamomile preparation, not a generic preparation query.

Produce exactly 3 concise, DISTINCT search queries that preserve the same user
intent while varying terminology. Use botanical, scientific, and traditional
phrasing where appropriate.

Return only JSON:
{"expanded_queries": ["query1", "query2", "query3"]}"""

GRADER_SYSTEM = """You are a strict relevance grader for herbal-health RAG.

Decide if a document contains ANY information that helps answer the user query.
Answer "yes" for any meaningful overlap; answer "no" only when unrelated.

Return only JSON:
{"score": "yes"} or {"score": "no"}"""

# --- Medical answer prompt -------------------------------------------------
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
   Malik Fikret, Ebru Tugce Polat, and Melisa Yildirim under the guidance of
   Prof. Dr. Ramazan KATIRCI.
7. TONE AND FORMAT: Calm, practical, concise. Use short paragraphs or tidy
   bullet points where helpful. Never bulletize a greeting.
8. CONVERSATIONAL CONTINUITY: Use the recent conversation turns provided to
   resolve follow-up references ("it", "that tea", "the recipe above") so the
   answer flows naturally from the earlier discussion."""

DIRECT_ANSWER_SYSTEM = """You are the conversational front-desk for an AI Herbalist Assistant.

Rules:
1) MATCH THE USER'S LANGUAGE exactly.
2) Be friendly, concise, and natural. Do NOT use bullet points for greetings.
3) If the user greets you, introduce yourself politely as an AI Herbalist
   Assistant and invite herbal questions.
4) If the user asks who made you, who created you, who your founders are, or
   asks for your full identity, answer clearly in the user's language:
   Developed by computer engineering students Malik Fikret, Ebru Tugce Polat,
   and Melisa Yildirim, under the guidance of Prof. Dr. Ramazan KATIRCI.
5) Do NOT invent product claims.
6) Do NOT add medical disclaimers, warnings, or advice to "consult a doctor".
7) Use the recent conversation turns to keep the reply coherent with the
   user's previous messages."""


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_route(raw: str) -> RouteLiteral:
    try:
        data = json.loads(_strip_fences(raw))
        route = str(data.get("route", "")).strip().upper()
        if route in ("VECTOR_SEARCH", "DIRECT_ANSWER"):
            return route  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    for line in raw.splitlines():
        token = line.strip().upper()
        if token in ("VECTOR_SEARCH", "DIRECT_ANSWER"):
            return token  # type: ignore[return-value]

    raise ValueError(f"Unparseable router output: {raw!r}")


def _extract_expanded_queries(raw: str) -> tuple[str, str, str]:
    data = json.loads(_strip_fences(raw))
    queries = data.get("expanded_queries")
    if not isinstance(queries, list) or len(queries) != 3:
        raise ValueError("expanded_queries must be a list of exactly 3 strings")

    cleaned = tuple(str(q).strip() for q in queries)
    if any(not q for q in cleaned):
        raise ValueError("expanded queries must be non-empty")
    if len({q.lower() for q in cleaned}) != 3:
        raise ValueError("expanded queries must be distinct")
    return cleaned  # type: ignore[return-value]


def _extract_score(raw: str) -> ScoreLiteral:
    try:
        data = json.loads(_strip_fences(raw))
        score = str(data.get("score", "")).strip().lower()
        if score in ("yes", "no"):
            return score  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    m = re.search(r'"score"\s*:\s*"(yes|no)"', raw, re.IGNORECASE)
    if m:
        return m.group(1).lower()  # type: ignore[return-value]

    for line in raw.splitlines():
        token = line.strip().lower()
        if token in ("yes", "no"):
            return token  # type: ignore[return-value]

    raise ValueError(f"Unparseable grader output: {raw!r}")


def _resolve_model_name(state: AgentState) -> str:
    candidate = str(state.get("model_name", "") or "").strip()
    return candidate or config.GROQ_MODEL


def _resolve_web_search_provider(state: AgentState) -> str:
    provider = str(state.get("web_search_provider", "") or "").strip()
    return provider or _DEFAULT_WEB_SEARCH_PROVIDER


def _get_required_env(var_name: str) -> str:
    value = os.getenv(var_name, "").strip()
    if not value:
        raise RuntimeError(f"{var_name} is not set. Please add it to your .env file.")
    return value


def _create_chat_model(*, model_name: str, temperature: float):
    if model_name in _GROQ_MODELS:
        return create_groq_llm(
            api_key=get_groq_api_key(),
            model_name=model_name,
            temperature=temperature,
        )
    if model_name == "gemini-1.5-flash":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=_get_required_env("GEMINI_API_KEY"),
        )
    if model_name == "deepseek-chat":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=_get_required_env("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
        )
    _logger.warning("Unknown model '%s'; falling back to %s", model_name, config.GROQ_MODEL)
    return create_groq_llm(
        api_key=get_groq_api_key(),
        model_name=config.GROQ_MODEL,
        temperature=temperature,
    )


@lru_cache(maxsize=32)
def _router_llm(model_name: str):
    return _create_chat_model(model_name=model_name, temperature=0.0)


@lru_cache(maxsize=32)
def _expansion_llm(model_name: str):
    return _create_chat_model(model_name=model_name, temperature=0.6)


@lru_cache(maxsize=32)
def _grader_llm(model_name: str):
    return _create_chat_model(model_name=model_name, temperature=0.0)


@lru_cache(maxsize=32)
def _generator_llm(model_name: str):
    return _create_chat_model(model_name=model_name, temperature=config.LLM_TEMPERATURE)


@lru_cache(maxsize=1)
def _retriever():
    from herbalist_assistant.rag.embeddings import create_embeddings
    from herbalist_assistant.rag.vectorstore import load_or_build_vectorstore, make_retriever

    embeddings = create_embeddings(config.EMBEDDING_MODEL)
    vectorstore = load_or_build_vectorstore(
        data_dir=config.DATA_DIR,
        persist_dir=config.CHROMA_DIR,
        embeddings=embeddings,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return make_retriever(vectorstore, k=config.RETRIEVER_K)


def reset_runtime_caches() -> None:
    """Invalidate all module-level LLM and retriever caches.

    Called by the admin "Re-index" action so that a fresh vectorstore is
    actually used by the live chat path (not just by the Streamlit caches).
    """
    _router_llm.cache_clear()
    _expansion_llm.cache_clear()
    _grader_llm.cache_clear()
    _generator_llm.cache_clear()
    _retriever.cache_clear()


def _document_key(doc: Document) -> str:
    explicit_id = getattr(doc, "id", None)
    if explicit_id:
        return f"id:{explicit_id}"
    meta_id = doc.metadata.get("id") if doc.metadata else None
    if meta_id is not None:
        return f"meta_id:{meta_id}"
    source = doc.metadata.get("source", "") if doc.metadata else ""
    page = doc.metadata.get("page", "")
    digest = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
    return f"src:{source!s}|page:{page!s}|sha256:{digest}"


def _dedupe_documents(documents: list[Document]) -> list[Document]:
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in documents:
        key = _document_key(doc)
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _format_chat_history(history: list[dict] | None) -> str:
    """Format recent turns for inclusion in LLM prompts.

    Returns an empty string when there is nothing to show. Truncates each
    message to keep prompt sizes bounded.
    """
    if not history:
        return ""

    turns = history[-_MAX_HISTORY_MESSAGES:]
    lines: list[str] = []
    for msg in turns:
        role = (msg.get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        label = "User" if role == "user" else "Assistant"
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        # Prevent a single huge assistant message from dominating the prompt.
        if len(content) > 800:
            content = content[:800] + "..."
        lines.append(f"{label}: {content}")

    return "\n".join(lines)


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


def expand_and_retrieve_node(state: AgentState) -> AgentState:
    """Expand into 3 search variants, retrieve from Chroma, and dedupe documents."""
    question = str(state.get("question", "")).strip()
    if not question:
        return {"expanded_queries": [], "documents": []}

    history_block = _format_chat_history(state.get("chat_history"))
    human_parts: list[str] = []
    if history_block:
        human_parts.append(f"Recent conversation:\n{history_block}")
    human_parts.append(f"Current user question:\n{question}")
    human_message = "\n\n".join(human_parts)

    try:
        response = _expansion_llm(_resolve_model_name(state)).invoke(
            [SystemMessage(content=EXPANSION_SYSTEM), HumanMessage(content=human_message)]
        )
        q1, q2, q3 = _extract_expanded_queries(getattr(response, "content", str(response)))
        expanded = [q1, q2, q3]
    except Exception:
        _logger.exception("Query expansion failed; falling back to original question only")
        expanded = [question]

    docs: list[Document] = []
    retriever = _retriever()
    for query in expanded:
        try:
            batch = retriever.invoke(query)
        except Exception:
            _logger.exception("Retriever.invoke failed for query=%r", query)
            continue
        if batch:
            docs.extend(batch)

    return {"expanded_queries": expanded, "documents": _dedupe_documents(docs)}


def _grade_document(question: str, document_text: str, *, model_name: str) -> ScoreLiteral:
    if not question.strip() or not document_text.strip():
        return "no"

    grader_human = f"User query:\n{question}\n\nDocument text:\n{document_text}"
    try:
        response = _grader_llm(model_name).invoke(
            [SystemMessage(content=GRADER_SYSTEM), HumanMessage(content=grader_human)]
        )
        return _extract_score(getattr(response, "content", str(response)))
    except Exception:
        _logger.exception("Grader LLM failed; keeping document (defaulting to 'yes')")
        # Fail-open: keep the document rather than drop it on a transient error.
        return "yes"


def grade_documents_node(state: AgentState) -> AgentState:
    """Keep only documents that pass CRAG grading ("yes")."""
    question = str(state.get("question", "")).strip()
    documents = state.get("documents", []) or []
    if not question or not documents:
        return {"documents": []}

    model_name = _resolve_model_name(state)
    filtered: list[Document] = []
    for doc in documents:
        if _grade_document(question, doc.page_content, model_name=model_name) == "yes":
            filtered.append(doc)
    return {"documents": filtered}


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


def _normalize_web_search_results(raw_results, *, provider_name: str) -> list[Document]:
    source_label = f"Web Search ({provider_name})"
    docs: list[Document] = []

    if isinstance(raw_results, str):
        text = raw_results.strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": source_label}))
        return docs

    if not isinstance(raw_results, list):
        return docs

    for item in raw_results:
        if isinstance(item, str):
            text = item.strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": source_label}))
            continue

        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "")).strip()
        snippet = str(
            item.get("content")
            or item.get("snippet")
            or item.get("body")
            or ""
        ).strip()
        url = str(item.get("url", "")).strip()
        if not (title or snippet or url):
            continue

        body_parts: list[str] = []
        if title:
            body_parts.append(f"Title: {title}")
        if snippet:
            body_parts.append(f"Snippet: {snippet}")
        if url:
            body_parts.append(f"URL: {url}")

        docs.append(
            Document(
                page_content="\n".join(body_parts),
                metadata={
                    "source": source_label,
                    "title": title,
                    "url": url,
                },
            )
        )

    return docs


def web_search_node(state: AgentState) -> AgentState:
    """Fallback web search when all local docs are filtered out."""
    question = str(state.get("question", "")).strip()
    if not question:
        return {}

    provider_name = _resolve_web_search_provider(state)
    try:
        if provider_name == "Tavily":
            from langchain_tavily_search import TavilySearchResults

            tavily_api_key = _get_required_env("TAVILY_API_KEY")
            try:
                search_tool = TavilySearchResults(
                    max_results=_WEB_SEARCH_RESULT_LIMIT,
                    tavily_api_key=tavily_api_key,
                )
            except TypeError:
                os.environ["TAVILY_API_KEY"] = tavily_api_key
                search_tool = TavilySearchResults(max_results=_WEB_SEARCH_RESULT_LIMIT)
            raw_results = search_tool.invoke(question)
        else:
            provider_name = _DEFAULT_WEB_SEARCH_PROVIDER
            search_tool = DuckDuckGoSearchRun()
            raw_results = search_tool.invoke(question)
    except Exception:
        _logger.exception("Web search failed with provider=%s", provider_name)
        return {}

    web_docs = _normalize_web_search_results(raw_results, provider_name=provider_name)
    if not web_docs:
        return {}
    return {"documents": web_docs}


def _sanitize_medical_answer(answer: str) -> str:
    """Remove meta/disclaimer lines that degrade user-facing answer quality."""
    banned_fragments = [
        "consult a healthcare professional",
        "consult your doctor",
        "talk to your doctor",
        "seek medical advice",
        "mentioned in the context",
        "according to the context",
        "based on the context",
        "provided context",
        "provided text",
        "doktora danış",
        "bir doktora danış",
    ]

    kept_lines: list[str] = []
    for line in (answer or "").splitlines():
        lowered = line.lower()
        if any(fragment in lowered for fragment in banned_fragments):
            continue
        kept_lines.append(line.rstrip())

    cleaned = "\n".join(kept_lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


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


def _route_from_router(state: AgentState) -> Literal["medical", "direct"]:
    return "medical" if bool(state.get("is_medical")) else "direct"


def _route_after_grading(state: AgentState) -> Literal["has_docs", "no_docs"]:
    return "has_docs" if bool(state.get("documents")) else "no_docs"


workflow = StateGraph(AgentState)

workflow.add_node("route_question", route_question)
workflow.add_node("direct_answer_node", direct_answer_node)
workflow.add_node("expand_and_retrieve_node", expand_and_retrieve_node)
workflow.add_node("grade_documents_node", grade_documents_node)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("generate_medical_answer_node", generate_medical_answer_node)

workflow.set_entry_point("route_question")
workflow.add_conditional_edges(
    "route_question",
    _route_from_router,
    {"medical": "expand_and_retrieve_node", "direct": "direct_answer_node"},
)
workflow.add_edge("direct_answer_node", END)
workflow.add_edge("expand_and_retrieve_node", "grade_documents_node")

# Explicit conditional branch after CRAG grading.
workflow.add_conditional_edges(
    "grade_documents_node",
    _route_after_grading,
    {
        "has_docs": "generate_medical_answer_node",
        "no_docs": "web_search_node",
    },
)
workflow.add_edge("web_search_node", "generate_medical_answer_node")
workflow.add_edge("generate_medical_answer_node", END)

app = workflow.compile()
