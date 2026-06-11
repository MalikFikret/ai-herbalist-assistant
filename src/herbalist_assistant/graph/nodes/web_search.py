"""Web search fallback node.

When CRAG grading drops every retrieved document, this node runs a web
search (Tavily or DuckDuckGo) and converts the results into ``Document``
objects so the medical-answer node can treat them like any other context.

Phase-3 changes
───────────────
* _WEB_SEARCH_RESULT_LIMIT: 4 → 6
* _build_web_search_queries: expands queries even on the first attempt.
* Web-result Documents carry ``web_filename`` for the Sources popover.
* ``web_search_queries`` added to the returned state slice.

Phase-4 changes (Priority Domain Search)
─────────────────────────────────────────
* Two-stage search strategy:
    Stage 1 — search only TRUSTED_HERB_DOMAINS (Turkish herbal sites).
    Stage 2 — search the open web ONLY if Stage 1 returns fewer than
               TRUSTED_DOMAIN_MIN_RESULTS Documents.
* Trusted results are marked with ``is_trusted_domain=True`` in metadata
  so the Sources popover can display a "✓ Trusted" badge in the future.
* Tavily uses ``include_domains`` for Stage 1 (native API support).
* DuckDuckGo uses ``site:`` query operators for Stage 1.
* Both providers fall through gracefully if the domain list is empty or
  if the trusted search itself raises an exception.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from herbalist_assistant import config
from herbalist_assistant.graph.extractors import _dedupe_documents, _format_chat_history
from herbalist_assistant.graph.runtime import (
    _expansion_llm,
    _get_required_env,
    _resolve_model_name,
    _resolve_web_search_provider,
)
from herbalist_assistant.graph.state import AgentState

_logger = logging.getLogger(__name__)

# ── tuneable constants ────────────────────────────────────────────────────────
_WEB_SEARCH_RESULT_LIMIT = 6
_PROVIDER_TAVILY = "Tavily"
_PROVIDER_DUCK = "DuckDuckGo"
_TAVILY_QUERY_MAX_LEN = 400


# ── system prompts ────────────────────────────────────────────────────────────

FIRST_ATTEMPT_WEB_QUERY_SYSTEM = """You write concise web search queries for an AI Herbalist assistant.

LANGUAGE RULE (MANDATORY):
- Detect the language of the user question.
- Generate primary queries in that same language.
- You may add ONE cross-language (English scientific/botanical) fallback query.

Generate 2-3 short, distinct queries that cover the core intent from different angles
(symptom-focused, herb-focused, preparation-focused). Never repeat the exact phrasing.

Return ONLY valid JSON - no markdown fences, no extra keys:
{
  "primary_queries": ["q1", "q2"],
  "fallback_queries": ["optional English botanical q"]
}"""

RETRY_WEB_QUERY_SYSTEM = """You rewrite web search queries for a herbal assistant.

The previous attempt did not yield grounded enough results.

LANGUAGE PRIORITY (MANDATORY):
- Detect the language of the original user question.
- Generate primary retry queries in that same language first.
- Optionally include additional fallback cross-language queries (for example
  English scientific terminology), but only after primary-language queries.

Produce NEW queries that are meaningfully different from the original phrasing,
mixing broader and more specific variants to improve recall and precision.

Avoid returning the exact original query. Return only JSON:
{
  "primary_queries": ["same-language q1", "same-language q2", "..."],
  "fallback_queries": ["optional cross-language q1", "..."]
}

Backward-compatible format is also accepted:
{"queries": ["q1", "q2", "q3"]}"""


# ── query helpers ─────────────────────────────────────────────────────────────

def _extract_queries(
    raw: str,
    original_question: str,
    *,
    max_primary: int = 5,
    max_fallback: int = 3,
) -> list[str]:
    """Parse LLM JSON output into a deduplicated query list."""
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = (
                text.removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
        data = json.loads(text)
    except Exception:
        _logger.debug("Could not parse LLM query JSON; using original question")
        return [original_question]

    if not isinstance(data, dict):
        return [original_question]

    seen: set[str] = {original_question.strip().lower()}
    result: list[str] = []

    def _append_unique(items: object, *, limit: int) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if len(result) >= limit:
                return
            q = str(item).strip()
            key = q.lower()
            if not q or key in seen:
                continue
            seen.add(key)
            result.append(q)

    _append_unique(data.get("primary_queries"), limit=max_primary)
    _append_unique(data.get("fallback_queries"), limit=max_primary + max_fallback)
    if not result:
        _append_unique(data.get("queries"), limit=3)

    return result or [original_question]


# Alias for backward compatibility with any external imports.
_extract_retry_queries = _extract_queries


def _build_web_search_queries(state: AgentState, question: str) -> list[str]:
    """Return an expanded list of search queries via LLM."""
    retries = int(state.get("generation_retries", 0) or 0)
    history_block = _format_chat_history(state.get("chat_history"))
    ui_language = str(state.get("ui_language", "")).strip()

    human_parts: list[str] = [f"User question:\n{question}"]
    if history_block:
        human_parts.insert(0, f"Recent conversation:\n{history_block}")
    if ui_language:
        human_parts.append(
            "UI language preference (hint only; prioritise question language):\n"
            f"{ui_language}"
        )
    if retries > 0:
        human_parts.append(f"Current retry count: {retries}")

    prompt = "\n\n".join(human_parts)
    system = RETRY_WEB_QUERY_SYSTEM if retries > 0 else FIRST_ATTEMPT_WEB_QUERY_SYSTEM

    try:
        response = _expansion_llm(_resolve_model_name(state)).invoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        )
        raw = getattr(response, "content", str(response))
        queries = _extract_queries(raw, question)
        _logger.debug("Web search queries (retries=%d): %s", retries, queries)
        return queries
    except Exception:
        _logger.exception("Web query expansion failed; using original question")
        return [question]


# ── DuckDuckGo site-restricted query builder (Phase-4) ───────────────────────

def _add_site_filter(query: str, domains: list[str]) -> str:
    """Append a ``site:`` OR filter so DuckDuckGo searches only trusted domains."""
    if not domains:
        return query
    site_filter = " OR ".join(f"site:{d}" for d in domains)
    # Keep within a sensible length; DuckDuckGo handles long queries poorly.
    restricted = f"{query} ({site_filter})"
    return restricted[:400] if len(restricted) > 400 else restricted


# ── result normalisation ──────────────────────────────────────────────────────

def _normalize_web_search_results(
    raw_results,
    *,
    provider_name: str,
    is_trusted: bool = False,
) -> list[Document]:
    """Convert raw provider output to ``Document`` objects with rich metadata.

    Phase-4: accepts ``is_trusted`` flag so every Document from a trusted-domain
    search is marked with ``is_trusted_domain=True`` in its metadata.
    """
    source_label = f"Web Search ({provider_name})"
    docs: list[Document] = []

    base_meta: dict = {
        "source": source_label,
        "web_filename": source_label,
        "is_trusted_domain": is_trusted,
    }

    # ── plain string (DuckDuckGo) ─────────────────────────────────────────────
    if isinstance(raw_results, str):
        text = raw_results.strip()
        if text:
            docs.append(Document(page_content=text, metadata=dict(base_meta)))
        return docs

    # ── dict (Tavily structured response) ────────────────────────────────────
    if isinstance(raw_results, dict):
        error_text = str(raw_results.get("error", "")).strip()
        if error_text:
            _logger.warning("%s returned error payload: %s", source_label, error_text)
            return docs

        answer_text = str(raw_results.get("answer", "")).strip()
        if answer_text:
            docs.append(
                Document(
                    page_content=f"Answer: {answer_text}",
                    metadata=dict(base_meta),
                )
            )

        if isinstance(raw_results.get("results"), list):
            raw_results = raw_results["results"]
        else:
            raw_results = [raw_results]

    if not isinstance(raw_results, list):
        return docs

    # ── list of result items ──────────────────────────────────────────────────
    for item in raw_results:
        if isinstance(item, str):
            text = item.strip()
            if text:
                docs.append(Document(page_content=text, metadata=dict(base_meta)))
            continue

        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "")).strip()
        snippet = str(
            item.get("content") or item.get("snippet") or item.get("body") or ""
        ).strip()
        url = str(item.get("url", "")).strip()
        score = item.get("score")

        if not (title or snippet or url):
            continue

        body_parts: list[str] = []
        if title:
            body_parts.append(f"Title: {title}")
        if snippet:
            body_parts.append(f"Snippet: {snippet}")
        if url:
            body_parts.append(f"URL: {url}")

        web_label = title if title else (url[:60] + "…" if len(url) > 60 else url)

        docs.append(
            Document(
                page_content="\n".join(body_parts),
                metadata={
                    "source": source_label,
                    "title": title,
                    "url": url,
                    "score": score,
                    "web_filename": web_label or source_label,
                    "is_trusted_domain": is_trusted,   # Phase-4
                },
            )
        )

    return docs


# ── provider wrappers — open web ──────────────────────────────────────────────

def _search_with_duck(queries: list[str], *, is_trusted: bool = False) -> list[Document]:
    search_tool = DuckDuckGoSearchRun()
    docs: list[Document] = []
    for query in queries:
        try:
            raw = search_tool.invoke(query)
            docs.extend(
                _normalize_web_search_results(
                    raw, provider_name=_PROVIDER_DUCK, is_trusted=is_trusted
                )
            )
        except Exception:
            _logger.warning("DuckDuckGo failed for query=%r", query, exc_info=True)
    return docs


def _search_with_tavily(
    queries: list[str],
    *,
    include_domains: list[str] | None = None,
    is_trusted: bool = False,
) -> list[Document]:
    from langchain_tavily import TavilySearch

    _get_required_env("TAVILY_API_KEY")

    # Phase-4: pass include_domains to Tavily when doing trusted-domain search.
    kwargs: dict = {"max_results": _WEB_SEARCH_RESULT_LIMIT}
    if include_domains:
        kwargs["include_domains"] = include_domains

    search_tool = TavilySearch(**kwargs)
    docs: list[Document] = []
    for query in queries:
        if len(query) > _TAVILY_QUERY_MAX_LEN:
            query = query[:_TAVILY_QUERY_MAX_LEN]
        try:
            raw = search_tool.invoke(query)
            docs.extend(
                _normalize_web_search_results(
                    raw, provider_name=_PROVIDER_TAVILY, is_trusted=is_trusted
                )
            )
        except Exception:
            _logger.warning("Tavily failed for query=%r", query, exc_info=True)
    return docs


# ── Phase-4: two-stage search helpers ────────────────────────────────────────

def _stage1_trusted(
    queries: list[str],
    domains: list[str],
    provider: str,
) -> list[Document]:
    """Search only within TRUSTED_HERB_DOMAINS."""
    if not domains:
        return []

    _logger.info("Stage-1 trusted search: domains=%s queries=%s", domains, queries)

    try:
        if provider == _PROVIDER_TAVILY:
            return _search_with_tavily(queries, include_domains=domains, is_trusted=True)
        else:
            # DuckDuckGo: prepend site: filters to every query
            restricted = [_add_site_filter(q, domains) for q in queries]
            return _search_with_duck(restricted, is_trusted=True)
    except Exception:
        _logger.warning("Stage-1 trusted search failed", exc_info=True)
        return []


def _stage2_general(
    queries: list[str],
    provider: str,
) -> list[Document]:
    """Search the open web (no domain restriction)."""
    _logger.info("Stage-2 open-web search: queries=%s", queries)

    try:
        if provider == _PROVIDER_TAVILY:
            try:
                return _search_with_tavily(queries, is_trusted=False)
            except Exception:
                _logger.warning("Tavily open-web failed; trying DuckDuckGo", exc_info=True)
                return _search_with_duck(queries, is_trusted=False)
        else:
            return _search_with_duck(queries, is_trusted=False)
    except Exception:
        _logger.exception("Stage-2 open-web search failed")
        return []


# ── web result grading (Risk-3 fix) ──────────────────────────────────────────
# Local docs use threshold=55. Web snippets are shorter and less structured,
# so we use a lower threshold (50) to avoid dropping too many results.
# If ALL docs fail grading we return an empty list so the caller can
# short-circuit to a polite fallback without wasting 3 downstream LLM calls.
_WEB_GRADE_THRESHOLD = 50
_WEB_MAX_GRADE_DOCS  = 6


def _grade_web_docs(
    docs: list[Document],
    question: str,
    model_name: str,
) -> list[Document]:
    """Grade web-search results and drop clearly irrelevant ones.

    Reuses ``_grade_document`` from the grading node (lazy import to avoid
    circular imports at module load time). Grading runs in parallel via
    ThreadPoolExecutor, same as the local-doc grading node.
    """
    # Lazy import — avoids circular dependency at module load time.
    from herbalist_assistant.graph.nodes.grading import _grade_document  # noqa: PLC0415

    if not docs or not question:
        return docs

    candidates = docs[:_WEB_MAX_GRADE_DOCS]
    workers = min(4, len(candidates))
    scored: list[tuple[int, Document]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_doc = {
            pool.submit(
                _grade_document,
                question,
                doc.page_content,
                model_name=model_name,
            ): doc
            for doc in candidates
        }
        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                score = future.result()
            except Exception:
                _logger.warning("Web doc grading future raised; keeping doc", exc_info=True)
                score = _WEB_GRADE_THRESHOLD + 1  # keep on unexpected error
            if score is None:
                # _grade_document returns None on LLM failure → keep the doc
                score = _WEB_GRADE_THRESHOLD + 1
            _logger.debug(
                "Web doc score=%s | %.60s…",
                score,
                doc.page_content.replace("\n", " "),
            )
            if score > _WEB_GRADE_THRESHOLD:
                scored.append((score, doc))

    if not scored:
        _logger.warning(
            "_grade_web_docs: all %d docs scored ≤ %d — returning empty list",
            len(candidates),
            _WEB_GRADE_THRESHOLD,
        )
        return []

    scored.sort(key=lambda pair: pair[0], reverse=True)
    kept = [doc for _, doc in scored]
    _logger.info(
        "_grade_web_docs: %d/%d web docs passed (threshold=%d)",
        len(kept),
        len(candidates),
        _WEB_GRADE_THRESHOLD,
    )
    return kept


# ── main node ─────────────────────────────────────────────────────────────────

def web_search_node(state: AgentState) -> AgentState:
    """Fallback web search with Phase-4 priority-domain strategy.

    Flow
    ────
    1. Build expanded queries via LLM.
    2. Stage 1 — search TRUSTED_HERB_DOMAINS only.
    3. If Stage 1 returns >= TRUSTED_DOMAIN_MIN_RESULTS docs → use them, done.
    4. Stage 2 — search the open web and MERGE with any Stage-1 results.
    5. Deduplicate the combined pool and return.
    """
    question = str(state.get("question", "")).strip()
    if not question:
        return {"documents": [], "web_search_queries": []}

    queries = _build_web_search_queries(state, question)
    provider = _resolve_web_search_provider(state)
    domains = list(config.TRUSTED_HERB_DOMAINS)
    min_hits = int(config.TRUSTED_DOMAIN_MIN_RESULTS)
    model_name = _resolve_model_name(state)

    # ── Stage 1: trusted domains ──────────────────────────────────────────────
    trusted_docs = _stage1_trusted(queries, domains, provider)
    trusted_deduped = _dedupe_documents(trusted_docs)

    _logger.info(
        "Stage-1 result: %d trusted docs (min_required=%d)",
        len(trusted_deduped),
        min_hits,
    )

    if len(trusted_deduped) >= min_hits:
        # Enough authoritative results — skip open web entirely.
        _logger.info("Stage-1 sufficient — skipping open-web search")
        graded = _grade_web_docs(trusted_deduped, question, model_name)
        if not graded:
            # All trusted docs failed grading → polite fallback, skip generation pipeline
            from herbalist_assistant.graph.nodes.grading import _generate_polite_fallback  # noqa: PLC0415
            _logger.warning("All trusted web docs rejected — generating polite fallback")
            fallback = _generate_polite_fallback(question, "no_sufficient_sources", model_name)
            return {"final_answer": fallback, "documents": [], "web_search_queries": queries}
        return {
            "documents": graded,
            "web_search_queries": queries,
        }

    # ── Stage 2: open web ─────────────────────────────────────────────────────
    general_docs = _stage2_general(queries, provider)
    combined = _dedupe_documents(trusted_docs + general_docs)

    _logger.info(
        "Final web docs: trusted=%d general=%d combined=%d",
        len(trusted_deduped),
        len(general_docs),
        len(combined),
    )

    if not combined:
        return {"documents": [], "web_search_queries": queries}

    graded = _grade_web_docs(combined, question, model_name)
    if not graded:
        # All combined docs failed grading → polite fallback, skip generation pipeline
        from herbalist_assistant.graph.nodes.grading import _generate_polite_fallback  # noqa: PLC0415
        _logger.warning("All web docs rejected after open-web search — generating polite fallback")
        fallback = _generate_polite_fallback(question, "no_sufficient_sources", model_name)
        return {"final_answer": fallback, "documents": [], "web_search_queries": queries}
    return {"documents": graded, "web_search_queries": queries}