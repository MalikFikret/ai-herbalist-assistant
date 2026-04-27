"""LangGraph orchestration package.

Submodules:
* ``state``         - shared ``AgentState`` schema + Literal aliases
* ``extractors``    - LLM-output parsers and text/document utilities
* ``runtime``       - cached LLM clients, retriever, runtime resolvers
* ``nodes``         - one module per node, each owning its system prompt
* ``advanced_graph``- the compiled ``StateGraph`` (entry point for callers)

Importing this package eagerly loads ``.env`` (via the parent
``herbalist_assistant`` package), which must happen before any
``langchain``/``langsmith`` import so tracing flags are picked up correctly.
"""

from __future__ import annotations

import herbalist_assistant  # noqa: F401  -- triggers .env load on first import
