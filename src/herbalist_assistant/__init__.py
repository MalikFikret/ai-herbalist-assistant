"""AI Herbalist Assistant package.

LangSmith / LangChain tracing relies on environment variables that LangChain
caches on first import (via ``functools.lru_cache``). If ``.env`` is loaded
*after* those modules are imported, tracing stays disabled for the lifetime
of the process. To make tracing reliable across every entry point
(`streamlit run`, ``python src/app.py``, ad-hoc scripts), we load ``.env``
here, in the package root, before any submodule (and therefore any
``langchain``/``langsmith`` import) can run.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parents[1]
_ENV_PATH = _REPO_ROOT / ".env"

load_dotenv(_ENV_PATH, override=False)

_logger = logging.getLogger("herbalist_assistant.tracing")


def _normalize_tracing_flags() -> None:
    """Mirror ``LANGSMITH_TRACING`` to ``LANGCHAIN_TRACING_V2`` (and vice versa).

    Different LangChain / LangSmith versions check different variable names.
    Setting both removes a common source of "I enabled tracing but nothing
    shows up" issues.
    """

    truthy = {"true", "1", "yes", "on"}

    smith = (os.environ.get("LANGSMITH_TRACING") or "").strip().lower()
    legacy = (os.environ.get("LANGCHAIN_TRACING_V2") or "").strip().lower()

    enabled = smith in truthy or legacy in truthy
    if enabled:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    if os.environ.get("LANGSMITH_API_KEY") and not os.environ.get("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
    if os.environ.get("LANGSMITH_PROJECT") and not os.environ.get("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]
    if os.environ.get("LANGSMITH_ENDPOINT") and not os.environ.get("LANGCHAIN_ENDPOINT"):
        os.environ["LANGCHAIN_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"]


_normalize_tracing_flags()


def verify_langsmith_tracing(timeout: float = 5.0) -> tuple[bool, str]:
    """Return ``(ok, message)`` describing the current tracing configuration.

    Performs a lightweight network probe against the configured LangSmith
    endpoint to surface common failures (revoked / wrong-region API keys)
    early instead of letting them disappear into a background thread as
    ``WARNING:langsmith.client:Failed to multipart ingest runs``.
    """

    if (os.environ.get("LANGSMITH_TRACING", "").lower() != "true"
            and os.environ.get("LANGCHAIN_TRACING_V2", "").lower() != "true"):
        return False, "LangSmith tracing is disabled (LANGSMITH_TRACING is not 'true')."

    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        return False, "LANGSMITH_API_KEY is not set."

    endpoint = (
        os.environ.get("LANGSMITH_ENDPOINT")
        or os.environ.get("LANGCHAIN_ENDPOINT")
        or "https://api.smith.langchain.com"
    )
    project = os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT") or "default"

    try:
        import requests

        # /info is anonymous on LangSmith Cloud, so it cannot tell us whether
        # the API key is valid. /sessions is tenant-scoped and is the same
        # surface that run ingestion uses, so a 401/403 here matches what the
        # background tracer would see when posting runs.
        resp = requests.get(
            f"{endpoint.rstrip('/')}/sessions",
            params={"limit": 1},
            headers={"x-api-key": api_key},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return True, (
                f"LangSmith tracing is enabled. endpoint={endpoint} project={project!r}."
            )
        if resp.status_code in (401, 403):
            return False, (
                "LangSmith API rejected the key with HTTP "
                f"{resp.status_code}. The key is invalid, revoked, or belongs "
                f"to a different region. endpoint={endpoint}. "
                "Generate a new API key in LangSmith -> Settings -> API Keys, "
                "and confirm LANGSMITH_ENDPOINT matches your workspace region "
                "(US: https://api.smith.langchain.com, "
                "EU: https://eu.api.smith.langchain.com)."
            )
        return False, (
            f"LangSmith /sessions returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    except Exception as exc:  # network errors, missing requests, etc.
        return False, f"Could not reach LangSmith at {endpoint}: {exc!r}"


def log_langsmith_status(level: int = logging.INFO) -> None:
    ok, message = verify_langsmith_tracing()
    if ok:
        _logger.log(level, message)
    else:
        _logger.warning("LangSmith tracing problem: %s", message)


__all__ = ["verify_langsmith_tracing", "log_langsmith_status"]
