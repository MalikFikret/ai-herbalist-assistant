#!/usr/bin/env python3
"""Automated RAG Evaluation System for the AI Herbalist Assistant.

Iterates through ``eval_dataset.json``, runs each question/language pair as a
standalone test case through the compiled LangGraph agent, collects internal
quality metrics (hallucination_score, answer_relevance_score), and uses an LLM
judge to compute an Accuracy/Similarity score against the language-specific
ideal answer.

Results are printed as a formatted console table and exported to a
timestamped CSV inside the ``evaluation_reports/`` directory.

Usage:
    python scripts/evaluate_system.py
"""

from __future__ import annotations

import argparse
import os
import asyncio
import csv
from datetime import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------------------------
# Ensure the package root is importable when running the script directly.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Trigger .env loading before any LangChain import.
import herbalist_assistant  # noqa: E402, F401

from langchain_groq import ChatGroq  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from herbalist_assistant.graph.advanced_graph import app as agent_graph_app  # noqa: E402
from herbalist_assistant.llm.groq import get_groq_api_key  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_PATH = _REPO_ROOT / "eval_dataset.json"
REPORTS_DIR = _REPO_ROOT / "evaluation_reports"
JUDGE_MODEL = "gemini-2.0-flash"
# Per-invocation timeout in seconds for the graph and judge calls.
GRAPH_TIMEOUT_SECONDS = 180
JUDGE_TIMEOUT_SECONDS = 60
# Pause between Groq API requests to avoid HTTP 429 rate-limit errors.
RATE_LIMIT_SLEEP_SECONDS = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
_logger = logging.getLogger("evaluate_system")

# ---------------------------------------------------------------------------
# LLM Judge prompt
# ---------------------------------------------------------------------------
ACCURACY_JUDGE_SYSTEM = """\
You are an expert evaluator for a herbal-health AI assistant.

Your task is to compare a GENERATED ANSWER against an IDEAL (reference) ANSWER
and rate the semantic similarity / factual accuracy on a scale from 0 to 100.

Scoring guidelines:
  - 90-100: Nearly identical factual content, same herbs/remedies mentioned.
  - 70-89:  Most key facts present, minor omissions or extra details.
  - 50-69:  Partially correct, some important herbs/facts missing.
  - 30-49:  Significant factual gaps or irrelevant content.
  - 0-29:   Mostly wrong, off-topic, or empty answer.

You MUST respond with ONLY a valid JSON object, no extra text:
{"score": <int 0-100>, "explanation": "<brief justification>"}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load and validate the evaluation dataset JSON."""
    if not path.exists():
        _logger.error("Dataset file not found: %s", path)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        _logger.error("Dataset must be a non-empty JSON array.")
        sys.exit(1)
    _logger.info("Loaded %d question(s) from %s", len(data), path.name)
    return data


#def _create_judge_llm() -> ChatGroq:
#    """Instantiate the LLM judge using ChatGroq."""
#    return ChatGroq(
#        model_name=JUDGE_MODEL,
#        temperature=0.0,
#        api_key=get_groq_api_key(),
#    )

def _create_judge_llm() -> ChatGoogleGenerativeAI:
    """Instantiate the LLM judge using Google Gemini 1.5 Flash."""
    # We will retrieve the key using the name in your `env` file
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        _logger.error("No Gemini/Google API key found in environment variables!")
    
    return ChatGoogleGenerativeAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        google_api_key=api_key,
    )


def _parse_judge_json(raw: str) -> dict[str, Any]:
    """Robustly extract a JSON object from the judge LLM response.

    Handles cases where the model wraps JSON in markdown code fences.
    """
    text = raw.strip()
    # Strip markdown code fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        _logger.warning("Could not parse judge response as JSON: %s", text[:200])
        return {"score": 0, "explanation": "Judge response was not valid JSON."}


async def _invoke_graph(question: str, timeout: float = GRAPH_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Run a single question through the compiled agent graph with a timeout.

    Returns the final AgentState dict.
    """
    initial_state = {
        "question": question,
        "chat_history": [],
        "model_name": "llama-3.3-70b-versatile",
    }
    try:
        result = await asyncio.wait_for(
            agent_graph_app.ainvoke(initial_state),
            timeout=timeout,
        )
        return result
    except asyncio.TimeoutError:
        _logger.error("Graph invocation timed out after %ds for: %.80s", timeout, question)
        return {
            "final_answer": "[TIMEOUT] The graph did not respond in time.",
            "hallucination_score": "error",
            "answer_relevance_score": "error",
        }
    except Exception as exc:
        _logger.exception("Graph invocation failed for: %.80s", question)
        return {
            "final_answer": f"[ERROR] {exc!r}",
            "hallucination_score": "error",
            "answer_relevance_score": "error",
        }


async def _judge_accuracy(
    judge: ChatGroq,
    generated_answer: str,
    ideal_answer: str,
    timeout: float = JUDGE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Use the LLM judge to score semantic accuracy of the generated answer."""
    prompt = (
        f"GENERATED ANSWER:\n{generated_answer}\n\n"
        f"IDEAL ANSWER:\n{ideal_answer}"
    )
    try:
        response = await asyncio.wait_for(
            judge.ainvoke([
                SystemMessage(content=ACCURACY_JUDGE_SYSTEM),
                HumanMessage(content=prompt),
            ]),
            timeout=timeout,
        )
        return _parse_judge_json(response.content)
    except asyncio.TimeoutError:
        _logger.error("Accuracy judge timed out.")
        return {"score": 0, "explanation": "Judge timed out."}
    except Exception as exc:
        _logger.exception("Accuracy judge failed.")
        return {"score": 0, "explanation": f"Judge error: {exc!r}"}


def _extract_answer(state: dict[str, Any]) -> str:
    """Pull the best available answer string out of a completed AgentState."""
    answer = str(state.get("final_answer", "") or "").strip()
    if answer:
        return answer
    answer = str(state.get("direct_answer", "") or "").strip()
    if answer:
        return answer
    return "(no answer generated)"


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _print_divider(width: int = 120) -> None:
    print("=" * width)


def _print_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted console summary table."""
    _print_divider()
    header = (
        f"{'ID':<8} {'Lang':<6} {'Topic':<25} "
        f"{'Latency':<10} {'Halluc.':<12} {'Relevance':<12} "
        f"{'Accuracy':<10} {'Explanation'}"
    )
    print(header)
    _print_divider()

    for r in results:
        accuracy_display = f"{r['accuracy_score']}%"
        latency_display = f"{r['latency_seconds']:.1f}s"
        print(
            f"{r['question_id']:<8} {r['language']:<6} {r['topic']:<25} "
            f"{latency_display:<10} {r['hallucination_score']:<12} {r['answer_relevance_score']:<12} "
            f"{accuracy_display:<10} {r['accuracy_explanation'][:45]}"
        )

    _print_divider()


def _print_language_averages(results: list[dict[str, Any]]) -> None:
    """Compute and print average accuracy per language."""
    lang_scores: dict[str, list[int]] = {}
    for r in results:
        lang = r["language"]
        lang_scores.setdefault(lang, []).append(r["accuracy_score"])

    print("\n--- Average System Accuracy by Language ---")
    for lang in sorted(lang_scores):
        scores = lang_scores[lang]
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"  {lang.upper():>4}: {avg:6.1f}%  (n={len(scores)})")
    print()


def _export_csv(results: list[dict[str, Any]], path: Path) -> None:
    """Write detailed results to a CSV file."""
    fieldnames = [
        "question_id",
        "language",
        "latency_seconds",
        "accuracy_score",
        "hallucination_score",
        "answer_relevance_score",
        "document_grade_score",
        "topic",
        "question",
        "ideal_answer",
        "generated_answer",
        "accuracy_explanation",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    _logger.info("Results exported to %s", path)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

async def main(target_lang: str) -> None:
    start_time = time.time()

    # Create the reports directory if it doesn't exist.
    REPORTS_DIR.mkdir(exist_ok=True)

    # Generate a timestamped output filename including the language
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_path = REPORTS_DIR / f"results_{target_lang}_{timestamp}.csv"

    # Select the dataset dynamically based on the requested language
    dataset_name = f"eval_dataset_{target_lang}.json"
    dataset_path = _REPO_ROOT / dataset_name
    
    dataset = _load_dataset(dataset_path)
    judge = _create_judge_llm()

    all_results: list[dict[str, Any]] = []
    total = len(dataset)
    _logger.info("Starting %s evaluation: %d test cases", target_lang.upper(), total)

    for idx, entry in enumerate(dataset, start=1):
        qid = entry["id"]
        topic = entry.get("topic", "unknown")
        lang = entry.get("language", target_lang)
        question_text = entry["question"]
        ideal_answer = entry["ideal_answer"]

        # ---- Step 1: Run the question through the graph ----
        _logger.info("[%d/%d] %s — Invoking graph...", idx, total, qid)

        graph_start = time.perf_counter()
        state = await _invoke_graph(question_text)
        latency_seconds = round(time.perf_counter() - graph_start, 2)

        generated_answer = _extract_answer(state)
        hallucination = str(state.get("hallucination_score", "N/A"))
        relevance = str(state.get("answer_relevance_score", "N/A"))

        # ------------------------------------------------------------
        selected_docs = state.get("selected_docs", [])
        document_grade_score = "yes" if isinstance(selected_docs, list) and len(selected_docs) > 0 else "no"
        # ------------------------------------------------------------

        _logger.info(
            "   latency=%.2fs  hallucination=%s  relevance=%s  docs_passed=%s  answer_len=%d",
            latency_seconds,
            hallucination,
            relevance,
            document_grade_score,
            len(generated_answer),
        )

        # Rate-limit pause after graph invocation (multiple LLM calls inside).
        _logger.info("   Sleeping %ds to respect rate limits...", RATE_LIMIT_SLEEP_SECONDS)
        time.sleep(RATE_LIMIT_SLEEP_SECONDS)

        # ---- Step 2: LLM Judge — accuracy against ideal answer ----
        _logger.info("[%d/%d] %s — Judging accuracy...", idx, total, qid)

        accuracy_result = await _judge_accuracy(
            judge,
            generated_answer,
            ideal_answer,
        )
        accuracy_score = int(accuracy_result.get("score", 0))
        accuracy_explanation = accuracy_result.get("explanation", "")

        _logger.info("   accuracy_score=%d%%", accuracy_score)

        # Rate-limit pause after judge call.
        time.sleep(RATE_LIMIT_SLEEP_SECONDS)

        # ---- Collect result row ----
        all_results.append(
            {
                "question_id": qid,
                "topic": topic,
                "language": lang,
                "question": question_text,
                "ideal_answer": ideal_answer,
                "generated_answer": generated_answer,
                "latency_seconds": latency_seconds,
                "hallucination_score": hallucination,
                "answer_relevance_score": relevance,
                "document_grade_score": document_grade_score,
                "accuracy_score": accuracy_score,
                "accuracy_explanation": accuracy_explanation,
            }
        )

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time
    _logger.info("Evaluation completed in %.1f seconds.", elapsed)

    print("\n")
    print("=" * 60)
    print(f"  AI HERBALIST ASSISTANT — {target_lang.upper()} EVALUATION REPORT")
    print(f"  Test cases: {total}  |  Elapsed: {elapsed:.1f}s")
    print(f"  Judge model: {JUDGE_MODEL}")
    print("=" * 60)
    print()

    _print_table(all_results)
    _print_language_averages(all_results)
    _export_csv(all_results, output_csv_path)

    print(f"Detailed results saved to: {output_csv_path}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Herbalist Assistant Evaluation.")
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "tr"],
        required=True,
        help="Specify the target language for evaluation (e.g., 'en' for English, 'tr' for Turkish)."
    )
    args = parser.parse_args()
    
    asyncio.run(main(args.lang))