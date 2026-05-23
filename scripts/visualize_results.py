#!/usr/bin/env python3
"""Generate a Comprehensive Technical Dashboard from RAG evaluation results.

Automatically finds the latest ``results_{lang}_*.csv`` in ``evaluation_reports/``,
computes key performance metrics, and renders a two-panel dashboard with combined quality metrics.

Usage:
    python scripts/visualize_results.py --lang en
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

REPORTS_DIR = _REPO_ROOT / "evaluation_reports"


def _find_latest_csv(directory: Path, lang: str) -> Path:
    """Find the most recent results CSV for the specified language."""
    if not directory.exists() or not directory.is_dir():
        print(f"ERROR: Reports directory not found: {directory}")
        sys.exit(1)

    pattern = f"results_{lang}_*.csv"
    csv_files = sorted(
        directory.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not csv_files:
        print(f"ERROR: No '{pattern}' files found in {directory}")
        sys.exit(1)

    return csv_files[0]


def _extract_timestamp(csv_path: Path) -> str:
    """Extract the YYYYMMDD_HHMMSS timestamp from a results filename."""
    match = re.search(r"results_[a-z]{2}_(\d{8}_\d{6})\.csv$", csv_path.name)
    return match.group(1) if match else "latest"


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read all rows from the CSV into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(value: str) -> float | None:
    """Convert a string to float safely."""
    try:
        return float(value.strip())
    except (ValueError, TypeError, AttributeError):
        return None


def _is_document_pass(raw: str) -> bool | None:
    """Determine if document_grade_score indicates a successful retrieval pass."""
    val = raw.strip().lower()
    if val == "yes":
        return True
    if val == "no":
        return False
    num = _safe_float(raw)
    if num is not None:
        if 0.0 < num <= 1.0:
            num = num * 100.0
        return num >= 70.0  # Threshold based on CRAG architecture
    return None


def _is_hallucination_pass(raw: str) -> bool | None:
    """Determine if a hallucination_score indicates a grounded answer."""
    val = raw.strip().lower()
    if val == "no":
        return True  # 'no' means no hallucinations found (PASS)
    if val == "yes":
        return False  # 'yes' means answer hallucinated (FAIL)
    num = _safe_float(raw)
    if num is not None:
        if 0.0 < num <= 1.0:
            num = num * 100.0
        return num >= 50.0
    return None


def _is_relevance_pass(raw: str) -> bool | None:
    """Determine if answer_relevance_score indicates a relevant answer."""
    val = raw.strip().lower()
    if val == "yes":
        return True
    if val == "no":
        return False
    num = _safe_float(raw)
    if num is not None:
        if 0.0 < num <= 1.0:
            num = num * 100.0
        return num >= 50.0
    return None


def _aggregate_metrics(rows: list[dict[str, str]]) -> dict:
    """Compute metrics from raw CSV rows including document grading."""
    latency_by_lang: dict[str, list[float]] = defaultdict(list)
    
    hallucination_pass = 0
    hallucination_total = 0
    
    relevance_pass = 0
    relevance_total = 0
    
    document_pass = 0
    document_total = 0

    for row in rows:
        lang = row.get("language", "").strip()
        if not lang:
            continue

        # Latency
        lat = _safe_float(row.get("latency_seconds", ""))
        if lat is not None:
            latency_by_lang[lang].append(lat)

        # Hallucination
        h_pass = _is_hallucination_pass(row.get("hallucination_score", ""))
        if h_pass is not None:
            hallucination_total += 1
            if h_pass:
                hallucination_pass += 1

        # Relevance
        r_pass = _is_relevance_pass(row.get("answer_relevance_score", ""))
        if r_pass is not None:
            relevance_total += 1
            if r_pass:
                relevance_pass += 1

        # Document Grading
        d_pass = _is_document_pass(row.get("document_grade_score", ""))
        if d_pass is not None:
            document_total += 1
            if d_pass:
                document_pass += 1

    return {
        "latency_by_lang": dict(latency_by_lang),
        "hallucination_pass": hallucination_pass,
        "hallucination_total": hallucination_total,
        "relevance_pass": relevance_pass,
        "relevance_total": relevance_total,
        "document_pass": document_pass,
        "document_total": document_total,
    }


def _generate_dashboard(metrics: dict, output_path: Path, target_lang: str) -> None:
    """Create a 2-panel technical dashboard with 3 side-by-side quality metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib is required. Run: pip install matplotlib")
        sys.exit(1)

    # Palette: Blue (Latency), Amber (Document), Green (Hallucination), Teal (Relevance)
    palette = ["#4e79a7", "#f28e2b", "#59a14f", "#76b7b2"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"AI Herbalist Assistant — Technical Performance Dashboard ({target_lang.upper()})",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    # ------------------------------------------------------------------
    # Subplot 1: Average Latency (Bar Chart)
    # ------------------------------------------------------------------
    ax1 = axes[0]
    latency = metrics.get("latency_by_lang", {})

    if latency and target_lang in latency:
        avg_lat = sum(latency[target_lang]) / len(latency[target_lang])
        count = len(latency[target_lang])

        bars1 = ax1.bar([target_lang.upper()], [avg_lat], color=[palette[0]], edgecolor="white", linewidth=1.2)
        for bar in bars1:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{avg_lat:.1f}s\n(n={count})",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        ax1.set_ylim(0, avg_lat * 1.35 if avg_lat else 10)
    else:
        ax1.text(0.5, 0.5, "No latency data", ha="center", va="center", fontsize=12, transform=ax1.transAxes)

    ax1.set_title("Average Latency", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Language", fontsize=11)
    ax1.set_ylabel("Seconds", fontsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    # ------------------------------------------------------------------
    # Subplot 2: Combined Quality Metrics (3 Bars Side-by-Side)
    # ------------------------------------------------------------------
    ax2 = axes[1]
    
    # Extract Document Grading stats
    d_pass = metrics.get("document_pass", 0)
    d_total = metrics.get("document_total", 0)
    d_rate = (d_pass / d_total * 100) if d_total > 0 else 0.0

    # Extract Hallucination stats
    h_pass = metrics.get("hallucination_pass", 0)
    h_total = metrics.get("hallucination_total", 0)
    h_rate = (h_pass / h_total * 100) if h_total > 0 else 0.0

    # Extract Relevance stats
    rel_pass = metrics.get("relevance_pass", 0)
    rel_total = metrics.get("relevance_total", 0)
    rel_rate = (rel_pass / rel_total * 100) if rel_total > 0 else 0.0

    if d_total > 0 or h_total > 0 or rel_total > 0:
        labels = ["Document Pass", "Hallucination Pass", "Relevance Pass"]
        rates = [d_rate, h_rate, rel_rate]
        counts = [d_total, h_total, rel_total]
        colors = [palette[1], palette[2], palette[3]]

        bars2 = ax2.bar(labels, rates, color=colors, edgecolor="white", linewidth=1.2, width=0.5)
        
        for bar, rate, n in zip(bars2, rates, counts):
            if n > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{rate:.0f}%\n(n={n})",
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                )
        ax2.set_ylim(0, 115)
    else:
        ax2.text(0.5, 0.5, "No quality data available", ha="center", va="center", fontsize=12, transform=ax2.transAxes)

    ax2.set_title("Quality Metrics Pass Rates", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylabel("Pass Rate (%)", fontsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    # ------------------------------------------------------------------
    # Save Output
    # ------------------------------------------------------------------
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize AI Herbalist Assistant Evaluation Results.")
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "tr"],
        required=True,
        help="Specify the target language to visualize."
    )
    args = parser.parse_args()

    csv_path = _find_latest_csv(REPORTS_DIR, args.lang)
    timestamp = _extract_timestamp(csv_path)
    dashboard_path = REPORTS_DIR / f"technical_dashboard_{args.lang}_{timestamp}.png"

    rows = _load_csv_rows(csv_path)
    if not rows:
        print("ERROR: CSV contains no data rows.")
        sys.exit(1)

    metrics = _aggregate_metrics(rows)
    _generate_dashboard(metrics, dashboard_path, args.lang)


if __name__ == "__main__":
    main()