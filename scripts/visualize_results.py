#!/usr/bin/env python3
"""Generate a Comprehensive Technical Dashboard from RAG evaluation results.

Automatically finds the latest ``results_*.csv`` in ``evaluation_reports/``,
computes key performance metrics, and renders a three-panel dashboard:

  1. Average Latency per Language (bar chart)
  2. Hallucination Pass Rate (pie chart — % of "grounded" answers)
  3. Answer Relevance Pass Rate per Language (bar chart)

The dashboard is saved as ``evaluation_reports/technical_dashboard_[TIMESTAMP].png``.

Usage:
    python scripts/visualize_results.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

REPORTS_DIR = _REPO_ROOT / "evaluation_reports"

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _find_latest_csv(directory: Path) -> Path:
    """Find the most recent results CSV in the reports directory.

    Files are sorted by modification time (newest first) so the caller
    always gets the latest evaluation run without specifying a filename.
    """
    if not directory.exists() or not directory.is_dir():
        print(f"ERROR: Reports directory not found: {directory}")
        print("Run  python scripts/evaluate_system.py  first to generate results.")
        sys.exit(1)

    csv_files = sorted(
        directory.glob("results_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not csv_files:
        print(f"ERROR: No results_*.csv files found in {directory}")
        print("Run  python scripts/evaluate_system.py  first to generate results.")
        sys.exit(1)

    latest = csv_files[0]
    print(f"Using latest CSV: {latest.name}  ({len(csv_files)} total report(s) found)")
    return latest


def _extract_timestamp(csv_path: Path) -> str:
    """Extract the YYYYMMDD_HHMMSS timestamp from a results filename.

    Falls back to 'latest' if the filename does not match the expected pattern.
    """
    match = re.search(r"results_(\d{8}_\d{6})\.csv$", csv_path.name)
    return match.group(1) if match else "latest"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read all rows from the CSV into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(value: str) -> float | None:
    """Convert a string to float, returning None for non-numeric values."""
    try:
        return float(value.strip())
    except (ValueError, TypeError, AttributeError):
        return None


def _is_hallucination_pass(raw: str) -> bool | None:
    """Determine if a hallucination_score indicates a "safe" (grounded) answer.

    The graph stores hallucination_score as:
      - "yes" = answer IS grounded in documents (pass / safe)
      - "no"  = answer contains hallucinations (fail)
      - "error" / "N/A" / numeric = indeterminate

    Returns True for pass, False for fail, None for indeterminate.
    """
    val = raw.strip().lower()
    if val == "yes":
        return True
    if val == "no":
        return False
    # Try numeric: treat >= 50 as pass (some configs return 0-100).
    num = _safe_float(raw)
    if num is not None:
        return num >= 50.0
    return None


def _is_relevance_pass(raw: str) -> bool | None:
    """Determine if answer_relevance_score indicates a relevant answer.

    Same convention as hallucination: "yes" = pass, "no" = fail.
    """
    val = raw.strip().lower()
    if val == "yes":
        return True
    if val == "no":
        return False
    num = _safe_float(raw)
    if num is not None:
        return num >= 50.0
    return None


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _aggregate_metrics(rows: list[dict[str, str]]) -> dict:
    """Compute per-language and global metrics from raw CSV rows.

    Returns a dict with keys:
      - latency_by_lang: {lang: [float, ...]}
      - hallucination_pass / hallucination_total: global counts
      - relevance_pass_by_lang / relevance_total_by_lang: per-language counts
    """
    latency_by_lang: dict[str, list[float]] = defaultdict(list)
    hallucination_pass = 0
    hallucination_total = 0
    relevance_pass_by_lang: dict[str, int] = defaultdict(int)
    relevance_total_by_lang: dict[str, int] = defaultdict(int)
    skipped = 0

    for row in rows:
        lang = row.get("language", "").strip()
        if not lang:
            skipped += 1
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
            relevance_total_by_lang[lang] += 1
            if r_pass:
                relevance_pass_by_lang[lang] += 1

    if skipped:
        print(f"Note: Skipped {skipped} row(s) with missing language data.")

    return {
        "latency_by_lang": dict(latency_by_lang),
        "hallucination_pass": hallucination_pass,
        "hallucination_total": hallucination_total,
        "relevance_pass_by_lang": dict(relevance_pass_by_lang),
        "relevance_total_by_lang": dict(relevance_total_by_lang),
    }


# ---------------------------------------------------------------------------
# Dashboard generation
# ---------------------------------------------------------------------------

def _generate_dashboard(metrics: dict, output_path: Path) -> None:
    """Create a 3-panel technical dashboard and save as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib is required.  pip install matplotlib")
        sys.exit(1)

    # Shared color palette.
    palette = ["#4e79a7", "#f28e2b", "#76b7b2", "#e15759", "#59a14f", "#edc949"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "AI Herbalist Assistant — Technical Performance Dashboard",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    # ------------------------------------------------------------------
    # Subplot 1: Average Latency per Language
    # ------------------------------------------------------------------
    ax1 = axes[0]
    latency = metrics["latency_by_lang"]

    if latency:
        langs = sorted(latency.keys())
        avg_latencies = [sum(latency[l]) / len(latency[l]) for l in langs]
        counts = [len(latency[l]) for l in langs]
        labels = [l.upper() for l in langs]
        colors = [palette[i % len(palette)] for i in range(len(langs))]

        bars1 = ax1.bar(labels, avg_latencies, color=colors, edgecolor="white", linewidth=1.2)
        for bar, avg, n in zip(bars1, avg_latencies, counts):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{avg:.1f}s\n(n={n})",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
        ax1.set_ylim(0, max(avg_latencies) * 1.35 if avg_latencies else 10)
    else:
        ax1.text(0.5, 0.5, "No latency data", ha="center", va="center", fontsize=12, transform=ax1.transAxes)

    ax1.set_title("Avg Latency per Language", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Language", fontsize=11)
    ax1.set_ylabel("Seconds", fontsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    # ------------------------------------------------------------------
    # Subplot 2: Hallucination Pass Rate (pie chart)
    # ------------------------------------------------------------------
    ax2 = axes[1]
    h_pass = metrics["hallucination_pass"]
    h_total = metrics["hallucination_total"]

    if h_total > 0:
        h_fail = h_total - h_pass
        pass_pct = (h_pass / h_total) * 100
        sizes = [h_pass, h_fail]
        pie_labels = [f"Grounded\n({h_pass})", f"Hallucinated\n({h_fail})"]
        pie_colors = ["#59a14f", "#e15759"]
        explode = (0.04, 0.04)

        wedges, texts, autotexts = ax2.pie(
            sizes, labels=pie_labels, colors=pie_colors, autopct="%1.1f%%",
            startangle=90, explode=explode, textprops={"fontsize": 10},
        )
        for at in autotexts:
            at.set_fontweight("bold")
            at.set_fontsize(11)

        ax2.set_title(
            f"Hallucination Pass Rate ({pass_pct:.0f}%)",
            fontsize=13, fontweight="bold", pad=12,
        )
    else:
        ax2.text(0.5, 0.5, "No hallucination\ndata available", ha="center", va="center", fontsize=12, transform=ax2.transAxes)
        ax2.set_title("Hallucination Pass Rate", fontsize=13, fontweight="bold", pad=12)

    # ------------------------------------------------------------------
    # Subplot 3: Answer Relevance Pass Rate per Language
    # ------------------------------------------------------------------
    ax3 = axes[2]
    rel_pass = metrics["relevance_pass_by_lang"]
    rel_total = metrics["relevance_total_by_lang"]

    all_langs = sorted(set(rel_total.keys()))
    if all_langs:
        pass_rates = []
        bar_labels = []
        bar_counts = []
        for l in all_langs:
            total = rel_total.get(l, 0)
            passed = rel_pass.get(l, 0)
            rate = (passed / total * 100) if total > 0 else 0.0
            pass_rates.append(rate)
            bar_labels.append(l.upper())
            bar_counts.append(total)

        colors3 = [palette[i % len(palette)] for i in range(len(all_langs))]
        bars3 = ax3.bar(bar_labels, pass_rates, color=colors3, edgecolor="white", linewidth=1.2)

        for bar, rate, n in zip(bars3, pass_rates, bar_counts):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{rate:.0f}%\n(n={n})",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
        ax3.set_ylim(0, 115)
    else:
        ax3.text(0.5, 0.5, "No relevance data", ha="center", va="center", fontsize=12, transform=ax3.transAxes)

    ax3.set_title("Answer Relevance Pass Rate", fontsize=13, fontweight="bold", pad=12)
    ax3.set_xlabel("Language", fontsize=11)
    ax3.set_ylabel("Pass Rate (%)", fontsize=11)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax3.set_axisbelow(True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard saved to: {output_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(metrics: dict) -> None:
    """Print a concise text summary of the aggregated metrics."""
    print("\n--- Technical Performance Summary ---")

    # Latency
    latency = metrics["latency_by_lang"]
    if latency:
        print("\n  Average Latency:")
        for lang in sorted(latency):
            vals = latency[lang]
            avg = sum(vals) / len(vals)
            print(f"    {lang.upper():>4}: {avg:6.2f}s  (n={len(vals)})")

    # Hallucination
    h_pass = metrics["hallucination_pass"]
    h_total = metrics["hallucination_total"]
    if h_total:
        pct = h_pass / h_total * 100
        print(f"\n  Hallucination Pass Rate: {h_pass}/{h_total} = {pct:.1f}%")
    else:
        print("\n  Hallucination Pass Rate: No data")

    # Relevance
    rel_pass = metrics["relevance_pass_by_lang"]
    rel_total = metrics["relevance_total_by_lang"]
    all_langs = sorted(set(rel_total.keys()))
    if all_langs:
        print("\n  Answer Relevance Pass Rate:")
        for lang in all_langs:
            t = rel_total[lang]
            p = rel_pass.get(lang, 0)
            pct = p / t * 100 if t else 0
            print(f"    {lang.upper():>4}: {p}/{t} = {pct:.1f}%")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Find the most recent results CSV automatically.
    csv_path = _find_latest_csv(REPORTS_DIR)

    # Derive a matching dashboard filename from the CSV timestamp.
    timestamp = _extract_timestamp(csv_path)
    dashboard_path = REPORTS_DIR / f"technical_dashboard_{timestamp}.png"

    print(f"Reading results from: {csv_path}")
    rows = _load_csv_rows(csv_path)
    print(f"Loaded {len(rows)} result row(s).")

    if not rows:
        print("ERROR: CSV contains no data rows.")
        sys.exit(1)

    metrics = _aggregate_metrics(rows)
    _print_summary(metrics)
    _generate_dashboard(metrics, dashboard_path)


if __name__ == "__main__":
    main()
