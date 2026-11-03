#!/usr/bin/env python3
"""Plot benchmark accuracy as a function of Huginn recurrence depth.

The input is evaluations/results.json, so the plot always reflects the
checkpointed evaluation records rather than a manually copied table.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Keep Matplotlib's cache inside the writable project directory on cluster nodes.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt


# ----------------------------- Configuration -----------------------------

RESULTS_FILE = Path(__file__).with_name("results.json")
OUTPUT_PNG = Path(__file__).with_name("recurrence_results.png")
OUTPUT_PDF = Path(__file__).with_name("recurrence_results.pdf")
RECURRENCE_VALUES = [1, 4, 8, 16, 32, 64]
FIGURE_SIZE = (9.0, 5.5)
DPI = 200
SHOW_ERROR_BARS = True

# ARC-Challenge (blue) and OpenBookQA (green) follow Figure 1's colors.
# The remaining colors are distinct additions for the new benchmarks.
BENCHMARKS = {
    "arc_challenge": {"label": "ARC Challenge", "color": "#1f77b4"},
    "openbookqa": {"label": "OpenBookQA", "color": "#2ca02c"},
    "hellaswag": {"label": "HellaSwag", "color": "#9467bd"},
    "winogrande": {"label": "WinoGrande", "color": "#e01313"},
    "mmlu_humanities": {"label": "MMLU Humanities", "color": "#8c564b"},
}


def metric_stderr_name(metric_name: str) -> str:
    """Convert lm-eval's metric name, e.g. acc_norm,none, to its stderr key."""
    return metric_name.replace(",none", "_stderr,none")


def load_series(results: dict, task_name: str) -> tuple[list[int], list[float], list[float]]:
    """Extract one task's completed scores and standard errors from lm-eval records."""
    by_recurrence = {
        run["recurrence"]: run
        for run in results["runs"]
        if run["task"] == task_name
    }

    recurrences, scores, errors = [], [], []
    for recurrence in RECURRENCE_VALUES:
        run = by_recurrence.get(recurrence)
        if run is None:
            continue
        metric_name = run["primary_metric"]
        task_results = run["results"][task_name]
        recurrences.append(recurrence)
        scores.append(100 * task_results[metric_name])
        errors.append(100 * task_results.get(metric_stderr_name(metric_name), 0.0))
    return recurrences, scores, errors


def main() -> None:
    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))

    fig, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    for task_name, style in BENCHMARKS.items():
        recurrences, scores, errors = load_series(results, task_name)
        if not recurrences:
            print(f"Skipping {task_name}: no completed results.")
            continue
        kwargs = {"color": style["color"], "label": style["label"], "marker": "o", "linewidth": 2.2}
        if SHOW_ERROR_BARS:
            axis.errorbar(recurrences, scores, yerr=errors, capsize=3, **kwargs)
        else:
            axis.plot(recurrences, scores, **kwargs)

    axis.set_xscale("log", base=2)
    axis.set_xticks(RECURRENCE_VALUES, [str(value) for value in RECURRENCE_VALUES])
    axis.set_xlabel("Test-time compute recurrence")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Scaling Test-Time Compute with Recurrent Depth")
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(title="Benchmark", frameon=False, ncols=2, loc="lower right")

    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
