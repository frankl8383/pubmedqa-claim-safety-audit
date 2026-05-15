"""
Plot PubMedQA calibration sanity-check results.

Usage:
    uv run python analysis/plot_calibration_sanity.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PLOT_MODELS = [
    "majority_train",
    "random_primary_best_7147e14",
    "agent_aux_a20f5b7",
    "tfidf_lr_selected",
    "frozen_biomedbert",
]
LABELS = {
    "majority_train": "Majority",
    "random_primary_best_7147e14": "Random primary",
    "agent_aux_a20f5b7": "Agent auxiliary",
    "tfidf_lr_selected": "TF-IDF LR",
    "frozen_biomedbert": "Frozen BiomedBERT",
}
COLORS = {
    "majority_train": "#7F7F7F",
    "random_primary_best_7147e14": "#E69F00",
    "agent_aux_a20f5b7": "#0072B2",
    "tfidf_lr_selected": "#009E73",
    "frozen_biomedbert": "#CC79A7",
}


def save_all(fig, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot calibration sanity check")
    parser.add_argument("--metrics", default="outputs/calibration_sanity/pubmedqa_calibration_metrics.tsv")
    parser.add_argument("--bins", default="outputs/calibration_sanity/pubmedqa_reliability_bins.tsv")
    parser.add_argument("--outdir", default="outputs/figures/calibration_sanity")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 1.1
    plt.rcParams["legend.frameon"] = False

    metrics = pd.read_csv(args.metrics, sep="\t")
    bins = pd.read_csv(args.bins, sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), gridspec_kw={"width_ratios": [1.18, 1.0]})

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1.0)
    for model in PLOT_MODELS:
        sub = bins[(bins["model"] == model) & (bins["n"] > 0)].copy()
        if sub.empty:
            continue
        sizes = 18 + 110 * sub["n"] / sub["n"].max()
        ax.scatter(
            sub["mean_confidence"],
            sub["accuracy"],
            s=sizes,
            color=COLORS[model],
            alpha=0.76,
            edgecolor="white",
            linewidth=0.4,
            label=LABELS[model],
        )
        ax.plot(sub["mean_confidence"], sub["accuracy"], color=COLORS[model], alpha=0.45, linewidth=1.0)
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("A  Reliability bins")

    ax = axes[1]
    m = metrics[metrics["model"].isin(PLOT_MODELS)].copy()
    m["_order"] = m["model"].map({model: i for i, model in enumerate(PLOT_MODELS)})
    m = m.sort_values("_order")
    y = np.arange(len(m))
    ax.barh(y - 0.18, m["ece"], height=0.34, color="#4C78A8", label="ECE")
    ax.barh(y + 0.18, m["high_conf_wrong_rate"], height=0.34, color="#E15759", label="High-conf wrong rate")
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[model] for model in m["model"]], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.48, float(max(m["ece"].max(), m["high_conf_wrong_rate"].max()) * 1.12)))
    ax.set_xlabel("Rate")
    ax.set_title("B  Calibration risk")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.18), ncol=2, fontsize=7, borderaxespad=0.0)

    fig.suptitle("PubMedQA calibration sanity check", y=1.02, fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.32, -0.05),
        ncol=3,
        fontsize=7.5,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.98), w_pad=2.6)
    save_all(fig, Path(args.outdir), "calibration_sanity")

    readme = """# Calibration Sanity Figure

Panel A shows top-label reliability bins. The dashed diagonal indicates perfect
top-label calibration. Bubble size is proportional to the number of predictions
in each bin. Panel B compares expected calibration error (ECE) and the fraction
of all predictions that were wrong with confidence >= 0.8.
"""
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote calibration sanity figure to {args.outdir}")


if __name__ == "__main__":
    main()
