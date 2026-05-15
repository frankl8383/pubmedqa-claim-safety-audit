#!/usr/bin/env python3
"""Plot same-trained context-ablation summary for the BMC MIDM manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "same_trained_context_ablation" / "same_trained_context_ablation_summary.tsv"
OUT_DIR = ROOT / "outputs" / "figures" / "same_trained_context_ablation"


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 1.1
plt.rcParams["legend.frameon"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


MODE_LABELS = {
    "question_context": "Question + context",
    "question_only": "Question only",
    "context_removed": "No context block",
    "context_only": "Context only",
    "shuffled_context": "Shuffled context",
    "question_shuffled": "Shuffled question",
}

CONFIG_LABELS = {
    "random_primary_best_7147e14": "Random primary",
    "agent_aux_a20f5b7": "Agent auxiliary",
}

PALETTE = {
    "random_primary_best_7147e14": "#6B7A90",
    "agent_aux_a20f5b7": "#B75D69",
}


def main() -> None:
    df = pd.read_csv(SUMMARY, sep="\t")
    modes = [
        "question_context",
        "question_only",
        "context_removed",
        "context_only",
        "shuffled_context",
        "question_shuffled",
    ]
    configs = ["random_primary_best_7147e14", "agent_aux_a20f5b7"]
    df["prompt_mode"] = pd.Categorical(df["prompt_mode"], categories=modes, ordered=True)
    df = df.sort_values(["config_name", "prompt_mode"])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=False)
    x = np.arange(len(modes))
    width = 0.36

    metrics = [
        ("macro_f1_mean", "Macro-F1", axes[0, 0]),
        ("brier_mean", "Brier score", axes[0, 1]),
        ("f1_no_mean", "No-class F1", axes[1, 0]),
        ("pred_no_frac_mean", "Predicted no fraction", axes[1, 1]),
    ]
    for metric, ylabel, ax in metrics:
        for offset, config in zip([-width / 2, width / 2], configs):
            part = df[df["config_name"].eq(config)].set_index("prompt_mode").loc[modes]
            y = part[metric].to_numpy()
            sd_col = metric.replace("_mean", "_sd")
            yerr = part[sd_col].fillna(0).to_numpy() if sd_col in part else None
            ax.bar(
                x + offset,
                y,
                width=width,
                yerr=yerr,
                capsize=2,
                color=PALETTE[config],
                edgecolor="white",
                linewidth=0.6,
                label=CONFIG_LABELS[config],
            )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[m] for m in modes], rotation=35, ha="right")
        ax.axvline(0.5, color="#D0D0D0", linewidth=0.8)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995))
    for label, ax in zip(["a", "b", "c", "d"], axes.flatten()):
        ax.text(-0.12, 0.98, label, transform=ax.transAxes, fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"same_trained_context_ablation.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
