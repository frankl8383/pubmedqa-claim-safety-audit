"""
Plot PubMedQA context-use sanity-check results.

Usage:
    uv run python analysis/plot_context_use_sanity.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODE_LABELS = {
    "question_context": "Question + context",
    "question_only": "Question only",
    "context_only": "Context only",
    "question_shuffled_context_eval": "Shuffled context",
}
MODEL_LABELS = {
    "tfidf_lr": "TF-IDF LR",
    "frozen_biomedbert": "Frozen BiomedBERT",
}
MODE_ORDER = ["question_context", "question_only", "context_only", "question_shuffled_context_eval"]
MODEL_ORDER = ["tfidf_lr", "frozen_biomedbert"]
COLORS = {
    "question_context": "#4C78A8",
    "question_only": "#72B7B2",
    "context_only": "#59A14F",
    "question_shuffled_context_eval": "#E15759",
}


def save_all(fig, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot context-use sanity check")
    parser.add_argument("--metrics", default="outputs/context_use_sanity/pubmedqa_context_use_metrics.tsv")
    parser.add_argument("--outdir", default="outputs/figures/context_use_sanity")
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

    df = pd.read_csv(args.metrics, sep="\t")
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["mode_label"] = df["mode"].map(MODE_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    x = np.arange(len(MODEL_ORDER))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MODE_ORDER))
    for offset, mode in zip(offsets, MODE_ORDER):
        rows = df.set_index(["model", "mode"])
        values = [rows.loc[(model, mode), "macro_f1"] for model in MODEL_ORDER]
        ax.bar(x + offset, values, width=width, color=COLORS[mode], label=MODE_LABELS[mode])
    ax.axhline(0.2375, color="#777777", linestyle="--", linewidth=1.0, label="Majority")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER], rotation=0)
    ax.set_ylim(0, 0.52)
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("A  Partial-input performance")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=1, fontsize=7)

    ax = axes[1]
    summary = pd.read_csv("outputs/context_use_sanity/pubmedqa_context_use_summary.tsv", sep="\t")
    summary = summary[summary["model"].isin(MODEL_ORDER)].copy()
    comparison_order = [
        "question_only vs question_context",
        "context_only vs question_context",
        "question_shuffled_context_eval vs question_context",
    ]
    y_positions = []
    labels = []
    values = []
    colors = []
    for model_idx, model in enumerate(MODEL_ORDER):
        for comp_idx, comp in enumerate(comparison_order):
            row = summary[(summary["model"] == model) & (summary["comparison"] == comp)].iloc[0]
            y_positions.append(model_idx * 4 + comp_idx)
            mode = comp.split(" vs ")[0]
            labels.append(f"{MODEL_LABELS[model]}\n{MODE_LABELS.get(mode, mode)}")
            values.append(row["macro_f1_delta"])
            colors.append(COLORS.get(mode, "#999999"))
    ax.barh(y_positions, values, color=colors, height=0.7)
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1 delta vs\nquestion + context")
    ax.set_xlim(-0.18, 0.04)
    ax.set_title("B  Context dependence")

    fig.suptitle("PubMedQA context-use sanity check", y=1.02, fontsize=11)
    fig.tight_layout()
    save_all(fig, Path(args.outdir), "context_use_sanity")

    readme = """# Context-Use Sanity Figure

Panel A compares validation macro-F1 under question+context, question-only,
context-only, and shuffled-context evaluation. Panel B shows the macro-F1
delta relative to question+context for each model family.

The key audit finding is that question-only baselines are unexpectedly strong,
while shuffled context reduces macro-F1, especially for frozen BiomedBERT.
"""
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote context-use sanity figure to {args.outdir}")


if __name__ == "__main__":
    main()
