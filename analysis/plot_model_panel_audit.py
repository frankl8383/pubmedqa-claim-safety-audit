"""
Plot BMC-facing model-panel audit figures.

This figure summarizes whether context-use artifacts and calibration trade-offs
appear across sparse, general, scientific, biomedical, and clinical reference
models. It is designed as a BMC MIDM main/supplement figure, not as a
leaderboard plot.

Usage:
    uv run python analysis/plot_model_panel_audit.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "outputs" / "model_panel_audit"
OUT = ROOT / "outputs" / "figures" / "model_panel_audit"

PALETTE = {
    "sparse": "#7A7A7A",
    "general": "#5B8DB8",
    "scientific": "#62A87C",
    "biomedical": "#B07AA1",
    "biomedical-link": "#D08C60",
    "clinical": "#8C6BB1",
}

MODE_COLORS = {
    "question_only vs question_context": "#5B8DB8",
    "context_only vs question_context": "#62A87C",
    "question_shuffled_context_eval vs question_context": "#D08C60",
}

MODE_LABELS = {
    "question_only vs question_context": "Question only",
    "context_only vs question_context": "Context only",
    "question_shuffled_context_eval vs question_context": "Shuffled context",
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.labelsize": 9,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
    })


def short_name(name: str) -> str:
    return name


def panel_label(ax, label: str) -> None:
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")


def panel_a(ax, qc: pd.DataFrame) -> None:
    qc = qc.sort_values("macro_f1", ascending=True)
    y = np.arange(len(qc))
    colors = [PALETTE.get(family, "#999999") for family in qc["family"]]
    ax.barh(y, qc["macro_f1"], color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels([short_name(x) for x in qc["display_name"]])
    ax.set_xlabel("Validation macro-F1")
    ax.set_title("Model-panel downstream behavior", loc="left")
    panel_label(ax, "a")
    ax.set_xlim(0, max(0.5, float(qc["macro_f1"].max()) + 0.05))
    for yi, val in zip(y, qc["macro_f1"]):
        ax.text(val + 0.008, yi, f"{val:.2f}", va="center", ha="left", fontsize=7)


def panel_b(ax, deltas: pd.DataFrame) -> None:
    order = (
        deltas[deltas["comparison"].eq("question_only vs question_context")]
        .sort_values("macro_f1_delta")["display_name"].tolist()
    )
    comparisons = list(MODE_COLORS)
    x = np.arange(len(order))
    width = 0.24
    for i, comp in enumerate(comparisons):
        sub = deltas[deltas["comparison"].eq(comp)].set_index("display_name").reindex(order)
        ax.bar(x + (i - 1) * width, sub["macro_f1_delta"], width=width, color=MODE_COLORS[comp], label=MODE_LABELS[comp])
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(x) for x in order], rotation=30, ha="right")
    ax.set_ylabel("Macro-F1 delta")
    ax.set_title("Partial-input and shuffled-context audit", loc="left")
    ax.legend(fontsize=6.8, ncol=1, loc="upper left")
    panel_label(ax, "b")


def panel_c(ax, qc: pd.DataFrame) -> None:
    for _, row in qc.iterrows():
        ax.scatter(row["macro_f1"], row["brier"], s=46, color=PALETTE.get(row["family"], "#999999"), edgecolor="white", linewidth=0.7)
        ax.text(row["macro_f1"] + 0.004, row["brier"] + 0.002, short_name(row["display_name"]), fontsize=7)
    ax.set_xlabel("Macro-F1")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Performance-calibration trade-off", loc="left")
    panel_label(ax, "c")


def panel_d(ax, qc: pd.DataFrame) -> None:
    qc = qc.sort_values("macro_f1", ascending=False)
    labels = ["pred_frac_yes", "pred_frac_no", "pred_frac_maybe"]
    colors = ["#C9D6E8", "#B9D9C0", "#E8C7B0"]
    bottom = np.zeros(len(qc))
    x = np.arange(len(qc))
    for label, color in zip(labels, colors):
        vals = qc[label].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, color=color, width=0.72, label=label.replace("pred_frac_", ""))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(x) for x in qc["display_name"]], rotation=30, ha="right")
    ax.set_ylabel("Predicted-label fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Predicted-label distribution", loc="left")
    ax.legend(fontsize=6.8, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    panel_label(ax, "d")


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(IN / "model_panel_metrics.tsv", sep="\t")
    deltas = pd.read_csv(IN / "model_panel_context_deltas.tsv", sep="\t")
    qc = metrics[metrics["input_mode"].eq("question_context")].copy()
    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.28], height_ratios=[1.0, 1.0])
    panel_a(fig.add_subplot(gs[0, 0]), qc)
    panel_b(fig.add_subplot(gs[0, 1]), deltas)
    panel_c(fig.add_subplot(gs[1, 0]), qc)
    panel_d(fig.add_subplot(gs[1, 1]), qc)
    for ext in ("png", "pdf", "svg"):
        path = OUT / f"model_panel_audit.{ext}"
        fig.savefig(path, dpi=320 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    readme = """# Model-panel audit figure

Generated by `analysis/plot_model_panel_audit.py`.

The figure is a claim-safety audit, not a leaderboard. It shows model-panel
macro-F1, partial-input sensitivity, calibration trade-offs, and predicted-label
distributions across sparse, general, scientific, biomedical, and clinical
reference models.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote model-panel figure to {OUT}")


if __name__ == "__main__":
    main()
