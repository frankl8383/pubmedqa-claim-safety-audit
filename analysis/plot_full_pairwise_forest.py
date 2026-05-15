"""
Create a full-control paired bootstrap forest plot for the BMC MIDM audit manuscript.

The figure avoids the earlier cherry-picking risk by showing agent auxiliary
against the strongest random/manual/reference controls, not only against the
BPB-selected random primary control.

Usage:
    uv run python analysis/plot_full_pairwise_forest.py \
      --table outputs/manuscript_tables/table7_full_pairwise_validation_audit.tsv \
      --outdir outputs/figures/full_pairwise_forest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPARISONS = [
    "Agent auxiliary - Random primary",
    "Agent auxiliary - Random macro-F1 best",
    "Agent auxiliary - Manual baseline",
    "Agent auxiliary - TF-IDF LR",
    "Agent auxiliary - Frozen BiomedBERT",
]

METRICS = [
    ("macro_f1", "Macro-F1", "Higher is better"),
    ("f1_no", "F1-no", "Higher is better"),
    ("accuracy", "Accuracy", "Higher is better"),
    ("brier", "Brier", "Lower is better; positive is worse for agent"),
]

COMPARISON_LABELS = {
    "Agent auxiliary - Random primary": "Random primary",
    "Agent auxiliary - Random macro-F1 best": "Random macro-F1 best",
    "Agent auxiliary - Manual baseline": "Manual baseline",
    "Agent auxiliary - TF-IDF LR": "TF-IDF LR",
    "Agent auxiliary - Frozen BiomedBERT": "Frozen BiomedBERT",
}

COLORS = {
    "Agent auxiliary - Random primary": "#5E81AC",
    "Agent auxiliary - Random macro-F1 best": "#74A9A4",
    "Agent auxiliary - Manual baseline": "#777777",
    "Agent auxiliary - TF-IDF LR": "#8DA781",
    "Agent auxiliary - Frozen BiomedBERT": "#B79B73",
}


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def read_rows(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    keep = table[
        table["comparison"].isin(COMPARISONS)
        & table["metric"].isin([metric for metric, _, _ in METRICS])
    ].copy()
    keep["comparison"] = pd.Categorical(keep["comparison"], COMPARISONS, ordered=True)
    keep["metric"] = pd.Categorical(keep["metric"], [metric for metric, _, _ in METRICS], ordered=True)
    keep = keep.sort_values(["metric", "comparison"]).reset_index(drop=True)
    return keep


def build_plot_table(rows: pd.DataFrame) -> pd.DataFrame:
    plot_rows = []
    for metric, metric_label, interpretation in METRICS:
        for comparison in COMPARISONS:
            subset = rows[(rows["metric"] == metric) & (rows["comparison"] == comparison)]
            if subset.empty:
                raise ValueError(f"missing row for {comparison} / {metric}")
            row = subset.iloc[0]
            plot_rows.append({
                "comparison": comparison,
                "comparison_label": COMPARISON_LABELS[comparison],
                "metric": metric,
                "metric_label": metric_label,
                "interpretation": interpretation,
                "diff": float(row["diff_left_minus_right"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "p_two_sided_sign": float(row["p_two_sided_sign"]),
                "n_examples": int(row["n_examples"]),
                "n_bootstrap": int(row["n_bootstrap"]),
            })
    return pd.DataFrame(plot_rows)


def save_all(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"{stem}.{ext}", bbox_inches="tight")


def plot_forest(plot_df: pd.DataFrame, outdir: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.65), sharey=True)
    y = np.arange(len(COMPARISONS))[::-1]
    y_labels = [COMPARISON_LABELS[c] for c in COMPARISONS]

    for ax, (metric, metric_label, interpretation) in zip(axes, METRICS):
        sub = plot_df[plot_df["metric"] == metric].copy()
        sub = sub.set_index("comparison").loc[COMPARISONS].reset_index()
        x = sub["diff"].to_numpy(dtype=float)
        low = sub["ci_low"].to_numpy(dtype=float)
        high = sub["ci_high"].to_numpy(dtype=float)
        xerr = np.vstack([x - low, high - x])

        for idx, row in sub.iterrows():
            ax.errorbar(
                row["diff"],
                y[idx],
                xerr=np.array([[row["diff"] - row["ci_low"]], [row["ci_high"] - row["diff"]]]),
                fmt="o",
                color=COLORS[row["comparison"]],
                ecolor=COLORS[row["comparison"]],
                elinewidth=1.15,
                capsize=2.7,
                markersize=4.6,
                zorder=3,
            )

        ax.axvline(0, color="#333333", linewidth=0.85, linestyle="--", zorder=1)
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.65, zorder=0)
        ax.set_title(metric_label)
        ax.set_xlabel("Aux - comparator")
        if metric == "brier":
            ax.text(
                0.5,
                -0.25,
                ">0 means worse for agent",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7.2,
                color="#8C2D04",
            )
        else:
            ax.text(
                0.5,
                -0.25,
                ">0 favors agent",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7.2,
                color="#2F5D3A",
            )

        pad = max(abs(low).max(), abs(high).max()) * 1.15
        pad = max(pad, 0.08)
        ax.set_xlim(-pad, pad)

    axes[0].set_yticks(y, y_labels)
    axes[0].set_ylabel("Comparator")
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)

    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    save_all(fig, outdir, "full_pairwise_forest")
    plt.close(fig)


def write_readme(outdir: Path) -> None:
    text = """# Full Pairwise Forest Plot

This figure shows agent auxiliary against the main random, manual, and
reference controls. It is intended to replace the narrower forest plot that
focused mainly on random primary and majority.

Important interpretation:

- Positive macro-F1, F1-no, and accuracy differences favor agent auxiliary.
- Positive Brier differences are worse for agent auxiliary because lower Brier
  is better.
- The figure supports a narrow class-behavior trade-off claim, not broad agent
  superiority.
"""
    (outdir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="outputs/manuscript_tables/table7_full_pairwise_validation_audit.tsv")
    parser.add_argument("--outdir", default="outputs/figures/full_pairwise_forest")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.table))
    plot_df = build_plot_table(rows)
    plot_df.to_csv(outdir / "full_pairwise_forest_plot_data.tsv", sep="\t", index=False)
    plot_forest(plot_df, outdir)
    write_readme(outdir)
    print(f"wrote full pairwise forest plot to {outdir}")


if __name__ == "__main__":
    main()
