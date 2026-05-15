"""
Plot validation BPB against downstream PubMedQA accuracy and macro-F1.

The figure is intended for the manuscript's proxy-metric limitation result:
lower language-model validation BPB does not necessarily imply better
minority-class downstream behavior.

Usage:
    uv run python analysis/plot_proxy_metric_mismatch.py \
      --results results_repeated_controls.tsv \
      --outdir outputs/figures/proxy_metric_mismatch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CONFIG_LABELS = {
    "manual_baseline_f1664dd": "Manual baseline",
    "manual_warmup10_da0044c": "Manual warmup",
    "random_accuracy_best_7e73835": "Random accuracy",
    "random_brier_best_8dc23be": "Random Brier",
    "random_macro_f1_best_8a1209b": "Random macro-F1",
    "random_primary_best_7147e14": "Random primary",
    "agent_aux_a20f5b7": "Agent auxiliary",
}

GROUP_COLORS = {
    "manual": "#6F6F6F",
    "random": "#C76D2A",
    "agent": "#2E76A8",
}

GROUP_MARKERS = {
    "manual": "s",
    "random": "o",
    "agent": "^",
}

HIGHLIGHT_CONFIGS = {
    "random_primary_best_7147e14",
    "agent_aux_a20f5b7",
}

PANEL_METRICS = [
    ("pubmedqa_acc", "Accuracy", 0.5533333333333333),
    ("pubmedqa_macro_f1", "Macro-F1", 0.23748211731044352),
]


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_all(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"{stem}.{ext}", bbox_inches="tight")


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config_name, group), subset in results.groupby(["config_name", "group"], sort=False):
        rows.append({
            "config_name": config_name,
            "display_name": CONFIG_LABELS.get(config_name, config_name),
            "group": group,
            "n_seeds": int(subset["seed"].nunique()),
            "lm_val_bpb_mean": subset["lm_val_bpb"].mean(),
            "lm_val_bpb_sd": subset["lm_val_bpb"].std(ddof=1),
            "accuracy_mean": subset["pubmedqa_acc"].mean(),
            "accuracy_sd": subset["pubmedqa_acc"].std(ddof=1),
            "macro_f1_mean": subset["pubmedqa_macro_f1"].mean(),
            "macro_f1_sd": subset["pubmedqa_macro_f1"].std(ddof=1),
            "brier_mean": subset["pubmedqa_brier"].mean(),
            "brier_sd": subset["pubmedqa_brier"].std(ddof=1),
        })
    summary = pd.DataFrame(rows)
    summary["lm_val_bpb_rank"] = summary["lm_val_bpb_mean"].rank(method="min", ascending=True).astype(int)
    summary["accuracy_rank"] = summary["accuracy_mean"].rank(method="min", ascending=False).astype(int)
    summary["macro_f1_rank"] = summary["macro_f1_mean"].rank(method="min", ascending=False).astype(int)
    summary["brier_rank"] = summary["brier_mean"].rank(method="min", ascending=True).astype(int)
    return summary.sort_values(["lm_val_bpb_mean", "config_name"]).reset_index(drop=True)


def write_summary_files(results: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    mean_columns = {
        "pubmedqa_acc": "accuracy_mean",
        "pubmedqa_macro_f1": "macro_f1_mean",
        "pubmedqa_brier": "brier_mean",
    }
    correlations = []
    for metric, label, _majority in PANEL_METRICS + [("pubmedqa_brier", "Brier", None)]:
        correlations.append({
            "metric": metric,
            "display_name": label,
            "seed_level_n": len(results),
            "seed_level_spearman": results["lm_val_bpb"].corr(results[metric], method="spearman"),
            "seed_level_pearson": results["lm_val_bpb"].corr(results[metric], method="pearson"),
            "config_mean_n": len(summary),
            "config_mean_spearman": summary["lm_val_bpb_mean"].corr(
                summary[mean_columns[metric]],
                method="spearman",
            ),
        })
    pd.DataFrame(correlations).to_csv(outdir / "proxy_metric_correlations.tsv", sep="\t", index=False)
    summary.to_csv(outdir / "proxy_metric_config_summary.tsv", sep="\t", index=False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def add_key_callouts(ax: plt.Axes, metric: str, summary: pd.DataFrame) -> None:
    offsets = {
        "pubmedqa_acc": {
            "random_primary_best_7147e14": (0.038, -0.016),
            "agent_aux_a20f5b7": (-0.14, -0.028),
        },
        "pubmedqa_macro_f1": {
            "random_primary_best_7147e14": (0.034, 0.014),
            "agent_aux_a20f5b7": (-0.14, 0.018),
        },
    }
    for config_name, (dx, dy) in offsets[metric].items():
        row = summary[summary["config_name"] == config_name].iloc[0]
        y_col = "accuracy_mean" if metric == "pubmedqa_acc" else "macro_f1_mean"
        ax.annotate(
            CONFIG_LABELS[config_name],
            xy=(row["lm_val_bpb_mean"], row[y_col]),
            xytext=(row["lm_val_bpb_mean"] + dx, row[y_col] + dy),
            textcoords="data",
            arrowprops={
                "arrowstyle": "-",
                "color": "#444444",
                "linewidth": 0.8,
                "shrinkA": 2,
                "shrinkB": 4,
            },
            fontsize=8,
            color="#222222",
            ha="left",
        )


def plot_panel(
    ax: plt.Axes,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str,
    label: str,
    majority: float,
) -> None:
    y_col = "accuracy_mean" if metric == "pubmedqa_acc" else "macro_f1_mean"
    y_sd_col = "accuracy_sd" if metric == "pubmedqa_acc" else "macro_f1_sd"

    for config_name, subset in results.groupby("config_name", sort=False):
        group = subset["group"].iloc[0]
        color = GROUP_COLORS.get(group, "#333333")
        marker = GROUP_MARKERS.get(group, "o")
        ax.scatter(
            subset["lm_val_bpb"],
            subset[metric],
            s=22,
            marker=marker,
            color=color,
            alpha=0.14,
            linewidths=0,
            zorder=1,
        )

    for _, row in summary.iterrows():
        group = row["group"]
        is_highlight = row["config_name"] in HIGHLIGHT_CONFIGS
        ax.errorbar(
            row["lm_val_bpb_mean"],
            row[y_col],
            xerr=row["lm_val_bpb_sd"],
            yerr=row[y_sd_col],
            fmt=GROUP_MARKERS.get(group, "o"),
            markersize=8.0 if is_highlight else 6.2,
            color=GROUP_COLORS.get(group, "#333333"),
            ecolor=GROUP_COLORS.get(group, "#333333"),
            markeredgecolor="#111111" if is_highlight else "white",
            markeredgewidth=0.9 if is_highlight else 0.5,
            elinewidth=1.1,
            capsize=2.8,
            alpha=0.96,
            zorder=3,
        )

    rho = results["lm_val_bpb"].corr(results[metric], method="spearman")
    pearson = results["lm_val_bpb"].corr(results[metric], method="pearson")
    ax.axhline(majority, color="#555555", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.text(
        0.99,
        majority + (0.004 if metric == "pubmedqa_acc" else 0.006),
        f"majority {label.lower()}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    ax.text(
        0.03,
        0.95,
        f"seed-level Spearman rho = {rho:+.2f}\nPearson r = {pearson:+.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.26", "alpha": 0.92},
    )
    ax.set_xlabel("Language-model validation BPB (lower is better)")
    ax.set_ylabel(f"PubMedQA validation {label}")
    ax.set_title(label)
    ax.grid(color="#DDDDDD", linewidth=0.7, alpha=0.8)
    ax.set_xlim(2.02, 2.66)
    arrow_y = 0.08 if metric == "pubmedqa_acc" else 0.16
    ax.annotate(
        "better BPB",
        xy=(2.07, arrow_y),
        xycoords=("data", "axes fraction"),
        xytext=(2.32, arrow_y),
        textcoords=("data", "axes fraction"),
        arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.9},
        fontsize=8,
        color="#444444",
        va="center",
    )
    add_key_callouts(ax, metric, summary)


def add_rank_inversion_note(ax: plt.Axes, summary: pd.DataFrame) -> None:
    proxy_winner = summary.sort_values("lm_val_bpb_rank").iloc[0]
    downstream_winner = summary.sort_values("macro_f1_rank").iloc[0]
    text = (
        f"BPB winner ranks {int(proxy_winner['macro_f1_rank'])}/7 by macro-F1\n"
        f"macro-F1 winner ranks {int(downstream_winner['lm_val_bpb_rank'])}/7 by BPB"
    )
    ax.text(
        0.98,
        0.95,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#222222",
        bbox={"facecolor": "#F7F7F7", "edgecolor": "#BDBDBD", "boxstyle": "round,pad=0.28"},
    )


def plot_proxy_mismatch(results: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.78, bottom=0.26, wspace=0.20)
    for idx, (ax, (metric, label, majority)) in enumerate(zip(axes, PANEL_METRICS)):
        plot_panel(ax, results, summary, metric, label, majority)
        panel_label(ax, chr(ord("A") + idx))
    add_rank_inversion_note(axes[1], summary)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=GROUP_MARKERS[group],
            color="none",
            markerfacecolor=color,
            markeredgecolor="#111111",
            markersize=7,
            label=group.capitalize(),
        )
        for group, color in GROUP_COLORS.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.105),
    )
    save_all(fig, outdir, "proxy_metric_mismatch")
    return fig


def write_manifest(outdir: Path) -> None:
    files = sorted(path.name for path in outdir.glob("*") if path.is_file())
    lines = [
        "# Proxy Metric Mismatch Figure Package",
        "",
        "Generated by `analysis/plot_proxy_metric_mismatch.py`.",
        "",
        "## Files",
        "",
    ]
    lines.extend(f"- `{name}`" for name in files)
    lines.extend([
        "",
        "## Notes",
        "",
        "- BPB is plotted on the x-axis; lower values are better for the language-model validation proxy.",
        "- Small points are individual repeated-seed runs.",
        "- Large points show five-seed means with standard-deviation error bars.",
        "- The figure highlights that the BPB-selected random primary run is not the macro-F1 winner.",
    ])
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PubMedQA proxy-metric mismatch")
    parser.add_argument("--results", default="results_repeated_controls.tsv")
    parser.add_argument("--outdir", default="outputs/figures/proxy_metric_mismatch")
    args = parser.parse_args()

    set_style()
    results = pd.read_csv(args.results, sep="\t")
    results = results[results["split"] == "val"].copy()
    if "phase" in results.columns:
        results = results[results["phase"] == "repeated_val"].copy()
    summary = summarize(results)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_summary_files(results, summary, outdir)
    fig = plot_proxy_mismatch(results, summary, outdir)
    plt.close(fig)
    write_manifest(outdir)
    print(f"wrote proxy metric mismatch figure package to {outdir}")


if __name__ == "__main__":
    main()
