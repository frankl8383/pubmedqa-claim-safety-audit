"""
Compute formal proxy-metric mismatch audit tables for repeated PubMedQA runs.

The audit asks whether selecting by language-model validation BPB would select
the same configurations as downstream PubMedQA metrics. It reports rank gaps,
direction mismatches, and pairwise Pareto conflicts.

Usage:
    uv run python analysis/audit_proxy_metric_mismatch.py \
      --results results_repeated_controls.tsv \
      --outdir outputs/innovation_audit/proxy_metric_mismatch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DISPLAY_NAMES = {
    "manual_baseline_f1664dd": "Manual baseline",
    "manual_warmup10_da0044c": "Manual warmup",
    "random_accuracy_best_7e73835": "Random accuracy",
    "random_brier_best_8dc23be": "Random Brier",
    "random_macro_f1_best_8a1209b": "Random macro-F1",
    "random_primary_best_7147e14": "Random primary",
    "agent_aux_a20f5b7": "Agent auxiliary",
    "prospective_aux_weight_0p02_20260506": "Prospective aux 0.02",
    "prospective_aux_weight_0p10_20260506": "Prospective aux 0.10",
}

DOWNSTREAM_METRICS = {
    "pubmedqa_acc": {"display": "Accuracy", "higher_is_better": True},
    "pubmedqa_macro_f1": {"display": "Macro-F1", "higher_is_better": True},
    "pubmedqa_brier": {"display": "Brier", "higher_is_better": False},
}


def rank_series(values: pd.Series, higher_is_better: bool) -> pd.Series:
    return values.rank(method="min", ascending=not higher_is_better).astype(int)


def summarize_configs(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby(["config_name", "group"], sort=False)
    summary = grouped.agg(
        n_seeds=("seed", "nunique"),
        lm_val_bpb_mean=("lm_val_bpb", "mean"),
        lm_val_bpb_sd=("lm_val_bpb", "std"),
        accuracy_mean=("pubmedqa_acc", "mean"),
        accuracy_sd=("pubmedqa_acc", "std"),
        macro_f1_mean=("pubmedqa_macro_f1", "mean"),
        macro_f1_sd=("pubmedqa_macro_f1", "std"),
        brier_mean=("pubmedqa_brier", "mean"),
        brier_sd=("pubmedqa_brier", "std"),
        tokens_M_mean=("tokens_M", "mean"),
    ).reset_index()
    summary["display_name"] = summary["config_name"].map(DISPLAY_NAMES).fillna(summary["config_name"])
    summary["proxy_rank_lm_val_bpb"] = rank_series(summary["lm_val_bpb_mean"], higher_is_better=False)
    summary["downstream_rank_accuracy"] = rank_series(summary["accuracy_mean"], higher_is_better=True)
    summary["downstream_rank_macro_f1"] = rank_series(summary["macro_f1_mean"], higher_is_better=True)
    summary["downstream_rank_brier"] = rank_series(summary["brier_mean"], higher_is_better=False)
    return summary.sort_values(["proxy_rank_lm_val_bpb", "config_name"]).reset_index(drop=True)


def downstream_column(metric: str) -> str:
    return {
        "pubmedqa_acc": "accuracy_mean",
        "pubmedqa_macro_f1": "macro_f1_mean",
        "pubmedqa_brier": "brier_mean",
    }[metric]


def downstream_rank_column(metric: str) -> str:
    return {
        "pubmedqa_acc": "downstream_rank_accuracy",
        "pubmedqa_macro_f1": "downstream_rank_macro_f1",
        "pubmedqa_brier": "downstream_rank_brier",
    }[metric]


def expected_spearman_sign(metric: str) -> int:
    # lm_val_bpb is lower-is-better. Higher-is-better downstream metrics should
    # therefore correlate negatively with BPB; lower-is-better metrics positively.
    return -1 if DOWNSTREAM_METRICS[metric]["higher_is_better"] else 1


def metric_value_better(left: float, right: float, metric: str) -> bool:
    if DOWNSTREAM_METRICS[metric]["higher_is_better"]:
        return left > right
    return left < right


def build_mismatch_summary(results: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    proxy_winner = summary.sort_values("lm_val_bpb_mean", ascending=True).iloc[0]
    k_configs = len(summary)
    for metric, meta in DOWNSTREAM_METRICS.items():
        value_col = downstream_column(metric)
        rank_col = downstream_rank_column(metric)
        downstream_winner = summary.sort_values(value_col, ascending=not meta["higher_is_better"]).iloc[0]
        seed_spearman = results["lm_val_bpb"].corr(results[metric], method="spearman")
        seed_pearson = results["lm_val_bpb"].corr(results[metric], method="pearson")
        config_spearman = summary["lm_val_bpb_mean"].corr(summary[value_col], method="spearman")
        expected_sign = expected_spearman_sign(metric)
        direction_mismatch = (seed_spearman * expected_sign) < 0
        proxy_rank_gap_for_downstream_winner = int(downstream_winner["proxy_rank_lm_val_bpb"]) - 1
        downstream_rank_gap_for_proxy_winner = int(proxy_winner[rank_col]) - 1
        normalized_rank_gap = (
            (proxy_rank_gap_for_downstream_winner + downstream_rank_gap_for_proxy_winner)
            / (2 * max(k_configs - 1, 1))
        )
        downstream_gain_over_proxy_winner = (
            downstream_winner[value_col] - proxy_winner[value_col]
            if meta["higher_is_better"]
            else proxy_winner[value_col] - downstream_winner[value_col]
        )

        rows.append({
            "downstream_metric": metric,
            "display_name": meta["display"],
            "higher_is_better": meta["higher_is_better"],
            "n_seed_runs": len(results),
            "n_configs": len(summary),
            "seed_level_spearman": seed_spearman,
            "seed_level_pearson": seed_pearson,
            "config_mean_spearman": config_spearman,
            "expected_spearman_sign": expected_sign,
            "direction_mismatch": direction_mismatch,
            "proxy_winner": proxy_winner["config_name"],
            "proxy_winner_display": proxy_winner["display_name"],
            "proxy_winner_downstream_rank": int(proxy_winner[rank_col]),
            "proxy_winner_downstream_value": proxy_winner[value_col],
            "downstream_winner": downstream_winner["config_name"],
            "downstream_winner_display": downstream_winner["display_name"],
            "downstream_winner_proxy_rank": int(downstream_winner["proxy_rank_lm_val_bpb"]),
            "downstream_winner_downstream_value": downstream_winner[value_col],
            "proxy_rank_gap_for_downstream_winner": proxy_rank_gap_for_downstream_winner,
            "downstream_rank_gap_for_proxy_winner": downstream_rank_gap_for_proxy_winner,
            "pmmi_normalized_rank_gap": normalized_rank_gap,
            "pmmi_downstream_gain_over_proxy_winner": downstream_gain_over_proxy_winner,
            "pmmi_direction_flag": bool(direction_mismatch),
            "rank_inversion": proxy_winner["config_name"] != downstream_winner["config_name"],
        })
    return pd.DataFrame(rows)


def build_pareto_conflicts(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    records = summary.to_dict("records")
    for metric, meta in DOWNSTREAM_METRICS.items():
        value_col = downstream_column(metric)
        for i, left in enumerate(records):
            for right in records[i + 1:]:
                left_worse_bpb = left["lm_val_bpb_mean"] > right["lm_val_bpb_mean"]
                right_worse_bpb = right["lm_val_bpb_mean"] > left["lm_val_bpb_mean"]
                if left_worse_bpb and metric_value_better(left[value_col], right[value_col], metric):
                    better_downstream = left
                    better_proxy = right
                elif right_worse_bpb and metric_value_better(right[value_col], left[value_col], metric):
                    better_downstream = right
                    better_proxy = left
                else:
                    continue
                rows.append({
                    "downstream_metric": metric,
                    "display_name": meta["display"],
                    "better_downstream_config": better_downstream["config_name"],
                    "better_downstream_display": better_downstream["display_name"],
                    "better_proxy_config": better_proxy["config_name"],
                    "better_proxy_display": better_proxy["display_name"],
                    "better_downstream_lm_val_bpb": better_downstream["lm_val_bpb_mean"],
                    "better_proxy_lm_val_bpb": better_proxy["lm_val_bpb_mean"],
                    "bpb_penalty": better_downstream["lm_val_bpb_mean"] - better_proxy["lm_val_bpb_mean"],
                    "better_downstream_value": better_downstream[value_col],
                    "better_proxy_value": better_proxy[value_col],
                    "downstream_gain": (
                        better_downstream[value_col] - better_proxy[value_col]
                        if meta["higher_is_better"]
                        else better_proxy[value_col] - better_downstream[value_col]
                    ),
                })
    return pd.DataFrame(rows)


def write_manifest(outdir: Path) -> None:
    lines = [
        "# Proxy-Metric Mismatch Audit",
        "",
        "Generated by `analysis/audit_proxy_metric_mismatch.py`.",
        "",
        "## Concept",
        "",
        "The audit evaluates whether a lower language-model validation BPB selects the",
        "same configurations as downstream PubMedQA metrics. It reports rank inversions,",
        "correlation direction mismatches, and Pareto conflicts where a configuration",
        "has worse BPB but better downstream behavior.",
        "",
        "## Files",
        "",
        "- `proxy_metric_rank_table.tsv`: repeated-seed config means and metric ranks.",
        "- `proxy_metric_mismatch_summary.tsv`: metric-level mismatch summary.",
        "- `proxy_metric_pareto_conflicts.tsv`: pairwise conflicts between BPB and downstream metrics.",
        "",
        "The mismatch summary includes a simple Proxy-Metric Mismatch Index (PMMI):",
        "`pmmi_normalized_rank_gap` averages the downstream rank gap of the BPB",
        "winner and the BPB rank gap of the downstream winner, scaled to [0, 1].",
        "`pmmi_downstream_gain_over_proxy_winner` reports the downstream gain of",
        "the downstream winner over the BPB-selected configuration.",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit proxy-metric mismatch")
    parser.add_argument("--results", default="results_repeated_controls.tsv")
    parser.add_argument("--outdir", default="outputs/innovation_audit/proxy_metric_mismatch")
    parser.add_argument(
        "--exclude-groups",
        default="",
        help="Comma-separated config groups to exclude, e.g. prospective.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.results, sep="\t")
    results = results[(results["split"] == "val") & (results["status"] == "ok")].copy()
    if args.exclude_groups.strip():
        excluded = {item.strip() for item in args.exclude_groups.split(",") if item.strip()}
        results = results[~results["group"].isin(excluded)].copy()

    rank_table = summarize_configs(results)
    mismatch_summary = build_mismatch_summary(results, rank_table)
    pareto_conflicts = build_pareto_conflicts(rank_table)

    rank_table.to_csv(outdir / "proxy_metric_rank_table.tsv", sep="\t", index=False)
    mismatch_summary.to_csv(outdir / "proxy_metric_mismatch_summary.tsv", sep="\t", index=False)
    pareto_conflicts.to_csv(outdir / "proxy_metric_pareto_conflicts.tsv", sep="\t", index=False)
    write_manifest(outdir)

    print(f"wrote proxy-metric mismatch audit to {outdir}")


if __name__ == "__main__":
    main()
