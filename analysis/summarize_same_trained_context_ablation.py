#!/usr/bin/env python3
"""Summarize same-trained PubMedQA context-ablation runs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = ROOT / "outputs" / "same_trained_context_ablation"
TABLE_DIR = ROOT / "outputs" / "manuscript_tables"


def load_summaries() -> pd.DataFrame:
    paths = sorted(ABLATION_DIR.glob("pubmedqa_same_trained_context_ablation_*.tsv"))
    if not paths:
        raise FileNotFoundError(f"No context-ablation summary TSVs under {ABLATION_DIR}")
    root_prefix = str(ROOT) + "/"
    for path in paths:
        text = path.read_text(encoding="utf-8")
        cleaned = text.replace(root_prefix, "")
        if cleaned != text:
            path.write_text(cleaned, encoding="utf-8")
    frames = [pd.read_csv(path, sep="\t") for path in paths]
    df = pd.concat(frames, ignore_index=True)
    df["prediction_file"] = df["prediction_file"].map(normalize_prediction_file)
    for col in ["accuracy", "macro_f1", "brier", "f1_yes", "f1_no", "f1_maybe"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    return df


def normalize_prediction_file(value: object) -> str:
    path = Path(str(value))
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = ["accuracy", "macro_f1", "brier", "f1_yes", "f1_no", "f1_maybe"]
    grouped = (
        df.groupby(["config_name", "prompt_mode"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            **{f"{col}_mean": (col, "mean") for col in value_cols},
            **{f"{col}_sd": (col, "std") for col in value_cols},
        )
        .sort_values(["config_name", "prompt_mode"])
    )

    baselines = grouped[grouped["prompt_mode"].eq("question_context")].set_index("config_name")
    for metric in ["accuracy", "macro_f1", "brier", "f1_yes", "f1_no", "f1_maybe"]:
        deltas = []
        for _, row in grouped.iterrows():
            base = baselines.loc[row["config_name"], f"{metric}_mean"]
            deltas.append(row[f"{metric}_mean"] - base)
        grouped[f"delta_{metric}_vs_question_context"] = deltas
    return grouped


def summarize_prediction_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, summary_row in df.iterrows():
        pred_path = Path(str(summary_row["prediction_file"]))
        if not pred_path.is_absolute():
            pred_path = ROOT / pred_path
        preds = pd.read_csv(pred_path, sep="\t")
        counts = Counter(preds["prediction"])
        n = len(preds)
        rows.append({
            "config_name": summary_row["config_name"],
            "seed": int(summary_row["seed"]),
            "prompt_mode": summary_row["prompt_mode"],
            "n_examples": n,
            "pred_yes_frac": counts.get("yes", 0) / n,
            "pred_no_frac": counts.get("no", 0) / n,
            "pred_maybe_frac": counts.get("maybe", 0) / n,
        })
    pred_df = pd.DataFrame(rows)
    return (
        pred_df.groupby(["config_name", "prompt_mode"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            pred_yes_frac_mean=("pred_yes_frac", "mean"),
            pred_no_frac_mean=("pred_no_frac", "mean"),
            pred_maybe_frac_mean=("pred_maybe_frac", "mean"),
        )
        .sort_values(["config_name", "prompt_mode"])
    )


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_summaries()
    aggregate = aggregate_metrics(df)
    pred_dist = summarize_prediction_distribution(df)
    merged = aggregate.merge(pred_dist, on=["config_name", "prompt_mode", "n_seeds"], how="left")

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ABLATION_DIR / "same_trained_context_ablation_runs.tsv", sep="\t", index=False)
    merged.to_csv(ABLATION_DIR / "same_trained_context_ablation_summary.tsv", sep="\t", index=False)
    merged.to_csv(TABLE_DIR / "table29_same_trained_context_ablation.tsv", sep="\t", index=False)

    print(f"runs={len(df)} configs={df['config_name'].nunique()} modes={df['prompt_mode'].nunique()}")
    print(f"wrote {ABLATION_DIR / 'same_trained_context_ablation_summary.tsv'}")
    print(f"wrote {TABLE_DIR / 'table29_same_trained_context_ablation.tsv'}")


if __name__ == "__main__":
    main()
