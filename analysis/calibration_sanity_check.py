"""
Calibration sanity checks for PubMedQA validation predictions.

This audit treats decoder label-score probabilities, TF-IDF probabilities, and
frozen-encoder probabilities as model score distributions. They are not
post-hoc calibrated probabilities. The goal is to compare reliability patterns,
not to claim clinical calibration.

Usage:
    uv run python analysis/calibration_sanity_check.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = ("yes", "no", "maybe")
REPEATED_MODELS = {
    "manual_baseline_f1664dd": "Manual baseline",
    "random_brier_best_8dc23be": "Random Brier best",
    "random_macro_f1_best_8a1209b": "Random macro-F1 best",
    "random_primary_best_7147e14": "Random primary",
    "agent_aux_a20f5b7": "Agent auxiliary",
}
REFERENCE_MODELS = {
    "tfidf_lr_selected": "TF-IDF LR",
    "frozen_biomedbert": "Frozen BiomedBERT",
    "majority_train": "Majority",
}
PLOT_MODELS = [
    "majority_train",
    "random_primary_best_7147e14",
    "agent_aux_a20f5b7",
    "tfidf_lr_selected",
    "frozen_biomedbert",
]


def entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def brier_score(truth: list[str], probs: np.ndarray) -> float:
    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    target = np.zeros_like(probs, dtype=float)
    for i, label in enumerate(truth):
        target[i, label_to_idx[label]] = 1.0
    return float(np.mean(np.sum((probs - target) ** 2, axis=1)))


def add_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    probs = df[[f"prob_{label}" for label in LABELS]].to_numpy(float)
    pred_idx = probs.argmax(axis=1)
    df = df.copy()
    df["confidence"] = probs.max(axis=1)
    df["prediction_from_prob"] = [LABELS[i] for i in pred_idx]
    # Use the stored prediction if present, but keep argmax sanity explicit.
    df["correct"] = df["prediction"].eq(df["truth"]).astype(int)
    df["entropy"] = entropy(probs)
    return df


def ece_from_df(df: pd.DataFrame, n_bins: int) -> tuple[float, float]:
    total = len(df)
    ece = 0.0
    mce = 0.0
    for bin_idx in range(n_bins):
        low = bin_idx / n_bins
        high = (bin_idx + 1) / n_bins
        if bin_idx == n_bins - 1:
            mask = (df["confidence"] >= low) & (df["confidence"] <= high)
        else:
            mask = (df["confidence"] >= low) & (df["confidence"] < high)
        if not mask.any():
            continue
        bin_df = df[mask]
        gap = abs(float(bin_df["correct"].mean()) - float(bin_df["confidence"].mean()))
        ece += len(bin_df) / total * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def reliability_bins(df: pd.DataFrame, n_bins: int) -> list[dict[str, object]]:
    rows = []
    for bin_idx in range(n_bins):
        low = bin_idx / n_bins
        high = (bin_idx + 1) / n_bins
        if bin_idx == n_bins - 1:
            mask = (df["confidence"] >= low) & (df["confidence"] <= high)
        else:
            mask = (df["confidence"] >= low) & (df["confidence"] < high)
        bin_df = df[mask]
        rows.append({
            "bin": bin_idx,
            "bin_low": low,
            "bin_high": high,
            "n": int(len(bin_df)),
            "mean_confidence": float(bin_df["confidence"].mean()) if len(bin_df) else np.nan,
            "accuracy": float(bin_df["correct"].mean()) if len(bin_df) else np.nan,
            "gap_accuracy_minus_confidence": (
                float(bin_df["correct"].mean() - bin_df["confidence"].mean()) if len(bin_df) else np.nan
            ),
        })
    return rows


def metric_row(model: str, display_name: str, group: str, df: pd.DataFrame, n_bins: int) -> dict[str, object]:
    probs = df[[f"prob_{label}" for label in LABELS]].to_numpy(float)
    ece, mce = ece_from_df(df, n_bins)
    wrong = df[df["correct"] == 0]
    correct = df[df["correct"] == 1]
    high_conf_wrong = wrong[wrong["confidence"] >= 0.8]
    return {
        "model": model,
        "display_name": display_name,
        "group": group,
        "n_predictions": len(df),
        "n_examples": df["pmid"].nunique(),
        "n_seeds": df["seed"].nunique() if "seed" in df.columns else 1,
        "accuracy": float(df["correct"].mean()),
        "brier": brier_score(df["truth"].tolist(), probs),
        "ece": ece,
        "mce": mce,
        "mean_confidence": float(df["confidence"].mean()),
        "mean_confidence_correct": float(correct["confidence"].mean()) if len(correct) else np.nan,
        "mean_confidence_wrong": float(wrong["confidence"].mean()) if len(wrong) else np.nan,
        "mean_entropy": float(df["entropy"].mean()),
        "high_conf_wrong_rate": len(high_conf_wrong) / len(df),
        "wrong_high_conf_fraction": len(high_conf_wrong) / len(wrong) if len(wrong) else 0.0,
    }


def load_repeated_predictions(pred_dir: Path, config: str, display_name: str) -> pd.DataFrame:
    rows = []
    for path in sorted(pred_dir.glob(f"pubmedqa_repeated_val_val_{config}_seed*.tsv")):
        seed = int(path.stem.rsplit("seed", 1)[1])
        df = pd.read_csv(path, sep="\t", dtype={"pmid": str})
        df["model"] = config
        df["display_name"] = display_name
        df["group"] = "autoresearch"
        df["seed"] = seed
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No repeated prediction files for {config}")
    return pd.concat(rows, ignore_index=True)


def load_tfidf(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype={"pmid": str})
    df = df[df["split"].eq("val")].copy()
    df["model"] = "tfidf_lr_selected"
    df["display_name"] = "TF-IDF LR"
    df["group"] = "reference"
    df["seed"] = 0
    return df


def load_frozen(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype={"id": str})
    df = df[(df["task"].eq("pubmedqa")) & (df["split"].eq("val"))].copy()
    df = df.rename(columns={"id": "pmid"})
    df["model"] = "frozen_biomedbert"
    df["display_name"] = "Frozen BiomedBERT"
    df["group"] = "reference"
    df["seed"] = 0
    return df


def load_majority(processed_val: Path, train_path: Path) -> pd.DataFrame:
    train = pd.read_json(train_path, lines=True, dtype={"pmid": str})
    val = pd.read_json(processed_val, lines=True, dtype={"pmid": str})
    majority = train["label"].value_counts().idxmax()
    rows = []
    for _, row in val.iterrows():
        out = {
            "pmid": str(row["pmid"]),
            "truth": row["label"],
            "prediction": majority,
            "prob_yes": 1.0 if majority == "yes" else 0.0,
            "prob_no": 1.0 if majority == "no" else 0.0,
            "prob_maybe": 1.0 if majority == "maybe" else 0.0,
            "model": "majority_train",
            "display_name": "Majority",
            "group": "baseline",
            "seed": 0,
        }
        rows.append(out)
    return pd.DataFrame(rows)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PubMedQA calibration sanity checks")
    parser.add_argument("--pred-dir", default="outputs/predictions")
    parser.add_argument("--tfidf-pred", default="outputs/tfidf_pubmedqa_predictions.tsv")
    parser.add_argument("--frozen-pred", default="outputs/biomedbert_pubmedqa_predictions.tsv")
    parser.add_argument("--cache", default="~/.cache/autoresearch_biomed")
    parser.add_argument("--outdir", default="outputs/calibration_sanity")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache).expanduser() / "processed"

    dfs = []
    dfs.append(load_majority(cache / "val.jsonl", cache / "train.jsonl"))
    for config, display in REPEATED_MODELS.items():
        dfs.append(load_repeated_predictions(Path(args.pred_dir), config, display))
    dfs.append(load_tfidf(Path(args.tfidf_pred)))
    dfs.append(load_frozen(Path(args.frozen_pred)))

    all_predictions = pd.concat(dfs, ignore_index=True)
    all_predictions = add_score_columns(all_predictions)
    all_predictions.to_csv(outdir / "pubmedqa_calibration_predictions.tsv", sep="\t", index=False)

    metric_rows = []
    bin_rows = []
    for model, model_df in all_predictions.groupby("model", sort=False):
        display = str(model_df["display_name"].iloc[0])
        group = str(model_df["group"].iloc[0])
        metric_rows.append(metric_row(model, display, group, model_df, args.n_bins))
        for row in reliability_bins(model_df, args.n_bins):
            row.update({"model": model, "display_name": display, "group": group})
            bin_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    order = {model: i for i, model in enumerate(["majority_train", *REPEATED_MODELS.keys(), "tfidf_lr_selected", "frozen_biomedbert"])}
    metrics["_order"] = metrics["model"].map(order).fillna(99)
    metrics = metrics.sort_values("_order").drop(columns="_order")
    metrics.to_csv(outdir / "pubmedqa_calibration_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(bin_rows).to_csv(outdir / "pubmedqa_reliability_bins.tsv", sep="\t", index=False)

    manuscript_models = [
        "majority_train",
        "random_primary_best_7147e14",
        "random_brier_best_8dc23be",
        "agent_aux_a20f5b7",
        "tfidf_lr_selected",
        "frozen_biomedbert",
    ]
    metrics[metrics["model"].isin(manuscript_models)].to_csv(
        "outputs/manuscript_tables/table14_calibration_sanity.tsv",
        sep="\t",
        index=False,
    )

    readme = """# PubMedQA Calibration Sanity Check

This audit compares top-label confidence, Brier score, expected calibration
error (ECE), maximum calibration error (MCE), and reliability bins on the locked
PubMedQA validation split.

Important caveat: decoder label-score probabilities and classifier probabilities
are uncalibrated score distributions. These metrics are used to audit relative
reliability patterns, not to claim deployable clinical calibration.
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote calibration sanity outputs to {outdir}")


if __name__ == "__main__":
    main()
