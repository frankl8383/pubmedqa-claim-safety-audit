"""
Build manuscript-ready TSV tables from the locked analysis outputs.

Usage:
    uv run python analysis/build_manuscript_tables.py \
      --outdir outputs/manuscript_tables
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd


PUBMEDQA_LABELS = ("yes", "no", "maybe")
SCIFACT_LABELS = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")
MODEL_ORDER = {
    "majority_train": 0,
    "manual_baseline_f1664dd": 1,
    "manual_warmup10_da0044c": 2,
    "random_accuracy_best_7e73835": 3,
    "random_brier_best_8dc23be": 4,
    "random_macro_f1_best_8a1209b": 5,
    "random_primary_best_7147e14": 6,
    "agent_aux_a20f5b7": 7,
    "tfidf_lr_selected": 8,
    "frozen_biomedbert": 9,
    "agent_aux_locked_post_freeze_test_audit": 10,
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def macro_f1_for_majority(rows: list[dict], labels: tuple[str, ...], majority_label: str) -> float:
    f1s = []
    for label in labels:
        tp = sum(row["label"] == label and majority_label == label for row in rows)
        fp = sum(row["label"] != label and majority_label == label for row in rows)
        fn = sum(row["label"] == label and majority_label != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1s) / len(f1s)


def dataset_split_rows(root: Path) -> list[dict]:
    rows = []
    pubmedqa_dir = Path.home() / ".cache" / "autoresearch_biomed" / "processed"
    for split in ("train", "val", "test"):
        split_rows = read_jsonl(pubmedqa_dir / f"{split}.jsonl")
        counts = Counter(row["label"] for row in split_rows)
        majority = counts.most_common(1)[0][0]
        out = {
            "dataset": "PubMedQA PQA-L",
            "split": split,
            "n": len(split_rows),
            "majority_label": majority,
            "majority_accuracy": counts[majority] / len(split_rows),
            "majority_macro_f1": macro_f1_for_majority(split_rows, PUBMEDQA_LABELS, majority),
        }
        for label in PUBMEDQA_LABELS:
            out[f"count_{label}"] = counts[label]
            out[f"frac_{label}"] = counts[label] / len(split_rows)
        rows.append(out)

    scifact_dir = Path.home() / ".cache" / "autoresearch_biomed" / "scifact" / "processed"
    for split in ("train", "dev"):
        split_rows = read_jsonl(scifact_dir / f"{split}.jsonl")
        counts = Counter(row["label"] for row in split_rows)
        majority = counts.most_common(1)[0][0]
        out = {
            "dataset": "SciFact cited-document",
            "split": split,
            "n": len(split_rows),
            "majority_label": majority,
            "majority_accuracy": counts[majority] / len(split_rows),
            "majority_macro_f1": macro_f1_for_majority(split_rows, SCIFACT_LABELS, majority),
        }
        for label in SCIFACT_LABELS:
            key = label.lower()
            out[f"count_{key}"] = counts[label]
            out[f"frac_{key}"] = counts[label] / len(split_rows)
        rows.append(out)
    return rows


def repeated_seed_table(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "outputs" / "repeated_controls_summary.tsv", sep="\t")
    columns = [
        "config_name", "group", "n", "seeds",
        "lm_val_bpb_mean", "lm_val_bpb_sd",
        "pubmedqa_acc_mean", "pubmedqa_acc_sd",
        "pubmedqa_macro_f1_mean", "pubmedqa_macro_f1_sd",
        "pubmedqa_brier_mean", "pubmedqa_brier_sd",
        "tokens_M_mean", "params_M_mean",
    ]
    available = [column for column in columns if column in df.columns]
    df = df[available].copy()
    group_order = {"manual": 0, "random": 1, "agent": 2}
    df["_group_order"] = df["group"].map(group_order).fillna(99)
    df = df.sort_values(["_group_order", "config_name"]).drop(columns=["_group_order"])
    return df


def reference_metric_table(root: Path) -> pd.DataFrame:
    metrics = pd.read_csv(root / "outputs" / "reference_baseline_audit" / "metrics_wide.tsv", sep="\t")
    rows = metrics[(metrics["task"] == "pubmedqa") & (metrics["split"].isin(["val", "test"]))].copy()
    main_models = {
        "majority_train",
        "random_primary_best_7147e14",
        "agent_aux_a20f5b7",
        "tfidf_lr_selected",
        "frozen_biomedbert",
    }
    rows = rows[rows["model"].isin(main_models)]
    keep = [
        "split", "model", "display_name", "group", "n_examples", "n_seeds",
        "accuracy", "accuracy_ci_low", "accuracy_ci_high",
        "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high",
        "brier", "brier_ci_low", "brier_ci_high",
        "f1_yes", "f1_no", "f1_maybe",
    ]
    rows = rows[keep]
    rows["uncertainty"] = "example_bootstrap_95ci"

    locked = pd.read_csv(root / "results_locked_replicates.tsv", sep="\t")
    locked = locked[(locked["phase"] == "post_freeze_test_audit") & (locked["status"] == "ok")]
    if not locked.empty:
        agent = {
            "split": "test",
            "model": "agent_aux_locked_post_freeze_test_audit",
            "display_name": "Agent auxiliary post-freeze test audit",
            "group": "agent",
            "n_examples": 150,
            "n_seeds": len(locked),
            "accuracy": locked["pubmedqa_acc"].mean(),
            "accuracy_ci_low": math.nan,
            "accuracy_ci_high": math.nan,
            "macro_f1": locked["pubmedqa_macro_f1"].mean(),
            "macro_f1_ci_low": math.nan,
            "macro_f1_ci_high": math.nan,
            "brier": locked["pubmedqa_brier"].mean(),
            "brier_ci_low": math.nan,
            "brier_ci_high": math.nan,
            "f1_yes": math.nan,
            "f1_no": math.nan,
            "f1_maybe": math.nan,
            "uncertainty": "seed_sd_available_in_results_locked_replicates",
        }
        rows = pd.concat([rows, pd.DataFrame([agent])], ignore_index=True)

    split_order = {"val": 0, "test": 1}
    rows["_split_order"] = rows["split"].map(split_order)
    rows["_model_order"] = rows["model"].map(MODEL_ORDER).fillna(99)
    return rows.sort_values(["_split_order", "_model_order"]).drop(columns=["_split_order", "_model_order"])


def pubmedqa_per_class_table(root: Path) -> pd.DataFrame:
    metrics = pd.read_csv(root / "outputs" / "reference_baseline_audit" / "metrics_wide.tsv", sep="\t")
    dist = pd.read_csv(root / "outputs" / "reference_baseline_audit" / "prediction_distribution.tsv", sep="\t")
    metrics = metrics[(metrics["task"] == "pubmedqa") & (metrics["split"] == "val")].copy()
    dist = dist[(dist["task"] == "pubmedqa") & (dist["split"] == "val")].copy()
    rows = []
    for _, row in metrics.iterrows():
        model = row["model"]
        for label in PUBMEDQA_LABELS:
            drow = dist[(dist["model"] == model) & (dist["label"] == label)].iloc[0]
            rows.append({
                "split": "val",
                "model": model,
                "display_name": row["display_name"],
                "group": row["group"],
                "label": label,
                "support_fraction": drow["support_fraction_mean"],
                "predicted_fraction": drow["predicted_fraction_mean"],
                f"precision": row[f"precision_{label}"],
                f"recall": row[f"recall_{label}"],
                f"f1": row[f"f1_{label}"],
                f"f1_ci_low": row[f"f1_{label}_ci_low"],
                f"f1_ci_high": row[f"f1_{label}_ci_high"],
            })
    out = pd.DataFrame(rows)
    out["_model_order"] = out["model"].map(MODEL_ORDER).fillna(99)
    out["_label_order"] = out["label"].map({label: i for i, label in enumerate(PUBMEDQA_LABELS)})
    return out.sort_values(["_model_order", "_label_order"]).drop(columns=["_model_order", "_label_order"])


def scifact_external_table(root: Path) -> pd.DataFrame:
    metrics = pd.read_csv(root / "outputs" / "reference_baseline_audit" / "metrics_wide.tsv", sep="\t")
    rows = metrics[(metrics["task"] == "scifact") & (metrics["split"] == "dev")].copy()
    keep = [
        "split", "model", "display_name", "group", "n_examples", "n_seeds",
        "accuracy", "accuracy_ci_low", "accuracy_ci_high",
        "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high",
        "brier", "brier_ci_low", "brier_ci_high",
        "f1_support", "f1_contradict", "f1_not_enough_info",
    ]
    rows = rows[keep]
    rows["_model_order"] = rows["model"].map(MODEL_ORDER).fillna(99)
    return rows.sort_values("_model_order").drop(columns="_model_order")


def paired_differences_table(root: Path) -> pd.DataFrame:
    diffs = pd.read_csv(root / "outputs" / "reference_baseline_audit" / "paired_bootstrap_diffs.tsv", sep="\t")
    key_metrics = {
        "accuracy", "macro_f1", "brier",
        "f1_yes", "f1_no", "f1_maybe",
        "f1_support", "f1_contradict", "f1_not_enough_info",
    }
    rows = diffs[diffs["metric"].isin(key_metrics)].copy()
    rows["comparison"] = rows["left_display"] + " - " + rows["right_display"]
    keep = [
        "task", "split", "comparison", "metric", "diff_left_minus_right",
        "ci_low", "ci_high", "p_two_sided_sign", "n_examples", "n_bootstrap",
    ]
    return rows[keep]


def full_pairwise_validation_audit_table(root: Path) -> pd.DataFrame:
    diffs = pd.read_csv(root / "outputs" / "prediction_level_audit_full" / "paired_bootstrap_diffs.tsv", sep="\t")
    agent_controls = {
        "random_primary_best_7147e14",
        "random_macro_f1_best_8a1209b",
        "random_brier_best_8dc23be",
        "random_accuracy_best_7e73835",
        "manual_baseline_f1664dd",
        "manual_warmup10_da0044c",
        "tfidf_lr_selected",
        "frozen_biomedbert",
        "majority_train",
    }
    key_metrics = {
        "accuracy", "macro_f1", "brier",
        "f1_yes", "f1_no", "f1_maybe",
        "precision_yes", "recall_yes",
        "precision_no", "recall_no",
        "precision_maybe", "recall_maybe",
    }
    rows = diffs[
        (diffs["left"] == "agent_aux_a20f5b7") &
        (diffs["right"].isin(agent_controls)) &
        (diffs["metric"].isin(key_metrics))
    ].copy()
    rows["comparison"] = rows["left_display"] + " - " + rows["right_display"]
    rows["_right_order"] = rows["right"].map(MODEL_ORDER).fillna(99)
    metric_order = {metric: idx for idx, metric in enumerate([
        "accuracy", "macro_f1", "brier",
        "f1_yes", "f1_no", "f1_maybe",
        "precision_yes", "recall_yes",
        "precision_no", "recall_no",
        "precision_maybe", "recall_maybe",
    ])}
    rows["_metric_order"] = rows["metric"].map(metric_order).fillna(99)
    keep = [
        "comparison", "right", "right_display", "metric",
        "diff_left_minus_right", "ci_low", "ci_high",
        "p_two_sided_sign", "n_examples", "n_bootstrap",
    ]
    return rows.sort_values(["_right_order", "_metric_order"])[keep]


def write_readme(outdir: Path) -> None:
    text = """# Manuscript Tables

Generated by `analysis/build_manuscript_tables.py`.

Tables:

- `table1_dataset_splits.tsv`: locked PubMedQA and SciFact split sizes and label distributions.
- `table2_repeated_seed_comparison.tsv`: 5-seed agent/manual/random validation comparison.
- `table3_pubmedqa_reference_baselines.tsv`: PubMedQA validation/test reference baseline metrics.
- `table4_pubmedqa_per_class_audit.tsv`: PubMedQA validation per-class audit.
- `table5_scifact_external_validation.tsv`: SciFact dev external stress-test reference baseline.
- `table6_key_paired_bootstrap_differences.tsv`: key paired bootstrap differences.
- `table7_full_pairwise_validation_audit.tsv`: agent auxiliary versus all key validation controls.
- `table8_manual_error_taxonomy_draft.tsv`: historical draft taxonomy for 24 candidate error-analysis examples.
- `table8_manual_error_taxonomy_reviewed.tsv`: historical reviewed taxonomy summary.
- `table9_manual_error_main_text_examples.tsv`: six historical illustrative main-text candidates.
- `table10_manual_error_revised_taxonomy.tsv`: conservative historical revised taxonomy.
- `table11_full_abstract_adjudication.tsv`: historical PubMed full-abstract review.
- `table12_manual_error_final_use_summary.tsv`: historical manual error-analysis use summary.
- `table13_context_use_sanity.tsv`: validation-only partial-input and shuffled-context audit generated by `analysis/context_use_sanity_check.py`.
- `table14_calibration_sanity.tsv`: validation-only top-label calibration audit generated by `analysis/calibration_sanity_check.py`.
- `table15_claim_safety_matrix.tsv`: allowed, conditional, and forbidden manuscript claims generated by `analysis/build_hypothesis_audit_package.py`.
- `table16_protocol_guardrail_checklist.tsv`: hypothesis-first audit guardrails generated by `analysis/build_hypothesis_audit_package.py`.
- `table17_retrospective_alignment_audit.tsv`: retrospective/prospective status of existing results after the hypothesis-first upgrade, generated by `analysis/build_retrospective_alignment_audit.py`.

Notes:

- PubMedQA test was already opened; new method selection must use validation or external data only.
- SciFact official test labels are not public; hyperparameters are selected on an inner train split and official dev is used only as an external stress test.
- Frozen BiomedBERT is a reference encoder baseline, not a compute-matched 5-minute run.
- Historical manual error-analysis rows are illustrative single-reviewer audit artifacts. Do not describe them as definitive expert adjudication unless a second independent reviewer confirms them.
- Calibration metrics are sanity-check diagnostics over uncalibrated label-score or classifier probabilities, not evidence of clinically reliable probabilities.
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manuscript tables")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(dataset_split_rows(root)).to_csv(outdir / "table1_dataset_splits.tsv", sep="\t", index=False)
    repeated_seed_table(root).to_csv(outdir / "table2_repeated_seed_comparison.tsv", sep="\t", index=False)
    reference_metric_table(root).to_csv(outdir / "table3_pubmedqa_reference_baselines.tsv", sep="\t", index=False)
    pubmedqa_per_class_table(root).to_csv(outdir / "table4_pubmedqa_per_class_audit.tsv", sep="\t", index=False)
    scifact_external_table(root).to_csv(outdir / "table5_scifact_external_validation.tsv", sep="\t", index=False)
    paired_differences_table(root).to_csv(outdir / "table6_key_paired_bootstrap_differences.tsv", sep="\t", index=False)
    full_pairwise_validation_audit_table(root).to_csv(outdir / "table7_full_pairwise_validation_audit.tsv", sep="\t", index=False)
    write_readme(outdir)
    print(f"wrote manuscript tables to {outdir}")


if __name__ == "__main__":
    main()
