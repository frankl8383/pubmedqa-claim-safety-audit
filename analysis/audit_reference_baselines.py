"""
Prediction-level audit for reference baselines and external stress testing.

This script complements analysis/prediction_level_audit.py. It brings together:

- 5-seed PubMedQA validation predictions for agent auxiliary and random primary;
- selected PubMedQA TF-IDF LogisticRegression predictions;
- selected PubMedQA frozen BiomedBERT predictions;
- selected SciFact TF-IDF LogisticRegression predictions;
- selected SciFact frozen BiomedBERT predictions;
- train-majority baselines for PubMedQA and SciFact.

For multi-seed models, point estimates are mean metrics across seeds. Bootstrap
intervals resample examples and recompute the seed-mean metric, so repeated
seeds are not treated as extra independent examples.

Usage:
    uv run python analysis/audit_reference_baselines.py \
      --outdir outputs/reference_baseline_audit \
      --bootstrap-iters 10000
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PUBMEDQA_LABELS = ("yes", "no", "maybe")
SCIFACT_LABELS = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")
BASE_METRICS = ("accuracy", "macro_f1", "brier")
MODEL_LABELS = {
    "agent_aux_a20f5b7": "Agent auxiliary",
    "random_primary_best_7147e14": "Random primary",
    "random_macro_f1_best_8a1209b": "Random macro-F1 best",
    "random_brier_best_8dc23be": "Random Brier best",
    "random_accuracy_best_7e73835": "Random accuracy best",
    "manual_baseline_f1664dd": "Manual baseline",
    "manual_warmup10_da0044c": "Manual warmup",
    "majority_train": "Majority",
    "tfidf_lr_selected": "TF-IDF LR",
    "tfidf_scifact_selected": "TF-IDF LR",
    "frozen_biomedbert": "Frozen BiomedBERT",
}
MODEL_GROUPS = {
    "agent_aux_a20f5b7": "agent",
    "random_primary_best_7147e14": "random",
    "random_macro_f1_best_8a1209b": "random",
    "random_brier_best_8dc23be": "random",
    "random_accuracy_best_7e73835": "random",
    "manual_baseline_f1664dd": "manual",
    "manual_warmup10_da0044c": "manual",
    "majority_train": "baseline",
    "tfidf_lr_selected": "traditional_baseline",
    "tfidf_scifact_selected": "traditional_baseline",
    "frozen_biomedbert": "biomedical_encoder",
}


@dataclass(frozen=True)
class Prediction:
    task: str
    split: str
    model: str
    seed: int
    example_id: str
    truth: str
    prediction: str
    probabilities: tuple[float, ...]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value.lower()).strip("_")


def labels_for_task(task: str) -> tuple[str, ...]:
    if task == "pubmedqa":
        return PUBMEDQA_LABELS
    if task == "scifact":
        return SCIFACT_LABELS
    raise ValueError(f"Unknown task: {task}")


def metrics_for_rows(rows: list[Prediction], labels: tuple[str, ...]) -> dict[str, float]:
    if not rows:
        return {metric: math.nan for metric in metric_names(labels)}

    y_true = [row.truth for row in rows]
    y_pred = [row.prediction for row in rows]
    n = len(rows)
    out: dict[str, float] = {
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / n,
    }

    f1s = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        suffix = safe_name(label)
        out[f"precision_{suffix}"] = precision
        out[f"recall_{suffix}"] = recall
        out[f"f1_{suffix}"] = f1
        out[f"support_{suffix}"] = sum(t == label for t in y_true)
        out[f"predicted_{suffix}"] = sum(p == label for p in y_pred)

    out["macro_f1"] = sum(f1s) / len(f1s)
    brier = 0.0
    for row in rows:
        for idx, label in enumerate(labels):
            target = 1.0 if row.truth == label else 0.0
            brier += (row.probabilities[idx] - target) ** 2
    out["brier"] = brier / n
    return out


def metric_names(labels: tuple[str, ...]) -> tuple[str, ...]:
    per_class = []
    for label in labels:
        suffix = safe_name(label)
        per_class.extend([f"precision_{suffix}", f"recall_{suffix}", f"f1_{suffix}"])
    return BASE_METRICS + tuple(per_class)


def parse_seed(path: Path) -> int:
    match = re.search(r"_seed(\d+)\.tsv$", path.name)
    if not match:
        raise ValueError(f"Could not parse seed from {path.name}")
    return int(match.group(1))


def load_repeated_pubmedqa(root: Path) -> list[Prediction]:
    rows = []
    repeated_models = (
        "manual_baseline_f1664dd",
        "manual_warmup10_da0044c",
        "random_accuracy_best_7e73835",
        "random_brier_best_8dc23be",
        "random_macro_f1_best_8a1209b",
        "random_primary_best_7147e14",
        "agent_aux_a20f5b7",
    )
    for model in repeated_models:
        pattern = f"pubmedqa_repeated_val_val_{model}_seed*.tsv"
        for path in sorted((root / "outputs" / "predictions").glob(pattern)):
            seed = parse_seed(path)
            for row in read_tsv(path):
                rows.append(Prediction(
                    task="pubmedqa",
                    split="val",
                    model=model,
                    seed=seed,
                    example_id=row["pmid"],
                    truth=row["truth"],
                    prediction=row["prediction"],
                    probabilities=(float(row["prob_yes"]), float(row["prob_no"]), float(row["prob_maybe"])),
                ))
    return rows


def load_tfidf_pubmedqa(root: Path) -> list[Prediction]:
    path = root / "outputs" / "tfidf_pubmedqa_predictions.tsv"
    rows = []
    for row in read_tsv(path):
        rows.append(Prediction(
            task="pubmedqa",
            split=row["split"],
            model="tfidf_lr_selected",
            seed=0,
            example_id=row["pmid"],
            truth=row["truth"],
            prediction=row["prediction"],
            probabilities=(float(row["prob_yes"]), float(row["prob_no"]), float(row["prob_maybe"])),
        ))
    return rows


def load_biomedbert_pubmedqa(root: Path) -> list[Prediction]:
    path = root / "outputs" / "biomedbert_pubmedqa_predictions.tsv"
    rows = []
    for row in read_tsv(path):
        rows.append(Prediction(
            task="pubmedqa",
            split=row["split"],
            model="frozen_biomedbert",
            seed=0,
            example_id=row["id"],
            truth=row["truth"],
            prediction=row["prediction"],
            probabilities=(float(row["prob_yes"]), float(row["prob_no"]), float(row["prob_maybe"])),
        ))
    return rows


def load_biomedbert_scifact(root: Path) -> list[Prediction]:
    path = root / "outputs" / "biomedbert_scifact_predictions.tsv"
    rows = []
    for row in read_tsv(path):
        rows.append(Prediction(
            task="scifact",
            split=row["split"],
            model="frozen_biomedbert",
            seed=0,
            example_id=row["id"],
            truth=row["truth"],
            prediction=row["prediction"],
            probabilities=(
                float(row["prob_support"]),
                float(row["prob_contradict"]),
                float(row["prob_not_enough_info"]),
            ),
        ))
    return rows


def load_tfidf_scifact(root: Path) -> list[Prediction]:
    path = root / "outputs" / "tfidf_scifact_predictions.tsv"
    if not path.exists():
        return []
    rows = []
    for row in read_tsv(path):
        rows.append(Prediction(
            task="scifact",
            split=row["split"],
            model="tfidf_scifact_selected",
            seed=0,
            example_id=row["id"],
            truth=row["truth"],
            prediction=row["prediction"],
            probabilities=(
                float(row["prob_support"]),
                float(row["prob_contradict"]),
                float(row["prob_not_enough_info"]),
            ),
        ))
    return rows


def add_majority_rows(rows: list[Prediction]) -> list[Prediction]:
    majority_by_task = {
        "pubmedqa": "yes",
        "scifact": "SUPPORT",
    }
    example_truth: dict[tuple[str, str, str], str] = {}
    for row in rows:
        key = (row.task, row.split, row.example_id)
        if key in example_truth and example_truth[key] != row.truth:
            raise ValueError(f"Truth mismatch for {key}: {example_truth[key]} vs {row.truth}")
        example_truth[key] = row.truth

    majority_rows = []
    for (task, split, example_id), truth in sorted(example_truth.items()):
        labels = labels_for_task(task)
        majority = majority_by_task[task]
        probs = tuple(1.0 if label == majority else 0.0 for label in labels)
        majority_rows.append(Prediction(
            task=task,
            split=split,
            model="majority_train",
            seed=0,
            example_id=example_id,
            truth=truth,
            prediction=majority,
            probabilities=probs,
        ))
    return majority_rows


def group_predictions(rows: list[Prediction]) -> dict[tuple[str, str, str, int], list[Prediction]]:
    grouped: dict[tuple[str, str, str, int], list[Prediction]] = defaultdict(list)
    for row in rows:
        grouped[(row.task, row.split, row.model, row.seed)].append(row)
    return {
        key: sorted(value, key=lambda row: row.example_id)
        for key, value in grouped.items()
    }


def rows_for_model(grouped: dict[tuple[str, str, str, int], list[Prediction]],
                   task: str, split: str, model: str) -> dict[int, list[Prediction]]:
    out = {}
    for (candidate_task, candidate_split, candidate_model, seed), rows in grouped.items():
        if (candidate_task, candidate_split, candidate_model) == (task, split, model):
            out[seed] = rows
    return out


def metric_seed_mean(seed_rows: dict[int, list[Prediction]], labels: tuple[str, ...],
                     metric: str, example_ids: list[str] | None = None) -> float:
    return seed_mean_all_metrics(seed_rows, labels, example_ids)[metric]


def seed_mean_all_metrics(seed_rows: dict[int, list[Prediction]], labels: tuple[str, ...],
                          example_ids: list[str] | None = None) -> dict[str, float]:
    values = []
    for rows in seed_rows.values():
        if example_ids is not None:
            by_id = {row.example_id: row for row in rows}
            rows = [by_id[example_id] for example_id in example_ids]
        values.append(metrics_for_rows(rows, labels))

    metric_keys = values[0].keys()
    return {
        metric: sum(seed_metrics[metric] for seed_metrics in values) / len(values)
        for metric in metric_keys
    }


def all_models(rows: list[Prediction]) -> list[tuple[str, str, str]]:
    return sorted({(row.task, row.split, row.model) for row in rows})


def common_example_ids(left: dict[int, list[Prediction]], right: dict[int, list[Prediction]] | None = None) -> list[str]:
    left_ids = {row.example_id for rows in left.values() for row in rows}
    if right is None:
        return sorted(left_ids)
    right_ids = {row.example_id for rows in right.values() for row in rows}
    return sorted(left_ids & right_ids)


def summarize_metrics(
    grouped: dict[tuple[str, str, str, int], list[Prediction]],
    n_iters: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    out = []
    for task, split, model in all_models([row for rows in grouped.values() for row in rows]):
        labels = labels_for_task(task)
        seed_rows = rows_for_model(grouped, task, split, model)
        ids = common_example_ids(seed_rows)
        ids_arr = np.array(ids)
        metrics = metric_names(labels)
        point = seed_mean_all_metrics(seed_rows, labels)
        samples = {metric: [] for metric in metrics}
        for _ in range(n_iters):
            sampled = ids_arr[rng.integers(0, len(ids_arr), size=len(ids_arr))].tolist()
            sampled_metrics = seed_mean_all_metrics(seed_rows, labels, sampled)
            for metric in metrics:
                samples[metric].append(sampled_metrics[metric])
        for metric in metrics:
            arr = np.array(samples[metric], dtype=float)
            out.append({
                "task": task,
                "split": split,
                "model": model,
                "display_name": MODEL_LABELS.get(model, model),
                "group": MODEL_GROUPS.get(model, "other"),
                "n_examples": len(ids),
                "n_seeds": len(seed_rows),
                "metric": metric,
                "point": point[metric],
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
                "bootstrap_sd": float(np.std(arr, ddof=1)),
                "n_bootstrap": n_iters,
            })
    return out


def summarize_confusion(grouped: dict[tuple[str, str, str, int], list[Prediction]]) -> list[dict[str, object]]:
    out = []
    for task, split, model in all_models([row for rows in grouped.values() for row in rows]):
        labels = labels_for_task(task)
        seed_rows = rows_for_model(grouped, task, split, model)
        per_seed_counts = []
        for rows in seed_rows.values():
            truth_counts = Counter(row.truth for row in rows)
            counts = {}
            for truth in labels:
                for pred in labels:
                    count = sum(row.truth == truth and row.prediction == pred for row in rows)
                    support = truth_counts[truth]
                    counts[(truth, pred)] = (count, support, count / support if support else math.nan)
            per_seed_counts.append(counts)
        for truth in labels:
            for pred in labels:
                values = [counts[(truth, pred)] for counts in per_seed_counts]
                count_values = [item[0] for item in values]
                support_values = [item[1] for item in values]
                rate_values = [item[2] for item in values]
                out.append({
                    "task": task,
                    "split": split,
                    "model": model,
                    "display_name": MODEL_LABELS.get(model, model),
                    "truth": truth,
                    "prediction": pred,
                    "support_mean": float(np.mean(support_values)),
                    "count_mean": float(np.mean(count_values)),
                    "row_rate_mean": float(np.mean(rate_values)),
                    "n_seeds": len(seed_rows),
                })
    return out


def summarize_distribution(grouped: dict[tuple[str, str, str, int], list[Prediction]]) -> list[dict[str, object]]:
    out = []
    for task, split, model in all_models([row for rows in grouped.values() for row in rows]):
        labels = labels_for_task(task)
        seed_rows = rows_for_model(grouped, task, split, model)
        for label in labels:
            suffix = safe_name(label)
            pred_counts = []
            support_counts = []
            fractions = []
            support_fracs = []
            for rows in seed_rows.values():
                n = len(rows)
                pred_count = sum(row.prediction == label for row in rows)
                support_count = sum(row.truth == label for row in rows)
                pred_counts.append(pred_count)
                support_counts.append(support_count)
                fractions.append(pred_count / n)
                support_fracs.append(support_count / n)
            out.append({
                "task": task,
                "split": split,
                "model": model,
                "display_name": MODEL_LABELS.get(model, model),
                "label": label,
                "label_key": suffix,
                "support_mean": float(np.mean(support_counts)),
                "support_fraction_mean": float(np.mean(support_fracs)),
                "predicted_mean": float(np.mean(pred_counts)),
                "predicted_sd": float(np.std(pred_counts, ddof=1)) if len(pred_counts) > 1 else 0.0,
                "predicted_fraction_mean": float(np.mean(fractions)),
                "predicted_fraction_sd": float(np.std(fractions, ddof=1)) if len(fractions) > 1 else 0.0,
                "n_seeds": len(seed_rows),
            })
    return out


def paired_diffs(
    grouped: dict[tuple[str, str, str, int], list[Prediction]],
    comparisons: list[tuple[str, str, str, str]],
    n_iters: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    out = []
    for task, split, left, right in comparisons:
        labels = labels_for_task(task)
        left_rows = rows_for_model(grouped, task, split, left)
        right_rows = rows_for_model(grouped, task, split, right)
        ids = common_example_ids(left_rows, right_rows)
        ids_arr = np.array(ids)
        if not ids:
            continue
        metrics = metric_names(labels)
        left_point = seed_mean_all_metrics(left_rows, labels, ids)
        right_point = seed_mean_all_metrics(right_rows, labels, ids)
        point = {metric: left_point[metric] - right_point[metric] for metric in metrics}
        samples = {metric: [] for metric in metrics}
        for _ in range(n_iters):
            sampled = ids_arr[rng.integers(0, len(ids_arr), size=len(ids_arr))].tolist()
            left_sample = seed_mean_all_metrics(left_rows, labels, sampled)
            right_sample = seed_mean_all_metrics(right_rows, labels, sampled)
            for metric in metrics:
                samples[metric].append(left_sample[metric] - right_sample[metric])
        for metric in metrics:
            arr = np.array(samples[metric], dtype=float)
            p_two_sided = 2 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))
            out.append({
                "task": task,
                "split": split,
                "left": left,
                "left_display": MODEL_LABELS.get(left, left),
                "right": right,
                "right_display": MODEL_LABELS.get(right, right),
                "metric": metric,
                "diff_left_minus_right": point[metric],
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
                "bootstrap_sd": float(np.std(arr, ddof=1)),
                "p_two_sided_sign": min(1.0, p_two_sided),
                "n_examples": len(ids),
                "n_bootstrap": n_iters,
            })
    return out


def metric_wide_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in metric_rows:
        key = (str(row["task"]), str(row["split"]), str(row["model"]))
        base = grouped.setdefault(key, {
            "task": row["task"],
            "split": row["split"],
            "model": row["model"],
            "display_name": row["display_name"],
            "group": row["group"],
            "n_examples": row["n_examples"],
            "n_seeds": row["n_seeds"],
        })
        metric = str(row["metric"])
        base[metric] = row["point"]
        base[f"{metric}_ci_low"] = row["ci_low"]
        base[f"{metric}_ci_high"] = row["ci_high"]
    return list(grouped.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit reference baselines")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260504)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(load_repeated_pubmedqa(root))
    rows.extend(load_tfidf_pubmedqa(root))
    rows.extend(load_biomedbert_pubmedqa(root))
    rows.extend(load_tfidf_scifact(root))
    rows.extend(load_biomedbert_scifact(root))
    rows.extend(add_majority_rows(rows))

    grouped = group_predictions(rows)
    rng = np.random.default_rng(args.bootstrap_seed)
    metric_rows = summarize_metrics(grouped, args.bootstrap_iters, rng)
    confusion_rows = summarize_confusion(grouped)
    distribution_rows = summarize_distribution(grouped)

    comparisons = [
        ("pubmedqa", "val", "agent_aux_a20f5b7", "random_primary_best_7147e14"),
        ("pubmedqa", "val", "agent_aux_a20f5b7", "random_macro_f1_best_8a1209b"),
        ("pubmedqa", "val", "agent_aux_a20f5b7", "random_brier_best_8dc23be"),
        ("pubmedqa", "val", "agent_aux_a20f5b7", "random_accuracy_best_7e73835"),
        ("pubmedqa", "val", "agent_aux_a20f5b7", "manual_baseline_f1664dd"),
        ("pubmedqa", "val", "agent_aux_a20f5b7", "manual_warmup10_da0044c"),
        ("pubmedqa", "val", "agent_aux_a20f5b7", "tfidf_lr_selected"),
        ("pubmedqa", "val", "agent_aux_a20f5b7", "frozen_biomedbert"),
        ("pubmedqa", "val", "agent_aux_a20f5b7", "majority_train"),
        ("pubmedqa", "val", "random_primary_best_7147e14", "majority_train"),
        ("pubmedqa", "val", "random_macro_f1_best_8a1209b", "majority_train"),
        ("pubmedqa", "val", "random_brier_best_8dc23be", "majority_train"),
        ("pubmedqa", "val", "random_accuracy_best_7e73835", "majority_train"),
        ("pubmedqa", "val", "manual_baseline_f1664dd", "majority_train"),
        ("pubmedqa", "val", "manual_warmup10_da0044c", "majority_train"),
        ("pubmedqa", "val", "frozen_biomedbert", "agent_aux_a20f5b7"),
        ("pubmedqa", "val", "frozen_biomedbert", "random_primary_best_7147e14"),
        ("pubmedqa", "val", "frozen_biomedbert", "majority_train"),
        ("pubmedqa", "val", "tfidf_lr_selected", "majority_train"),
        ("pubmedqa", "val", "tfidf_lr_selected", "agent_aux_a20f5b7"),
        ("pubmedqa", "test", "frozen_biomedbert", "majority_train"),
        ("pubmedqa", "test", "tfidf_lr_selected", "majority_train"),
        ("scifact", "dev", "tfidf_scifact_selected", "majority_train"),
        ("scifact", "dev", "frozen_biomedbert", "tfidf_scifact_selected"),
        ("scifact", "dev", "frozen_biomedbert", "majority_train"),
    ]
    diff_rows = paired_diffs(grouped, comparisons, args.bootstrap_iters, rng)

    write_tsv(outdir / "metrics_long.tsv", metric_rows)
    write_tsv(outdir / "metrics_wide.tsv", metric_wide_rows(metric_rows))
    write_tsv(outdir / "confusion_summary.tsv", confusion_rows)
    write_tsv(outdir / "prediction_distribution.tsv", distribution_rows)
    write_tsv(outdir / "paired_bootstrap_diffs.tsv", diff_rows)
    write_tsv(outdir / "audit_metadata.tsv", [{
        "bootstrap_iters": args.bootstrap_iters,
        "bootstrap_seed": args.bootstrap_seed,
        "n_prediction_rows": len(rows),
        "models": ",".join(sorted({row.model for row in rows})),
        "tasks": ",".join(sorted({row.task for row in rows})),
    }])

    print(f"wrote reference-baseline audit to {outdir}")


if __name__ == "__main__":
    main()
