"""
Vectorized full paired bootstrap audit for PubMedQA validation predictions.

This complements `prediction_level_audit.py` with a wider comparison set:
agent auxiliary versus all key repeated-seed manual/random controls plus
single-seed TF-IDF and frozen BiomedBERT reference baselines. It keeps bootstrap
resampling at the example level and averages metrics across seeds for repeated
models.

Usage:
    uv run python analysis/full_pairwise_validation_audit.py \
      --outdir outputs/prediction_level_audit_full \
      --bootstrap-iters 10000
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LABELS = ("yes", "no", "maybe")
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
REPEATED_MODELS = (
    "manual_baseline_f1664dd",
    "manual_warmup10_da0044c",
    "random_accuracy_best_7e73835",
    "random_brier_best_8dc23be",
    "random_macro_f1_best_8a1209b",
    "random_primary_best_7147e14",
    "agent_aux_a20f5b7",
)
MODEL_LABELS = {
    "majority_val": "Majority",
    "manual_baseline_f1664dd": "Manual baseline",
    "manual_warmup10_da0044c": "Manual warmup",
    "random_accuracy_best_7e73835": "Random accuracy best",
    "random_brier_best_8dc23be": "Random Brier best",
    "random_macro_f1_best_8a1209b": "Random macro-F1 best",
    "random_primary_best_7147e14": "Random primary",
    "agent_aux_a20f5b7": "Agent auxiliary",
    "tfidf_lr_selected": "TF-IDF LR",
    "frozen_biomedbert": "Frozen BiomedBERT",
}
MODEL_GROUPS = {
    "majority_val": "baseline",
    "manual_baseline_f1664dd": "manual",
    "manual_warmup10_da0044c": "manual",
    "random_accuracy_best_7e73835": "random",
    "random_brier_best_8dc23be": "random",
    "random_macro_f1_best_8a1209b": "random",
    "random_primary_best_7147e14": "random",
    "agent_aux_a20f5b7": "agent",
    "tfidf_lr_selected": "traditional_baseline",
    "frozen_biomedbert": "biomedical_encoder",
}
PAIRWISE_COMPARISONS = (
    ("agent_aux_a20f5b7", "random_primary_best_7147e14"),
    ("agent_aux_a20f5b7", "random_macro_f1_best_8a1209b"),
    ("agent_aux_a20f5b7", "random_brier_best_8dc23be"),
    ("agent_aux_a20f5b7", "random_accuracy_best_7e73835"),
    ("agent_aux_a20f5b7", "manual_baseline_f1664dd"),
    ("agent_aux_a20f5b7", "manual_warmup10_da0044c"),
    ("agent_aux_a20f5b7", "tfidf_lr_selected"),
    ("agent_aux_a20f5b7", "frozen_biomedbert"),
    ("agent_aux_a20f5b7", "majority_val"),
)
METRICS = (
    "accuracy", "macro_f1", "brier",
    "precision_yes", "recall_yes", "f1_yes", "pred_frac_yes",
    "precision_no", "recall_no", "f1_no", "pred_frac_no",
    "precision_maybe", "recall_maybe", "f1_maybe", "pred_frac_maybe",
)


@dataclass(frozen=True)
class ModelPredictions:
    model: str
    pmids: tuple[str, ...]
    truth: np.ndarray
    pred: np.ndarray
    probs: np.ndarray


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


def label_id(value: str) -> int:
    return LABEL_TO_ID[value]


def probs_from_row(row: dict[str, str]) -> list[float]:
    return [float(row[f"prob_{label}"]) for label in LABELS]


def align_rows(rows: list[dict[str, str]], id_key: str) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    rows = sorted(rows, key=lambda row: str(row[id_key]))
    pmids = tuple(str(row[id_key]) for row in rows)
    truth = np.array([label_id(row["truth"]) for row in rows], dtype=np.int16)
    pred = np.array([label_id(row["prediction"]) for row in rows], dtype=np.int16)
    probs = np.array([probs_from_row(row) for row in rows], dtype=np.float64)
    return pmids, truth, pred, probs


def load_repeated_model(root: Path, model: str) -> ModelPredictions:
    seed_truth = []
    seed_pred = []
    seed_probs = []
    pmids_ref = None
    for path in sorted((root / "outputs" / "predictions").glob(f"pubmedqa_repeated_val_val_{model}_seed*.tsv")):
        pmids, truth, pred, probs = align_rows(read_tsv(path), "pmid")
        if pmids_ref is None:
            pmids_ref = pmids
        elif pmids != pmids_ref:
            raise ValueError(f"PMID order mismatch for {model}: {path}")
        seed_truth.append(truth)
        seed_pred.append(pred)
        seed_probs.append(probs)
    if pmids_ref is None:
        raise FileNotFoundError(f"No repeated prediction files for {model}")
    truth_stack = np.stack(seed_truth, axis=0)
    if not np.all(truth_stack == truth_stack[0]):
        raise ValueError(f"Truth mismatch across seeds for {model}")
    return ModelPredictions(
        model=model,
        pmids=pmids_ref,
        truth=truth_stack[0],
        pred=np.stack(seed_pred, axis=0),
        probs=np.stack(seed_probs, axis=0),
    )


def load_single_seed_pubmedqa(root: Path, path: Path, model: str, variant_prefix: str, id_key: str) -> ModelPredictions:
    rows = [
        row for row in read_tsv(path)
        if row["split"] == "val" and (row.get("variant") or row.get("model", "")).startswith(variant_prefix)
    ]
    if not rows:
        raise ValueError(f"No selected validation rows for {model} in {path}")
    pmids, truth, pred, probs = align_rows(rows, id_key)
    return ModelPredictions(
        model=model,
        pmids=pmids,
        truth=truth,
        pred=pred[None, :],
        probs=probs[None, :, :],
    )


def make_majority(source: ModelPredictions) -> ModelPredictions:
    counts = np.bincount(source.truth, minlength=len(LABELS))
    majority = int(np.argmax(counts))
    pred = np.full((1, source.truth.shape[0]), majority, dtype=np.int16)
    probs = np.zeros((1, source.truth.shape[0], len(LABELS)), dtype=np.float64)
    probs[:, :, majority] = 1.0
    return ModelPredictions(
        model="majority_val",
        pmids=source.pmids,
        truth=source.truth,
        pred=pred,
        probs=probs,
    )


def align_all(models: dict[str, ModelPredictions]) -> dict[str, ModelPredictions]:
    common = sorted(set.intersection(*(set(model.pmids) for model in models.values())))
    if not common:
        raise ValueError("No common PMIDs across models")
    aligned = {}
    for name, model in models.items():
        index = {pmid: idx for idx, pmid in enumerate(model.pmids)}
        idx = np.array([index[pmid] for pmid in common], dtype=np.int64)
        aligned[name] = ModelPredictions(
            model=name,
            pmids=tuple(common),
            truth=model.truth[idx],
            pred=model.pred[:, idx],
            probs=model.probs[:, idx, :],
        )
    truths = [model.truth for model in aligned.values()]
    if any(not np.array_equal(truths[0], truth) for truth in truths[1:]):
        raise ValueError("Truth mismatch across aligned models")
    return aligned


def metric_samples(model: ModelPredictions, indices: np.ndarray | None = None) -> dict[str, np.ndarray]:
    n = model.truth.shape[0]
    if indices is None:
        indices = np.arange(n, dtype=np.int64)[None, :]
    truth = model.truth[indices]
    pred = model.pred[:, indices]
    probs = model.probs[:, indices, :]
    n_boot = indices.shape[0]
    n_seeds = model.pred.shape[0]

    out: dict[str, np.ndarray] = {}
    out["accuracy"] = (pred == truth[None, :, :]).mean(axis=2).mean(axis=0)

    brier_by_example = np.zeros((n_seeds, n_boot, n), dtype=np.float64)
    for label_id_ in range(len(LABELS)):
        target = (truth == label_id_).astype(np.float64)
        brier_by_example += (probs[:, :, :, label_id_] - target[None, :, :]) ** 2
    out["brier"] = brier_by_example.mean(axis=2).mean(axis=0)

    f1s = []
    for label_name, label_id_ in LABEL_TO_ID.items():
        truth_is = truth == label_id_
        pred_is = pred == label_id_
        tp = (pred_is & truth_is[None, :, :]).sum(axis=2).astype(np.float64)
        fp = (pred_is & ~truth_is[None, :, :]).sum(axis=2).astype(np.float64)
        fn = (~pred_is & truth_is[None, :, :]).sum(axis=2).astype(np.float64)
        predicted = pred_is.sum(axis=2).astype(np.float64)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
        out[f"precision_{label_name}"] = precision.mean(axis=0)
        out[f"recall_{label_name}"] = recall.mean(axis=0)
        out[f"f1_{label_name}"] = f1.mean(axis=0)
        out[f"pred_frac_{label_name}"] = (predicted / n).mean(axis=0)
        f1s.append(out[f"f1_{label_name}"])
    out["macro_f1"] = np.vstack(f1s).mean(axis=0)
    return out


def point_metrics(model: ModelPredictions) -> dict[str, float]:
    return {metric: float(values[0]) for metric, values in metric_samples(model).items()}


def bootstrap_indices(n_examples: int, n_iters: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n_examples, size=(n_iters, n_examples), dtype=np.int64)


def summarize_models(models: dict[str, ModelPredictions], samples: dict[str, dict[str, np.ndarray]], n_iters: int) -> list[dict[str, object]]:
    rows = []
    for model_name, model in models.items():
        point = point_metrics(model)
        for metric in METRICS:
            arr = samples[model_name][metric]
            rows.append({
                "model": model_name,
                "display_name": MODEL_LABELS.get(model_name, model_name),
                "group": MODEL_GROUPS.get(model_name, "other"),
                "metric": metric,
                "point": point[metric],
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
                "bootstrap_sd": float(np.std(arr, ddof=1)),
                "n_examples": len(model.pmids),
                "n_seeds": model.pred.shape[0],
                "n_bootstrap": n_iters,
            })
    return rows


def paired_diffs(samples: dict[str, dict[str, np.ndarray]], models: dict[str, ModelPredictions], n_iters: int) -> list[dict[str, object]]:
    points = {name: point_metrics(model) for name, model in models.items()}
    rows = []
    for left, right in PAIRWISE_COMPARISONS:
        for metric in METRICS:
            diff = points[left][metric] - points[right][metric]
            arr = samples[left][metric] - samples[right][metric]
            p_two_sided = 2 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))
            rows.append({
                "left": left,
                "left_display": MODEL_LABELS.get(left, left),
                "right": right,
                "right_display": MODEL_LABELS.get(right, right),
                "comparison": f"{MODEL_LABELS.get(left, left)} - {MODEL_LABELS.get(right, right)}",
                "metric": metric,
                "diff_left_minus_right": diff,
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
                "bootstrap_sd": float(np.std(arr, ddof=1)),
                "p_two_sided_sign": min(1.0, p_two_sided),
                "n_examples": len(models[left].pmids),
                "left_n_seeds": models[left].pred.shape[0],
                "right_n_seeds": models[right].pred.shape[0],
                "n_bootstrap": n_iters,
            })
    return rows


def wide_metric_rows(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in long_rows:
        base = grouped.setdefault(str(row["model"]), {
            "model": row["model"],
            "display_name": row["display_name"],
            "group": row["group"],
            "n_examples": row["n_examples"],
            "n_seeds": row["n_seeds"],
            "n_bootstrap": row["n_bootstrap"],
        })
        metric = str(row["metric"])
        base[metric] = row["point"]
        base[f"{metric}_ci_low"] = row["ci_low"]
        base[f"{metric}_ci_high"] = row["ci_high"]
    return list(grouped.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Full paired validation audit")
    parser.add_argument("--outdir", default="outputs/prediction_level_audit_full")
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260504)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    models = {model: load_repeated_model(root, model) for model in REPEATED_MODELS}
    models["majority_val"] = make_majority(models["agent_aux_a20f5b7"])
    models["tfidf_lr_selected"] = load_single_seed_pubmedqa(
        root,
        root / "outputs" / "tfidf_pubmedqa_predictions.tsv",
        "tfidf_lr_selected",
        "tfidf_lr_",
        "pmid",
    )
    models["frozen_biomedbert"] = load_single_seed_pubmedqa(
        root,
        root / "outputs" / "biomedbert_pubmedqa_predictions.tsv",
        "frozen_biomedbert",
        "frozen_mean_zscore_lr_",
        "id",
    )
    models = align_all(models)

    rng = np.random.default_rng(args.bootstrap_seed)
    indices = bootstrap_indices(len(next(iter(models.values())).pmids), args.bootstrap_iters, rng)
    samples = {name: metric_samples(model, indices) for name, model in models.items()}

    outdir = Path(args.outdir)
    metric_long = summarize_models(models, samples, args.bootstrap_iters)
    diff_rows = paired_diffs(samples, models, args.bootstrap_iters)
    write_tsv(outdir / "metrics_long.tsv", metric_long)
    write_tsv(outdir / "metrics_wide.tsv", wide_metric_rows(metric_long))
    write_tsv(outdir / "paired_bootstrap_diffs.tsv", diff_rows)
    write_tsv(outdir / "audit_metadata.tsv", [{
        "n_examples": len(next(iter(models.values())).pmids),
        "bootstrap_iters": args.bootstrap_iters,
        "bootstrap_seed": args.bootstrap_seed,
        "models": ",".join(models.keys()),
        "comparisons": ",".join(f"{left}:{right}" for left, right in PAIRWISE_COMPARISONS),
    }])
    print(f"wrote full pairwise validation audit to {outdir}")


if __name__ == "__main__":
    main()
