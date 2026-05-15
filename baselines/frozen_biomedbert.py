"""
Frozen biomedical encoder baselines for PubMedQA and SciFact.

The encoder is never fine-tuned. Texts are embedded once with a biomedical BERT
model, then a small z-scored LogisticRegression classifier is tuned on the
locked PubMedQA validation split or, for SciFact, an inner split from train.
This provides a domain-specific reference baseline, not a compute-matched
competitor to the 5-minute autoresearch runs.

Examples:
    uv run python baselines/frozen_biomedbert.py --task pubmedqa \
      --out outputs/biomedbert_pubmedqa_metrics.tsv \
      --pred-out outputs/biomedbert_pubmedqa_predictions.tsv

    uv run python prepare_scifact.py
    uv run python baselines/frozen_biomedbert.py --task scifact \
      --out outputs/biomedbert_scifact_metrics.tsv \
      --pred-out outputs/biomedbert_scifact_predictions.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np


PUBMEDQA_LABELS = ("yes", "no", "maybe")
SCIFACT_LABELS = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")
DEFAULT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pubmedqa_text(row: dict) -> str:
    return f"Question: {row['question']}\nContext: {row['context']}"


def load_pubmedqa(cache: Path) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], tuple[str, ...]]:
    splits = {}
    ids = {}
    labels = {}
    for split in ("train", "val", "test"):
        rows = read_jsonl(cache / "processed" / f"{split}.jsonl")
        splits[split] = [pubmedqa_text(row) for row in rows]
        ids[split] = [str(row["pmid"]) for row in rows]
        labels[split] = [row["label"] for row in rows]
    return splits, ids, labels, PUBMEDQA_LABELS


def load_scifact(cache: Path) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], tuple[str, ...]]:
    splits = {}
    ids = {}
    labels = {}
    for split in ("train", "dev"):
        rows = read_jsonl(cache / "processed" / f"{split}.jsonl")
        splits[split] = [row["text"] for row in rows]
        ids[split] = [row["id"] for row in rows]
        labels[split] = [row["label"] for row in rows]
    return splits, ids, labels, SCIFACT_LABELS


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def mean_pool(hidden, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def embed_texts(
    texts: list[str],
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
    pooling: str,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            if pooling == "cls":
                pooled = output.last_hidden_state[:, 0]
            elif pooling == "mean":
                pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            else:
                raise ValueError(f"Unsupported pooling mode: {pooling}")
            chunks.append(pooled.detach().cpu().float().numpy())
            print(f"embedded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(chunks)


def load_or_embed(
    split: str,
    texts: list[str],
    task: str,
    model_name: str,
    outdir: Path,
    device: str,
    batch_size: int,
    max_length: int,
    pooling: str,
    reuse: bool,
) -> np.ndarray:
    emb_dir = outdir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{task}_{safe_name(model_name)}_{pooling}_len{max_length}_{split}.npz"
    path = emb_dir / stem
    if reuse and path.exists():
        data = np.load(path)
        print(f"loaded embeddings {path}")
        return data["embeddings"]

    embeddings = embed_texts(texts, model_name, device, batch_size, max_length, pooling)
    np.savez_compressed(path, embeddings=embeddings)
    print(f"wrote embeddings {path}")
    return embeddings


def macro_f1(y_true: list[str], y_pred: list[str], labels: tuple[str, ...]) -> tuple[float, dict[str, float]]:
    f1s = []
    by_label = {}
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        by_label[label] = f1
    return sum(f1s) / len(f1s), by_label


def brier_score(y_true: list[str], probs: list[list[float]], labels: tuple[str, ...]) -> float:
    label_to_idx = {label: i for i, label in enumerate(labels)}
    total = 0.0
    for truth, row_probs in zip(y_true, probs):
        target = [0.0] * len(labels)
        target[label_to_idx[truth]] = 1.0
        total += sum((p - t) ** 2 for p, t in zip(row_probs, target))
    return total / len(y_true)


def align_proba(clf, raw_probs, labels: tuple[str, ...]) -> list[list[float]]:
    classes = clf.classes_.tolist()
    class_to_idx = {label: i for i, label in enumerate(classes)}
    return [[float(row[class_to_idx[label]]) for label in labels] for row in raw_probs]


def ignore_known_sklearn_matmul_warnings():
    return warnings.catch_warnings()


def fit_classifier(clf, x: np.ndarray, y: list[str]):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        clf.fit(x, y)
    if any(warning.category.__name__ == "ConvergenceWarning" for warning in caught):
        raise RuntimeError("LogisticRegression did not converge")
    return clf


def predict_with_probs(clf, x: np.ndarray, labels: tuple[str, ...]) -> tuple[list[str], list[list[float]]]:
    with ignore_known_sklearn_matmul_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        pred = clf.predict(x).tolist()
        raw_probs = clf.predict_proba(x)
    if not np.isfinite(raw_probs).all():
        raise ValueError("Classifier produced non-finite probabilities")
    return pred, align_proba(clf, raw_probs, labels)


def metrics_row(
    task: str,
    model_name: str,
    variant: str,
    split: str,
    y_true: list[str],
    y_pred: list[str],
    probs: list[list[float]],
    labels: tuple[str, ...],
    selected: bool = False,
) -> dict[str, object]:
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    mf1, class_f1 = macro_f1(y_true, y_pred, labels)
    row = {
        "task": task,
        "encoder": model_name,
        "variant": variant,
        "split": split,
        "selected_for_final_eval": selected,
        "n": len(y_true),
        "acc": acc,
        "macro_f1": mf1,
        "brier": brier_score(y_true, probs, labels),
    }
    for label in labels:
        row[f"f1_{safe_name(label.lower())}"] = class_f1[label]
    return row


def majority_metrics(task: str, model_name: str, split: str, train_labels: list[str],
                     y_true: list[str], labels: tuple[str, ...]) -> dict[str, object]:
    majority = Counter(train_labels).most_common(1)[0][0]
    pred = [majority] * len(y_true)
    probs = [[1.0 if label == majority else 0.0 for label in labels] for _ in y_true]
    return metrics_row(task, model_name, "majority_train", split, y_true, pred, probs, labels)


def write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(
    path: Path,
    task: str,
    model_name: str,
    variant: str,
    split: str,
    ids: list[str],
    y_true: list[str],
    y_pred: list[str],
    probs: list[list[float]],
    labels: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prob_fields = [f"prob_{safe_name(label.lower())}" for label in labels]
    fieldnames = ["task", "encoder", "variant", "split", "id", "truth", "prediction"] + prob_fields
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        for row_id, truth, prediction, prob in zip(ids, y_true, y_pred, probs):
            row = {
                "task": task,
                "encoder": model_name,
                "variant": variant,
                "split": split,
                "id": row_id,
                "truth": truth,
                "prediction": prediction,
            }
            for label, value in zip(labels, prob):
                row[f"prob_{safe_name(label.lower())}"] = value
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen biomedical encoder baselines")
    parser.add_argument("--task", choices=["pubmedqa", "scifact"], required=True)
    parser.add_argument("--pubmedqa-cache", default="~/.cache/autoresearch_biomed")
    parser.add_argument("--scifact-cache", default="~/.cache/autoresearch_biomed/scifact")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pred-out", required=True)
    parser.add_argument("--workdir", default="outputs/frozen_encoder")
    parser.add_argument("--no-reuse-embeddings", action="store_true")
    parser.add_argument(
        "--scifact-selection",
        choices=["inner_train_val", "dev"],
        default="inner_train_val",
        help="For SciFact, select LR hyperparameters on an inner split by default; dev is final stress-test only.",
    )
    parser.add_argument("--scifact-inner-val-frac", type=float, default=0.2)
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def make_lr(c_value: float, class_weight: str | None):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight=class_weight,
                max_iter=5000,
                tol=1e-3,
                solver="saga",
                random_state=20260504,
            ),
        )

    if args.task == "pubmedqa":
        texts, ids, y, labels = load_pubmedqa(Path(args.pubmedqa_cache).expanduser())
        eval_splits = ("val", "test")
        select_split = "val"
        selected_suffix = "selected_on_val"
    else:
        texts, ids, y, labels = load_scifact(Path(args.scifact_cache).expanduser())
        eval_splits = ("dev",)
        select_split = "dev" if args.scifact_selection == "dev" else "inner_val"
        selected_suffix = "selected_on_scifact_inner_val" if args.scifact_selection == "inner_train_val" else "selected_on_dev"

    device = resolve_device(args.device)
    print(f"task={args.task} encoder={args.model} device={device} pooling={args.pooling}")
    workdir = Path(args.workdir)

    embeddings = {
        split: load_or_embed(
            split=split,
            texts=split_texts,
            task=args.task,
            model_name=args.model,
            outdir=workdir,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
            reuse=not args.no_reuse_embeddings,
        )
        for split, split_texts in texts.items()
    }

    metric_rows = []
    for split in eval_splits:
        metric_rows.append(majority_metrics(args.task, args.model, split, y["train"], y[split], labels))

    if args.task == "scifact" and args.scifact_selection == "inner_train_val":
        indices = np.arange(len(y["train"]))
        train_idx, inner_val_idx = train_test_split(
            indices,
            test_size=args.scifact_inner_val_frac,
            random_state=20260504,
            stratify=y["train"],
        )
        select_train_x = embeddings["train"][train_idx]
        select_train_y = [y["train"][int(i)] for i in train_idx]
        select_eval_x = embeddings["train"][inner_val_idx]
        select_eval_y = [y["train"][int(i)] for i in inner_val_idx]
    else:
        select_train_x = embeddings["train"]
        select_train_y = y["train"]
        select_eval_x = embeddings[select_split]
        select_eval_y = y[select_split]

    candidates = []
    for class_weight in (None, "balanced"):
        for c_value in (0.01, 0.1, 1.0, 10.0):
            variant = f"frozen_{args.pooling}_zscore_lr_c{c_value}_cw{class_weight or 'none'}"
            clf = make_lr(c_value, class_weight)
            try:
                fit_classifier(clf, select_train_x, select_train_y)
            except RuntimeError as exc:
                print(f"skipping {variant}: {exc}", flush=True)
                continue
            pred, probs = predict_with_probs(clf, select_eval_x, labels)
            row = metrics_row(args.task, args.model, variant, select_split, select_eval_y, pred, probs, labels)
            metric_rows.append(row)
            candidates.append((row["macro_f1"], -row["brier"], variant, c_value, class_weight))

    candidates.sort(key=lambda item: (-item[0], -item[1]))
    if not candidates:
        raise RuntimeError("No converged frozen-encoder LR candidates")
    best_variant = candidates[0][2]
    best_c = candidates[0][3]
    best_class_weight = candidates[0][4]
    best_clf = make_lr(best_c, best_class_weight)
    fit_classifier(best_clf, embeddings["train"], y["train"])

    pred_path = Path(args.pred_out)
    if pred_path.exists():
        pred_path.unlink()
    selected_variant = best_variant + "_" + selected_suffix

    for split in eval_splits:
        pred, probs = predict_with_probs(best_clf, embeddings[split], labels)
        metric_rows.append(
            metrics_row(
                args.task,
                args.model,
                selected_variant,
                split,
                y[split],
                pred,
                probs,
                labels,
                selected=True,
            )
        )
        write_predictions(pred_path, args.task, args.model, selected_variant, split, ids[split], y[split], pred, probs, labels)

    write_metrics(Path(args.out), metric_rows)
    print(f"selected_on_{select_split}={best_variant}")
    print(f"wrote {args.out}")
    print(f"wrote {args.pred_out}")


if __name__ == "__main__":
    main()
