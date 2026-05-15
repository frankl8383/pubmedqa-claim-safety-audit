"""
TF-IDF baselines for the locked SciFact external stress-test split.

The official SciFact test labels are not public. This script tunes a small
TF-IDF + LogisticRegression grid on an inner split from SciFact train and then
reports the official dev split once. It is a reference stress-test baseline,
not validation of the autoresearch-trained small causal LM.

Usage:
    uv run python prepare_scifact.py
    uv run python baselines/tfidf_scifact.py \
      --cache ~/.cache/autoresearch_biomed/scifact \
      --out outputs/tfidf_scifact_metrics.tsv \
      --pred-out outputs/tfidf_scifact_predictions.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


LABELS = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def macro_f1(y_true: list[str], y_pred: list[str]) -> tuple[float, dict[str, float]]:
    f1s = []
    by_label = {}
    for label in LABELS:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        by_label[label] = f1
    return sum(f1s) / len(f1s), by_label


def brier_score(y_true: list[str], probs: list[list[float]]) -> float:
    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    total = 0.0
    for truth, row_probs in zip(y_true, probs):
        target = [0.0] * len(LABELS)
        target[label_to_idx[truth]] = 1.0
        total += sum((p - t) ** 2 for p, t in zip(row_probs, target))
    return total / len(y_true)


def metric_row(model: str, split: str, y_true: list[str], y_pred: list[str], probs: list[list[float]]) -> dict[str, object]:
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    mf1, class_f1 = macro_f1(y_true, y_pred)
    return {
        "model": model,
        "split": split,
        "n": len(y_true),
        "acc": acc,
        "macro_f1": mf1,
        "brier": brier_score(y_true, probs),
        "f1_support": class_f1["SUPPORT"],
        "f1_contradict": class_f1["CONTRADICT"],
        "f1_not_enough_info": class_f1["NOT_ENOUGH_INFO"],
    }


def aligned_predict_proba(pipe, texts: list[str]) -> list[list[float]]:
    raw_probs = pipe.predict_proba(texts)
    classes = pipe.named_steps["clf"].classes_.tolist()
    class_to_idx = {label: i for i, label in enumerate(classes)}
    return [[float(row[class_to_idx[label]]) for label in LABELS] for row in raw_probs]


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TF-IDF SciFact stress-test baselines")
    parser.add_argument("--cache", default="~/.cache/autoresearch_biomed/scifact")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pred-out", required=True)
    parser.add_argument("--seed", type=int, default=20260506)
    args = parser.parse_args()

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise SystemExit("Install scikit-learn first: uv add scikit-learn") from exc

    cache = Path(args.cache).expanduser()
    train_rows = read_jsonl(cache / "processed" / "train.jsonl")
    dev_rows = read_jsonl(cache / "processed" / "dev.jsonl")

    x_train_all = [row["text"] for row in train_rows]
    y_train_all = [row["label"] for row in train_rows]
    x_dev = [row["text"] for row in dev_rows]
    y_dev = [row["label"] for row in dev_rows]

    train_idx, inner_idx = train_test_split(
        list(range(len(train_rows))),
        test_size=0.2,
        random_state=args.seed,
        stratify=y_train_all,
    )
    x_train = [x_train_all[i] for i in train_idx]
    y_train = [y_train_all[i] for i in train_idx]
    x_inner = [x_train_all[i] for i in inner_idx]
    y_inner = [y_train_all[i] for i in inner_idx]

    metric_rows: list[dict[str, object]] = []
    majority = Counter(y_train_all).most_common(1)[0][0]
    for split_name, y_split in (("inner_val", y_inner), ("dev", y_dev)):
        pred = [majority] * len(y_split)
        probs = [[1.0 if label == majority else 0.0 for label in LABELS] for _ in pred]
        metric_rows.append(metric_row("majority_train", split_name, y_split, pred, probs))

    candidates = []
    for class_weight in (None, "balanced"):
        for c_value in (0.01, 0.1, 1.0, 10.0):
            for ngram_range in ((1, 1), (1, 2)):
                name = f"tfidf_lr_c{c_value}_ng{ngram_range[0]}{ngram_range[1]}_cw{class_weight or 'none'}"
                pipe = Pipeline([
                    ("tfidf", TfidfVectorizer(
                        lowercase=True,
                        ngram_range=ngram_range,
                        min_df=1,
                        max_features=50000,
                        sublinear_tf=True,
                    )),
                    ("clf", LogisticRegression(
                        C=c_value,
                        class_weight=class_weight,
                        max_iter=5000,
                        solver="lbfgs",
                    )),
                ])
                pipe.fit(x_train, y_train)
                inner_pred = pipe.predict(x_inner).tolist()
                inner_probs = aligned_predict_proba(pipe, x_inner)
                inner_metrics = metric_row(name, "inner_val", y_inner, inner_pred, inner_probs)
                candidates.append((inner_metrics["macro_f1"], inner_metrics["brier"], name, pipe, inner_metrics))
                metric_rows.append(inner_metrics)

    candidates.sort(key=lambda item: (-float(item[0]), float(item[1])))
    best_name = candidates[0][2]
    best_pipe = candidates[0][3]

    final_pipe = best_pipe
    dev_pred = final_pipe.predict(x_dev).tolist()
    dev_probs = aligned_predict_proba(final_pipe, x_dev)
    metric_rows.append(metric_row(best_name + "_selected_on_inner_train", "dev", y_dev, dev_pred, dev_probs))

    pred_rows = []
    for row, truth, prediction, probs in zip(dev_rows, y_dev, dev_pred, dev_probs):
        pred_rows.append({
            "model": best_name,
            "split": "dev",
            "id": row["id"],
            "truth": truth,
            "prediction": prediction,
            "prob_support": probs[LABELS.index("SUPPORT")],
            "prob_contradict": probs[LABELS.index("CONTRADICT")],
            "prob_not_enough_info": probs[LABELS.index("NOT_ENOUGH_INFO")],
        })

    write_tsv(
        Path(args.out),
        metric_rows,
        ["model", "split", "n", "acc", "macro_f1", "brier", "f1_support", "f1_contradict", "f1_not_enough_info"],
    )
    write_tsv(
        Path(args.pred_out),
        pred_rows,
        ["model", "split", "id", "truth", "prediction", "prob_support", "prob_contradict", "prob_not_enough_info"],
    )

    print(f"selected_on_inner_train={best_name}")
    print(f"wrote {args.out}")
    print(f"wrote {args.pred_out}")


if __name__ == "__main__":
    main()
