"""
Summarize PubMedQA prediction TSV files.

The input format is produced by eval_pubmedqa.write_predictions_tsv().

Usage:
    uv run python analysis/summarize_pubmedqa_predictions.py \
      --pred outputs/predictions/pubmedqa_post_freeze_test_audit_seed42.tsv \
      --out outputs/prediction_audit_seed42.tsv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict


LABELS = ("yes", "no", "maybe")


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def group_key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("model", "model"), row.get("split", "split")


def per_class_metrics(model: str, split: str, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = []
    for label in LABELS:
        tp = sum(r["truth"] == label and r["prediction"] == label for r in rows)
        fp = sum(r["truth"] != label and r["prediction"] == label for r in rows)
        fn = sum(r["truth"] == label and r["prediction"] != label for r in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append({
            "model": model,
            "split": split,
            "metric": "per_class",
            "label": label,
            "support": sum(r["truth"] == label for r in rows),
            "predicted": sum(r["prediction"] == label for r in rows),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count": "",
        })
    return out


def confusion_rows(model: str, split: str, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = []
    for truth in LABELS:
        for pred in LABELS:
            out.append({
                "model": model,
                "split": split,
                "metric": "confusion",
                "label": f"{truth}->{pred}",
                "support": "",
                "predicted": "",
                "precision": "",
                "recall": "",
                "f1": "",
                "count": sum(r["truth"] == truth and r["prediction"] == pred for r in rows),
            })
    return out


def distribution_rows(model: str, split: str, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    truth_counts = Counter(r["truth"] for r in rows)
    pred_counts = Counter(r["prediction"] for r in rows)
    out = []
    for label in LABELS:
        out.append({
            "model": model,
            "split": split,
            "metric": "distribution",
            "label": label,
            "support": truth_counts[label],
            "predicted": pred_counts[label],
            "precision": "",
            "recall": "",
            "f1": "",
            "count": "",
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PubMedQA prediction TSV")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = read_rows(args.pred)
    by_group = defaultdict(list)
    for row in rows:
        by_group[group_key(row)].append(row)

    output_rows = []
    for (model, split), group_rows in sorted(by_group.items()):
        output_rows.extend(per_class_metrics(model, split, group_rows))
        output_rows.extend(confusion_rows(model, split, group_rows))
        output_rows.extend(distribution_rows(model, split, group_rows))

    fieldnames = ["model", "split", "metric", "label", "support", "predicted", "precision", "recall", "f1", "count"]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
