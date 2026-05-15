"""
Summarize validation-only prospective hypothesis-first mini-runs.

This script intentionally reads only rows for phase
`prospective_aux_weight_20260506` and writes a separate artifact directory. It
does not update the original repeated-control manuscript tables.

Usage:
    uv run python analysis/summarize_prospective_mini_run.py \
      --phase prospective_aux_weight_20260506
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


LABELS = ("yes", "no", "maybe")
ROOT = Path(__file__).resolve().parents[1]


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
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def sd(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    m = mean(values)
    return (sum((value - m) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def f1_by_label(rows: list[dict[str, str]], label: str) -> dict[str, float]:
    tp = sum(row["truth"] == label and row["prediction"] == label for row in rows)
    fp = sum(row["truth"] != label and row["prediction"] == label for row in rows)
    fn = sum(row["truth"] == label and row["prediction"] != label for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def load_prediction_metrics(pred_path: Path) -> dict[str, float]:
    rows = read_tsv(pred_path)
    out: dict[str, float] = {}
    f1s = []
    for label in LABELS:
        metrics = f1_by_label(rows, label)
        suffix = label
        out[f"precision_{suffix}"] = metrics["precision"]
        out[f"recall_{suffix}"] = metrics["recall"]
        out[f"f1_{suffix}"] = metrics["f1"]
        out[f"pred_frac_{suffix}"] = sum(row["prediction"] == label for row in rows) / len(rows)
        f1s.append(metrics["f1"])
    out["macro_f1_from_predictions"] = mean(f1s)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize prospective mini-run outputs")
    parser.add_argument("--phase", default="prospective_aux_weight_20260506")
    parser.add_argument("--results", default="results_repeated_controls.tsv")
    parser.add_argument("--outdir", default="outputs/prospective_mini_run")
    args = parser.parse_args()

    outdir = ROOT / args.outdir
    results_path = ROOT / args.results
    rows = [
        row for row in read_tsv(results_path)
        if row.get("phase") == args.phase and row.get("status") == "ok"
    ]
    if not rows:
        raise SystemExit(f"No ok rows found for phase={args.phase}")

    enriched = []
    for row in rows:
        pred_path = ROOT / row["predictions"]
        pred_metrics = load_prediction_metrics(pred_path)
        enriched_row = dict(row)
        for key, value in pred_metrics.items():
            enriched_row[key] = f"{value:.6f}"
        enriched.append(enriched_row)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in enriched:
        groups[row["config_name"]].append(row)

    metrics = [
        "lm_val_bpb",
        "pubmedqa_acc",
        "pubmedqa_macro_f1",
        "pubmedqa_brier",
        "tokens_M",
        "f1_yes",
        "f1_no",
        "f1_maybe",
        "pred_frac_yes",
        "pred_frac_no",
        "pred_frac_maybe",
    ]
    summary_rows = []
    for config_name, config_rows in sorted(groups.items()):
        summary: dict[str, object] = {
            "phase": args.phase,
            "split": "val",
            "config_name": config_name,
            "group": config_rows[0]["group"],
            "n": len(config_rows),
            "seeds": ",".join(row["seed"] for row in config_rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in config_rows]
            summary[f"{metric}_mean"] = f"{mean(values):.6f}"
            summary[f"{metric}_sd"] = f"{sd(values):.6f}"
        summary_rows.append(summary)

    write_tsv(outdir / "prospective_mini_run_rows.tsv", enriched)
    write_tsv(outdir / "prospective_mini_run_summary.tsv", summary_rows)
    write_tsv(ROOT / "outputs" / "manuscript_tables" / "table18_prospective_mini_run.tsv", summary_rows)

    readme = """# Prospective Mini-Run Outputs

This directory summarizes the validation-only prospective hypothesis-first
auxiliary-weight mini-run.

Interpretation boundaries:

- PubMedQA test was not used.
- Results demonstrate prospective protocol feasibility and class-behavior
  trade-offs only.
- These rows should not be merged into the original repeated-control evidence
  without explicitly labeling them as post-review prospective exploratory
  runs.
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote prospective mini-run summary to {outdir}")


if __name__ == "__main__":
    main()
