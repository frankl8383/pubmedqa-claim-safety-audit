"""
Locked PubMedQA evaluation utilities.

The training script imports evaluate_pubmedqa() and passes the in-memory model.
When run directly, this file reports split statistics and majority baselines; it
does not train or load a checkpoint.
"""

import argparse
import csv
import json
import math
import os
from collections import Counter

import torch
import torch.nn.functional as F

from prepare_biomed import LABELS, MAX_SEQ_LEN, PROCESSED_DIR, Tokenizer, format_prompt, read_jsonl


CONTEXT_ABLATION_MODES = (
    "question_context",
    "question_only",
    "context_removed",
    "context_only",
    "shuffled_context",
    "question_shuffled",
)


def _softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [x / total for x in exps]


def _macro_f1(y_true, y_pred, labels=LABELS):
    f1s = []
    by_label = {}
    for label in labels:
        tp = sum(yt == label and yp == label for yt, yp in zip(y_true, y_pred))
        fp = sum(yt != label and yp == label for yt, yp in zip(y_true, y_pred))
        fn = sum(yt == label and yp != label for yt, yp in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        by_label[label] = f1
    return sum(f1s) / len(f1s), by_label


def _brier_score(y_true, probabilities, labels=LABELS):
    label_to_idx = {label: i for i, label in enumerate(labels)}
    total = 0.0
    for truth, probs in zip(y_true, probabilities):
        target = [0.0] * len(labels)
        target[label_to_idx[truth]] = 1.0
        total += sum((p - t) ** 2 for p, t in zip(probs, target))
    return total / len(y_true) if y_true else 0.0


def format_prompt_for_mode(row, mode="question_context", paired_row=None):
    """Return a PubMedQA prompt variant for context-use sanity checks.

    The default mode exactly matches the training/evaluation prompt. Other modes
    preserve the answer-label scoring setup while perturbing only the input
    fields before the answer marker.
    """
    if mode not in CONTEXT_ABLATION_MODES:
        raise ValueError(f"Unsupported context-ablation mode: {mode}")
    paired_row = paired_row or row

    if mode == "question_context":
        return format_prompt(row)
    if mode == "question_only":
        return (
            "<|question|>\n"
            f"{row['question']}\n\n"
            "<|context|>\n\n"
            "<|answer|>\n"
        )
    if mode == "context_removed":
        return (
            "<|question|>\n"
            f"{row['question']}\n\n"
            "<|answer|>\n"
        )
    if mode == "context_only":
        return (
            "<|question|>\n\n"
            "<|context|>\n"
            f"{row['context']}\n\n"
            "<|answer|>\n"
        )
    if mode == "shuffled_context":
        return (
            "<|question|>\n"
            f"{row['question']}\n\n"
            "<|context|>\n"
            f"{paired_row['context']}\n\n"
            "<|answer|>\n"
        )
    if mode == "question_shuffled":
        return (
            "<|question|>\n"
            f"{paired_row['question']}\n\n"
            "<|context|>\n"
            f"{row['context']}\n\n"
            "<|answer|>\n"
        )
    raise AssertionError(f"Unhandled mode: {mode}")


def _score_label(model, tokenizer, prompt, label, device, max_seq_len=MAX_SEQ_LEN):
    prompt_ids = tokenizer.encode(prompt)
    label_ids = tokenizer.encode(label)
    if not prompt_ids or not label_ids:
        return float("-inf")

    max_prompt_len = max_seq_len - len(label_ids)
    if max_prompt_len < 1:
        raise ValueError("Label tokenization is longer than max sequence length")
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    ids = prompt_ids + label_ids
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    targets = torch.tensor(ids[1:], dtype=torch.long, device=device)
    logits = model(x)
    log_probs = F.log_softmax(logits[0].float(), dim=-1)

    start = len(prompt_ids) - 1
    token_scores = []
    for i, target_id in enumerate(targets[start:start + len(label_ids)]):
        token_scores.append(log_probs[start + i, target_id].item())
    return sum(token_scores) / len(token_scores)


@torch.no_grad()
def evaluate_pubmedqa(
    model,
    tokenizer,
    split="val",
    max_examples=None,
    max_seq_len=MAX_SEQ_LEN,
    prompt_mode="question_context",
    shuffle_offset=37,
):
    """Evaluate yes/no/maybe by length-normalized label log-probability."""
    path = os.path.join(PROCESSED_DIR, f"{split}.jsonl")
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[:max_examples]
    if not rows:
        raise ValueError(f"No PubMedQA rows available for split={split!r}")
    if prompt_mode not in CONTEXT_ABLATION_MODES:
        raise ValueError(f"Unsupported context-ablation mode: {prompt_mode}")

    device = next(model.parameters()).device
    model.eval()

    y_true = []
    y_pred = []
    probabilities = []
    predictions = []
    n_rows = len(rows)
    offset = shuffle_offset % n_rows
    if offset == 0 and n_rows > 1:
        offset = 1
    for idx, row in enumerate(rows):
        paired_row = rows[(idx + offset) % n_rows]
        prompt = format_prompt_for_mode(row, prompt_mode, paired_row=paired_row)
        scores = [_score_label(model, tokenizer, prompt, label, device, max_seq_len=max_seq_len) for label in LABELS]
        probs = _softmax(scores)
        pred = LABELS[max(range(len(LABELS)), key=lambda i: scores[i])]
        truth = row["label"]

        y_true.append(truth)
        y_pred.append(pred)
        probabilities.append(probs)
        predictions.append({
            "pmid": row["pmid"],
            "truth": truth,
            "prediction": pred,
            "prompt_mode": prompt_mode,
            "paired_pmid": paired_row["pmid"] if prompt_mode in {"shuffled_context", "question_shuffled"} else "",
            "scores": {label: scores[i] for i, label in enumerate(LABELS)},
            "probabilities": {label: probs[i] for i, label in enumerate(LABELS)},
        })

    accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    macro_f1, class_f1 = _macro_f1(y_true, y_pred)
    brier = _brier_score(y_true, probabilities)

    return {
        "pubmedqa_acc": accuracy,
        "pubmedqa_macro_f1": macro_f1,
        "pubmedqa_brier": brier,
        "pubmedqa_class_f1": class_f1,
        "pubmedqa_n": len(y_true),
        "pubmedqa_prompt_mode": prompt_mode,
        "pubmedqa_predictions": predictions,
    }


def majority_baseline(split):
    rows = read_jsonl(os.path.join(PROCESSED_DIR, f"{split}.jsonl"))
    counts = Counter(row["label"] for row in rows)
    majority = counts.most_common(1)[0][0]
    y_true = [row["label"] for row in rows]
    y_pred = [majority] * len(rows)
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    macro_f1, class_f1 = _macro_f1(y_true, y_pred)
    return {
        "split": split,
        "n": len(rows),
        "counts": dict(counts),
        "majority_label": majority,
        "majority_acc": acc,
        "majority_macro_f1": macro_f1,
        "majority_class_f1": class_f1,
    }


def write_predictions_tsv(path, predictions):
    fieldnames = [
        "pmid",
        "truth",
        "prediction",
        "score_yes",
        "score_no",
        "score_maybe",
        "prob_yes",
        "prob_no",
        "prob_maybe",
        "prompt_mode",
        "paired_pmid",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in predictions:
            writer.writerow({
                "pmid": row["pmid"],
                "truth": row["truth"],
                "prediction": row["prediction"],
                "score_yes": row["scores"]["yes"],
                "score_no": row["scores"]["no"],
                "score_maybe": row["scores"]["maybe"],
                "prob_yes": row["probabilities"]["yes"],
                "prob_no": row["probabilities"]["no"],
                "prob_maybe": row["probabilities"]["maybe"],
                "prompt_mode": row.get("prompt_mode", "question_context"),
                "paired_pmid": row.get("paired_pmid", ""),
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report locked PubMedQA split statistics and majority baselines")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    args = parser.parse_args()

    # Also checks that tokenizer exists, so users get an early setup error.
    Tokenizer.from_directory()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        print(json.dumps(majority_baseline(split), indent=2, ensure_ascii=False))
