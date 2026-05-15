"""
Context-use sanity checks for PubMedQA validation.

The goal is to test whether reference baselines rely on the PubMedQA context or
can obtain similar behavior from question wording and label priors. These
post-freeze analyses use validation only and must not be used for PubMedQA test
selection.

Outputs:
    outputs/context_use_sanity/pubmedqa_context_use_metrics.tsv
    outputs/context_use_sanity/pubmedqa_context_use_predictions.tsv
    outputs/context_use_sanity/pubmedqa_context_use_summary.tsv
    outputs/manuscript_tables/table13_context_use_sanity.tsv

Usage:
    uv run python analysis/context_use_sanity_check.py --run-frozen
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np


LABELS = ("yes", "no", "maybe")
DEFAULT_ENCODER = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
TEXT_MODES = ("question_context", "question_only", "context_only")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_splits(cache: Path) -> dict[str, list[dict]]:
    return {split: read_jsonl(cache / "processed" / f"{split}.jsonl") for split in ("train", "val")}


def shuffled_contexts(rows: list[dict], seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    contexts = [row["context"] for row in rows]
    indices = np.arange(len(contexts))
    for _ in range(10):
        perm = rng.permutation(indices)
        if len(perm) <= 1 or np.all(perm != indices):
            return [contexts[int(i)] for i in perm]
    # Deterministic fallback for very small arrays.
    return contexts[1:] + contexts[:1]


def texts_for(rows: list[dict], mode: str, shuffled: bool = False, seed: int = 20260505) -> list[str]:
    if shuffled:
        contexts = shuffled_contexts(rows, seed)
    else:
        contexts = [row["context"] for row in rows]
    texts = []
    for row, context in zip(rows, contexts):
        if mode == "question_context":
            texts.append(f"Question: {row['question']}\nContext: {context}")
        elif mode == "question_only":
            texts.append(f"Question: {row['question']}")
        elif mode == "context_only":
            texts.append(f"Context: {context}")
        else:
            raise ValueError(f"Unknown text mode: {mode}")
    return texts


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


def pred_distribution(y_pred: list[str]) -> dict[str, float]:
    n = len(y_pred)
    counts = Counter(y_pred)
    return {f"pred_frac_{label}": counts[label] / n for label in LABELS}


def metrics_row(
    model: str,
    mode: str,
    train_mode: str,
    eval_mode: str,
    y_true: list[str],
    y_pred: list[str],
    probs: list[list[float]],
    selected_variant: str,
) -> dict[str, object]:
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    mf1, class_f1 = macro_f1(y_true, y_pred)
    row = {
        "model": model,
        "mode": mode,
        "train_mode": train_mode,
        "eval_mode": eval_mode,
        "split": "val",
        "selected_variant": selected_variant,
        "n": len(y_true),
        "accuracy": acc,
        "macro_f1": mf1,
        "brier": brier_score(y_true, probs),
        "f1_yes": class_f1["yes"],
        "f1_no": class_f1["no"],
        "f1_maybe": class_f1["maybe"],
    }
    row.update(pred_distribution(y_pred))
    return row


def align_proba(raw_probs, classes: list[str]) -> list[list[float]]:
    class_to_idx = {label: i for i, label in enumerate(classes)}
    return [[float(row[class_to_idx[label]]) for label in LABELS] for row in raw_probs]


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


def run_tfidf(splits: dict[str, list[dict]], pred_rows: list[dict]) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    y = {split: [row["label"] for row in rows] for split, rows in splits.items()}
    ids = {split: [str(row["pmid"]) for row in rows] for split, rows in splits.items()}
    metric_rows = []

    majority = Counter(y["train"]).most_common(1)[0][0]
    majority_pred = [majority] * len(y["val"])
    majority_probs = [[1.0 if label == majority else 0.0 for label in LABELS] for _ in y["val"]]
    metric_rows.append(metrics_row(
        model="majority_train",
        mode="majority",
        train_mode="label_prior",
        eval_mode="label_prior",
        y_true=y["val"],
        y_pred=majority_pred,
        probs=majority_probs,
        selected_variant="majority_train",
    ))

    selected_pipes = {}
    for mode in TEXT_MODES:
        x_train = texts_for(splits["train"], mode)
        x_val = texts_for(splits["val"], mode)
        candidates = []
        for class_weight in (None, "balanced"):
            for c_value in (0.01, 0.1, 1.0, 10.0):
                for ngram_range in ((1, 1), (1, 2)):
                    variant = f"tfidf_lr_{mode}_c{c_value}_ng{ngram_range[0]}{ngram_range[1]}_cw{class_weight or 'none'}"
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
                    pipe.fit(x_train, y["train"])
                    val_pred = pipe.predict(x_val).tolist()
                    val_probs = align_proba(pipe.predict_proba(x_val), pipe.named_steps["clf"].classes_.tolist())
                    row = metrics_row("tfidf_lr", mode, mode, mode, y["val"], val_pred, val_probs, variant)
                    candidates.append((row["macro_f1"], -row["brier"], variant, pipe, row))
        candidates.sort(key=lambda item: (-item[0], -item[1]))
        best_macro_f1, _, best_variant, best_pipe, best_row = candidates[0]
        selected_pipes[mode] = (best_variant, best_pipe)
        best_row["selected_variant"] = best_variant
        metric_rows.append(best_row)

        val_pred = best_pipe.predict(x_val).tolist()
        val_probs = align_proba(best_pipe.predict_proba(x_val), best_pipe.named_steps["clf"].classes_.tolist())
        for row, truth, pred, probs in zip(splits["val"], y["val"], val_pred, val_probs):
            pred_rows.append({
                "model": "tfidf_lr",
                "mode": mode,
                "split": "val",
                "pmid": row["pmid"],
                "truth": truth,
                "prediction": pred,
                "prob_yes": probs[0],
                "prob_no": probs[1],
                "prob_maybe": probs[2],
            })

    # Corrupt only validation contexts while keeping the question-context model fixed.
    best_variant, best_pipe = selected_pipes["question_context"]
    shuffled_val = texts_for(splits["val"], "question_context", shuffled=True)
    shuffled_pred = best_pipe.predict(shuffled_val).tolist()
    shuffled_probs = align_proba(best_pipe.predict_proba(shuffled_val), best_pipe.named_steps["clf"].classes_.tolist())
    metric_rows.append(metrics_row(
        model="tfidf_lr",
        mode="question_shuffled_context_eval",
        train_mode="question_context",
        eval_mode="question_shuffled_context",
        y_true=y["val"],
        y_pred=shuffled_pred,
        probs=shuffled_probs,
        selected_variant=best_variant,
    ))
    for row, truth, pred, probs in zip(splits["val"], y["val"], shuffled_pred, shuffled_probs):
        pred_rows.append({
            "model": "tfidf_lr",
            "mode": "question_shuffled_context_eval",
            "split": "val",
            "pmid": row["pmid"],
            "truth": truth,
            "prediction": pred,
            "prob_yes": probs[0],
            "prob_no": probs[1],
            "prob_maybe": probs[2],
        })

    return metric_rows


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


def embed_texts(texts: list[str], model_name: str, device: str, batch_size: int, max_length: int) -> np.ndarray:
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
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            chunks.append(pooled.detach().cpu().float().numpy())
            print(f"embedded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(chunks)


def load_or_embed_context_use(
    texts: list[str],
    split: str,
    mode: str,
    model_name: str,
    outdir: Path,
    device: str,
    batch_size: int,
    max_length: int,
    reuse: bool,
) -> np.ndarray:
    emb_dir = outdir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pubmedqa_contextuse_{safe_name(model_name)}_mean_len{max_length}_{mode}_{split}.npz"
    path = emb_dir / stem
    if reuse and path.exists():
        data = np.load(path)
        print(f"loaded embeddings {path}")
        return data["embeddings"]
    embeddings = embed_texts(texts, model_name, device, batch_size, max_length)
    np.savez_compressed(path, embeddings=embeddings)
    print(f"wrote embeddings {path}")
    return embeddings


def fit_lr(clf, x: np.ndarray, y: list[str]):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        clf.fit(x, y)
    if any(w.category.__name__ == "ConvergenceWarning" for w in caught):
        raise RuntimeError("LogisticRegression did not converge")
    return clf


def predict_lr(clf, x: np.ndarray) -> tuple[list[str], list[list[float]]]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        pred = clf.predict(x).tolist()
        raw_probs = clf.predict_proba(x)
    if not np.isfinite(raw_probs).all():
        raise ValueError("non-finite LR probabilities")
    return pred, align_proba(raw_probs, clf.classes_.tolist())


def run_frozen(
    splits: dict[str, list[dict]],
    pred_rows: list[dict],
    outdir: Path,
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
    reuse: bool,
) -> list[dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = {split: [row["label"] for row in rows] for split, rows in splits.items()}
    metric_rows = []

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

    embeddings = {}
    for mode in TEXT_MODES:
        embeddings[mode] = {}
        for split in ("train", "val"):
            embeddings[mode][split] = load_or_embed_context_use(
                texts_for(splits[split], mode),
                split,
                mode,
                model_name,
                outdir,
                device,
                batch_size,
                max_length,
                reuse,
            )
    shuffled_val_emb = load_or_embed_context_use(
        texts_for(splits["val"], "question_context", shuffled=True),
        "val",
        "question_shuffled_context",
        model_name,
        outdir,
        device,
        batch_size,
        max_length,
        reuse,
    )

    selected = {}
    for mode in TEXT_MODES:
        candidates = []
        for class_weight in (None, "balanced"):
            for c_value in (0.01, 0.1, 1.0, 10.0):
                variant = f"frozen_biomedbert_{mode}_zscore_lr_c{c_value}_cw{class_weight or 'none'}"
                clf = make_lr(c_value, class_weight)
                try:
                    fit_lr(clf, embeddings[mode]["train"], y["train"])
                    pred, probs = predict_lr(clf, embeddings[mode]["val"])
                except RuntimeError as exc:
                    print(f"skipping {variant}: {exc}", flush=True)
                    continue
                row = metrics_row("frozen_biomedbert", mode, mode, mode, y["val"], pred, probs, variant)
                candidates.append((row["macro_f1"], -row["brier"], variant, clf, row))
        candidates.sort(key=lambda item: (-item[0], -item[1]))
        if not candidates:
            raise RuntimeError(f"No converged frozen encoder candidates for {mode}")
        _, _, best_variant, best_clf, best_row = candidates[0]
        selected[mode] = (best_variant, best_clf)
        metric_rows.append(best_row)
        pred, probs = predict_lr(best_clf, embeddings[mode]["val"])
        for row, truth, prediction, prob in zip(splits["val"], y["val"], pred, probs):
            pred_rows.append({
                "model": "frozen_biomedbert",
                "mode": mode,
                "split": "val",
                "pmid": row["pmid"],
                "truth": truth,
                "prediction": prediction,
                "prob_yes": prob[0],
                "prob_no": prob[1],
                "prob_maybe": prob[2],
            })

    best_variant, qc_clf = selected["question_context"]
    shuffled_pred, shuffled_probs = predict_lr(qc_clf, shuffled_val_emb)
    metric_rows.append(metrics_row(
        model="frozen_biomedbert",
        mode="question_shuffled_context_eval",
        train_mode="question_context",
        eval_mode="question_shuffled_context",
        y_true=y["val"],
        y_pred=shuffled_pred,
        probs=shuffled_probs,
        selected_variant=best_variant,
    ))
    for row, truth, prediction, prob in zip(splits["val"], y["val"], shuffled_pred, shuffled_probs):
        pred_rows.append({
            "model": "frozen_biomedbert",
            "mode": "question_shuffled_context_eval",
            "split": "val",
            "pmid": row["pmid"],
            "truth": truth,
            "prediction": prediction,
            "prob_yes": prob[0],
            "prob_no": prob[1],
            "prob_maybe": prob[2],
        })

    return metric_rows


def build_summary(metrics: list[dict]) -> list[dict]:
    by_model_mode = {(row["model"], row["mode"]): row for row in metrics}
    rows = []
    for model in sorted({row["model"] for row in metrics}):
        if model == "majority_train":
            continue
        qc = by_model_mode[(model, "question_context")]
        for mode in ("question_only", "context_only", "question_shuffled_context_eval"):
            row = by_model_mode[(model, mode)]
            rows.append({
                "model": model,
                "comparison": f"{mode} vs question_context",
                "macro_f1_delta": row["macro_f1"] - qc["macro_f1"],
                "accuracy_delta": row["accuracy"] - qc["accuracy"],
                "brier_delta": row["brier"] - qc["brier"],
                "question_context_macro_f1": qc["macro_f1"],
                "comparison_macro_f1": row["macro_f1"],
                "question_context_accuracy": qc["accuracy"],
                "comparison_accuracy": row["accuracy"],
                "question_context_brier": qc["brier"],
                "comparison_brier": row["brier"],
            })
    return rows


def write_readme(outdir: Path) -> None:
    text = """# PubMedQA Context-Use Sanity Check

This post-freeze validation-only audit tests whether reference baselines depend
on PubMedQA abstract context or can obtain similar behavior from partial input.

Input modes:

- `question_context`: train/evaluate on question plus abstract context.
- `question_only`: train/evaluate on question text only.
- `context_only`: train/evaluate on abstract context only.
- `question_shuffled_context_eval`: train on normal question+context, then
  evaluate with each validation question paired to another validation context.

Interpretation:

- Strong question-only performance suggests question wording or label-prior
  artifacts.
- Context-only performance suggests the abstract contains label signal even
  without the explicit question.
- A small drop under shuffled-context evaluation suggests the model is not
  strongly using question-context alignment.

This audit uses PubMedQA validation only and must not guide PubMedQA test
selection.
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PubMedQA context-use sanity checks")
    parser.add_argument("--cache", default="~/.cache/autoresearch_biomed")
    parser.add_argument("--outdir", default="outputs/context_use_sanity")
    parser.add_argument("--run-frozen", action="store_true")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--no-reuse-embeddings", action="store_true")
    args = parser.parse_args()

    cache = Path(args.cache).expanduser()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits = load_splits(cache)

    pred_rows = []
    metric_rows = run_tfidf(splits, pred_rows)
    if args.run_frozen:
        device = resolve_device(args.device)
        print(f"running frozen encoder context-use audit on device={device}")
        metric_rows.extend(run_frozen(
            splits=splits,
            pred_rows=pred_rows,
            outdir=outdir,
            model_name=args.encoder,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            reuse=not args.no_reuse_embeddings,
        ))

    summary_rows = build_summary(metric_rows)
    write_rows(outdir / "pubmedqa_context_use_metrics.tsv", metric_rows)
    write_rows(outdir / "pubmedqa_context_use_predictions.tsv", pred_rows)
    write_rows(outdir / "pubmedqa_context_use_summary.tsv", summary_rows)
    write_rows(Path("outputs/manuscript_tables/table13_context_use_sanity.tsv"), summary_rows)
    write_readme(outdir)
    print(f"wrote context-use sanity outputs to {outdir}")


if __name__ == "__main__":
    main()
