"""
Frozen encoder model-panel audit for PubMedQA validation.

This post-freeze audit asks whether claim-safety concerns persist across model
families rather than only in the small autoresearch model. It evaluates sparse
TF-IDF and frozen transformer encoders under identical input-mode checks:

- question + abstract context
- question only
- context only
- question + shuffled validation context

Hyperparameters are selected on an inner split of the PubMedQA training set by
default, and the locked PubMedQA validation set is used only for audit
reporting. The PubMedQA test split is not used.

Outputs:
    outputs/model_panel_audit/model_panel_metrics.tsv
    outputs/model_panel_audit/model_panel_predictions.tsv
    outputs/model_panel_audit/model_panel_context_deltas.tsv
    outputs/model_panel_audit/model_panel_brier_decomposition.tsv
    outputs/model_panel_audit/model_panel_failures.tsv
    outputs/manuscript_tables/table19_model_panel_audit.tsv
    outputs/manuscript_tables/table20_model_panel_context_calibration.tsv

Example:
    uv run python analysis/model_panel_audit.py --model-set core --device auto
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = ("yes", "no", "maybe")
TEXT_MODES = ("question_context", "question_only", "context_only")
SHUFFLED_MODE = "question_shuffled_context_eval"


@dataclass(frozen=True)
class EncoderSpec:
    model_id: str
    display_name: str
    family: str
    citation_key: str


ENCODERS = {
    "bert": EncoderSpec("bert-base-uncased", "BERT", "general", "Devlin2019BERT"),
    "scibert": EncoderSpec("allenai/scibert_scivocab_uncased", "SciBERT", "scientific", "Beltagy2019SciBERT"),
    "biobert": EncoderSpec("dmis-lab/biobert-v1.1", "BioBERT", "biomedical", "Lee2020BioBERT"),
    "biomedbert": EncoderSpec(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "BiomedBERT",
        "biomedical",
        "Gu2021PubMedBERT",
    ),
    "biolinkbert": EncoderSpec("michiyasunaga/BioLinkBERT-base", "BioLinkBERT", "biomedical-link", "Yasunaga2022LinkBERT"),
    "clinicalbert": EncoderSpec("emilyalsentzer/Bio_ClinicalBERT", "BioClinicalBERT", "clinical", "Alsentzer2019ClinicalBERT"),
}

MODEL_SETS = {
    "minimal": ["biomedbert"],
    "core": ["bert", "scibert", "biobert", "biomedbert", "biolinkbert", "clinicalbert"],
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pubmedqa(cache: Path) -> dict[str, list[dict]]:
    return {split: read_jsonl(cache / "processed" / f"{split}.jsonl") for split in ("train", "val")}


def shuffled_contexts(rows: list[dict], seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    contexts = [row["context"] for row in rows]
    indices = np.arange(len(contexts))
    for _ in range(20):
        perm = rng.permutation(indices)
        if len(perm) <= 1 or np.all(perm != indices):
            return [contexts[int(i)] for i in perm]
    return contexts[1:] + contexts[:1]


def texts_for(rows: list[dict], mode: str, *, shuffled: bool = False, seed: int = 20260507) -> list[str]:
    contexts = shuffled_contexts(rows, seed) if shuffled else [row["context"] for row in rows]
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
    values = []
    by_label = {}
    for label in LABELS:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values.append(f1)
        by_label[label] = f1
    return float(sum(values) / len(values)), by_label


def brier_score(y_true: list[str], probs: np.ndarray) -> float:
    target = np.zeros_like(probs, dtype=float)
    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    for i, label in enumerate(y_true):
        target[i, label_to_idx[label]] = 1.0
    return float(np.mean(np.sum((probs - target) ** 2, axis=1)))


def entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def ece_mce(y_true: list[str], y_pred: list[str], probs: np.ndarray, n_bins: int) -> tuple[float, float]:
    confidence = probs.max(axis=1)
    correct = np.array([a == b for a, b in zip(y_true, y_pred)], dtype=float)
    total = len(y_true)
    ece = 0.0
    mce = 0.0
    for bin_idx in range(n_bins):
        low = bin_idx / n_bins
        high = (bin_idx + 1) / n_bins
        mask = (confidence >= low) & (confidence <= high if bin_idx == n_bins - 1 else confidence < high)
        if not np.any(mask):
            continue
        gap = abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
        ece += float(mask.mean()) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def grouped_multiclass_brier_decomposition(
    y_true: list[str],
    y_pred: list[str],
    probs: np.ndarray,
    n_bins: int,
) -> dict[str, float]:
    """Grouped multiclass Brier decomposition using predicted label/confidence bins.

    This is a diagnostic approximation: forecasts are grouped by predicted
    label and confidence bin, then decomposed into reliability, resolution, and
    uncertainty components. It is used as a calibration audit, not a clinical
    reliability claim.
    """

    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    target = np.zeros_like(probs, dtype=float)
    for i, label in enumerate(y_true):
        target[i, label_to_idx[label]] = 1.0
    overall = target.mean(axis=0)
    uncertainty = float(np.sum(overall * (1.0 - overall)))

    confidence = probs.max(axis=1)
    group_keys = []
    for pred, conf in zip(y_pred, confidence):
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        group_keys.append((pred, bin_idx))

    reliability = 0.0
    resolution = 0.0
    n = len(y_true)
    for key in sorted(set(group_keys)):
        idx = np.array([i for i, item in enumerate(group_keys) if item == key], dtype=int)
        weight = len(idx) / n
        prob_bar = probs[idx].mean(axis=0)
        outcome_bar = target[idx].mean(axis=0)
        reliability += weight * float(np.sum((prob_bar - outcome_bar) ** 2))
        resolution += weight * float(np.sum((outcome_bar - overall) ** 2))
    brier = brier_score(y_true, probs)
    residual = brier - (reliability - resolution + uncertainty)
    return {
        "brier": brier,
        "uncertainty": uncertainty,
        "resolution": float(resolution),
        "reliability": float(reliability),
        "decomposition_residual": float(residual),
        "n_groups": len(set(group_keys)),
    }


def pred_distribution(y_pred: list[str]) -> dict[str, float]:
    counts = Counter(y_pred)
    n = len(y_pred)
    return {f"pred_frac_{label}": counts[label] / n for label in LABELS}


def metric_row(
    model_id: str,
    display_name: str,
    family: str,
    input_mode: str,
    train_mode: str,
    eval_mode: str,
    selected_variant: str,
    y_true: list[str],
    y_pred: list[str],
    probs: np.ndarray,
    n_bins: int,
) -> dict[str, object]:
    accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    mf1, class_f1 = macro_f1(y_true, y_pred)
    ece, mce = ece_mce(y_true, y_pred, probs, n_bins)
    confidence = probs.max(axis=1)
    correct = np.array([a == b for a, b in zip(y_true, y_pred)], dtype=bool)
    row: dict[str, object] = {
        "task": "pubmedqa",
        "split": "val",
        "model_id": model_id,
        "display_name": display_name,
        "family": family,
        "input_mode": input_mode,
        "train_mode": train_mode,
        "eval_mode": eval_mode,
        "selected_variant": selected_variant,
        "n": len(y_true),
        "accuracy": float(accuracy),
        "macro_f1": mf1,
        "brier": brier_score(y_true, probs),
        "ece": ece,
        "mce": mce,
        "mean_confidence": float(confidence.mean()),
        "mean_entropy": float(entropy(probs).mean()),
        "high_conf_wrong_rate": float(((confidence >= 0.8) & (~correct)).mean()),
    }
    for label in LABELS:
        row[f"f1_{label}"] = class_f1[label]
    row.update(pred_distribution(y_pred))
    return row


def align_proba(raw_probs: np.ndarray, classes: list[str]) -> np.ndarray:
    class_to_idx = {label: i for i, label in enumerate(classes)}
    return np.array([[float(row[class_to_idx[label]]) for label in LABELS] for row in raw_probs], dtype=float)


def fit_classifier(clf, x: np.ndarray, y: list[str]):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        clf.fit(x, y)
    if any(w.category.__name__ == "ConvergenceWarning" for w in caught):
        raise RuntimeError("LogisticRegression did not converge")
    return clf


def predict_classifier(clf, x: np.ndarray) -> tuple[list[str], np.ndarray]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        pred = clf.predict(x).tolist()
        raw_probs = clf.predict_proba(x)
    if not np.isfinite(raw_probs).all():
        raise ValueError("Classifier produced non-finite probabilities")
    return pred, align_proba(raw_probs, clf.classes_.tolist())


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inner_split_indices(labels: list[str], val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    train_idx, inner_val_idx = train_test_split(
        indices,
        test_size=val_frac,
        random_state=seed,
        stratify=labels,
    )
    return np.array(train_idx, dtype=int), np.array(inner_val_idx, dtype=int)


def tune_lr(
    x_train_full: np.ndarray,
    y_train: list[str],
    train_idx: np.ndarray,
    inner_val_idx: np.ndarray,
    model_prefix: str,
) -> tuple[object, str]:
    from sklearn.linear_model import LogisticRegression
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
                random_state=20260507,
            ),
        )

    candidates = []
    select_train_y = [y_train[int(i)] for i in train_idx]
    select_eval_y = [y_train[int(i)] for i in inner_val_idx]
    for class_weight in (None, "balanced"):
        for c_value in (0.01, 0.1, 1.0, 10.0):
            variant = f"{model_prefix}_zscore_lr_c{c_value}_cw{class_weight or 'none'}"
            clf = make_lr(c_value, class_weight)
            try:
                fit_classifier(clf, x_train_full[train_idx], select_train_y)
                pred, probs = predict_classifier(clf, x_train_full[inner_val_idx])
            except RuntimeError as exc:
                print(f"skipping {variant}: {exc}", flush=True)
                continue
            row_mf1, _ = macro_f1(select_eval_y, pred)
            row_brier = brier_score(select_eval_y, probs)
            candidates.append((row_mf1, -row_brier, variant, c_value, class_weight))
    if not candidates:
        raise RuntimeError(f"No converged LR candidates for {model_prefix}")
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    _, _, best_variant, best_c, best_weight = candidates[0]
    final_clf = make_lr(best_c, best_weight)
    fit_classifier(final_clf, x_train_full, y_train)
    return final_clf, best_variant + "_selected_on_inner_train"


def run_tfidf(
    splits: dict[str, list[dict]],
    train_idx: np.ndarray,
    inner_val_idx: np.ndarray,
    n_bins: int,
    pred_rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    y_train = [row["label"] for row in splits["train"]]
    y_val = [row["label"] for row in splits["val"]]
    rows = []
    decomp_rows = []
    selected_qc = None

    for mode in TEXT_MODES:
        x_train_text = texts_for(splits["train"], mode)
        x_val_text = texts_for(splits["val"], mode)
        candidates = []
        select_train_y = [y_train[int(i)] for i in train_idx]
        select_eval_y = [y_train[int(i)] for i in inner_val_idx]
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
                    pipe.fit([x_train_text[int(i)] for i in train_idx], select_train_y)
                    pred = pipe.predict([x_train_text[int(i)] for i in inner_val_idx]).tolist()
                    probs = align_proba(pipe.predict_proba([x_train_text[int(i)] for i in inner_val_idx]), pipe.named_steps["clf"].classes_.tolist())
                    mf1, _ = macro_f1(select_eval_y, pred)
                    candidates.append((mf1, -brier_score(select_eval_y, probs), variant, pipe))
        candidates.sort(key=lambda item: (-item[0], -item[1]))
        _, _, best_variant, best_pipe = candidates[0]
        best_pipe.fit(x_train_text, y_train)
        pred = best_pipe.predict(x_val_text).tolist()
        probs = align_proba(best_pipe.predict_proba(x_val_text), best_pipe.named_steps["clf"].classes_.tolist())
        rows.append(metric_row("tfidf_lr", "TF-IDF LR", "sparse", mode, mode, mode, best_variant, y_val, pred, probs, n_bins))
        decomp = grouped_multiclass_brier_decomposition(y_val, pred, probs, n_bins)
        decomp.update({"model_id": "tfidf_lr", "display_name": "TF-IDF LR", "input_mode": mode})
        decomp_rows.append(decomp)
        if mode == "question_context":
            selected_qc = (best_variant, best_pipe)
        for row, truth, prediction, prob in zip(splits["val"], y_val, pred, probs):
            pred_rows.append({
                "model_id": "tfidf_lr",
                "display_name": "TF-IDF LR",
                "family": "sparse",
                "input_mode": mode,
                "split": "val",
                "pmid": str(row["pmid"]),
                "truth": truth,
                "prediction": prediction,
                "prob_yes": prob[0],
                "prob_no": prob[1],
                "prob_maybe": prob[2],
            })

    assert selected_qc is not None
    best_variant, best_pipe = selected_qc
    shuffled_text = texts_for(splits["val"], "question_context", shuffled=True)
    pred = best_pipe.predict(shuffled_text).tolist()
    probs = align_proba(best_pipe.predict_proba(shuffled_text), best_pipe.named_steps["clf"].classes_.tolist())
    rows.append(metric_row("tfidf_lr", "TF-IDF LR", "sparse", SHUFFLED_MODE, "question_context", "question_shuffled_context", best_variant, y_val, pred, probs, n_bins))
    decomp = grouped_multiclass_brier_decomposition(y_val, pred, probs, n_bins)
    decomp.update({"model_id": "tfidf_lr", "display_name": "TF-IDF LR", "input_mode": SHUFFLED_MODE})
    decomp_rows.append(decomp)
    for row, truth, prediction, prob in zip(splits["val"], y_val, pred, probs):
        pred_rows.append({
            "model_id": "tfidf_lr",
            "display_name": "TF-IDF LR",
            "family": "sparse",
            "input_mode": SHUFFLED_MODE,
            "split": "val",
            "pmid": str(row["pmid"]),
            "truth": truth,
            "prediction": prediction,
            "prob_yes": prob[0],
            "prob_no": prob[1],
            "prob_maybe": prob[2],
        })
    return rows, decomp_rows


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
    model_id: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        print(f"fast tokenizer failed for {model_id}: {exc!r}; retrying with use_fast=False", flush=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        except Exception as slow_exc:
            if "biobert-base-cased" not in model_id.lower():
                raise
            print(
                f"slow tokenizer failed for {model_id}: {slow_exc!r}; "
                "using bert-base-cased tokenizer fallback for BioBERT",
                flush=True,
            )
            tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained(model_id)
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
            print(f"{safe_name(model_id)} embedded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(chunks)


def load_or_embed(
    texts: list[str],
    split: str,
    mode: str,
    spec: EncoderSpec,
    outdir: Path,
    device: str,
    batch_size: int,
    max_length: int,
    reuse: bool,
) -> np.ndarray:
    emb_dir = outdir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    path = emb_dir / f"pubmedqa_panel_{safe_name(spec.model_id)}_mean_len{max_length}_{mode}_{split}.npz"
    if reuse and path.exists():
        data = np.load(path)
        print(f"loaded embeddings {path}", flush=True)
        return data["embeddings"]
    embeddings = embed_texts(texts, spec.model_id, device, batch_size, max_length)
    np.savez_compressed(path, embeddings=embeddings)
    print(f"wrote embeddings {path}", flush=True)
    return embeddings


def run_encoder(
    spec: EncoderSpec,
    splits: dict[str, list[dict]],
    train_idx: np.ndarray,
    inner_val_idx: np.ndarray,
    outdir: Path,
    device: str,
    batch_size: int,
    max_length: int,
    reuse: bool,
    n_bins: int,
    pred_rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    y_train = [row["label"] for row in splits["train"]]
    y_val = [row["label"] for row in splits["val"]]
    rows = []
    decomp_rows = []
    selected_qc = None

    embeddings: dict[str, dict[str, np.ndarray]] = {}
    for mode in TEXT_MODES:
        embeddings[mode] = {
            "train": load_or_embed(texts_for(splits["train"], mode), "train", mode, spec, outdir, device, batch_size, max_length, reuse),
            "val": load_or_embed(texts_for(splits["val"], mode), "val", mode, spec, outdir, device, batch_size, max_length, reuse),
        }
    shuffled_val = load_or_embed(
        texts_for(splits["val"], "question_context", shuffled=True),
        "val",
        "question_shuffled_context",
        spec,
        outdir,
        device,
        batch_size,
        max_length,
        reuse,
    )

    for mode in TEXT_MODES:
        clf, selected_variant = tune_lr(
            embeddings[mode]["train"],
            y_train,
            train_idx,
            inner_val_idx,
            f"frozen_{safe_name(spec.model_id)}_{mode}",
        )
        pred, probs = predict_classifier(clf, embeddings[mode]["val"])
        rows.append(metric_row(spec.model_id, spec.display_name, spec.family, mode, mode, mode, selected_variant, y_val, pred, probs, n_bins))
        decomp = grouped_multiclass_brier_decomposition(y_val, pred, probs, n_bins)
        decomp.update({"model_id": spec.model_id, "display_name": spec.display_name, "input_mode": mode})
        decomp_rows.append(decomp)
        if mode == "question_context":
            selected_qc = (selected_variant, clf)
        for row, truth, prediction, prob in zip(splits["val"], y_val, pred, probs):
            pred_rows.append({
                "model_id": spec.model_id,
                "display_name": spec.display_name,
                "family": spec.family,
                "input_mode": mode,
                "split": "val",
                "pmid": str(row["pmid"]),
                "truth": truth,
                "prediction": prediction,
                "prob_yes": prob[0],
                "prob_no": prob[1],
                "prob_maybe": prob[2],
            })

    assert selected_qc is not None
    selected_variant, qc_clf = selected_qc
    pred, probs = predict_classifier(qc_clf, shuffled_val)
    rows.append(metric_row(spec.model_id, spec.display_name, spec.family, SHUFFLED_MODE, "question_context", "question_shuffled_context", selected_variant, y_val, pred, probs, n_bins))
    decomp = grouped_multiclass_brier_decomposition(y_val, pred, probs, n_bins)
    decomp.update({"model_id": spec.model_id, "display_name": spec.display_name, "input_mode": SHUFFLED_MODE})
    decomp_rows.append(decomp)
    for row, truth, prediction, prob in zip(splits["val"], y_val, pred, probs):
        pred_rows.append({
            "model_id": spec.model_id,
            "display_name": spec.display_name,
            "family": spec.family,
            "input_mode": SHUFFLED_MODE,
            "split": "val",
            "pmid": str(row["pmid"]),
            "truth": truth,
            "prediction": prediction,
            "prob_yes": prob[0],
            "prob_no": prob[1],
            "prob_maybe": prob[2],
        })
    return rows, decomp_rows


def build_context_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, model_df in metrics.groupby("model_id", sort=False):
        qc = model_df[model_df["input_mode"].eq("question_context")]
        if qc.empty:
            continue
        qc_row = qc.iloc[0]
        for _, row in model_df.iterrows():
            if row["input_mode"] == "question_context":
                continue
            rows.append({
                "model_id": model_id,
                "display_name": row["display_name"],
                "family": row["family"],
                "comparison": f"{row['input_mode']} vs question_context",
                "macro_f1_delta": row["macro_f1"] - qc_row["macro_f1"],
                "accuracy_delta": row["accuracy"] - qc_row["accuracy"],
                "brier_delta": row["brier"] - qc_row["brier"],
                "ece_delta": row["ece"] - qc_row["ece"],
                "question_context_macro_f1": qc_row["macro_f1"],
                "comparison_macro_f1": row["macro_f1"],
                "question_context_brier": qc_row["brier"],
                "comparison_brier": row["brier"],
            })
    return pd.DataFrame(rows)


def build_manuscript_tables(metrics: pd.DataFrame, deltas: pd.DataFrame, outdir: Path) -> None:
    table_dir = Path("outputs/manuscript_tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    qc = metrics[metrics["input_mode"].eq("question_context")].copy()
    qc = qc.sort_values(["family", "display_name"])
    panel_cols = [
        "display_name", "family", "accuracy", "macro_f1", "f1_yes", "f1_no", "f1_maybe",
        "brier", "ece", "mean_confidence", "pred_frac_yes", "pred_frac_no", "pred_frac_maybe",
    ]
    qc[panel_cols].to_csv(table_dir / "table19_model_panel_audit.tsv", sep="\t", index=False)

    keep = deltas[deltas["comparison"].isin([
        "question_only vs question_context",
        "context_only vs question_context",
        "question_shuffled_context_eval vs question_context",
    ])].copy()
    keep_cols = [
        "display_name", "family", "comparison", "macro_f1_delta", "accuracy_delta",
        "brier_delta", "ece_delta", "question_context_macro_f1", "comparison_macro_f1",
    ]
    keep[keep_cols].to_csv(table_dir / "table20_model_panel_context_calibration.tsv", sep="\t", index=False)

    # Keep copies in the model-panel folder too.
    qc[panel_cols].to_csv(outdir / "table19_model_panel_audit.tsv", sep="\t", index=False)
    keep[keep_cols].to_csv(outdir / "table20_model_panel_context_calibration.tsv", sep="\t", index=False)


def write_readme(outdir: Path, model_keys: list[str]) -> None:
    lines = [
        "# PubMedQA Frozen Encoder Model-Panel Audit",
        "",
        "This is a post-freeze validation-only audit. It does not use the PubMedQA test split.",
        "",
        "## Models",
        "",
    ]
    for key in model_keys:
        if key == "tfidf_lr":
            lines.append("- TF-IDF logistic regression: sparse lexical reference baseline.")
        else:
            spec = ENCODERS[key]
            lines.append(f"- {spec.display_name}: `{spec.model_id}` ({spec.family}; citation key `{spec.citation_key}`).")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The goal is not to find a new PubMedQA SOTA model. The audit tests whether",
        "context-use artifacts, calibration trade-offs, and class-behavior shifts are",
        "visible across sparse, general-domain, scientific, biomedical, and clinical",
        "reference encoders.",
    ])
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PubMedQA model-panel audit")
    parser.add_argument("--cache", default="~/.cache/autoresearch_biomed")
    parser.add_argument("--outdir", default="outputs/model_panel_audit")
    parser.add_argument("--model-set", choices=sorted(MODEL_SETS), default="core")
    parser.add_argument("--models", default="", help="Comma-separated encoder keys; overrides --model-set. Use 'tfidf_lr' to include TF-IDF.")
    parser.add_argument("--skip-tfidf", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--inner-val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--no-reuse-embeddings", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits = load_pubmedqa(Path(args.cache).expanduser())
    y_train = [row["label"] for row in splits["train"]]
    train_idx, inner_val_idx = inner_split_indices(y_train, args.inner_val_frac, args.seed)

    if args.models.strip():
        model_keys = [item.strip() for item in args.models.split(",") if item.strip()]
    else:
        model_keys = list(MODEL_SETS[args.model_set])
    if not args.skip_tfidf and "tfidf_lr" not in model_keys:
        model_keys = ["tfidf_lr", *model_keys]

    metric_rows: list[dict] = []
    pred_rows: list[dict] = []
    decomp_rows: list[dict] = []
    failures: list[dict] = []

    if "tfidf_lr" in model_keys:
        rows, decomp = run_tfidf(splits, train_idx, inner_val_idx, args.n_bins, pred_rows)
        metric_rows.extend(rows)
        decomp_rows.extend(decomp)

    device = resolve_device(args.device)
    for key in model_keys:
        if key == "tfidf_lr":
            continue
        if key not in ENCODERS:
            failures.append({"model_key": key, "model_id": key, "error": "unknown model key"})
            continue
        spec = ENCODERS[key]
        print(f"running encoder={spec.display_name} model_id={spec.model_id} device={device}", flush=True)
        try:
            rows, decomp = run_encoder(
                spec=spec,
                splits=splits,
                train_idx=train_idx,
                inner_val_idx=inner_val_idx,
                outdir=outdir,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                reuse=not args.no_reuse_embeddings,
                n_bins=args.n_bins,
                pred_rows=pred_rows,
            )
            metric_rows.extend(rows)
            decomp_rows.extend(decomp)
        except Exception as exc:  # Keep the panel robust if one public model is unavailable.
            failures.append({"model_key": key, "model_id": spec.model_id, "display_name": spec.display_name, "error": repr(exc)})
            print(f"FAILED {spec.display_name}: {exc!r}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(outdir / "model_panel_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(pred_rows).to_csv(outdir / "model_panel_predictions.tsv", sep="\t", index=False)
    pd.DataFrame(decomp_rows).to_csv(outdir / "model_panel_brier_decomposition.tsv", sep="\t", index=False)
    pd.DataFrame(failures).to_csv(outdir / "model_panel_failures.tsv", sep="\t", index=False)

    deltas = build_context_deltas(metrics)
    deltas.to_csv(outdir / "model_panel_context_deltas.tsv", sep="\t", index=False)
    build_manuscript_tables(metrics, deltas, outdir)
    write_readme(outdir, model_keys)
    print(f"wrote model-panel audit to {outdir}", flush=True)


if __name__ == "__main__":
    main()
