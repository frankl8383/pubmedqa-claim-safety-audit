"""
Biomedical data preparation and runtime utilities for PubMedQA autoresearch.

Usage:
    uv run prepare_biomed.py

Data, tokenizer, and fixed splits are stored in ~/.cache/autoresearch_biomed/.
The default MVP uses PubMedQA's expert-labeled PQA-L file. The downloader tries
the official GitHub raw URL first, then a GitHub proxy that is often reachable
from networks where raw.githubusercontent.com is blocked.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict

import requests
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants locked for biomedical autoresearch
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = int(os.environ.get("AUTORESEARCH_MAX_SEQ_LEN", "1024"))
TIME_BUDGET = int(os.environ.get("AUTORESEARCH_TIME_BUDGET", "300"))
EVAL_TOKENS = int(os.environ.get("AUTORESEARCH_EVAL_TOKENS", str(256 * 1024)))

CACHE_DIR = os.environ.get(
    "AUTORESEARCH_BIOMED_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_biomed"),
)
RAW_DIR = os.path.join(CACHE_DIR, "raw")
PROCESSED_DIR = os.path.join(CACHE_DIR, "processed")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
LM_DIR = os.path.join(CACHE_DIR, "lm")

PQA_L_FILENAME = "ori_pqal.json"
PQA_L_URLS = [
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json",
    "https://gh-proxy.com/https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json",
]

VOCAB_SIZE = int(os.environ.get("AUTORESEARCH_BIOMED_VOCAB_SIZE", "4096"))
SPLIT_SEED = 20260503
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"
LABELS = ("yes", "no", "maybe")


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    for path in [RAW_DIR, PROCESSED_DIR, TOKENIZER_DIR, LM_DIR]:
        os.makedirs(path, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(urls, out_path, timeout=60):
    if os.path.exists(out_path):
        print(f"Data: found {out_path}")
        return

    last_error = None
    tmp_path = out_path + ".tmp"
    for url in urls:
        print(f"Data: downloading {url}")
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, out_path)
            print(f"Data: saved {out_path}")
            return
        except requests.RequestException as exc:
            last_error = exc
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print(f"  failed: {exc}")

    raise RuntimeError(
        "Could not download PubMedQA PQA-L. Place ori_pqal.json manually at "
        f"{out_path} and rerun. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# PubMedQA parsing and serialization
# ---------------------------------------------------------------------------

def normalize_pubmedqa_record(pmid, record):
    question = str(record.get("QUESTION", "")).strip()
    contexts = record.get("CONTEXTS") or []
    labels = record.get("LABELS") or []
    long_answer = str(record.get("LONG_ANSWER", "")).strip()
    final_decision = str(record.get("final_decision", "")).strip().lower()
    year = record.get("YEAR")

    if final_decision not in LABELS:
        raise ValueError(f"Unexpected PubMedQA label {final_decision!r} for PMID {pmid}")

    sections = []
    for i, context in enumerate(contexts):
        text = str(context).strip()
        if not text:
            continue
        heading = str(labels[i]).strip() if i < len(labels) else f"SECTION_{i + 1}"
        sections.append(f"{heading}: {text}")

    return {
        "pmid": str(pmid),
        "question": question,
        "context": "\n".join(sections),
        "long_answer": long_answer,
        "label": final_decision,
        "year": year,
    }


def load_pubmedqa_labeled(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    examples = [normalize_pubmedqa_record(pmid, record) for pmid, record in raw.items()]
    examples.sort(key=lambda x: x["pmid"])
    return examples


def stratified_split(examples, seed=SPLIT_SEED, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for example in examples:
        by_label[example["label"]].append(example)

    splits = {"train": [], "val": [], "test": []}
    for label, rows in sorted(by_label.items()):
        rows = list(rows)
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        splits["train"].extend(rows[:n_train])
        splits["val"].extend(rows[n_train:n_train + n_val])
        splits["test"].extend(rows[n_train + n_val:])

    for split_name, rows in splits.items():
        rows.sort(key=lambda x: x["pmid"])
        print(f"{split_name:5s}: {len(rows):4d} {dict(Counter(r['label'] for r in rows))}")
    return splits


def format_prompt(example):
    return (
        "<|question|>\n"
        f"{example['question']}\n\n"
        "<|context|>\n"
        f"{example['context']}\n\n"
        "<|answer|>\n"
    )


def serialize_example(example, include_long_answer=False):
    text = format_prompt(example) + example["label"] + "\n"
    if include_long_answer and example.get("long_answer"):
        text += "\n<|long_answer|>\n" + example["long_answer"] + "\n"
    return text


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_splits(splits, include_long_answer=False):
    for split_name, rows in splits.items():
        out_rows = []
        for row in rows:
            item = dict(row)
            item["text"] = serialize_example(row, include_long_answer=include_long_answer)
            out_rows.append(item)
        write_jsonl(os.path.join(PROCESSED_DIR, f"{split_name}.jsonl"), out_rows)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled by train_tokenizer()."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        prepend_id = None
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend_id is not None:
                ids.insert(0, prepend_id)
            return ids

        if isinstance(text, list):
            rows = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend_id is not None:
                for row in rows:
                    row.insert(0, prepend_id)
            return rows

        raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids):
        return self.enc.decode(ids)


def iter_training_texts():
    train_path = os.path.join(PROCESSED_DIR, "train.jsonl")
    if not os.path.exists(train_path):
        raise FileNotFoundError("Missing processed train split. Run data preparation first.")
    for row in read_jsonl(train_path):
        yield row["text"]


def train_tokenizer():
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    print("Tokenizer: training PubMedQA BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(iter_training_texts(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="pubmedqa-rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    special_set = set(SPECIAL_TOKENS)
    token_bytes = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        token_bytes.append(0 if token_str in special_set else len(token_str.encode("utf-8")))
    torch.save(torch.tensor(token_bytes, dtype=torch.int32), token_bytes_path)

    test = "PubMedQA answer: yes/no/maybe"
    assert enc.decode(enc.encode_ordinary(test)) == test
    print(f"Tokenizer: trained in {time.time() - t0:.1f}s, vocab_size={enc.n_vocab}")


def get_token_bytes(device="cpu"):
    return torch.load(os.path.join(TOKENIZER_DIR, "token_bytes.pt"), map_location=device)


def encode_lm_splits():
    tokenizer = Tokenizer.from_directory()
    for split in ["train", "val", "test"]:
        jsonl_path = os.path.join(PROCESSED_DIR, f"{split}.jsonl")
        out_path = os.path.join(LM_DIR, f"{split}_tokens.pt")
        if os.path.exists(out_path):
            print(f"LM tokens: found {out_path}")
            continue

        rows = read_jsonl(jsonl_path)
        ids = []
        for row in rows:
            ids.extend(tokenizer.encode(row["text"], prepend=tokenizer.get_bos_token_id()))
        tensor = torch.tensor(ids, dtype=torch.long)
        torch.save(tensor, out_path)
        print(f"LM tokens: {split:5s} {len(rows):4d} examples, {tensor.numel():,} tokens")


# ---------------------------------------------------------------------------
# Runtime dataloader and BPB metric
# ---------------------------------------------------------------------------

def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_token_stream(split):
    path = os.path.join(LM_DIR, f"{split}_tokens.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing token stream {path}. Run uv run prepare_biomed.py first.")
    return torch.load(path, map_location="cpu").long()


def _take_window(tokens, start, length):
    n = tokens.numel()
    if n >= start + length:
        return tokens[start:start + length]
    idx = (torch.arange(length) + start) % n
    return tokens[idx]


def make_dataloader(tokenizer, B, T, split, seed=42):
    """Infinite random-window dataloader over a fixed token stream."""
    del tokenizer
    device = _device()
    tokens = load_token_stream(split)
    if tokens.numel() < T + 2:
        repeats = math.ceil((T + 2) / tokens.numel()) + 1
        tokens = tokens.repeat(repeats)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + (0 if split == "train" else 1_000_000))
    max_start = max(1, tokens.numel() - (T + 1))

    cpu_batch = torch.empty((B, T + 1), dtype=torch.long, pin_memory=(device == "cuda"))
    inputs = torch.empty((B, T), dtype=torch.long, device=device)
    targets = torch.empty((B, T), dtype=torch.long, device=device)
    epoch = 1
    draws = 0

    while True:
        starts = torch.randint(0, max_start, (B,), generator=generator)
        for row, start in enumerate(starts.tolist()):
            cpu_batch[row] = _take_window(tokens, start, T + 1)
        inputs.copy_(cpu_batch[:, :-1], non_blocking=True)
        targets.copy_(cpu_batch[:, 1:], non_blocking=True)
        draws += B
        if draws * T >= tokens.numel():
            epoch += 1
            draws = 0
        yield inputs, targets, epoch


@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """Deterministic validation bits per byte on the locked biomedical val split."""
    del tokenizer
    device = next(model.parameters()).device
    token_bytes = get_token_bytes(device=device)
    tokens = load_token_stream("val")
    if tokens.numel() < MAX_SEQ_LEN + 2:
        repeats = math.ceil((MAX_SEQ_LEN + 2) / tokens.numel()) + 1
        tokens = tokens.repeat(repeats)

    steps = max(1, EVAL_TOKENS // (batch_size * MAX_SEQ_LEN))
    total_nats = 0.0
    total_bytes = 0
    offset = 0
    for _ in range(steps):
        rows = []
        for _row in range(batch_size):
            rows.append(_take_window(tokens, offset, MAX_SEQ_LEN + 1))
            offset = (offset + MAX_SEQ_LEN) % tokens.numel()
        batch = torch.stack(rows).to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]
        loss_flat = model(x, y, reduction="none").reshape(-1)
        y_flat = y.reshape(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(include_long_answer=False, force=False):
    ensure_dirs()
    raw_path = os.path.join(RAW_DIR, PQA_L_FILENAME)
    if force:
        for path in [
            os.path.join(TOKENIZER_DIR, "tokenizer.pkl"),
            os.path.join(TOKENIZER_DIR, "token_bytes.pt"),
            os.path.join(LM_DIR, "train_tokens.pt"),
            os.path.join(LM_DIR, "val_tokens.pt"),
            os.path.join(LM_DIR, "test_tokens.pt"),
        ]:
            if os.path.exists(path):
                os.remove(path)

    download_file(PQA_L_URLS, raw_path)
    examples = load_pubmedqa_labeled(raw_path)
    print(f"Data: loaded {len(examples)} PubMedQA PQA-L examples")
    print(f"Data: label distribution {dict(Counter(e['label'] for e in examples))}")

    splits = stratified_split(examples)
    write_splits(splits, include_long_answer=include_long_answer)
    train_tokenizer()
    encode_lm_splits()

    metadata = {
        "dataset": "PubMedQA PQA-L",
        "source_urls": PQA_L_URLS,
        "raw_file": raw_path,
        "raw_sha256": sha256_file(raw_path),
        "split_seed": SPLIT_SEED,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "label_counts": {k: dict(Counter(row["label"] for row in v)) for k, v in splits.items()},
        "max_seq_len": MAX_SEQ_LEN,
        "time_budget": TIME_BUDGET,
        "eval_tokens": EVAL_TOKENS,
        "vocab_size": VOCAB_SIZE,
        "include_long_answer": include_long_answer,
    }
    with open(os.path.join(CACHE_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Done. Cache directory: {CACHE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PubMedQA data for biomedical autoresearch")
    parser.add_argument("--include-long-answer", action="store_true", help="Append PubMedQA long_answer text after the label during LM training")
    parser.add_argument("--force", action="store_true", help="Regenerate tokenizer and token streams")
    args = parser.parse_args()

    try:
        prepare(include_long_answer=args.include_long_answer, force=args.force)
    except Exception as exc:
        print(f"prepare_biomed.py failed: {exc}", file=sys.stderr)
        raise
