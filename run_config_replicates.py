"""
Run fair repeated-seed comparisons for pre-specified PubMedQA configurations.

The runner keeps the training/evaluation code fixed and varies only explicit
configuration values supplied in JSON files. This is intended for validation
audits after PubMedQA test freeze; it does not perform model selection from test
metrics.

Usage:
    uv run python run_config_replicates.py \
      --phase repeated_val \
      --split val \
      --seeds 42,43,44,45,46 \
      --configs configs/repeated_seed_controls/*.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "results_repeated_controls.tsv"
AUDIT_ONLY_RESULTS_FILE = ROOT / "outputs" / "audit_only" / "results_repeated_controls_test_audit.tsv"
HEADER = (
    "phase\tsplit\tconfig_name\tgroup\tseed\ttrain_code_commit\tsource_commit\t"
    "lm_val_bpb\tpubmedqa_acc\tpubmedqa_macro_f1\tpubmedqa_brier\tmemory_gb\t"
    "params_M\ttokens_M\tstatus\tlog\tpredictions\tdescription\n"
)

METRIC_PATTERNS = {
    "val_bpb": re.compile(r"^val_bpb:\s+([0-9.]+)", re.MULTILINE),
    "pubmedqa_acc": re.compile(r"^pubmedqa_acc:\s+([0-9.]+)", re.MULTILINE),
    "pubmedqa_macro_f1": re.compile(r"^pubmedqa_macro_f1:\s+([0-9.]+)", re.MULTILINE),
    "pubmedqa_brier": re.compile(r"^pubmedqa_brier:\s+([0-9.]+)", re.MULTILINE),
    "peak_vram_mb": re.compile(r"^peak_vram_mb:\s+([0-9.]+)", re.MULTILINE),
    "num_params_M": re.compile(r"^num_params_M:\s+([0-9.]+)", re.MULTILINE),
    "total_tokens_M": re.compile(r"^total_tokens_M:\s+([0-9.]+)", re.MULTILINE),
}

HP_ENV_PREFIX = "AUTORESEARCH_"
HP_KEYS = {
    "ASPECT_RATIO",
    "HEAD_DIM",
    "WINDOW_PATTERN",
    "TOTAL_BATCH_SIZE",
    "EMBEDDING_LR",
    "UNEMBEDDING_LR",
    "MATRIX_LR",
    "SCALAR_LR",
    "WEIGHT_DECAY",
    "WARMUP_RATIO",
    "WARMDOWN_RATIO",
    "FINAL_LR_FRAC",
    "DEPTH",
    "DEVICE_BATCH_SIZE",
    "QA_AUX_WEIGHT",
    "QA_AUX_BATCH_SIZE",
    "QA_AUX_START_PROGRESS",
}


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=check)


def git_stdout(args: list[str]) -> str | None:
    """Return git stdout, or None when the release has no git metadata."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def short_head() -> str:
    release_id = os.environ.get("PACKAGE_RELEASE_ID")
    if release_id:
        return release_id
    return git_stdout(["rev-parse", "--short", "HEAD"]) or "unknown-release"


def require_clean_tree() -> None:
    status = git_stdout(["status", "--short"])
    if status is None:
        raise SystemExit(
            "Git metadata is unavailable in this release directory. Re-run with "
            "--allow-dirty for archived-package reproduction, or set "
            "PACKAGE_RELEASE_ID to the repository release/tag identifier."
        )
    if status:
        raise SystemExit(f"Working tree is not clean:\n{status}")


def ensure_results_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")


def require_test_audit_confirmation(split: str, confirmed: bool) -> None:
    if split == "test" and not confirmed:
        raise SystemExit(
            "PubMedQA test split is frozen and may only be used for post-freeze "
            "audit-only runs. Re-run with --confirm-test-audit-only if this is "
            "intentional; test outputs will be written under outputs/audit_only/."
        )


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    for required in ("name", "group", "source_commit", "description", "hyperparameters"):
        if required not in config:
            raise ValueError(f"{path} missing required key: {required}")
    return config


def parse_log(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    metrics: dict[str, float] = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.search(text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def env_for_config(config: dict[str, object], seed: int, split: str, pred_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["AUTORESEARCH_RUN_SEED"] = str(seed)
    env["PUBMEDQA_EVAL_SPLIT"] = split
    env["PUBMEDQA_PRED_OUT"] = str(pred_path)
    env["AUTORESEARCH_CONFIG_NAME"] = str(config["name"])
    for key, value in config["hyperparameters"].items():  # type: ignore[union-attr]
        key_upper = key.upper()
        if key_upper not in HP_KEYS:
            raise ValueError(f"{config['name']} has unsupported hyperparameter: {key}")
        env[HP_ENV_PREFIX + key_upper] = str(value)
    return env


def append_row(
    results_file: Path,
    phase: str,
    split: str,
    seed: int,
    train_code_commit: str,
    config: dict[str, object],
    metrics: dict[str, float],
    status: str,
    log_path: Path,
    pred_path: Path,
) -> None:
    ensure_results_file(results_file)
    memory_gb = metrics.get("peak_vram_mb", 0.0) / 1024
    row = [
        phase,
        split,
        str(config["name"]),
        str(config["group"]),
        str(seed),
        train_code_commit,
        str(config["source_commit"]),
        f"{metrics.get('val_bpb', math.nan):.6f}",
        f"{metrics.get('pubmedqa_acc', math.nan):.6f}",
        f"{metrics.get('pubmedqa_macro_f1', math.nan):.6f}",
        f"{metrics.get('pubmedqa_brier', math.nan):.6f}",
        f"{memory_gb:.1f}",
        f"{metrics.get('num_params_M', math.nan):.1f}",
        f"{metrics.get('total_tokens_M', math.nan):.1f}",
        status,
        str(log_path.relative_to(ROOT)),
        str(pred_path.relative_to(ROOT)),
        str(config["description"]),
    ]
    with results_file.open("a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")


def train_once(
    phase: str,
    split: str,
    seed: int,
    train_code_commit: str,
    config: dict[str, object],
) -> tuple[str, Path, Path, dict[str, float]]:
    config_name = str(config["name"])
    if split == "test":
        base_dir = ROOT / "outputs" / "audit_only"
        log_path = base_dir / "logs" / f"run_biomed_{phase}_{split}_{config_name}_seed{seed}.log"
        pred_path = base_dir / "predictions" / f"pubmedqa_{phase}_{split}_{config_name}_seed{seed}.tsv"
    else:
        log_path = ROOT / f"run_biomed_{phase}_{split}_{config_name}_seed{seed}.log"
        pred_path = ROOT / "outputs" / "predictions" / f"pubmedqa_{phase}_{split}_{config_name}_seed{seed}.tsv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    env = env_for_config(config, seed, split, pred_path)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            ["uv", "run", "train_biomed.py"],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    metrics = parse_log(log_path)
    status = "ok" if proc.returncode == 0 and "val_bpb" in metrics else "crash"
    return status, log_path, pred_path, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pre-specified config seed replicates")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--confirm-test-audit-only",
        action="store_true",
        help="Required for frozen PubMedQA test split audit-only runs.",
    )
    args = parser.parse_args()

    require_test_audit_confirmation(args.split, args.confirm_test_audit_only)
    if not args.allow_dirty:
        require_clean_tree()
    results_file = AUDIT_ONLY_RESULTS_FILE if args.split == "test" else RESULTS_FILE
    ensure_results_file(results_file)
    train_code_commit = short_head()
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    configs = [load_config(Path(path)) for path in args.configs]

    print(f"train_code_commit={train_code_commit}")
    print(f"phase={args.phase} split={args.split} seeds={seeds}")
    print(f"results_file={results_file.relative_to(ROOT)}")
    print("configs=" + ",".join(str(config["name"]) for config in configs))

    for config in configs:
        for seed in seeds:
            print(f"[{args.phase} {args.split} {config['name']} seed={seed}] training...", flush=True)
            status, log_path, pred_path, metrics = train_once(
                args.phase,
                args.split,
                seed,
                train_code_commit,
                config,
            )
            append_row(results_file, args.phase, args.split, seed, train_code_commit, config, metrics, status, log_path, pred_path)
            print(
                f"[{args.phase} {args.split} {config['name']} seed={seed}] {status} "
                f"lm_val_bpb={metrics.get('val_bpb', math.nan):.6f} "
                f"acc={metrics.get('pubmedqa_acc', math.nan):.6f} "
                f"macro_f1={metrics.get('pubmedqa_macro_f1', math.nan):.6f} "
                f"brier={metrics.get('pubmedqa_brier', math.nan):.6f} "
                f"log={log_path.name}",
                flush=True,
            )


if __name__ == "__main__":
    main()
