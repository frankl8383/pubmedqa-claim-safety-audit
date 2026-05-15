#!/usr/bin/env python3
"""Run same-trained context ablation for retained small-LM configurations.

Each run trains one retained configuration once, then evaluates the same
in-memory model on normal and perturbed PubMedQA validation prompts.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "same_trained_context_ablation"
CONFIGS = [
    "configs/repeated_seed_controls/random_primary_best_7147e14.json",
    "configs/repeated_seed_controls/agent_aux_a20f5b7.json",
]
MODES = [
    "question_context",
    "question_only",
    "context_removed",
    "context_only",
    "shuffled_context",
    "question_shuffled",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PUBMEDQA_CONTEXT_ABLATION_OUT_DIR"] = str(OUT_DIR)
    env["PUBMEDQA_CONTEXT_ABLATION_MODES"] = ",".join(MODES)
    cmd = [
        "uv",
        "run",
        "python",
        "run_config_replicates.py",
        "--phase",
        "same_trained_context_ablation_20260508",
        "--split",
        "val",
        "--seeds",
        "42,43,44,45,46",
        "--configs",
        *CONFIGS,
        "--allow-dirty",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    subprocess.run(
        ["uv", "run", "python", "analysis/summarize_same_trained_context_ablation.py"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
