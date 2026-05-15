"""
Build a unified run registry for PubMedQA autoresearch audits.

The registry records which split was used, whether PubMedQA test had been
opened, where metrics/predictions live, and whether the row belongs to model
selection, validation audit, or post-freeze test audit.

Usage:
    uv run python analysis/build_run_registry.py --out outputs/run_registry.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


REGISTRY_COLUMNS = [
    "run_id",
    "source_file",
    "phase",
    "split",
    "config_name",
    "group",
    "seed",
    "commit",
    "source_commit",
    "selection_stage",
    "opened_test",
    "metrics_file",
    "prediction_file",
    "status",
    "description",
]


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str)


def row_dict(**kwargs: object) -> dict[str, object]:
    row = {col: "" for col in REGISTRY_COLUMNS}
    row.update(kwargs)
    return row


def selection_stage(split: str, phase: str, source_file: str) -> str:
    if split == "test":
        return "post_freeze_audit_only"
    if "prospective" in phase:
        return "prospective_validation_demo"
    if source_file == "results_biomed.tsv":
        return "search_validation"
    if "repeated" in phase or source_file == "results_repeated_controls.tsv":
        return "repeated_validation_audit"
    if "validation" in phase:
        return "locked_validation_replicate"
    return "validation_audit"


def load_single_search() -> list[dict[str, object]]:
    path = ROOT / "results_biomed.tsv"
    if not path.exists():
        return []
    rows = []
    df = read_tsv(path)
    for idx, row in df.iterrows():
        commit = row.get("commit", "")
        rows.append(row_dict(
            run_id=f"search_{idx:03d}_{commit}",
            source_file=path.name,
            phase="single_run_search",
            split="val",
            config_name=f"single_run_{commit}",
            group=infer_group(row.get("description", "")),
            seed="",
            commit=commit,
            source_commit=commit,
            selection_stage="search_validation",
            opened_test="false",
            metrics_file=path.name,
            prediction_file="",
            status=row.get("status", ""),
            description=row.get("description", ""),
        ))
    return rows


def infer_group(description: str) -> str:
    desc = str(description).lower()
    if desc.startswith("agent"):
        return "agent"
    if desc.startswith("random"):
        return "random"
    return "manual"


def load_repeated_controls(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    df = read_tsv(path)
    for _, row in df.iterrows():
        split = row.get("split", "")
        phase = row.get("phase", "")
        config_name = row.get("config_name", "")
        seed = row.get("seed", "")
        commit = row.get("train_code_commit", "")
        rows.append(row_dict(
            run_id=f"{phase}_{split}_{config_name}_seed{seed}",
            source_file=path.name,
            phase=phase,
            split=split,
            config_name=config_name,
            group=row.get("group", ""),
            seed=seed,
            commit=commit,
            source_commit=row.get("source_commit", ""),
            selection_stage=selection_stage(split, phase, path.name),
            opened_test=str(split == "test").lower(),
            metrics_file=str(path.relative_to(ROOT)),
            prediction_file=row.get("predictions", ""),
            status=row.get("status", ""),
            description=row.get("description", ""),
        ))
    return rows


def load_locked(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    df = read_tsv(path)
    for _, row in df.iterrows():
        split = row.get("split", "")
        phase = row.get("phase", "")
        seed = row.get("seed", "")
        rows.append(row_dict(
            run_id=f"{phase}_{split}_locked_agent_seed{seed}",
            source_file=path.name,
            phase=phase,
            split=split,
            config_name="locked_agent_protocol",
            group="agent",
            seed=seed,
            commit=row.get("commit", ""),
            source_commit="a20f5b7",
            selection_stage=selection_stage(split, phase, path.name),
            opened_test=str(split == "test").lower(),
            metrics_file=str(path.relative_to(ROOT)),
            prediction_file="",
            status=row.get("status", ""),
            description=row.get("description", ""),
        ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified run registry")
    parser.add_argument("--out", default="outputs/run_registry.tsv")
    args = parser.parse_args()

    rows = []
    rows.extend(load_single_search())
    rows.extend(load_repeated_controls(ROOT / "results_repeated_controls.tsv"))
    rows.extend(load_repeated_controls(ROOT / "outputs" / "audit_only" / "results_repeated_controls_test_audit.tsv"))
    rows.extend(load_locked(ROOT / "results_locked_replicates.tsv"))
    rows.extend(load_locked(ROOT / "outputs" / "audit_only" / "results_locked_replicates_test_audit.tsv"))

    out = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)
    print(f"wrote {len(out)} registry rows to {out_path}")


if __name__ == "__main__":
    main()
