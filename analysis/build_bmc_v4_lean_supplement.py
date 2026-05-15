#!/usr/bin/env python3
"""Build a lean BMC MIDM additional-file bundle.

The full reproducibility package contains predictions and logs and should live
in an archived repository release. This lean bundle is intended for the journal
submission system, where individual additional files should remain small.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "outputs" / "bmc_midm_v4_2_lean_additional_files_20260508"
ZIP_PATH = ROOT.parent / "Additional_file_1.zip"

FILES = [
    "docs/same_trained_context_ablation_20260508.md",
    "docs/test_split_freeze_notice.md",
    "docs/deprecated_locked_replicates_notice.md",
    "docs/prospective_mini_run_provenance_20260506.md",
    "outputs/manuscript_tables/table15_claim_safety_matrix.tsv",
    "outputs/manuscript_tables/table16_protocol_guardrail_checklist.tsv",
    "outputs/manuscript_tables/table17_retrospective_alignment_audit.tsv",
    "outputs/manuscript_tables/table18_prospective_mini_run.tsv",
    "outputs/manuscript_tables/table19_model_panel_audit.tsv",
    "outputs/manuscript_tables/table20_model_panel_context_calibration.tsv",
    "outputs/manuscript_tables/table21_pmmi_original7.tsv",
    "outputs/manuscript_tables/table22_pmmi_with_prospective.tsv",
    "outputs/manuscript_tables/table23_supervisor_case_audit_taxonomy.tsv",
    "outputs/manuscript_tables/table24_supervisor_case_audit_main_examples.tsv",
    "outputs/manuscript_tables/table29_same_trained_context_ablation.tsv",
    "outputs/manuscript_tables/table4_pubmedqa_per_class_audit.tsv",
    "outputs/manuscript_tables/table7_full_pairwise_validation_audit.tsv",
    "outputs/model_panel_audit/model_panel_predictions.tsv",
    "outputs/model_panel_audit/model_panel_brier_decomposition.tsv",
    "outputs/prediction_level_audit_full/metrics_long.tsv",
    "outputs/prediction_level_audit_full/metrics_wide.tsv",
    "outputs/prediction_level_audit_full/paired_bootstrap_diffs.tsv",
    "outputs/prediction_level_audit_full/audit_metadata.tsv",
    "outputs/run_registry.tsv",
    "outputs/hypothesis_audit/experiment_card_index.tsv",
    "outputs/split_manifest/pubmedqa_pqal_split_manifest_README.md",
    "outputs/split_manifest/pubmedqa_pqal_split_manifest.tsv",
    "outputs/split_manifest/pubmedqa_pqal_checksums.tsv",
    "outputs/supervisor_case_audit/supervisor_case_audit_public_supplement_60.tsv",
    "outputs/supervisor_case_audit/supervisor_case_audit_provenance_sanitized.md",
    "outputs/same_trained_context_ablation/same_trained_context_ablation_summary.tsv",
    "outputs/figures/same_trained_context_ablation/same_trained_context_ablation.png",
]


def copy_file(rel: str) -> None:
    src = ROOT / rel
    if not src.exists():
        return
    dst = PACKAGE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)
    for rel in FILES:
        copy_file(rel)

    readme = """# BMC MIDM v4.2 Lean Additional Files

This bundle contains manuscript-facing supplementary tables, sanitized
supervisor-informed case-audit outputs, same-trained context-ablation summary
data, full pairwise/per-class audit tables, model-panel prediction and Brier
decomposition summaries, run registry, split manifest/checksums, and selected
provenance notes. It intentionally excludes large per-run prediction folders,
raw run logs, virtual environments, model caches, and downloaded model
weights. Those belong in the archived public repository release. Some evidence
paths listed in supplementary checklists refer to files in the tagged public
release rather than files duplicated inside this lean BMC additional file. See
`docs/artifact_path_map.tsv` for the boundary between this Additional file and
the full public release.
"""
    (PACKAGE / "README.md").write_text(readme, encoding="utf-8")

    artifact_map = """artifact_group\tlocation\tnotes
Manuscript-facing supplementary tables\tAdditional file 1\tIncludes claim matrix, guardrail checklist, PMMI tables, model-panel summaries, same-trained context-ablation summary, paired/per-class audit tables, and sanitized case-audit summaries.
Split manifest and checksums\tAdditional file 1 and tagged public release\tIncluded so reviewers can verify PubMedQA PQA-L split membership and processed-file checksums without opening the full archive.
Run registry\tAdditional file 1 and tagged public release\tIncluded as a compact registry of retained runs and audit-only test rows; raw run logs are omitted from the minimal public release.
Model-panel predictions and Brier decomposition\tAdditional file 1 and tagged public release\tIncluded because they support the reference-line and calibration/context-use audit.
Same-trained context-ablation figure and summary\tAdditional file 1 and tagged public release\tIncluded because they support the context-use guardrail in the main manuscript.
Sanitized supervisor-informed case audit\tAdditional file 1 and tagged public release\tIncludes public 60-case supplement and sanitized provenance; raw private workbooks are excluded.
Experiment cards and detailed configuration files\tTagged public release / full reproducibility archive\tReferenced by supplementary checklists but not duplicated in the lean Additional file except for the experiment-card index.
Per-seed prediction files\tTagged public release / full reproducibility archive\tExcluded from the lean Additional file to keep the journal supplement small; available in the release archive.
Analysis and figure-generation scripts\tTagged public release / full reproducibility archive\tNot duplicated in the lean Additional file; used to regenerate tables and figures from release outputs.
Virtual environments, downloaded model weights, and large caches\tExcluded\tRegenerable from public sources and documented scripts; excluded for size and redistribution reasons.
"""
    path_map = PACKAGE / "docs" / "artifact_path_map.tsv"
    path_map.parent.mkdir(parents=True, exist_ok=True)
    path_map.write_text(artifact_map, encoding="utf-8")

    checksum = PACKAGE / "SHA256.txt"
    if checksum.exists():
        checksum.unlink()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(PACKAGE))
    digest = sha256(ZIP_PATH)
    checksum.write_text(f"{digest}  {ZIP_PATH.name}\n", encoding="utf-8")
    print(f"Lean supplement: {PACKAGE}")
    print(f"Zip: {ZIP_PATH}")
    print((PACKAGE / "SHA256.txt").read_text().strip())


if __name__ == "__main__":
    main()
