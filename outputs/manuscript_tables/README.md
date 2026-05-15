# Manuscript Tables for BMC MIDM v4

Generated and curated for the BMC Medical Informatics and Decision Making
claim-safety audit manuscript.

## Main-Manuscript Candidate Tables

- `table25_bmc_study_design_claim_boundary.tsv`: study design, split governance, and claim boundary.
- `table26_bmc_primary_claim_boundary_clean.tsv`: primary retained-configuration and paired-difference audit with submission-ready labels.
- `table27_bmc_compressed_claim_safety_matrix.tsv`: compressed claim-safety matrix for the main manuscript.
- `table28_bmc_model_panel_context_calibration_clean.tsv`: model-panel context-use and calibration summary with submission-ready labels.
- `table23_supervisor_case_audit_taxonomy.tsv`: 60-case supervisor-informed illustrative case-audit taxonomy.

## Supplementary Tables

- `table1_dataset_splits.tsv`: PubMedQA and SciFact split governance and label distributions.
- `table2_repeated_seed_comparison.tsv`: retained manual/random/agent repeated-seed validation comparison.
- `table15_claim_safety_matrix.tsv`: full allowed, conditional, and forbidden manuscript claims.
- `table19_model_panel_audit.tsv`: full sparse/frozen model-panel validation reference lines.
- `table20_model_panel_context_calibration.tsv`: full model-panel context-use and calibration summary.
- `table7_full_pairwise_validation_audit.tsv`: full paired bootstrap validation audit.
- `table13_context_use_sanity.tsv`: earlier TF-IDF/BiomedBERT context-use sanity audit.
- `table14_calibration_sanity.tsv`: earlier calibration sanity diagnostics.
- `table16_protocol_guardrail_checklist.tsv`: protocol guardrail checklist.
- `table17_retrospective_alignment_audit.tsv`: retrospective/prospective alignment audit.
- `table18_prospective_mini_run.tsv`: validation-only pre-specified auxiliary-weight mini-run.
- `table21_pmmi_original7.tsv`: original-seven Proxy-Metric Mismatch Index.
- `table22_pmmi_with_prospective.tsv`: PMMI including prospective mini-run settings.
- `table24_supervisor_case_audit_main_examples.tsv`: main-text candidate examples from the 60-case audit.

## Historical Tables

Tables `table8` through `table12` are retained as historical author-adjudicated
manual-error development artifacts from the pre-supervisor-audit phase. They
should not be cited as primary BMC v4 evidence and should not be described as
independent expert consensus.

## Reporting Boundaries

- PubMedQA test results are post-freeze audit artifacts only.
- SciFact is an external reference stress test, not validation of the PubMedQA
  autoresearch-trained small language model.
- Frozen biomedical encoders are reference lines, not compute-matched
  five-minute autoresearch runs.
- The 60-case supervisor-informed audit is illustrative and standardized after supervisor-informed review. It is not independent blinded expert consensus,
  PubMedQA relabeling, or evidence of improved biomedical reasoning.
- Calibration metrics are sanity-check diagnostics over model-family-specific
  probabilities, not clinical reliability estimates.
