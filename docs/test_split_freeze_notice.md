# PubMedQA Test Split Freeze Notice

Date opened: 2026-05-03

The PubMedQA test split was opened only after selecting the frozen agent
protocol and running validation-seed replication.

From this point onward, PubMedQA test metrics must not be used for:

- architecture changes
- training-objective changes
- tokenizer changes
- prompt-format changes
- hyperparameter selection
- calibration tuning
- model-selection decisions

Future PubMedQA test results are audit-only after opening. Any new method
development must use the locked validation split or a new external dataset such
as SciFact.

Implementation guardrails added after review:

- `run_config_replicates.py --split test` now requires
  `--confirm-test-audit-only`.
- `run_locked_replicates.py` is deprecated for new manuscript evidence because
  it does not load explicit JSON configurations. Use
  `run_config_replicates.py` plus `configs/repeated_seed_controls/*.json`
  instead. See `docs/deprecated_locked_replicates_notice.md`.
- Future test predictions and appended metric rows should be written under:

```text
outputs/audit_only/
```

- Future validation repeated controls continue to use the original validation
  result tables. Raw run logs are omitted from the minimal public repository.

Important metric naming note:

- `lm_val_bpb` means language-model BPB on the locked validation token stream.
- PubMedQA `accuracy`, `macro-F1`, and `Brier` may be computed on either the
  validation or test split, depending on the `split` column.
- Final-test rows still report `lm_val_bpb`; this is not test-set BPB.
- PubMedQA test should be described as a post-freeze audit-only split,
  not as an untouched blind test.
