# Deprecated Locked Replicates Notice

Date: 2026-05-06

`run_locked_replicates.py` is retained only as a historical provenance wrapper
for early frozen PubMedQA audit runs. It is no longer the recommended
reproduction entry point.

The current neutral training default in `train_biomed.py` sets
`AUTORESEARCH_QA_AUX_WEIGHT=0.0`. Because the legacy wrapper does not load a
pre-specified JSON configuration, re-running it can silently execute the
neutral no-auxiliary protocol rather than the intended `agent_aux_a20f5b7`
configuration.

All current manuscript-facing repeated-seed and post-freeze audit-only runs
should use the JSON-configured runner:

```bash
uv run python run_config_replicates.py \
  --phase repeated_val \
  --split val \
  --seeds 42,43,44,45,46 \
  --configs configs/repeated_seed_controls/agent_aux_a20f5b7.json
```

For frozen PubMedQA test prediction export, use audit-only mode:

```bash
uv run python run_config_replicates.py \
  --phase audit_only_test \
  --split test \
  --seeds 42,43,44 \
  --configs configs/repeated_seed_controls/agent_aux_a20f5b7.json \
  --confirm-test-audit-only
```

Historical `results_locked_replicates.tsv` rows may still be cited as legacy
audit artifacts, but the manuscript should identify
`run_config_replicates.py` plus `configs/repeated_seed_controls/*.json` as the
reproducible entry point for current evidence.
