# Prospective Mini-Run Provenance Statement

Date: 2026-05-06

This note documents the local provenance of the validation-only
hypothesis-first mini-run. It is intended for the reproducibility appendix and
does not claim external registration.

## Scope

The mini-run was performed after the hypothesis-first protocol had been
formalized. It was designed to demonstrate that the protocol could be used
prospectively for a small validation-only experiment. It was not used for
PubMedQA test-set selection and does not validate a globally better biomedical
QA model.

## Pre-Run Artifacts

The following files were authored before the run outputs were generated,
according to local filesystem timestamps:

```text
2026-05-06 14:27:21 +0800  configs/prospective_hypothesis_first/aux_weight_0p02_20260506.json
2026-05-06 14:27:35 +0800  configs/prospective_hypothesis_first/aux_weight_0p10_20260506.json
2026-05-06 14:27:56 +0800  docs/prospective_hypothesis_first_mini_run_20260506.md
2026-05-06 14:28:13 +0800  outputs/experiment_cards/prospective_aux_weight_0p02_20260506.md
2026-05-06 14:28:31 +0800  outputs/experiment_cards/prospective_aux_weight_0p10_20260506.md
```

These artifacts specified the auxiliary-weight comparison, expected benefits,
expected harms, pass/fail gates, validation-only split, seeds 42-46, and the
restriction that PubMedQA test metrics were not to be used.

## Run Outputs

The run outputs were generated after the pre-run artifacts:

```text
2026-05-06 15:29:06 +0800  results_repeated_controls.tsv
2026-05-06 17:25:49 +0800  outputs/prospective_mini_run/prospective_mini_run_summary.tsv
2026-05-06 17:26:32 +0800  docs/prospective_hypothesis_first_mini_run_results_20260506.md
```

The current repository state at the time of packaging reported:

```text
git rev-parse --short HEAD
325e90f
```

The worktree contained uncommitted manuscript and audit artifacts, so this
timestamp evidence should be described as local provenance rather than as an
external registration or external timestamped record.

## Execution Command

The mini-run used the explicit JSON-configured replication runner:

```bash
uv run python run_config_replicates.py \
  --phase prospective_aux_weight_20260506 \
  --split val \
  --seeds 42,43,44,45,46 \
  --configs \
    configs/prospective_hypothesis_first/aux_weight_0p02_20260506.json \
    configs/prospective_hypothesis_first/aux_weight_0p10_20260506.json \
  --allow-dirty
```

No `--split test` command was used for this mini-run. The `--allow-dirty`
flag permitted execution while manuscript and audit artifacts were still
uncommitted. It did not change split locking, test-freeze guardrails, seeds, or
configuration values, but it means the provenance evidence is local rather
than externally registered.

## Run Registry

The rebuilt `outputs/run_registry.tsv` indexes the 10 prospective mini-run
rows with phase `prospective_aux_weight_20260506`, split `val`, and
`opened_test=false`. These rows link the validation metrics file, prediction
TSVs, configuration names, and seeds. Raw run logs are omitted from the
minimal public repository.

## Safe Manuscript Wording

Use:

```text
The prospective protocol and configuration files were authored before
execution according to local file timestamps and run-registry order; however,
this was not externally registered.
```

Avoid:

```text
The mini-run was externally registered before execution.
The mini-run confirms the agent strategy.
The mini-run identifies the best biomedical QA model.
```
