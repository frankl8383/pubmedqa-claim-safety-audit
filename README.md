# PubMedQA Claim-Safety Audit Package

This repository contains the minimal public reproducibility package for the
BMC Medical Informatics and Decision Making submission:

**A claim-safety methods audit of low-resource autonomous biomedical NLP
experimentation: a PubMedQA case study**

The project is a health-informatics claim-safety and reproducibility audit of
low-resource biomedical NLP experimentation. It is not a PubMedQA leaderboard,
a clinical decision-support evaluation, or an agent-superiority benchmark.

## Release

Tagged release:

<https://github.com/frankl8383/pubmedqa-claim-safety-audit/releases/tag/v1.0.1-bmc-midm-clean>

This branch is intentionally minimal. It retains the code, configuration files,
processed public-benchmark metadata, prediction files, metric tables, main
figures, split manifest/checksums, test-freeze notice, same-trained
context-ablation outputs, model-panel summaries, and sanitized
supervisor-informed case-audit materials needed to audit the manuscript claims.

It intentionally excludes raw local workbooks, historical internal review
materials, old venue drafts, root-level run logs, virtual environments,
downloaded model weights, and regenerable embedding caches.

## Repository Layout

- `analysis/`: scripts used to summarize predictions, regenerate tables, and
  regenerate figures.
- `baselines/`: sparse and frozen-encoder reference baseline scripts.
- `configs/`: retained JSON configuration files used for repeated validation
  and validation-only mini-run checks.
- `docs/`: final manuscript source and short provenance notices.
- `outputs/`: compact results, predictions, figures, split manifests, and
  sanitized case-audit materials.
- `prepare_biomed.py`, `train_biomed.py`, `eval_pubmedqa.py`,
  `run_config_replicates.py`: core reproduction entry points.

## Clean-Room Reproduction Command

For archived releases without `.git` metadata, use the release tag and
`--allow-dirty`:

```bash
PACKAGE_RELEASE_ID=v1.0.1-bmc-midm-clean \
uv run python run_config_replicates.py \
  --phase repeated_val \
  --split val \
  --seeds 42,43,44,45,46 \
  --configs configs/repeated_seed_controls/*.json \
  --allow-dirty
```

Full small-language-model training requires macOS on Apple Silicon with MPS
support. Analysis-only table and figure regeneration is cross-platform after
dependencies are installed.

## Data Boundary

PubMedQA and SciFact are public benchmark datasets and remain available from
their original sources. This repository includes processed metadata and
prediction outputs needed to audit the manuscript claims. The split manifest
provides PMID-level split membership, labels, years, and SHA-256 hashes of
question and context fields rather than redistributing full abstract/context
text.

Large local caches, virtual environments, downloaded model-weight caches, and
regenerable embedding arrays are excluded because they can be regenerated from
documented scripts and public sources.

## License

Code and package metadata are released under the MIT License. Public benchmark
datasets and pretrained models remain subject to their original licenses and
terms.
