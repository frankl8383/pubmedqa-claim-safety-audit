# Hypothesis-First Audit Package

This package converts the PubMedQA autoresearch project into a claim-safe
hypothesis-first audit workflow.

Files:

- `claim_safety_matrix.tsv`: allowed, conditional, and forbidden manuscript claims.
- `protocol_guardrail_checklist.tsv`: data, evaluation, seed, context,
  calibration, reproduction-entry, and disclosure guardrails.
- `experiment_card_index.tsv`: key configuration cards in `outputs/experiment_cards/`.

The central interpretation is that agent auxiliary is an auditable
class-behavior intervention, not proof of general superiority over random or
manual search. The hypothesis-first protocol retrospectively grades the current
evidence and should govern future runs prospectively.
