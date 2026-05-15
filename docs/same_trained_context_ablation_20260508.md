# Same-Trained Context-Ablation Audit

Date: 2026-05-08

## Rationale

Earlier context-use checks used sparse and frozen-encoder reference models. Those
checks were useful guardrails, but they did not directly test whether the two
small autoresearch configurations used in the main claim boundary depended on
question/context alignment. This experiment evaluates the same trained small-LM
checkpoint immediately after each training run under multiple validation-input
perturbations.

## Design

Configurations:

- `random_primary_best_7147e14`
- `agent_aux_a20f5b7`

Seeds:

- 42, 43, 44, 45, 46

Split:

- PubMedQA validation only.

Prompt modes:

- `question_context`: original question plus original context.
- `question_only`: original question with empty context field.
- `context_removed`: original question with the context block removed.
- `context_only`: original context with empty question field.
- `shuffled_context`: original question with a deterministic mismatched context.
- `question_shuffled`: deterministic mismatched question with original context.

The same in-memory trained model was evaluated across all prompt modes before
the training process exited. No PubMedQA test split was used.

## Key results

For `agent_aux_a20f5b7`, question-only evaluation reduced macro-F1 from 0.3210
to 0.2716 and worsened Brier score from 0.6703 to 0.7621. Shuffled-context and
question-shuffled modes changed macro-F1 only modestly (-0.0120 and +0.0043,
respectively). Context-only evaluation was close to the original prompt
macro-F1, which limits any claim that the small model was robustly using
question-context alignment.

For `random_primary_best_7147e14`, the original prompt remained yes-collapsed.
Question-only and context-removed perturbations increased no predictions and
macro-F1, but also worsened Brier score. This pattern is interpreted as
prompt-format sensitivity rather than improved biomedical reasoning.

## Output files

- `outputs/same_trained_context_ablation/same_trained_context_ablation_runs.tsv`
- `outputs/same_trained_context_ablation/same_trained_context_ablation_summary.tsv`
- `outputs/manuscript_tables/table29_same_trained_context_ablation.tsv`
- `outputs/figures/same_trained_context_ablation/same_trained_context_ablation.png`
- `outputs/figures/same_trained_context_ablation/same_trained_context_ablation.pdf`
- `outputs/figures/same_trained_context_ablation/same_trained_context_ablation.svg`

## Manuscript boundary

Allowed: same-trained context-ablation results strengthen the claim-safety
guardrail and limit abstract-grounding claims.

Not allowed: these results prove abstract-grounded biomedical reasoning,
clinical usefulness, or agent superiority.
