# Experiment Card: Prospective Auxiliary Weight 0.02

- Config: `prospective_aux_weight_0p02_20260506`
- Stage: prospective hypothesis-first mini-run
- Split: PubMedQA validation only
- Seeds: 42, 43, 44, 45, 46
- Written before execution: yes

## Hypothesis

A weaker delayed class-balanced answer-label auxiliary objective may preserve
some no-class recovery while reducing the accuracy and calibration harm seen in
the earlier agent auxiliary weight 0.05 configuration.

## Expected Direction

- Macro-F1: possibly above random primary but likely below or near agent 0.05.
- Accuracy: expected to be closer to random primary than agent 0.05.
- Brier/ECE: expected to be less harmed than agent 0.05 if the weight is not
  too weak.
- Maybe class: expected to remain unresolved.

## Risk Ledger

- The auxiliary signal may be too weak to change class behavior.
- Any macro-F1 gain may still reflect label-prior correction rather than
  abstract-grounded biomedical reasoning.
- Validation-only results must not trigger PubMedQA test reuse.

## Claim Gate

Allowed claim: this configuration prospectively tests a weaker class-behavior
intervention under the hypothesis-first protocol.

Forbidden claim: this configuration validates the agent strategy or identifies
the best low-resource biomedical QA model.
