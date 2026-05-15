# Experiment Card: Prospective Auxiliary Weight 0.10

- Config: `prospective_aux_weight_0p10_20260506`
- Stage: prospective hypothesis-first mini-run
- Split: PubMedQA validation only
- Seeds: 42, 43, 44, 45, 46
- Written before execution: yes

## Hypothesis

A stronger delayed class-balanced answer-label auxiliary objective may produce
more no-class recovery than weaker auxiliary settings, but may also increase
overcorrection of yes examples and worsen calibration diagnostics.

## Expected Direction

- Macro-F1: may increase if no-class recovery dominates, but can fall if
  overcorrection harms yes examples.
- Accuracy: expected to decrease if yes recall drops.
- Brier/ECE: expected to worsen if the model becomes more confidently shifted
  toward minority labels.
- Maybe class: expected to remain unresolved.

## Risk Ledger

- The weight may create a label-prior correction rather than better evidence
  use.
- Stronger auxiliary pressure may worsen accuracy and Brier without improving
  macro-F1.
- Any result must be reported as a validation-only prospective audit.

## Claim Gate

Allowed claim: this configuration prospectively tests whether stronger
class-balanced auxiliary pressure amplifies the observed class-behavior
trade-off.

Forbidden claim: this configuration validates biomedical reasoning or provides
external confirmation of agent superiority.
