# Experiment Card: Random macro-F1 best control

- Config: `random_macro_f1_best_8a1209b`
- Group: `random`
- Stage: repeated validation audit
- Seeds: 42, 43, 44, 45, 46

## Hypothesis

A random configuration selected by single-run macro-F1 should test whether agent gains exceed a strong downstream random control.

## Expected Direction

Higher macro-F1 than BPB-selected random primary, but potentially worse BPB and calibration.

## Risk Ledger

Selection on a single validation run may overstate downstream improvement and requires repeated seeds.

## Repeated-Seed Outcome

| metric | mean | sd |
| --- | ---: | ---: |
| lm_val_bpb | 2.5318 | 0.0955 |
| accuracy | 0.5253 | 0.0378 |
| macro-F1 | 0.3112 | 0.0299 |
| Brier | 0.6227 | 0.0292 |
