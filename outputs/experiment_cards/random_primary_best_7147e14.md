# Experiment Card: Random primary BPB-selected control

- Config: `random_primary_best_7147e14`
- Group: `random`
- Stage: repeated validation audit
- Seeds: 42, 43, 44, 45, 46

## Hypothesis

The BPB-selected five-layer random configuration should provide the primary proxy-metric control under the fixed training budget.

## Expected Direction

Lower lm_val_bpb and stable accuracy near majority baseline.

## Risk Ledger

May preserve yes-collapse and fail minority-class macro-F1 despite good proxy loss and Brier.

## Repeated-Seed Outcome

| metric | mean | sd |
| --- | ---: | ---: |
| lm_val_bpb | 2.1403 | 0.2237 |
| accuracy | 0.5547 | 0.0119 |
| macro-F1 | 0.2573 | 0.0225 |
| Brier | 0.5651 | 0.0058 |
