# Experiment Card: Manual baseline

- Config: `manual_baseline_f1664dd`
- Group: `manual`
- Stage: repeated validation audit
- Seeds: 42, 43, 44, 45, 46

## Hypothesis

The locked manual baseline anchors the audit and estimates seed variance without agent or random modifications.

## Expected Direction

Moderate macro-F1 and stable crash-free execution.

## Risk Ledger

If omitted, agent/random comparisons are not interpretable.

## Repeated-Seed Outcome

| metric | mean | sd |
| --- | ---: | ---: |
| lm_val_bpb | 2.3824 | 0.0410 |
| accuracy | 0.5053 | 0.0292 |
| macro-F1 | 0.3105 | 0.0278 |
| Brier | 0.6356 | 0.0223 |
