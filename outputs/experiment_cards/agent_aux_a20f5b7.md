# Experiment Card: Agent auxiliary

- Config: `agent_aux_a20f5b7`
- Group: `agent`
- Stage: repeated validation audit
- Seeds: 42, 43, 44, 45, 46

## Hypothesis

A delayed class-balanced answer-label auxiliary objective may reduce yes-collapse and improve PubMedQA macro-F1 under a fixed 5-minute budget.

## Expected Direction

Higher macro-F1 and no-class F1; possible lower accuracy, worse BPB, and worse calibration.

## Risk Ledger

The intervention may correct label priors rather than improve biomedical evidence reasoning; maybe-class behavior may remain unresolved.

## Repeated-Seed Outcome

| metric | mean | sd |
| --- | ---: | ---: |
| lm_val_bpb | 2.3770 | 0.2318 |
| accuracy | 0.4960 | 0.0507 |
| macro-F1 | 0.3291 | 0.0421 |
| Brier | 0.6532 | 0.0339 |

## Volatility and Claim Gate

| gate | result | evidence |
| --- | --- | --- |
| Class-behavior gate vs BPB-selected random primary | pass | macro-F1 diff 0.0718 [0.0316, 0.1090]; F1-no diff 0.2925 [0.2025, 0.3716] |
| Global superiority gate | fail | macro-F1 diff vs random macro-F1 best 0.0179 [-0.0207, 0.0549] |
| Calibration gate | fail | Brier diff vs random primary 0.0881; ECE agent 0.1614 vs random primary 0.0321 |
| Allowed claim | allowed | Agent generated a no-class recovery hypothesis with a metric trade-off. |
| Forbidden claim | forbidden | Agent found a generally better biomedical QA model. |
