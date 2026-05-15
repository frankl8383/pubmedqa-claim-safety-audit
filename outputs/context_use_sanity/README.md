# PubMedQA Context-Use Sanity Check

This post-freeze validation-only audit tests whether reference baselines depend
on PubMedQA abstract context or can obtain similar behavior from partial input.

Input modes:

- `question_context`: train/evaluate on question plus abstract context.
- `question_only`: train/evaluate on question text only.
- `context_only`: train/evaluate on abstract context only.
- `question_shuffled_context_eval`: train on normal question+context, then
  evaluate with each validation question paired to another validation context.

Interpretation:

- Strong question-only performance suggests question wording or label-prior
  artifacts.
- Context-only performance suggests the abstract contains label signal even
  without the explicit question.
- A small drop under shuffled-context evaluation suggests the model is not
  strongly using question-context alignment.

This audit uses PubMedQA validation only and must not guide PubMedQA test
selection.
