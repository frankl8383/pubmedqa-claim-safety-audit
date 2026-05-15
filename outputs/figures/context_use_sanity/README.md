# Context-Use Sanity Figure

Panel A compares validation macro-F1 under question+context, question-only,
context-only, and shuffled-context evaluation. Panel B shows the macro-F1
delta relative to question+context for each model family.

The key audit finding is that question-only baselines are unexpectedly strong,
while shuffled context reduces macro-F1, especially for frozen BiomedBERT.
