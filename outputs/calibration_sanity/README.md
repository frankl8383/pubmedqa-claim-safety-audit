# PubMedQA Calibration Sanity Check

This audit compares top-label confidence, Brier score, expected calibration
error (ECE), maximum calibration error (MCE), and reliability bins on the locked
PubMedQA validation split.

Important caveat: decoder label-score probabilities and classifier probabilities
are uncalibrated score distributions. These metrics are used to audit relative
reliability patterns, not to claim deployable clinical calibration.
