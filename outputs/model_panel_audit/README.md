# PubMedQA Frozen Encoder Model-Panel Audit

This is a post-freeze validation-only audit. It does not use the PubMedQA test split.

## Models

- TF-IDF logistic regression: sparse lexical reference baseline.
- BERT: `bert-base-uncased` (general; citation key `Devlin2019BERT`).
- SciBERT: `allenai/scibert_scivocab_uncased` (scientific; citation key `Beltagy2019SciBERT`).
- BioBERT: `dmis-lab/biobert-v1.1` (biomedical; citation key `Lee2020BioBERT`).
- BiomedBERT: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (biomedical; citation key `Gu2021PubMedBERT`).
- BioLinkBERT: `michiyasunaga/BioLinkBERT-base` (biomedical-link; citation key `Yasunaga2022LinkBERT`).
- BioClinicalBERT: `emilyalsentzer/Bio_ClinicalBERT` (clinical; citation key `Alsentzer2019ClinicalBERT`).

## Interpretation

The goal is not to find a new PubMedQA SOTA model. The audit tests whether
context-use artifacts, calibration trade-offs, and class-behavior shifts are
visible across sparse, general-domain, scientific, biomedical, and clinical
reference encoders.
