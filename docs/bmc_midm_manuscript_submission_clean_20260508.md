# A claim-safety methods audit of low-resource autonomous biomedical NLP experimentation: a PubMedQA case study

## Authors

Zihao Liu^1,2, Bing Li^1*

^1 Department of Respiratory and Critical Care Medicine, Shanghai Pulmonary Hospital, School of Medicine, Tongji University, Shanghai 200433, China

^2 School of Medicine, Tongji University, Shanghai 200092, China

*Correspondence: Bing Li, Department of Respiratory and Critical Care Medicine, Shanghai Pulmonary Hospital, School of Medicine, Tongji University, No. 507 Zhengmin Road, Shanghai 200433, China. Email: libing044162@163.com

Zihao Liu ORCID: https://orcid.org/0009-0008-5010-2298

Bing Li ORCID: https://orcid.org/0000-0003-0234-044X

## Abstract

**Background:** Autonomous tools can modify biomedical natural language processing (NLP) training code and select candidate configurations using proxy metrics. In biomedical evidence question answering, apparent gains may reflect seed effects, class-prior shifts, opened test sets, poor calibration, or weak context use rather than biomedical improvement.

**Methods:** We evaluated a claim-safety audit workflow using PubMedQA as a non-clinical benchmark case study. The study audited selected manual, random, and agent-derived PubMedQA configurations rather than benchmarking search algorithms. Primary evidence came from locked validation-split analyses (n=150) with repeated seeds, class-wise metrics, paired bootstrap intervals, prediction-level inspection, language-model bits-per-byte proxy-metric mismatch analysis, reference-panel and same-trained context-use checks, calibration diagnostics, sparse and frozen-encoder reference baselines, and a locally pre-specified validation-only mini-run that was not externally registered. The PubMedQA test split had been opened and was restricted to post-freeze audit-only use. A purposively sampled 60-case supervisor-informed illustrative audit was also integrated.

**Results:** The agent-derived delayed class-balanced auxiliary objective increased validation macro-F1 versus the bits-per-byte-selected random primary control, mainly through no-class recovery, defined as increased no-class F1 and increased predicted-no fraction. This came with lower accuracy, worse Brier score and expected calibration error diagnostics, unresolved maybe-class behavior, and small or uncertain differences versus the strongest retained random/manual controls. Proxy-metric mismatch analysis showed that bits-per-byte selection was poorly aligned with downstream macro-F1 in the original retained-configuration audit. Same-trained context ablation showed mixed perturbation sensitivity: agent auxiliary question-only evaluation reduced macro-F1 from 0.3210 to 0.2716 and worsened Brier score from 0.6703 to 0.7621, whereas shuffled-question and shuffled-context perturbations changed macro-F1 modestly. A sparse/frozen model panel supplied reference lines and showed additional context-use and calibration trade-offs. The 60-case illustrative audit supported failure-mode taxonomy development but not independent relabeling or expert consensus.

**Conclusions:** The agent-derived objective produced a bounded class-behavior hypothesis, not validated biomedical improvement. Similar low-resource biomedical NLP benchmark results should remain hypothesis-generating unless repeated-seed, control, proxy-metric, context-use, calibration, test-freeze, and case-review gates support stronger claims.

**Keywords:** medical informatics; biomedical natural language processing; reproducibility; machine learning; calibration.

## Background

Biomedical NLP systems are commonly developed through iterative experimentation: preprocessing is changed, training code is edited, candidate configurations are retained or discarded, and manuscript claims are built around the metrics that appear to improve. When autonomous code-modifying agents enter this loop, the risk is not only that a model score may be noisy [1,2]. The larger health-informatics risk is that a weak experimental signal may be reported as biomedical improvement before its evidence is strong enough.

This reporting risk matters for evidence-question-answering benchmarks that may influence downstream evidence-screening workflows [3]. A system that appears to improve a biomedical benchmark may have shifted class priors, exploited question wording, degraded calibration, or benefited from an opened test set [4]. Such results should not be reported as evidence understanding, clinical usefulness, or agent superiority without explicit guardrails. Reporting standards and transparency proposals for medical artificial intelligence emphasize the need to describe data, intended use, model development, evaluation, and limitations clearly [5-8]. Model cards and datasheets make a similar point for reusable machine-learning artifacts [9,10]. In biomedical NLP, strong domain reference models such as BioBERT, SciBERT, BiomedBERT, BioLinkBERT, and BioClinicalBERT also make it unsafe to interpret a small-model result without stronger reference lines [11-16].

PubMedQA is a useful non-clinical stress test for this problem. It asks yes/no/maybe questions about PubMed abstracts and contains an expert-labeled PQA-L subset [17]. The task is small, imbalanced, and sensitive to the rare maybe label. Its question-answering format also creates a grounding challenge: performance may come from question wording or label priors rather than robust use of the abstract context. These properties do not invalidate PubMedQA. They make it a suitable case for asking whether a low-resource autonomous biomedical NLP experiment can support safe claims.

In health informatics, benchmark-derived biomedical NLP claims can influence evidence-screening, triage, and reporting about biomedical information systems even when no model is deployed clinically. Audit methods that bound unsupported improvement claims are therefore part of responsible health information technology evaluation.

The objective of this study was therefore not to build a deployable PubMedQA system. It was to evaluate a claim-safety audit workflow for low-resource autonomous biomedical NLP experimentation. The central question was how an apparent model improvement should be bounded when repeated-seed evaluation, stronger controls, proxy-metric mismatch, context-use checks, calibration diagnostics, test-freeze rules, and supervisor-informed case review tell different stories.

## Methods

### Study aim and audit object

This was an in silico medical informatics methods and audit study. It did not involve patient-level records, identifiable private information, clinician-patient interaction, healthcare deployment, or clinical decision-support evaluation. The object of evaluation was the autonomous experimentation workflow and the safety of claims made from its outputs.

The study audited selected configurations rather than benchmarking search algorithms. The agent phase occurred after earlier manual and random exploration and changed the objective space by introducing a delayed weak class-balanced auxiliary answer-label objective. The resulting agent-derived configuration was therefore treated as a candidate hypothesis requiring audit, not as evidence that an agent-search algorithm outperformed random or manual search.

The autonomous experimentation harness was a local code-modifying workflow around the small-language-model training and evaluation scripts. It could inspect workspace code, configuration files, run logs, and validation-facing summaries such as BPB, accuracy, macro-F1, and Brier score, and it could propose or edit training-code changes within the low-resource run budget. It was not treated as an independent search comparator. Human review retained candidate manual, random, and agent-derived configurations for later audit, and the current manuscript-facing evidence was regenerated from explicit JSON configurations, validation-only repeated seeds, and post-freeze test-use guardrails. Full training used a fixed five-minute Apple Silicon/MPS budget per small-language-model run; analysis-level reproduction uses the tagged release scripts and processed outputs.

The audit workflow is summarized in Figure 1.

### Data sources and split governance

The primary dataset was PubMedQA PQA-L [17]. A locked split contained 700 training, 150 validation, and 150 test examples. The split was generated reproducibly from the PubMedQA PQA-L source by stratified label splitting with seed 20260503; split manifests and checksum files are provided in the repository release. The validation split contained 83 yes, 51 no, and 16 maybe examples; the test split contained 83 yes, 50 no, and 17 maybe examples. The yes-majority baseline achieved 0.5533 accuracy and 0.2375 macro-F1 on both validation and test. Main evidence was based on validation analyses. Table 1 summarizes the split governance and claim boundary.

The PubMedQA test split had been opened during an earlier audit stage. After the freeze notice, no prompt, objective, calibration decision, model-selection decision, or manuscript-facing method change used PubMedQA test metrics. Test results were restricted to post-freeze audit artifacts and were not used as blind confirmation.

The audit timeline was as follows: manual and random exploration preceded the agent-derived candidate; PubMedQA test results were opened during the initial audit on May 3, 2026; test-freeze rules then restricted all later test use to audit-only reporting; the claim-safety protocol was retrospectively formalized; the auxiliary-weight mini-run was locally pre-specified and run on validation data only on May 6, 2026; the same-trained context-ablation audit was added on validation data only; and the submission package was assembled without using test metrics for prompt, objective, calibration, or model-selection decisions.

SciFact was used as an external reference stress test for related scientific evidence-classification behavior [18]. It did not test the PubMedQA autoresearch-trained small causal language model and does not validate the PubMedQA agent-derived intervention.

### Retained configurations

The autoresearch harness trained small causal language models under a fixed 5-minute Apple Silicon budget. This budget was a low-resource stress-test constraint rather than a search for the best possible biomedical model. PubMedQA label scoring used normalized yes/no/maybe answer-label probabilities.

Seven retained manual, random, and agent configurations were evaluated across validation seeds 42-46. The BPB-selected random primary configuration was the main random control. Additional random controls were selected by accuracy, Brier score, and macro-F1. Manual controls included a baseline and warmup configuration. The agent-derived candidate was the delayed class-balanced auxiliary objective.

Configuration labels such as random macro-F1 best denote the original retention criterion or configuration identity from the earlier search history. They are not claims of repeated-seed superiority.

Current reproduction uses explicit JSON configurations through `run_config_replicates.py`. The older locked-replicate wrapper is retained only as historical provenance because it does not explicitly load the current agent auxiliary configuration.

### Repeated-seed and paired-bootstrap audit

Validation outcomes included language-model validation bits-per-byte (BPB), accuracy, macro-F1, class-wise F1, predicted-label distribution, Brier score, expected calibration error (ECE), mean confidence, entropy, and high-confidence wrong rate. Repeated validation seeds were used because NLP scores can vary materially across runs [19]. Pairwise comparisons used paired bootstrap resampling over validation examples with 10,000 iterations [20,21]. The resampling unit was the aligned PubMedQA example or PMID; sampling was unstratified, used seed 20260504, and reported percentile 95% intervals. For repeated-seed configurations, predictions and probabilities were aligned by PMID and metrics were averaged across the five seeds within each bootstrap sample; single-run reference baselines were included as reference lines, not as equally replicated competitors. The resulting intervals were descriptive uncertainty intervals conditional on the retained configurations and validation setting. They were not confirmatory significance tests, search-adjusted intervals, or evidence of generalizable PubMedQA model superiority.

### Proxy-metric mismatch and PMMI

The primary proxy metric was language-model validation BPB, where lower is better. Downstream PubMedQA outcomes included accuracy, macro-F1, and Brier score. Proxy-metric mismatch was summarized with rank inversion, correlations, Pareto conflicts, and a simple Proxy-Metric Mismatch Index (PMMI). PMMI averaged two scaled rank gaps: the downstream rank gap of the BPB-selected configuration and the BPB rank gap of the downstream winner. A value near zero indicates agreement between proxy and downstream winners; a larger value indicates that proxy and downstream selection identify different configurations.

PMMI was computed for the original seven retained configurations and for an expanded nine-configuration set that included two later locally pre-specified auxiliary-weight mini-run settings. The expanded analysis was used to examine whether the protocol continued to flag accuracy and calibration harms when BPB and macro-F1 appeared more aligned. PMMI was a descriptive audit index defined for this case study rather than a validated general-purpose metric.

### Context-use and calibration checks

Context-use checks were intended to prevent direct claims that PubMedQA macro-F1 implies abstract-grounded biomedical reasoning. Sparse and frozen reference models were evaluated with question plus context, question only, context only, and question plus shuffled validation context. Shuffled-context evaluation paired validation questions with mismatched validation contexts. These analyses probed artifact sensitivity; they were not used to relabel PubMedQA.

A same-trained context-ablation audit was then added for the two retained small-language-model configurations most central to the claim boundary: BPB-selected random primary and agent auxiliary. This was a later validation-only rerun of the same JSON configurations, not a reuse of the primary prediction files. Therefore, its original question-plus-context condition served as the within-run reference for perturbation direction and was not intended to replace the primary repeated-seed summary. For each validation seed, the final in-memory trained model was evaluated before process exit under six prompt modes: original question plus context, question only with an empty context block, question with the context block removed, context only with an empty question field, question plus deterministically shuffled validation context, and deterministically shuffled validation question plus original context. This design perturbed inputs within the same trained model and seed, avoiding a confound between context mode and independently trained checkpoints. It used only the validation split.

Calibration diagnostics included Brier score, ECE, mean confidence, high-confidence wrong rate, and grouped multiclass Brier decomposition. Multiclass Brier score was computed as the mean summed squared distance between predicted class probabilities and one-hot labels. ECE was confidence-based multiclass ECE with 10 equal-width confidence bins. Because probability sources differed across label-scored language models, logistic regression heads, and frozen encoders, these diagnostics were interpreted as validation-set sanity checks rather than clinical reliability estimates [22,23].

### Model-panel reference audit

TF-IDF logistic regression and frozen transformer encoders were used as reference lines and audit objects, not as compute-matched competitors. The model panel included BERT, SciBERT, BioBERT, BiomedBERT, BioLinkBERT, and BioClinicalBERT [11-16,24-26]. For each encoder, mean-pooled frozen representations were fed into logistic regression. Hyperparameters were selected on an inner stratified split of the PubMedQA training data, after which the selected model was refit on the full training split and evaluated on the locked validation split. The PubMedQA test split was not used.

### Supervisor-informed illustrative case audit

A purposively sampled 60-case PubMedQA validation audit was integrated after the repeated-seed prediction audit. Cases were sampled from five behavior strata: agent correction of random-primary yes-collapse in no-labelled examples, agent overcorrection of clear or qualified yes-labelled examples, maybe or qualified-label cases, both-models-correct controls, and both-models-wrong or ambiguous controls. PubMed identifiers, dataset-provided question/context fields, and model prediction distributions were available for each case.

The resulting table was supervisor-informed and standardized into controlled vocabulary fields. The normalized fields included abstract-conclusion support, evidence type, linguistic cue, agent-effect category, label ambiguity, context sufficiency, safe manuscript use, reviewer confidence, and a brief note. This audit is supervisor-informed and illustrative. It is not independent blinded two-reviewer consensus, relabeling of PubMedQA, or a new reference standard.

### Pre-specified validation-only mini-run

After formalization of the hypothesis-first protocol, two auxiliary-weight settings were locally pre-specified for the random-primary backbone: 0.02 and 0.10. This mini-run was not externally registered. Experiment cards and JSON configurations were written before execution. Each setting was evaluated on validation seeds 42-46 only. No PubMedQA test command was used. The mini-run tested whether the protocol could pre-specify expected benefits and harms before execution; it was not intended to find a new best model.

### Reproducibility package and software availability

The audit package includes configuration files, run registry, prediction files, metric tables, analysis scripts, figure-generation scripts, split manifest and checksum files, test-freeze notice, deprecated-runner notice, PMMI outputs, model-panel outputs, and supervisor-informed case-audit provenance files. Large local caches, virtual environments, and downloaded model weights are excluded from the release package because they can be regenerated from documented scripts and public sources. The Python code requires Python >=3.10 and uses scikit-learn, PyTorch, and Hugging Face Transformers [24-26]. The public release is available at https://github.com/frankl8383/pubmedqa-claim-safety-audit/releases/tag/v1.0.1-bmc-midm-clean.

## Results

### Claim-safety workflow outputs

The audit produced a 24-item claim-safety matrix, 12-item guardrail checklist, run registry, experiment cards, test-freeze notice, deprecated-runner notice, PMMI tables, model-panel outputs, and supervisor-informed case-audit taxonomy. The claim matrix allowed one narrow performance statement: the agent-derived auxiliary objective produced no-class recovery or a class-behavior shift relative to the BPB-selected random primary control in the locked PubMedQA validation audit. It forbade agent-superiority claims, clinical utility claims, blind-test confirmation claims, sufficient-BPB-proxy claims, abstract-grounded reasoning claims, and independent expert-consensus claims unless additional evidence is added.

### Repeated-seed claim boundary

Across five validation seeds, the BPB-selected random primary configuration had mean BPB 2.1403, accuracy 0.5547, macro-F1 0.2573, Brier 0.5651, and ECE 0.0321. The agent auxiliary configuration had worse BPB (2.3770), lower accuracy (0.4960), worse Brier (0.6532), and higher ECE (0.1614), but higher macro-F1 (0.3291). This was a trade-off rather than global improvement.

The primary retained-configuration audit is summarized in Table 2 and Figure 2. Paired bootstrap analysis clarified the boundary. Agent auxiliary exceeded random primary by +0.0718 macro-F1 (95% interval +0.0316 to +0.1090), largely through no-class F1 improvement (+0.2925, +0.2025 to +0.3716). However, accuracy decreased (-0.0587, -0.1187 to +0.0013), and Brier score worsened (+0.0881, +0.0488 to +0.1286; positive indicates worse Brier for the agent auxiliary configuration). Agent auxiliary had only small uncertain macro-F1 differences versus random macro-F1-best (+0.0179, -0.0207 to +0.0549) and manual baseline (+0.0186, -0.0187 to +0.0563). These results supported the compressed claim-safety matrix in Table 3.

### Prediction-level class behavior and case-audit taxonomy

Random primary predicted yes for 97.6% of validation predictions and had no-class F1 0.0622. Agent auxiliary reduced the predicted yes fraction to 66.8%, increased the predicted no fraction to 30.1%, and reached no-class F1 0.3548. This explained the macro-F1 improvement relative to the BPB-selected random primary control, but it also reduced yes recall and left the maybe class nearly unresolved (F1-maybe 0.0167).

The 60-case supervisor-informed audit supported a behavioral taxonomy. It categorized 13 cases as corrected yes-collapse, 12 as overcorrected yes, 15 as maybe unresolved, 10 as both wrong or ambiguous, eight as no meaningful change, and two as increased instability. Eight cases were judged suitable for main-text illustration, 42 for supplementary reporting, and 10 for exclusion from manuscript examples. Thirty-seven cases were high-confidence, 22 moderate-confidence, and one low-confidence. These categories illustrate failure modes and claim-safety risks; they do not establish independent PubMedQA relabeling or improved biomedical reasoning. The sanitized case-audit table is provided in Additional file 1.

### Proxy-metric mismatch

In the original seven-configuration audit, the BPB-selected random primary configuration ranked sixth of seven by macro-F1, while the macro-F1 winner, agent auxiliary, ranked fifth of seven by BPB. The original-seven macro-F1 PMMI was 0.75, and the downstream macro-F1 gain of the macro-F1 winner over the BPB winner was 0.0718. The seed-level BPB versus macro-F1 Spearman correlation was +0.39, opposite the expected negative direction for a lower-is-better proxy and a higher-is-better downstream metric (Figure 3). This does not show that BPB is generally invalid. It shows that in this low-resource PubMedQA audit, BPB alone was incomplete for selecting configurations with better minority-class macro-F1.

When the two locally pre-specified mini-run settings were included, the BPB winner changed to auxiliary weight 0.02, which ranked second of nine by macro-F1 but ninth of nine by accuracy and seventh of nine by Brier. The expanded PMMI therefore sharpened a different lesson: even when BPB and macro-F1 look more aligned, accuracy and calibration gates can still fail.

### Model-panel reference lines and context/calibration trade-offs

The model panel helped bound interpretation of the small autoresearch model. BiomedBERT achieved the highest PubMedQA validation macro-F1 (0.4351), followed by BERT (0.3896), BioBERT (0.3828), BioLinkBERT (0.3691), TF-IDF logistic regression (0.3516), SciBERT (0.3357), and BioClinicalBERT (0.3089). The agent auxiliary configuration remained below the strongest biomedical frozen-encoder reference line.

The panel also showed that better macro-F1 did not imply uniformly better calibration or evidence grounding (Table 4 and Figure 4). BiomedBERT had the strongest macro-F1 but Brier 0.7034 and ECE 0.2575 in this validation audit. BioBERT had lower macro-F1 but lower Brier (0.5861) and ECE (0.1246). Context-use checks further limited grounding claims: BiomedBERT achieved question-only macro-F1 0.4441 versus 0.4351 with question plus context, while shuffled context reduced macro-F1 by 0.1386. Strong question-only behavior does not invalidate PubMedQA, but it prevents direct claims that macro-F1 proves abstract-grounded reasoning.

### Same-trained context-ablation audit

The same-trained ablation extended this guardrail to the small autoresearch models themselves. For random primary, the original question-plus-context mode remained yes-collapsed, with mean macro-F1 0.2392 and predicted no fraction 0.0067. Question-only and context-removed prompts increased macro-F1 to 0.2793 and 0.2804 by increasing predicted no fractions to 0.2973 and 0.2240, but Brier score worsened from 0.5712 to 0.5939 and 0.5897. This was prompt-format sensitivity, not improved evidence reasoning.

For agent auxiliary, the original question-plus-context mode within the later ablation rerun had macro-F1 0.3210, Brier 0.6703, no-class F1 0.3381, and predicted no fraction 0.3307. These absolute values differed slightly from the primary repeated-seed audit because the ablation used a separate same-configuration validation rerun designed to compare prompt perturbations within each trained model. Question-only evaluation reduced macro-F1 to 0.2716 and worsened Brier to 0.7621. Removing the context block also reduced macro-F1 to 0.2861 and Brier to 0.6990. However, context-only, shuffled-context, and question-shuffled evaluations were close to the ablation rerun's original macro-F1 (0.3187, 0.3089, and 0.3252). Figure 5 summarizes these perturbation results. They strengthen the context-use guardrail and limit grounding claims: the agent auxiliary objective produced a class-behavior shift, but the available evidence does not show robust question-context-grounded biomedical reasoning.

### SciFact reference stress test and mini-run

On SciFact development data, the majority baseline achieved macro-F1 0.1929, TF-IDF logistic regression achieved macro-F1 0.4264, and frozen BiomedBERT achieved macro-F1 0.4381. This supported the use of external reference stress testing for related scientific evidence-classification behavior. It did not validate the PubMedQA autoresearch-trained small causal language model.

The locally pre-specified auxiliary-weight mini-run completed all 10 planned validation runs. Auxiliary weight 0.02 had mean BPB 2.0018, accuracy 0.4613, macro-F1 0.3202, Brier 0.6437, and no-class F1 0.3913 across five seeds. Auxiliary weight 0.10 had BPB 2.2813, accuracy 0.4640, macro-F1 0.3154, Brier 0.6671, and no-class F1 0.2948. The mini-run showed that the protocol could pre-specify expected class-behavior and calibration trade-offs before execution in a local validation-only demonstration. It did not produce a globally better model and did not use PubMedQA test metrics.

## Discussion

This audit shows how an apparent improvement in a biomedical NLP benchmark can fail multiple claim-safety gates. The retained agent-derived auxiliary objective changed class behavior and partially reduced yes-collapse, but it also degraded accuracy and Brier/ECE diagnostics, left maybe-class behavior unresolved, did not exceed the strongest frozen biomedical reference line, and did not establish abstract-grounded biomedical reasoning. The scientific contribution is therefore not a better PubMedQA model. It is a reproducibility-oriented, internally auditable workflow for bounding biomedical improvement claims from low-resource autonomous experimentation, consistent with reporting-transparency principles for medical AI and reusable model artifacts [5-10].

The agent-derived objective did generate a plausible training hypothesis. It changed the model's output distribution in a way that improved no-class F1 relative to the BPB-selected random primary control. The supervisor-informed case audit made this pattern easier to interpret: some no-labelled examples with negative, no-effect, or limited-utility conclusions were corrected away from yes-collapse. At the same time, clear or qualified yes-labelled examples were sometimes overcorrected toward no, and maybe/qualified cases remained unresolved. This pattern is useful for understanding the mechanism of the intervention, but it is not evidence that the agent learned biomedical reasoning or produced a stronger PubMedQA system [1,2,17].

The PMMI analysis formalized a broader risk. The proxy metric used for selection was not sufficient for safe downstream reporting. In the original retained-configuration audit, the BPB-selected model had favorable BPB, accuracy, and Brier behavior but weak macro-F1 and no-class recovery. In the expanded audit, the BPB winner was closer to the macro-F1 winner but still failed accuracy and calibration gates. These results support a multidimensional audit view: proxy loss, accuracy, macro-F1, class-wise behavior, and calibration can each imply different claims, particularly when seed-level variation and descriptive uncertainty intervals are part of the audit [19-21].

Context-use and calibration checks are central to this framing. Strong question-only behavior in the reference panel and mixed same-trained perturbation behavior in the small autoresearch models show that PubMedQA scores cannot be automatically interpreted as abstract-grounded evidence use, especially in the presence of dataset artifacts and partial-input signals [4]. Calibration diagnostics show that a minority-class recovery signal can coexist with worse probability reliability [22,23]. These two findings are especially relevant to biomedical NLP audit and evidence-screening evaluation settings, where claims about both evidence grounding and uncertainty require caution.

The study should not be read as an agent algorithm benchmark. The agent phase was small, occurred after manual and random exploration, and did not share a blind, identical search space with random search. The appropriate conclusion is that one agent-derived candidate objective generated a bounded hypothesis that required rigorous audit. Future studies that compare agentic search methods should prospectively define search spaces, budgets, information access, stopping rules, and evaluation criteria before the agent run [1,2].

### Practical recommendations

Autonomous biomedical NLP experiments should report candidate improvements as hypothesis-generating until they pass at least six gates: repeated validation seeds, strongest available controls, reference baseline comparison, prediction-level class audit, context-use or partial-input audit, and calibration diagnostics [5-10,19-23]. If a test split has been opened, it should be reported only as post-freeze audit evidence. If case review is used, reviewer qualifications, independence, and disagreement handling should be stated explicitly.

### Limitations

This exploratory audit used one small PubMedQA split, an opened test set, retrospective protocol formalization, and a small non-comparative agent phase. The supervisor-informed case audit, context ablation, calibration diagnostics, and SciFact analysis were claim-safety checks rather than independent consensus, mechanism proof, clinical reliability estimation, or external validation; independent reproduction remains needed.

### Future work

Useful next steps include independent reproduction on another hardware/software environment, repeated inner splits or seeds for the frozen-encoder panel, and a blinded two-reviewer case audit with agreement statistics. A prospective agent-search comparison would require a locked search space, equal budgets, pre-specified stopping rules, and no PubMedQA test use.

## Conclusions

One agent-derived auxiliary objective produced measurable no-class recovery relative to the BPB-selected random primary control, but not a generally better biomedical QA model. Lower validation BPB was an incomplete proxy for downstream claim safety, and stronger frozen encoders showed that the small autoresearch model was best treated as an audit object. Similar low-resource biomedical NLP benchmark results should therefore be reported as bounded hypotheses unless repeated-seed, control, proxy-metric, context-use, calibration, test-freeze, and case-review gates support stronger claims.

## Figures

**Figure 1. Claim-safety audit workflow.** The workflow combines locked data, repeated validation seeds, pairwise controls, prediction-level audit, proxy-metric mismatch analysis, context-use checks, calibration diagnostics, run registry, and forbidden-claim rules. It was used to grade the original search retrospectively and was later operationally demonstrated in a validation-only mini-run.

**Figure 2. Primary repeated-seed claim boundary.** The paired-bootstrap forest plot summarizes validation-set differences between the agent auxiliary configuration and key retained controls. Positive Brier-score differences indicate worse Brier score for the first-listed model, so the figure separates the macro-F1/no-class F1 signal from accuracy and calibration harms.

**Figure 3. Proxy-metric mismatch under repeated PubMedQA validation seeds.** Lower language-model validation BPB did not identify the original-seven macro-F1 winner. The figure is a descriptive rank audit for this case study, not evidence that BPB is generally invalid.

**Figure 4. Model-panel context-use and calibration audit.** Panel a shows validation macro-F1 across sparse, general-domain, scientific, biomedical, and clinical reference models; panel b shows macro-F1 changes under partial-input and shuffled-context audits; panel c compares macro-F1 with Brier score; and panel d shows predicted-label distributions. These reference lines bound interpretation of the small autoresearch model but are not compute-matched competitors.

**Figure 5. Same-trained context-ablation audit.** The two retained small-language-model configurations were rerun with the same JSON settings and evaluated under six prompt modes after each training run without retraining. 'No context block' denotes the `context_removed` mode in which the original question was retained but the context block was removed. Bars show means across five validation seeds, and error bars show across-seed standard deviation. Deterministic shuffled-context and shuffled-question modes were created within the validation split. The original question-plus-context condition is the within-rerun reference for perturbation direction, not a replacement for the primary repeated-seed audit.

## Tables

**Table 1. Study design, split governance, and claim boundary.** This table summarizes the audit object, data split status, opened-test handling, retained-configuration scope, and the claims explicitly excluded by design.

**Table 2. Primary retained-configuration audit.** This table reports the retained configurations most relevant to the claim boundary, including repeated-seed summary metrics and paired differences that separate no-class F1 recovery from accuracy and calibration harms. ECE is a descriptive 10-bin confidence-based diagnostic.

**Table 3. Compressed claim-safety matrix.** This table lists the main allowed, bounded, and forbidden manuscript claims, with the evidence required before stronger biomedical NLP claims could be made.

**Table 4. Model-panel and context/calibration audit.** This table summarizes sparse and frozen-encoder reference lines, context-use checks, and calibration diagnostics used to bound interpretation of the small autoresearch model.

The supervisor-informed case-audit taxonomy summary and the sanitized 60-case table are provided in Additional file 1.

## Additional files

Additional file 1. File format: .zip. Title: Supplementary audit tables and provenance notes. Description: Full claim matrix, PMMI/model-panel/context-ablation outputs, sanitized 60-case audit, artifact path map, and test-freeze/deprecated-runner notes.

The full reproducibility package is provided in the tagged repository release.

## Abbreviations

BPB: bits per byte; ECE: expected calibration error; F1: harmonic mean of precision and recall; LR: logistic regression; MIDM: Medical Informatics and Decision Making; NLP: natural language processing; PMMI: Proxy-Metric Mismatch Index; PQA-L: PubMedQA labeled subset; QA: question answering; TF-IDF: term frequency-inverse document frequency.

## Declarations

### Ethics approval and consent to participate

Not applicable. This study used public benchmark datasets and did not involve human participants, human data, human tissue, animals, patient-level records, identifiable private information, clinician-patient interaction, healthcare deployment, or clinical decision-support evaluation.

### Consent for publication

Not applicable.

### Availability of data and materials

PubMedQA and SciFact are publicly available. Project name: PubMedQA Claim-Safety Audit Package. Processed data, code, predictions, metrics, figures, and provenance files are available in the tagged release [27]. Archived version: `v1.0.1-bmc-midm-clean`; the release notes identify the current release commit and SHA-256 checksums for the journal-facing and full reproducibility release assets. The MIT-licensed Python code requires Python >=3.10 and uses `uv` for dependency management; analysis regeneration is cross-platform, whereas full small-language-model training requires macOS Apple Silicon/MPS. Caches, virtual environments, and downloaded model weights are excluded.

### Competing interests

The authors declare no competing interests.

### Funding

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

### Authors' contributions

Z.L. conceived the computational audit, implemented the experimental harness, ran the experiments, analyzed outputs, curated data and code artifacts, prepared figures and tables, and drafted the manuscript. B.L. supervised the study, contributed to the health-informatics framing, provided supervisor-informed case-audit review, revised the interpretation, and reviewed the manuscript. Both authors approved the final manuscript.

### Acknowledgements

The authors thank the developers and maintainers of the public PubMedQA, SciFact, scikit-learn, Hugging Face Transformers, and biomedical encoder resources used in this audit.

### Declaration of generative AI and AI-assisted technologies

Generative AI tools were used only for language editing during manuscript preparation. The authors conducted the experiments, analyses, case review, interpretation, and final manuscript decisions, and take full responsibility for the content. AI tools were not used as authors, independent adjudicators, expert consensus reviewers, or final scientific decision-makers.

## References

1. Huang Q, Vora J, Liang P, Leskovec J. MLAgentBench: Evaluating language agents on machine learning experimentation. Proc ICML. 2024;235:20271-20309.
2. Chan JS, Chowdhury N, Jaffe O, Aung J, Sherburn D, Mays E, et al. MLE-bench: Evaluating machine learning agents on machine learning engineering. arXiv:2410.07095. 2024.
3. Marshall IJ, Wallace BC. Toward systematic review automation: a practical guide to using machine learning tools in research synthesis. Syst Rev. 2019;8:163.
4. Gururangan S, Swayamdipta S, Levy O, Schwartz R, Bowman S, Smith NA. Annotation artifacts in natural language inference data. Proc NAACL-HLT. 2018:107-112.
5. Norgeot B, Quer G, Beaulieu-Jones BK, Torkamani A, Dias R, Gianfrancesco M, et al. Minimum information about clinical artificial intelligence modeling: the MI-CLAIM checklist. Nat Med. 2020;26:1320-1324.
6. Hernandez-Boussard T, Bozkurt S, Ioannidis JPA, Shah NH. MINIMAR (MINimum Information for Medical AI Reporting): developing reporting standards for artificial intelligence in health care. J Am Med Inform Assoc. 2020;27:2011-2015.
7. Mongan J, Moy L, Kahn CE Jr. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): A guide for authors and reviewers. Radiol Artif Intell. 2020;2:e200029.
8. Collins GS, Dhiman P, Navarro CLA, Ma J, Hooft L, Reitsma JB, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378.
9. Mitchell M, Wu S, Zaldivar A, Barnes P, Vasserman L, Hutchinson B, et al. Model cards for model reporting. Proc FAT*. 2019:220-229.
10. Gebru T, Morgenstern J, Vecchione B, Vaughan JW, Wallach H, Daume H III, et al. Datasheets for datasets. Commun ACM. 2021;64:86-92.
11. Devlin J, Chang MW, Lee K, Toutanova K. BERT: Pre-training of deep bidirectional transformers for language understanding. Proc NAACL-HLT. 2019:4171-4186.
12. Lee J, Yoon W, Kim S, Kim D, Kim S, So CH, et al. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics. 2020;36:1234-1240.
13. Beltagy I, Lo K, Cohan A. SciBERT: A pretrained language model for scientific text. Proc EMNLP-IJCNLP. 2019:3615-3620.
14. Gu Y, Tinn R, Cheng H, Lucas M, Usuyama N, Liu X, et al. Domain-specific language model pretraining for biomedical natural language processing. ACM Trans Comput Healthc. 2022;3:1-23.
15. Yasunaga M, Leskovec J, Liang P. LinkBERT: Pretraining language models with document links. Proc ACL. 2022:8003-8016.
16. Alsentzer E, Murphy JR, Boag W, Weng WH, Jindi D, Naumann T, et al. Publicly available clinical BERT embeddings. Proc Clinical NLP Workshop. 2019:72-78.
17. Jin Q, Dhingra B, Liu Z, Cohen W, Lu X. PubMedQA: A dataset for biomedical research question answering. Proc EMNLP-IJCNLP. 2019:2567-2577.
18. Wadden D, Lin S, Lo K, Wang LL, van Zuylen M, Cohan A, et al. Fact or fiction: verifying scientific claims. Proc EMNLP. 2020:7534-7550.
19. Reimers N, Gurevych I. Reporting score distributions makes a difference: performance study of LSTM-networks for sequence tagging. Proc EMNLP. 2017:338-348.
20. Dror R, Baumer G, Shlomov S, Reichart R. The hitchhiker's guide to testing statistical significance in natural language processing. Proc ACL. 2018:1383-1392.
21. Efron B, Tibshirani RJ. An Introduction to the Bootstrap. New York: Chapman and Hall/CRC; 1993.
22. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. Proc ICML. 2017:1321-1330.
23. Ovadia Y, Fertig E, Ren J, Nado Z, Sculley D, Nowozin S, et al. Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. Proc NeurIPS. 2019;32.
24. Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825-2830.
25. Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, et al. PyTorch: an imperative style, high-performance deep learning library. Proc NeurIPS. 2019;32.
26. Wolf T, Debut L, Sanh V, Chaumond J, Delangue C, Moi A, Cistac P, et al. Transformers: State-of-the-art natural language processing. Proc EMNLP System Demonstrations. 2020:38-45.
27. Liu Z, Li B. PubMedQA Claim-Safety Audit Package. GitHub. Version v1.0.1-bmc-midm-clean. 2026. Available from: https://github.com/frankl8383/pubmedqa-claim-safety-audit/releases/tag/v1.0.1-bmc-midm-clean. Accessed 13 May 2026.
