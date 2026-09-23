# RCAEval RE2-OB shift audit protocol

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

Status: locked after the public baseline result and before feature-level drift
statistics were computed.

## Question

The public baseline exposed two weaknesses: every modality performed markedly
worse when repetition 1 was held out, and `delay` was the weakest unseen fault
for the metric-led models. This audit asks which observable distribution
differences accompany those errors.

It is a diagnostic audit, not a new model experiment. It reuses the frozen
out-of-fold predictions and does not fit a classifier, tune a threshold or
change a hyperparameter.

## Locked inputs

- 90-case incident JSONL, SHA-256
  `d234d0682fcdc19594d4172904a7edb79842c93dd012bf82eaf427beb952d8c5`;
- public baseline result, SHA-256
  `9c7f8095e2d3cacb0c86d32f70e4375014d31d5338d516c1598e145147e7fb4d`;
- public baseline specification, SHA-256
  `574751ca0c032965865ebcce6618e7098237b8836e4d87baa4108a12953a3439`.

Any hash mismatch aborts the audit.

## Metric diagnostics

For each already-defined fold, training-only feature distributions are used to
calculate, per test incident:

1. the fraction of metric features outside the training minimum/maximum;
2. mean absolute z-score using the training mean and standard deviation;
3. mean absolute robust z-score using the training median and interquartile
   range.

Zero is the pre-registered sentinel for a missing numeric feature. Absolute
z-scores are capped at 20 for aggregation. Results are reported overall and
for `pre_mean`, `post_mean` and `mean_delta` feature blocks. The 15 features
with the highest mean absolute training z-score are listed per fold.

## Log diagnostics

Each log line is lowercased, then UUIDs, IPv4 addresses, hexadecimal values and
numbers are replaced with fixed placeholders and whitespace is collapsed. A
template seen only in a test fold is considered unseen. The audit reports:

- line-weighted unseen-template fraction;
- unique unseen-template fraction;
- L1 distance between each incident's log-source proportions and the mean
  training source proportions;
- the ten most frequent unseen templates per fold.

## Unsupervised metric-signal localisation

For each test incident, `mean_delta` features are standardised using only the
training fold. Each candidate service receives the maximum absolute z-score of
its service-prefixed features. The audit reports whether the annotated
root-cause service is ranked in the top 1 or top 3. This is a descriptive
signal check, not the official RCAEval method and not a trained prediction.

## Error association

The frozen prediction table is joined with the diagnostics. Correct and wrong
predictions are compared using group means only. These associations must not be
described as causal or statistically confirmatory.

The exact machine-readable settings are in
`experiment/configs/rcaeval-re2-ob-shift-audit-spec.json`.
