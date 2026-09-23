# Local traditional-ML baseline v1 - 28 August 2026

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

## Purpose

Evaluate the first learned root-cause classifiers on the frozen 60-incident
local campaign, while preventing log-line leakage and using identical
incident-level folds for a fair comparison between logs alone and logs plus
metrics.

## Dataset and leakage controls

The input is `local-campaign-v1.jsonl`, SHA-256
`dd45cf4378e01caa5f9d3b698bba07870dad9f8d7306a1bedbe56ddf6d75ace6`.
It contains 60 unique incidents and 60 unique split groups, with 20 incidents
for each of the three local fault classes and a union of 45 metric features.

The raw incident text could not be used directly. It contained the rule
detector's own log records, including its `predicted_fault` value. The model
input therefore applies an auditable, non-destructive sanitisation step:

| Leakage control | Count |
|---|---:|
| Original log records | 19,729 |
| Rule-detector records removed | 4,249 |
| Fault-control toggle records removed | 40 |
| Injected `fault_type` fields stripped | 160 |
| Operational log records retained | 15,440 |

All 60 records retained non-empty operational evidence. Exact
`database_connection_failure` and `http_500_failure` target strings were absent
after sanitisation. The application lifecycle message `service_stopped`
remained in all 20 service-stop incidents; this is a genuine service shutdown
record rather than a rule-detector prediction, but it is also a strong
closed-set signature and is treated as a limitation.

The original incident JSONL was not changed. Sanitisation occurs only inside
the ML input pipeline, and its counts are written to the result artifact.

## Evaluation design

- `StratifiedGroupKFold`, five folds, shuffled with seed `20260825`;
- one complete incident is the unit of splitting;
- each fold contains 48 training records and 12 test records;
- each test fold contains four incidents from each class;
- both models use exactly the same fold assignments;
- TF-IDF vocabulary, metric vectorisation and scaling are fitted only on the
  training portion of each fold;
- every reported prediction is out-of-fold.

The logs-only model uses word unigrams and bigrams, `min_df=2`,
`max_df=0.98`, at most 10,000 TF-IDF features, and logistic regression with
`C=1.0`. The combined model uses the same text branch plus the 45 metric
features transformed by sparse `StandardScaler`, followed by the same
classifier.

## Results

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Fold F1 mean +/- SD |
|---|---:|---:|---:|---:|---:|
| Logs only | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 +/- 0.000 |
| Logs + metrics | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 +/- 0.000 |

Both models correctly classified all 60 out-of-fold incidents. Each confusion
matrix therefore contains 20 observations on each diagonal cell and zero
off-diagonal observations. Adding metrics changed neither accuracy nor Macro
F1; the observed difference is exactly `0.000`.

The mean predicted-class confidence was 0.882 for logs only and 0.991 for logs
plus metrics. This is descriptive only: the experiment did not evaluate
probability calibration, and the higher confidence must not be reported as an
accuracy improvement.

## Feature interpretation

The full-data fits used for feature interpretation are separate from the
out-of-fold evaluation. The logs-only fit produced 669 features. Its strongest
positive evidence included Python stack-trace terms for database failures,
`internal server error` for HTTP 500, and connection/application-startup terms
for service stoppage.

The logs-plus-metrics fit produced 714 features. Its strongest features were
more directly aligned with the controlled mechanisms:

- database failure: database-health post-fault latency and availability change;
- HTTP 500: HTTP 500 count/rate and request failure-count change;
- service stop: liveness and order-metrics post-fault latency.

This improves the semantic quality of the evidence used by the model, but the
classification score is already at its ceiling, so it does not demonstrate a
quantitative performance improvement.

The historical figure for class balance, paired fold comparison and confusion
matrices is recorded at `data/results/figures/local-ml-baseline-v1.png`. It is
not included in this source-only repository; see the
[reproducibility and artifact-availability notes](../REPRODUCIBILITY.md).

## Interpretation against the hypotheses

The result does not support the current form of H1: the learned models match
the frozen rule baseline's Macro F1 of 1.000 but cannot exceed it. It also does
not support H2 under this test condition: adding metrics produces a Macro-F1
change of zero. These are valid ceiling-limited findings, not failed
experiments.

The three failure classes are highly separable because they were generated by
one system, one campaign and fixed fault mechanisms. Five-fold performance on
this dataset estimates interpolation within that campaign; it does not
establish cross-system, cross-workload or unseen-fault generalisation. The
model also performs incident-level post-window classification, whereas the
rule detector performs online detection, so their latency cannot be compared
directly.

## Next experimental step

Before considering deep learning, produce an independent robustness campaign
with changed workload and fault-window settings. Keep the current model and
preprocessing frozen, then use the new campaign as a true external holdout.
Useful secondary ablations are removal of explicit lifecycle signatures and
controlled corruption or truncation of one modality. These tests can break the
current ceiling and determine whether metrics improve robustness when logs are
incomplete or noisy.

## Reproduction

Install the isolated analysis dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
```

Run the same evaluation without overwriting the v1 artifacts:

```powershell
.\.venv\Scripts\python.exe -m experiment.ml_baseline `
    --output data\results\local-ml-baseline-reproduction.json `
    --figure data\results\figures\local-ml-baseline-reproduction.png
```

The frozen machine-readable result is
`data/results/local-ml-baseline-v1.json`, and the implementation is
`experiment/ml_baseline.py`.
