# Gated robust-fusion v2 development and freeze

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

Date: 2026-08-30

Subsequent status: the candidate has now completed its independent v4
confirmation. See the
[v4 confirmation report](2026-08-30-local-holdout-v4-confirmation.md).

## Research question

Can a training-only robust metric representation and an explicit
out-of-distribution (OOD) fallback prevent the workload-sensitive failure seen
in the frozen v1 logs-plus-metrics model without weakening the logs-only
baseline?

This is a development result, not a new confirmation result. The v3 labels and
the v1-to-v3 contribution audit informed the candidate design, so v3 is used
only as a post-hoc diagnostic. A newly collected v4 campaign is required for
confirmation.

## Locked design

The machine-readable design was written before implementation and fitting in
`experiment/configs/robust-fusion-v2-spec.json`.

The candidate:

- retains the v1 logs-only TF-IDF and logistic-regression branch;
- removes the absolute `successful_requests`, `total_requests`,
  `failure_count`, `http_500_count`, and `window_seconds` metric families;
- adds pre-, post-, and delta success/failure rates;
- clips each engineered metric to min/max values learned from the current
  training fold or the full v1 fit;
- scales metrics by the training median and interquartile range, using 1.0 for
  a zero IQR;
- uses robust early fusion when at most 10% of engineered metrics are outside
  their fitted ranges;
- falls back to the logs-only model when the OOD fraction is strictly greater
  than 10%.

The 45 raw v1 metric features become 36 engineered features: 15 absolute
features are removed and six rate features are added.

## Leakage and integrity controls

- Candidate design and the 0.10 routing threshold were locked before fitting.
- v1 development used five shared `StratifiedGroupKFold` splits.
- Feature engineering ranges, medians, IQRs, TF-IDF vocabulary and classifiers
  were fitted on each training fold only.
- The final candidate was fitted on v1 only and then frozen.
- v3 prediction called neither `fit` nor `fit_transform` and did not select a
  feature or threshold.
- Source, runtime, specification and model hashes are verified before model
  deserialisation.
- The model hash was unchanged before and after v3 diagnostic prediction.
- A cross-process regression test verifies that bundles created with
  `python -m` can be loaded through the canonical Python module.

The complete test suite passed 85 tests. The warnings were existing
joblib/NumPy deprecation warnings; there were no failures.

## Results

| Evaluation | Records | Macro F1 | Accuracy | Fusion coverage | Logs fallback |
|---|---:|---:|---:|---:|---:|
| v1 grouped out-of-fold development | 60 | 1.000 | 1.000 | 98.3% | 1.7% |
| v1 logs-only comparator | 60 | 1.000 | 1.000 | n/a | n/a |
| v3 post-hoc diagnostic | 30 | 1.000 | 1.000 | 33.3% | 66.7% |
| v3 logs-only comparator | 30 | 1.000 | 1.000 | n/a | n/a |

All three classes achieved precision, recall and F1 of 1.000 in the routed v3
diagnostic. Routing by actual class was:

| Actual class | Records | Fallback records | Mean OOD fraction | OOD range |
|---|---:|---:|---:|---:|
| Database connection failure | 10 | 4 | 0.110 | 0.056–0.194 |
| HTTP 500 failure | 10 | 6 | 0.139 | 0.056–0.306 |
| Service stopped | 10 | 10 | 0.259 | 0.222–0.444 |

As a supplemental no-fit diagnostic, the robust fusion branch without routing
scored Macro F1 0.9666 on v3. It made one database-to-service-stop error. That
incident had an OOD fraction of 0.1944, so the locked gate routed it to the
logs-only branch and corrected the final prediction. This shows that both the
robust metric transformation and the fallback contributed: the metric-only
redesign removed nine of the ten database errors seen in the original v1
combined model, while routing protected the remaining case.

## Interpretation

The original combined-model collapse from Macro F1 1.000 to 0.5556 is not
reproduced by this candidate on the diagnostic data. Removing absolute request
volume, bounding extrapolation and routing shifted metric observations appear
to address the mechanism identified by the contribution audit.

However, the v3 coverage of 33.3% is low. The perfect routed score is therefore
not evidence that the fusion branch generalises broadly: two thirds of v3
predictions used the already-strong logs-only fallback. The research value of
the next experiment is not only classification F1 but also whether the
candidate can retain useful fusion coverage on genuinely unseen data.

## Frozen artifacts

- Bundle: `data/models/gated-robust-fusion-v2/`
- Model SHA-256:
  `336ef2a404bcd0db3e783638081a6b541e153f79b3cd7276b56423d5b7d99c85`
- Manifest SHA-256:
  `e4c0426e5fa744cb797f981600de55fd506530f640356dfe76ee1fa3629a8b9c`
- Development result:
  `data/results/gated-robust-fusion-v2-development.json`
- Development result SHA-256:
  `ac9149b2c23af198cd1dd41abd40c3ac88a743e1205defa573667ae4b64efc6f`
- Figure:
  `data/results/figures/gated-robust-fusion-v2-development.png`
- Figure SHA-256:
  `e8aa87d24228c71ed4d8396c17926114833c0fbf9649965d76761f1be55e7b42`

## Confirmation decision

At the time of this development report, the candidate was ready for a v4
confirmation campaign but was not yet a confirmed replacement for the original
baseline. v4 has since completed; the original preregistered requirements are
preserved below for auditability.

v4 must:

1. be collected after the candidate freeze time;
2. use new incident IDs, split groups, campaign run IDs and raw telemetry;
3. keep the same local system, three fault classes and raw metric schema;
4. pass the duration, telemetry-latency, recovery, detector-pair, provenance
   and runtime-restoration gates already used for v3;
5. be evaluated once with the frozen candidate, reporting Macro F1, accuracy,
   per-class F1, fusion coverage, fallback rate and OOD fractions;
6. trigger no feature, threshold or classifier change after labels are seen.

Any candidate change after v4 inspection requires a newly frozen model and a
new confirmation dataset.
