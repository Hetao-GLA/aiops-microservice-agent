# RCAEval RE2-TT service-delta v2 confirmation protocol

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

Status: locked before downloading or evaluating any RE2-TT telemetry values.

## Purpose

RE2-OB and RE2-SS were both used to develop service-delta v2. The locked
development gate selected only `empirical_tail_top2_service_score`. RE2-TT
(Train Ticket) is the untouched third system used for one independent
confirmation attempt. The fixed full-metric logistic regression is the
reference.

## Dataset and modalities

Use all 90 RE2-TT incidents: five root-cause services, six fault types and
three repetitions for every service/fault pair. All 90 cases have metrics,
but `re2tt_ts-auth-service_cpu_1` is marked as having no logs. Both frozen
algorithms use metrics only, so the locked download contains only
`inject_time.txt` and `metrics.parquet`. No missing log is fabricated and no
case is dropped. The incident window remains 60 seconds before through 120
seconds after injection, inclusive.

## Frozen candidate and reference

The candidate is the empirical-tail Top-2 service score exactly as selected
on OB and SS. It uses only `__mean_delta` values, fits its empirical tails
within each training fold, does not use target labels to fit scores, and has
no tunable or fitted cross-system parameters.

The reference is the unchanged full pre/post/delta logistic regression. All
preprocessing and coefficients are fitted inside each training fold. Neither
algorithm may be changed after RE2-TT results are seen. The soft-vote candidate
is not eligible as a fallback.

## Evaluation and gate

Run repetition-held-out and fault-type-held-out evaluation on shared folds.
Across the two designs, all of the following must hold:

- mean Macro F1 difference from the reference is at least 0;
- the worse design is no more than 0.05 below the reference;
- candidate Macro F1 is at least 0.75 in both designs;
- every repetition fold has Macro F1 at least 0.55;
- held-out-delay Macro F1 is no more than 0.02 below the reference;
- Top-3 accuracy is at least 0.95 in both designs;
- no predicted class exceeds 40% of the 90 cases in either design;
- exact numeric tie fraction is at most 5% in either design.

Every check must pass. A pass supports three-system confirmation for this
algorithm under the stated RCAEval service-localisation setup. A failure is
reported as a failed confirmation; TT will not become another tuning set in
this phase.

Exact hashes, class lists and numerical limits are in
`experiment/configs/rcaeval-re2-tt-service-delta-v2-confirmation-spec.json`.
