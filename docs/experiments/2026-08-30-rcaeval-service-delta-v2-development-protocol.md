# RCAEval service-delta v2 development protocol

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

Status: locked before v2 candidate predictions were generated.

## Context

The v1 robust service-delta score failed RE2-SS confirmation because its hard
cap saturated on 89 of 90 cases. Maximum aggregation then produced many
zero-margin ties, all candidate errors occurred in ties, and lexicographic
resolution biased predictions toward `carts`.

RE2-OB and RE2-SS are now both development systems. Candidate v2 may be
confirmed only on a third untouched system.

## Fixed candidates

### Empirical-tail Top-2 service score

Only `__mean_delta` features are used. For each feature, the training-fold
median is the centre. A test value receives the two-sided empirical surprise

`-log10((1 + number of training deviations at least as large) / (n + 1))`.

There is no hard cap. Each service's primary score is the mean of its two
largest feature surprises, which prevents one saturated feature from deciding
the result. Ties are resolved using, in order, mean surprise across all service
features, fraction outside the training range, mean log-transformed training
z, and finally service name. No target values fit the score.

### Full/delta fixed soft vote

The full pre/post/delta logistic branch and the delta-only logistic branch are
fitted separately within each training fold. Their class probabilities are
averaged with fixed weights 0.5 and 0.5. Classifier and preprocessing settings
are inherited unchanged from the previous experiments. The weights are not
tuned.

## Evaluation

Both candidates are evaluated on the existing repetition-held-out and
fault-type-held-out folds for RE2-OB and RE2-SS. The stored full-metric
logistic predictions are reused as the reference. Report every system/design
Macro F1, Top-1, Top-3, repetition fold, delay fold, predicted-class balance
and exact numeric tie rate.

## Promotion gate

Across the four system/design combinations, a candidate must satisfy all:

- mean Macro F1 improvement over reference at least 0;
- worst system/design difference from reference at least -0.05;
- every system/design Macro F1 at least 0.75;
- every repetition-fold Macro F1 at least 0.55;
- mean delay difference from reference at least -0.02;
- every Top-3 accuracy at least 0.95;
- no predicted class may exceed 40% of a 90-case result;
- exact numeric tie fraction at most 5%.

If both pass, prefer the empirical-tail score because it does not fit target
coefficients. Otherwise, the fixed soft vote may advance. Passing only permits
confirmation on an untouched third system; it is not itself confirmation.

The exact rules and frozen input hashes are in
`experiment/configs/rcaeval-service-delta-v2-development-spec.json`.
