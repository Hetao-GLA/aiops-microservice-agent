# RCAEval RE2-OB delta candidates protocol

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

Status: locked before candidate predictions were generated.

## Purpose

The frozen shift audit found that repetition 1 differs mainly in absolute
pre-window metric levels, while service-grouped mean-delta signals remain
informative. This development experiment tests whether removing `pre_mean` and
`post_mean` improves stability without sacrificing unseen-fault performance.

This is post-hoc model development on RE2-OB. It cannot confirm an improvement
on the same data that motivated the candidates.

## Fixed reference and candidates

The fixed reference is the already-published metrics-only model. Its stored
out-of-fold predictions are reused without refitting.

Three candidates are evaluated:

1. `delta_only_lr`: DictVectorizer, sparse standard scaling and logistic
   regression using only `__mean_delta` metric features;
2. `logs_plus_delta_lr`: the same delta block combined with the frozen TF-IDF
   log configuration;
3. `robust_service_delta_score`: a label-free parameter fit that divides each
   delta's absolute deviation from the training median by the training IQR,
   caps it at 20 and assigns each service its maximum service-prefixed score.

The robust scorer selects among the five root-cause service classes observed
in the training fold. Equal scores are resolved lexicographically. It uses no
training target values to fit weights.

All missing numeric values use the already-fixed zero sentinel. Every fitted
transformer uses only the training fold. There is no hyperparameter search.

## Evaluation

The candidates use the same repetition-held-out and fault-type-held-out folds
as the frozen public baseline. Report top-1 accuracy, Macro F1, per-service F1,
top-3 accuracy and confusion matrices. Repetition 1 and held-out `delay` must
be reported separately.

## Pre-registered promotion gate

A candidate must satisfy all four conditions:

- repetition-held-out Macro F1 at least 0.804922;
- repetition-1 Macro F1 at least 0.650000;
- fault-type-held-out Macro F1 at least 0.835535;
- held-out-delay Macro F1 at least 0.457143.

If several pass, select the first in this fixed order:

1. `delta_only_lr`;
2. `robust_service_delta_score`;
3. `logs_plus_delta_lr`.

The selected candidate may proceed to confirmation on a new public system or
suite. Passing this development gate is not itself confirmation.

## Locked inputs

The machine-readable specification records and verifies the incident,
baseline and shift-audit hashes:
`experiment/configs/rcaeval-re2-ob-delta-candidates-spec.json`.
