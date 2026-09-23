# RCAEval RE2-SS robust-delta confirmation protocol

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

Status: locked before downloading RE2-SS telemetry or generating predictions.

## Purpose

This experiment tests whether the robust service-grouped delta algorithm
selected on Online Boutique (`RE2-OB`) reproduces on the independent Sock Shop
system (`RE2-SS`) without algorithm changes or hyperparameter tuning.

It confirms algorithm behaviour across systems. It is not a pre-trained model
transfer: training-only distribution statistics and the fixed reference
classifier are fitted anew inside each RE2-SS fold.

## Fixed data selection

- all and only the 90 RCAEval `RE2-SS` cases with logs;
- five root-cause services: carts, catalogue, orders, payment and user;
- six fault types and three repetitions of every service/fault pair;
- 60 seconds before through 120 seconds after injection, inclusive;
- logs and metrics downloaded for schema completeness; traces excluded;
- target remains the original root-cause service and is not mapped to local
  fault classes.

## Frozen algorithms

The candidate implementation is frozen at SHA-256
`89697793842a7766ea68a73d8fa08601fa505b1b44031ab9487bfd2157db7b5f`.
It uses only mean-delta metrics, training-fold medians and IQRs, a score cap of
20 and the maximum service-prefixed score. It fits no target coefficients.

The full-metric logistic-regression reference implementation is frozen at
SHA-256
`90b8017f3209c03fd28c5fde9fb61d3d19216f955012e881108b2f7906367dc7`.
It uses the same pre/post/delta feature construction and fixed classifier as
the RE2-OB public baseline.

Any implementation, index or development-result hash mismatch aborts the
confirmation.

## Evaluation

Both algorithms use identical repetition-held-out and fault-type-held-out
folds. Report complete out-of-fold Top-1 accuracy, Macro F1, per-service F1,
Top-3 accuracy and confusion matrices. Report every repetition and fault fold,
including delay.

## Confirmation gate

All of the following must pass:

- candidate repetition-held-out Macro F1 at least 0.80;
- candidate fault-type-held-out Macro F1 at least 0.80;
- candidate worst repetition-fold Macro F1 at least 0.65;
- candidate held-out-delay Macro F1 at least 0.55;
- candidate Top-3 accuracy at least 0.95 in both designs;
- candidate Macro F1 no lower than the fixed reference in either design;
- candidate held-out-delay Macro F1 no lower than the reference.

Failure of any item means the RE2-OB improvement is not confirmed on RE2-SS.
No threshold may be changed after seeing results.

The exact machine-readable protocol is
`experiment/configs/rcaeval-re2-ss-confirmation-spec.json`.
