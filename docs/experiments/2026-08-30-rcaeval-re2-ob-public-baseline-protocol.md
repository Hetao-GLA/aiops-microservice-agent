# RCAEval RE2-OB public baseline protocol

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

Status: locked before downloading or modelling the full 90-case subset.

## Purpose

This experiment tests whether the project's conventional log/metric pipeline
can localise a root-cause service on an independent public microservice
benchmark. It does not merge public labels into the local three-fault task.

## Fixed selection

- Source: RCAEval Hugging Face Parquet mirror.
- Index: `data/external/rcaeval/cases.parquet`, SHA-256
  `c49a288920dbba2e8e724679a14636d5c7eb2b45426bba14007ef79a6c0ab1bb`.
- Subset: all and only the 90 `RE2-OB` cases with logs.
- Balance required before evaluation: five root-cause services, six fault
  types and exactly three repetitions of each service/fault pair.
- Modalities: logs and metrics. Traces are excluded because the current
  project pipeline does not consume traces.
- Incident window: injection minus 60 seconds through injection plus 120
  seconds, with inclusive endpoints.

The exact machine-readable settings are in
`experiment/configs/rcaeval-re2-ob-public-baseline-spec.json`.

## Target and features

The target is the original RCAEval `root_cause_service` field with five
classes. The model receives only telemetry-derived features:

1. logs-only: TF-IDF word unigrams/bigrams;
2. metrics-only: per-series pre mean, post mean and mean delta;
3. logs plus metrics: the union of the same two feature blocks.

Case names, dataset metadata, fault labels and target fields are not model
inputs. Container/service identity inside telemetry is retained because the
origin of an observation is an intended root-cause-localisation signal.

All vectorisers, scaling and classification are fitted inside each training
fold. Hyperparameters are fixed in advance; there is no public-data tuning.

## Evaluation

Primary evaluation is a three-fold repetition-held-out design. Fold `k`
tests all 30 cases whose repetition is `k` and trains on the remaining 60.
Every service/fault combination therefore appears in training and test, but
no repeated execution crosses as the same case.

A harder six-fold fault-type-held-out stress test holds out one complete
fault type at a time. This measures localisation under an unseen fault
mechanism and must be reported separately from the primary result.

Report top-1 accuracy, macro F1, per-service F1, top-3 accuracy and confusion
matrices. Report every modality even if one is clearly worse.

## Validity limits

- This is service classification, whereas the official RCAEval benchmark can
  rank more granular root-cause indicators; the scores are not directly
  comparable with its Avg@k tables.
- Only Online Boutique is included, so cross-system generalisation is not
  tested.
- Traces are deliberately absent.
- A high primary score can reflect recurring patterns of known fault types;
  the fault-type-held-out result is the more demanding robustness check.
