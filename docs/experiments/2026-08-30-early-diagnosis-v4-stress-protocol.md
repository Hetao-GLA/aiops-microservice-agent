# Early-diagnosis v4 stress-test protocol

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

Date locked: 2026-08-30

This protocol was written after the v4 confirmation result and before any
early-window model prediction. It is therefore explicitly post-hoc and cannot
serve as a second independent confirmation.

The analysis reconstructs each of the 30 accepted v4 incidents at four fixed
cutoffs: one, three, five and eight seconds after fault injection. Each variant
retains the complete ten-second pre-fault telemetry baseline and only logs and
probes timestamped at or before its cutoff. Since every realised v4 fault
lasted more than eight seconds, these variants exclude recovery evidence.

Probe summaries preserve the original pre/post/delta schema. If a numerical
probe field has no observation in the truncated post-fault phase, its post mean
is set to the preregistered failure sentinel 0.0; the delta is then calculated
against the observed pre-fault mean. Raw telemetry hashes must match the v4
provenance before a variant is built.

The frozen gated candidate, its internal logs-only model and its ungated robust
fusion branch are evaluated without fitting. Mandatory outputs are Macro F1,
accuracy, per-class F1, OOD routing coverage, fallback rate and OOD-fraction
distribution at every cutoff, with the complete retrospective v4 result shown
only as a reference point.

Cutoffs are not selected or discarded after prediction. The four windows reuse
the same incidents and are a descriptive time curve, not independent samples.
No model, threshold, feature or imputation rule may be changed on the basis of
this analysis while retaining an independent-test claim.

The machine-readable protocol is
`experiment/configs/early-diagnosis-v4-stress-spec.json`.
