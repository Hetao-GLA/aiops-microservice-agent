# Robust fusion v2 locked development specification

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

This specification was written after the v1-to-v3 contribution audit and
before fitting the new candidate on v1.

The model keeps the v1 logs-only branch unchanged. Its metric branch removes
absolute and duplicated request-volume families, adds success/failure rates,
clips engineered metrics to training-only ranges and scales by training median
and interquartile range.

For each incident it calculates the fraction of engineered metrics outside
their fitted training ranges. If that fraction is strictly greater than 0.10,
the system routes the incident to the logs-only model. Otherwise it uses robust
early fusion. Routing coverage and fallback rate are mandatory outputs.

All preprocessing is fitted inside each v1 development fold. The final
candidate is fitted on v1 only. v3 may be used only as an explicitly post-hoc
diagnostic because its labels and errors informed this design. No result on v3
can be called independent confirmation for this candidate.

The candidate must be frozen before collecting v4. v4 will be the first
eligible confirmation dataset and must report Macro F1, accuracy, coverage and
fallback rate. Any threshold or feature change after inspecting v4 creates a
new candidate and requires another dataset.

The machine-readable specification is
`experiment/configs/robust-fusion-v2-spec.json`.
