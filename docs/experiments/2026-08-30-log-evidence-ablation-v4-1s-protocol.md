# +1-second log-evidence ablation protocol

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

Date locked: 2026-08-30

This post-hoc analysis asks why logs-only classification is already perfect one
second after fault injection. It uses the previously locked +1-second v4
variants and does not modify their metric features or time windows.

Six log views are fixed before prediction:

1. all operational logs;
2. `order-service` logs only;
3. workload-client logs only;
4. all logs with direct operational signatures masked;
5. masked `order-service` logs only;
6. masked workload-client logs only.

The same case-insensitive masking rule is applied to every class. It replaces
database/Postgres/asyncpg, connection/refusal, HTTP, error/failure,
unavailable/stopped/shutdown/timeout, healthy/success/creation and
exception/traceback substrings. Every three-digit status code from 100 to 599
is also replaced. The replacement token is not present in the frozen training
vocabulary and is therefore ignored by TF-IDF.

All six variants must be reported. No term, source or result may be selected or
discarded after prediction. The frozen gated candidate, logs-only branch and
ungated robust-fusion branch are evaluated without fitting or tuning. Metrics,
OOD routing and per-class outcomes remain descriptive because v4 and the +1
second result were already observed.

This is a dependency/ablation test, not a natural operational distribution:
masking tokens at inference time also creates lexical shift. Its purpose is to
identify how much the strong early result depends on explicit source messages,
not to estimate deployed accuracy.

The machine-readable protocol is
`experiment/configs/log-evidence-ablation-v4-1s-spec.json`.
