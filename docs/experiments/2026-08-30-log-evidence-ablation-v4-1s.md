# +1-second log-evidence ablation on v4

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

Date: 2026-08-30

## Decision

The perfect +1-second logs-only result does not depend on a single log source
or only on the preregistered list of explicit fault words. It depends on the
combined evidence pattern across the application service and workload client.

With all sources present, uniformly masking direct database, connection, HTTP
status, success/failure, availability and exception terms left Macro F1 at
1.000. Application-service logs alone scored 0.8222, workload logs alone
0.5556, and masked workload logs 0.1667. The combination of sources therefore
contains structural and contextual evidence beyond the masked lexical terms.

The ungated fusion branch stayed at Macro F1 0.5556 for all six log variants.
At +1 second its metric contribution dominates strongly enough that changing
log evidence does not alter its aggregate result.

## Locked design

The six views were fixed before prediction:

1. all logs;
2. `order-service` only;
3. workload client only;
4. masked all logs;
5. masked `order-service` only;
6. masked workload client only.

The same mask was applied to every incident and class. It removed
database/Postgres/asyncpg, connection/refusal, HTTP, error/failure,
availability/stopping/timeout, health/success/creation and
exception/traceback substrings, plus every three-digit 100–599 status code.
Metric features, +1-second windows, labels and the frozen models were unchanged.

This is a post-hoc dependency test. Inference-time masking creates an
artificial lexical shift and is not an estimate of a naturally instrumented
production system.

## Results

| Log evidence | Mean lines/incident | Gated Macro F1 | Logs Macro F1 | Ungated Macro F1 | Mean logs confidence |
|---|---:|---:|---:|---:|---:|
| All sources | 269.10 | 1.0000 | 1.0000 | 0.5556 | 0.798 |
| Application only | 151.27 | 0.8222 | 0.8222 | 0.5556 | 0.666 |
| Workload only | 82.53 | 0.5556 | 0.5556 | 0.5556 | 0.583 |
| Masked all sources | 269.10 | 1.0000 | 1.0000 | 0.5556 | 0.710 |
| Masked application only | 151.27 | 0.8222 | 0.8222 | 0.5556 | 0.691 |
| Masked workload only | 82.53 | 0.1667 | 0.1667 | 0.5556 | 0.448 |

OOD routing was identical across variants because only log text changed:
fusion coverage remained 0.1667 and fallback rate 0.8333.

## Class-level findings

| Log evidence | Database F1 | HTTP 500 F1 | Service-stop F1 |
|---|---:|---:|---:|
| All sources | 1.0000 | 1.0000 | 1.0000 |
| Application only | 0.6667 | 1.0000 | 0.8000 |
| Workload only | 0.0000 | 0.6667 | 1.0000 |
| Masked all sources | 1.0000 | 1.0000 | 1.0000 |
| Masked application only | 0.6667 | 0.8000 | 1.0000 |
| Masked workload only | 0.0000 | 0.5000 | 0.0000 |

Application-only logs mapped five database failures to service stop. Workload
logs alone mapped every database failure to HTTP 500 but retained all service
stops. After masking workload-only logs, all 30 incidents were predicted as
HTTP 500.

Masking all sources reduced mean prediction confidence from about 0.798 to
0.710 without changing labels. This indicates that the masked terms contribute
strength but are not the only discriminative evidence. Remaining evidence may
include service presence/absence, endpoint and logger context, message shape,
request timing density and unmasked vocabulary. This ablation does not identify
their causal importance individually.

## Interpretation

The local task remains relatively easy for text classification because the
faults generate immediate, distributed operational patterns. A deeper text
model is not justified by these results: TF-IDF already extracts sufficient
information from combined sources.

The scientifically useful negative result is that metrics do not rescue
source-restricted logs at +1 second. The ungated fusion score is unchanged
across all log views, and the gated candidate largely follows its logs-only
fallback. Future metric work should model short-window missingness and temporal
sequences directly rather than reuse full-window aggregate features.

## Artifacts

- Specification:
  `experiment/configs/log-evidence-ablation-v4-1s-spec.json`
- Result:
  `data/results/log-evidence-ablation-v4-1s.json`
- Result SHA-256:
  `639b2abdca4ad6ecc2a55fedc8b4320f07dabf2b59275614c877bc67c894b2f4`
- Figure:
  `data/results/figures/log-evidence-ablation-v4-1s.png`
- Figure SHA-256:
  `aceab318ba13a582d45f0a15f9114edacdd59a7c6ba4a2e232303c49ecc14c69`

All six derived JSONL datasets and their hashes are recorded in the result
artifact. The frozen candidate hash remained unchanged.

## Next step

The local analysis has now reached a defensible depth: complete-window
confirmation, early-window stress testing and evidence-source ablation all
agree that logs are strong and metrics are workload-sensitive.

The next major evidence gain should come from a separate public task rather
than another local classifier. The preferred public expansion is the complete
RCAEval `RE2-OB` subset, retaining its original root-cause service and fault
labels. It contains substantially more cases than the six-case importer pilot
and must be evaluated separately from the local three-class task.
