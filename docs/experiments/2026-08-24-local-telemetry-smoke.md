# Local telemetry smoke experiments - 24 August 2026

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

## Purpose

Validate that each of the three controlled fault classes produces an isolated
Ground Truth record, host-side probes, bounded Docker logs and a run-specific
detector archive suitable for later incident-schema conversion.

## Configuration

| Fault | Incident | Telemetry run | Pre / post | Interval |
|---|---|---|---:|---:|
| HTTP 500 | `INC-DB8294E49724` | `batch-20260824T141601Z-e7bb5442` | 2 s / 4 s | 0.5 s |
| Database connection failure | `INC-35A841916D93` | `batch-20260824T142054Z-58c85e15` | 2 s / 6 s | 0.5 s |
| Service stopped | `INC-E46F4733E7EE` | `batch-20260824T141846Z-722315b8` | 2 s / 6 s | 0.5 s |

Each controlled fault lasted 3 seconds. The database rerun used an automatically
resolved request timeout of 0.4 seconds, which is 80% of the probe interval.

This short window was used only to validate the collector. Dataset-production
runs should use the fixed default windows or another value declared before the
batch.

## Collection result

| Fault | Probe rows | Unavailable probes | Docker log rows | Collector / Docker errors |
|---|---:|---:|---:|---:|
| HTTP 500 | 54 | 0 | 153 | 0 / 0 |
| Database connection failure | 90 | 9 | 220 | 0 / 0 |
| Service stopped | 69 | 15 | 160 | 0 / 0 |

The HTTP 500 window recorded three `forced_http_500` application events and a
maximum five-second HTTP 500 rate of 0.6. Database failure produced nine
unavailable database-health probes, including two explicit HTTP 503 responses.
Service stoppage produced 15 unavailable probes while the API container was
down. These unavailable observations were retained rather than discarded.

PostgreSQL produced no log line within the short HTTP 500 and service-stop
windows; this is an observed zero rather than a collection failure.

## Rule-baseline evaluation

Evaluation used only the run-specific detection archive. Historical records in
the shared detector file were excluded.

| Fault | Correct | FP / FN | Precision / Recall / F1 | Detection latency | Recovery latency |
|---|---:|---:|---:|---:|---:|
| HTTP 500 | 1 / 1 | 0 / 0 | 1.000 / 1.000 / 1.000 | 3.596 s | 3.599 s |
| Database connection failure | 1 / 1 | 0 / 0 | 1.000 / 1.000 / 1.000 | 4.511 s | 0.641 s |
| Service stopped | 1 / 1 | 0 / 0 | 1.000 / 1.000 / 1.000 | 4.470 s | 2.101 s |

## Conclusion

The local collection chain is operational for all three fault classes. It
captures model-ready evidence while keeping Ground Truth, raw telemetry and
rule detections separate. The database pilot also demonstrated why probe
timeouts must remain below the sampling interval: the initial four-second
timeout created a blind spot during a three-second fault, while the corrected
0.4-second timeout captured nine unavailable observations.

The three approved raw packages were subsequently converted into incident
schema v1. The resulting dataset contains three records, one per fault class,
with 533 Docker-log rows, 213 probe rows and 45 common metric features. All
Ground Truth, telemetry-count, collector-error, split-group and detection-pair
quality checks passed. Its top-level fields match the RCAEval incident schema,
while its task and local labels remain separate.

The next task is the production campaign of at least 20 incidents per fault
class with one frozen configuration.
