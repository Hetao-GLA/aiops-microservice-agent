# Service Stop Batch Baseline

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

## Purpose

This exploratory batch evaluates whether the upgraded rule detector can
distinguish an unavailable API service from a database connection failure.

## Configuration

- Date: 10 August 2026
- Trials: 3
- Fault class: `service_stopped`
- Root cause: `api_container_stopped`
- Fault implementation: stop the `api` Docker Compose service
- Fault duration: 8 seconds
- Cooldown between trials: 4 seconds
- Workload: one simulated order request per second
- Detector: `operations-health-rule-v2`
- Probe order: service liveness, followed by database health
- Detector interval: 1 second
- Detector HTTP timeout: 4 seconds
- Ground-truth timing source: Docker `State.FinishedAt`

## Aggregate results

| Metric | Result |
|---|---:|
| Ground-truth incidents | 3 |
| Detection incidents | 3 |
| Correctly classified | 3 |
| Missed incidents | 0 |
| False-positive detections | 0 |
| Event precision | 1.0000 |
| Event recall | 1.0000 |
| Event F1 | 1.0000 |
| Classification accuracy | 1.0000 |
| Mean detection latency | 2.814 s |
| Median detection latency | 4.132 s |
| Mean recovery-detection latency | 2.577 s |

## Trial-level results

| Ground-truth incident | Detection incident | Detection latency | Recovery latency | Predicted class |
|---|---|---:|---:|---|
| `INC-DD2DB38F70DB` | `DET-42A73D8024F6` | 4.132 s | 3.933 s | `service_stopped` |
| `INC-F4E28E74076B` | `DET-C0CEA3775D7A` | 0.109 s | 1.745 s | `service_stopped` |
| `INC-662181FA678E` | `DET-EE080FDC3A1E` | 4.201 s | 2.054 s | `service_stopped` |

## Interpretation

The detector correctly separated service unavailability from database
unavailability in all three exploratory trials. The latency variation reflects
two network behaviours: some liveness probes fail immediately when the
container is absent, while others remain open until the four-second client
timeout. The detector configuration must therefore remain fixed when comparing
later methods.

## Limitations and next steps

- Three trials validate the pipeline but do not support statistical inference.
- No-fault observation time is still needed for a stronger false-positive
  estimate.
- The rule relies on an explicit liveness endpoint and therefore represents a
  favourable baseline condition.
- The third fault class, application HTTP 500, should be implemented next.
- After all three classes exist, mixed-class batches should be randomised to
  test classification rather than isolated single-class detection.
