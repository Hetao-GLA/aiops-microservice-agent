# Database Failure Batch Baseline

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

## Purpose

This exploratory batch validates the automated experiment runner, independent
ground truth, rule-based detector, and evaluator before collecting the larger
dataset required for the final study.

## Configuration

- Date: 10 August 2026
- Trials: 3
- Fault class: `database_connection_failure`
- Fault implementation: stop the PostgreSQL container
- Fault duration: 8 seconds
- Cooldown between trials: 4 seconds
- Workload: one simulated order request per second
- Detector: `database-health-rule-v1`
- Detector interval: 1 second
- Detector HTTP timeout: 4 seconds
- Ground-truth timing source: Docker `State.FinishedAt`
- Ground-truth file: `batch-20260810-02.jsonl`
- Detection file: `batch-20260810-02.jsonl`

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
| Mean detection latency | 3.677 s |
| Median detection latency | 4.079 s |
| Mean recovery-detection latency | 1.343 s |

## Trial-level results

| Ground-truth incident | Detection incident | Detection latency | Recovery latency | Correct class |
|---|---|---:|---:|:---:|
| `INC-3E38FE00C18E` | `DET-205EAF007606` | 4.079 s | 0.500 s | Yes |
| `INC-EC990BD4D37A` | `DET-BEF760313EDC` | 2.076 s | 2.288 s | Yes |
| `INC-1A49BA5C611B` | `DET-DC0E4E40543F` | 4.877 s | 1.241 s | Yes |

## Timing correction

An earlier pilot batch produced one negative latency because the ground-truth
start event was timestamped after the blocking `docker compose stop` command
returned. The detector could observe the failure while that command was still
running. The injector was corrected to use the affected container's Docker
`State.FinishedAt` timestamp. This batch contains the corrected, non-negative
timing measurements.

## Interpretation

The rule baseline reliably detected this single, clearly observable fault in
the three exploratory repetitions. Detection latency varies because a failed
health request can remain open until the detector's HTTP timeout. The current
result establishes a reproducible baseline configuration; it does not yet show
that the detector will perform equally well for other faults or workloads.

## Limitations and next steps

- Three trials are sufficient for pipeline validation but not statistical
  inference.
- Only one fault class has been evaluated.
- The current rule predicts database unavailability, not every possible
  low-level root cause.
- A no-fault observation period is required for a stronger false-positive
  estimate.
- The next implementation target is the service/container-stop fault, followed
  by HTTP 500 injection.
