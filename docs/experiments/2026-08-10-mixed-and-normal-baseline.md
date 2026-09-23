# Mixed-fault and Normal-operation Baseline

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

## Purpose

This experiment tests the three-class rule baseline in a reproducibly shuffled
mixed batch rather than in isolated single-class batches. A separate normal
operation period provides an initial false-positive observation.

## Normal-operation observation

- Duration: 30 seconds
- Workload: one simulated order request per second
- Health samples: 30
- Healthy samples: 30
- Unhealthy samples: 0
- Detection incidents: 0
- Observed false positives: 0
- Result: passed

This short observation verifies the pipeline but is not long enough to estimate
a production false-positive rate.

## Mixed-batch configuration

- Random seed: `20260810`
- Repetitions per class: 2
- Total incidents: 6
- Fault duration: 8 seconds
- Cooldown: 8 seconds
- Detector: `operations-health-rule-v2`
- Ground-truth timing:
  - container faults: Docker `State.FinishedAt`;
  - application fault: application `changed_at`.

The generated schedule was:

1. `service_stop`
2. `database_disconnect`
3. `http_500`
4. `http_500`
5. `database_disconnect`
6. `service_stop`

## Aggregate results

| Metric | Result |
|---|---:|
| Ground-truth incidents | 6 |
| Detection incidents | 6 |
| Correct classifications | 6 |
| Missed incidents | 0 |
| False-positive detections | 0 |
| Event precision | 1.0000 |
| Event recall | 1.0000 |
| Event F1 | 1.0000 |
| Classification accuracy | 1.0000 |
| Mean detection latency | 2.431 s |
| Median detection latency | 2.764 s |
| Mean recovery-detection latency | 1.815 s |

## Per-class results

| Fault class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `database_connection_failure` | 2 | 1.0000 | 1.0000 | 1.0000 |
| `http_500_failure` | 2 | 1.0000 | 1.0000 | 1.0000 |
| `service_stopped` | 2 | 1.0000 | 1.0000 | 1.0000 |

## Confusion matrix

| Expected / predicted | Database failure | HTTP 500 | Service stopped | Missed |
|---|---:|---:|---:|---:|
| Database failure | 2 | 0 | 0 | 0 |
| HTTP 500 | 0 | 2 | 0 | 0 |
| Service stopped | 0 | 0 | 2 | 0 |

## Timing-boundary treatment

One service-stop detection timestamp preceded Docker `State.FinishedAt` by
0.015 seconds. A service port can become unreachable immediately before the
container process records its final exit timestamp. The evaluator preserves
the raw value and, for aggregation, maps negative values within a documented
0.5-second boundary tolerance to zero. One timing value was adjusted in this
batch. Larger negative values would remain visible as timing anomalies.

## Interpretation

The detector correctly separated all three failure classes in a mixed order.
This confirms that probe precedence works as intended:

```text
liveness unavailable -> service_stopped
liveness healthy + database unhealthy -> database_connection_failure
liveness and database healthy + HTTP 500 threshold -> http_500_failure
```

These results demonstrate a functioning experimental baseline, not evidence
that the rules generalise beyond the controlled scenarios.

## Limitations and next steps

- Two incidents per class are insufficient for statistical inference.
- The 30-second normal period provides only an initial false-positive check.
- Fault durations and workload are currently fixed.
- The rule detector uses explicit health endpoints and a deliberately chosen
  application threshold.
- The rule configuration should now be frozen before collecting a larger
  baseline dataset.
- Next, add structured dataset export at the incident level and begin the
  traditional machine-learning baseline.
