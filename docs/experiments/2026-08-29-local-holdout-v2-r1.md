# Frozen-model local holdout v2-r1 - 29 August 2026

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

## Outcome

This run exposed a real robustness signal, but it is **not accepted as the
confirmatory holdout** because three incidents violated collection conditions.
The original artifacts are retained and must not be rewritten.

The v1 pipelines were frozen before v2 collection. Their artifact hash is
`7e228631ed1465a35a5aec1bcfbf82e2d0ed41030f98744a54006d1c0828f15b`.
No fit, feature selection or tuning was performed on v2.

## Prespecified condition change

| Setting | v1 training | v2-r1 test attempt |
|---|---:|---:|
| Incidents | 60 (20/class) | 30 (10/class) |
| Workload sleep after each request | 1 s | 0.25 s |
| Target fault duration | 8 s | 12 s |
| Pre/post capture | 10/10 s | 10/10 s |
| Probe interval / timeout | 1/0.8 s | 1/0.8 s |

The workload container used the same image. Its interval was restored to one
second after collection, HTTP-500 injection was off, and the post-run database
health check passed.

## Data integrity

The converter produced 30 balanced incidents, 18,799 Docker-log records, 2,790
probe records and the same 45 metric features as v1. Its six existing quality
checks passed: unique IDs and split groups, complete ground-truth pairs,
telemetry-count agreement, no collector errors and complete detection archives.

Those checks were insufficient. A later timing audit found:

- `INC-53BB2946394C`: service stop lasted 1,767.29 seconds;
- `INC-C2A5320B0F76`: database interruption lasted 2,751.68 seconds;
- `INC-D9BF00C55652`: nominal fault duration was 13.11 seconds, but mean
  post-window liveness latency was 346,696 ms.

The target was 12 seconds and request timeout was 0.8 seconds. Host
suspension/scheduling affected collection, so the complete set is not a valid
fixed-condition test set.

The campaign runner now supports maximum actual fault duration and maximum
derived latency acceptance criteria, allowing future campaigns to fail before
publishing an incident dataset.

## Frozen-model results (exploratory only)

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logs only | 1.000 | 1.000 | 1.000 | 1.000 |
| Logs + metrics | 0.667 | 0.500 | 0.667 | 0.556 |

Logs-only classified all 30 records correctly. The combined model classified
HTTP-500 and service-stop records correctly, but classified all ten database
failures as service stoppage. Its mean predicted-class probability rounded to
1.000 despite those errors, so it is not a reliability guarantee.

The invalid incidents do not explain the whole pattern. After excluding them,
all eight remaining database records are still classified as service stoppage.
On that post-hoc, imbalanced 27-record subset, logs-only Macro F1 remains 1.000
and combined Macro F1 is approximately 0.564. This subset is exploratory and
does not replace a confirmatory test.

## Post-hoc distribution audit

For the seven database records unaffected by duration or latency violations,
mean five-second pre-window request count rose from 4.94 in v1 to about 19.45
in v2-r1. Mean post-window database-health latency rose from about 341 ms to
about 434 ms.

This is consistent with the metric branch extrapolating under workload shift
and overpowering useful log evidence. It is not causal attribution because
workload and fault duration changed together. Adding a modality can reduce
robustness even when it raised confidence on the original campaign.

## Rule detector

The raw evaluation reports perfect event/classification counts, but its mean
detection latency of 68.769 seconds is contaminated by timing violations. After
excluding the three invalid incidents, the exploratory subset has mean
detection latency 3.503 seconds, median 4.135 seconds and mean recovery
detection latency 1.750 seconds. These online latencies are not directly
comparable with retrospective ML classification over full incident windows.

## Interpretation and next experiment

The next confirmatory campaign should:

1. keep both frozen v1 models unchanged;
2. change only workload or only duration;
3. enforce duration and probe-latency limits before conversion;
4. run in smaller validated blocks to reduce host-suspension damage;
5. disclose v2-r1 and treat the next run as prospective confirmation, not an
   untouched first test.

Deep learning is still not the next step. The immediate problem is
covariate-shift robustness and trustworthy data collection, not model capacity.

## Artifacts

- Protocol: `data/results/local-campaign-v2-r1-protocol.json`
- Processed data: `data/processed/incidents/local-campaign-v2-r1.jsonl`
- Dataset manifest: `data/processed/incidents/local-campaign-v2-r1-manifest.json`
- Frozen result: `data/results/local-frozen-holdout-v2-r1.json`
- Figure: `data/results/figures/local-frozen-holdout-v2-r1.png`
- Failed first attempt: `data/results/local-campaign-v2-protocol.json`
