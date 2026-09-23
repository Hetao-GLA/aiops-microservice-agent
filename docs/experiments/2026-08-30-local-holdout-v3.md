# Frozen-model workload-only holdout v3 - 30 August 2026

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

## Purpose

Test whether the two v1 incident classifiers remain reliable when only the
workload interval changes. This is a prospective confirmation of the robustness
pattern observed in the invalid v2-r1 attempt; it is not presented as an
untouched first external test.

## Locked design

The models were fitted on the 60-incident v1 campaign and frozen before any
holdout collection. Their artifact SHA-256 is
`7e228631ed1465a35a5aec1bcfbf82e2d0ed41030f98744a54006d1c0828f15b`.
No fitting, vocabulary construction, scaling, feature selection or
hyperparameter tuning used v3.

| Setting | v1 training | v3 confirmation |
|---|---:|---:|
| Incidents | 60 (20/class) | 30 (10/class) |
| Workload sleep after each request | 1 s | 0.25 s |
| Fault duration | 8 s | 8 s |
| Pre/post capture | 10/10 s | 10/10 s |
| Probe interval / timeout | 1/0.8 s | 1/0.8 s |
| Metric schema | 45 features | Same 45 features |

The 30 incidents were collected as ten independently accepted blocks. Each
block contained one database failure, one HTTP-500 failure and one service
stop in a seeded order. The workload was restored to one second after every
block. Model prediction was deferred until all ten blocks passed.

## Quality gates

Every block had to satisfy all of the following before the next block could
start:

- complete fault start/end and recovery;
- one complete detector start/end pair per incident and no false positives;
- actual fault duration at most 20 seconds;
- every derived latency feature at most 1,000 ms;
- unique incident IDs and split groups;
- telemetry counts matching metadata, no collector errors and complete
  run-specific detection archives;
- unchanged configuration and orchestration hashes;
- restored workload image/interval and successful post-run database health.

All ten blocks passed. Maximum observed fault duration was 10.174 seconds and
maximum derived metric latency was 395.819 ms. This directly addresses the
host-suspension contamination found in v2-r1.

## Dataset

The accepted aggregate contains:

- 30 incidents, balanced 10/10/10 across the three known fault classes;
- 17,659 Docker-log records and 2,568 probe records;
- 45 metric features in every model-compatible record;
- dataset SHA-256
  `c6383041fea9e3e5ab4f55bf69d91152bf21607164b7ac4b1f3832cd50f23800`;
- ten distinct telemetry run IDs and 30 unique split groups.

Leakage sanitisation removed 2,127 rule-detector rows and 20 control-plane
toggle rows, stripped 317 injected `fault_type` fields and retained 15,512
operational log rows. All records remained non-empty. Exact target labels were
absent except for the genuine `service_stopped` lifecycle message.

## Frozen-model results

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Errors |
|---|---:|---:|---:|---:|---:|
| Logs only | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| Logs + metrics | 0.667 | 0.500 | 0.667 | 0.556 | 10 |

The Macro-F1 difference, combined minus logs-only, is `-0.444444`. The
logs-only model classified every incident correctly. The combined model
classified every database failure as service stoppage while correctly
classifying the other two classes.

Mean predicted-class probability was about 0.89 for logs-only. For the combined
model it rounded to 1.00, with a minimum of 0.99, despite ten errors. These
uncalibrated probabilities are therefore unsafe as confidence guarantees.

The confusion matrices are stored in
`data/results/figures/local-frozen-holdout-v3.png`.

## Workload-shift evidence

For database-failure incidents, the mean five-second pre-window request count
rose from 4.94 in v1 to 19.34 in v3. Mean post-window database-health latency
rose from 340.87 ms to 369.63 ms, and mean post-window liveness latency rose
from 4.08 ms to 14.60 ms.

The repeated database-to-service-stop error is therefore consistent with the
metric branch extrapolating beyond its training distribution and dominating
useful log evidence. This is an inference from the observed covariate shift,
not yet a causal feature-contribution decomposition.

Because fault duration was held at eight seconds, the v3 result isolates the
workload condition more cleanly than v2-r1. The v2-r1 timing violations are not
present in v3 and cannot explain this repeated failure.

## Rule detector

The frozen online rule detector matched and correctly labelled all 30
incidents:

| Metric | Result |
|---|---:|
| Event precision / recall / F1 | 1.000 / 1.000 / 1.000 |
| Classification accuracy | 1.000 |
| Mean detection latency | 2.670 s |
| Median detection latency | 2.867 s |
| Mean recovery-detection latency | 2.068 s |

Five small negative boundary differences were adjusted under the previously
defined 0.5-second timestamp tolerance. Rule latency and retrospective
full-window ML classification remain different tasks and should not be used
for a direct speed comparison.

## Interpretation against the hypotheses

- H1, learned diagnosis outperforming rules, is not supported. Logs-only ties
  the rule baseline on classification, while the combined model is worse.
- H2, adding metrics improving diagnosis, is contradicted under the workload
  shift: Macro F1 falls by 0.444.
- H3, an Agent workflow reducing MTTR, is not tested by this campaign.

The negative H2 result is scientifically useful. It shows that multimodal input
is not automatically more robust and that confidence can increase while
accuracy decreases.

## Limitations

- One local system and three known injected faults; no cross-system or
  unknown-fault generalisation.
- Ten examples per class remain a small test set.
- Complete incident windows include recovery evidence; this is retrospective
  classification, not early online diagnosis.
- The workload interval is sleep after a completed request, not exact
  throughput.
- v3 used blockwise workload-container recreation and warm-up to enforce quality
  gates, whereas v1 was one continuous campaign. This is a procedural nuisance
  variable even though workload interval was the only target system condition
  deliberately changed.
- v2-r1 revealed the pattern first, so v3 is prospective confirmation rather
  than a blind initial discovery.

## Next step

Do not tune a replacement model and report it on v3 as if v3 were still
untouched. First perform a frozen-model contribution and out-of-range feature
audit to explain the failure. Any redesigned scaling, clipping, modality
gating or confidence-rejection mechanism becomes a new model and requires a
separate v4 confirmation set.

After the robustness mechanism is understood, the software phase should add
the Agent diagnosis/runbook/approval workflow. Deep learning is still not
justified by the current data volume or research bottleneck.

The completed
[frozen contribution audit](2026-08-30-frozen-contribution-audit-v1-to-v3.md)
shows that the log branch remains correct, while load-dependent absolute-count
features drive a much larger metric margin toward service stoppage.

## Artifacts

- Master protocol: `experiment/configs/local-campaign-v3-blocks.json`
- Campaign summary: `data/results/local-campaign-v3-summary.json`
- Processed data: `data/processed/incidents/local-campaign-v3.jsonl`
- Dataset manifest: `data/processed/incidents/local-campaign-v3-manifest.json`
- Frozen-model result: `data/results/local-frozen-holdout-v3.json`
- Rule evaluation: `data/results/evaluation-local-campaign-v3.json`
- Confusion matrices: `data/results/figures/local-frozen-holdout-v3.png`
