# Local holdout v4: gated robust-fusion confirmation

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

Date: 2026-08-30

## Decision

The frozen `gated-robust-fusion-v2` candidate passed its preregistered v4
accuracy confirmation under a same-system, known-fault, workload-only shift.
It classified all 30 incidents correctly, with Accuracy and Macro F1 of 1.000.

The result confirms the safety value of the OOD gate, but it does not establish
an accuracy benefit from metric fusion. Only five incidents used the fusion
branch; 25 used the logs-only fallback. The candidate should therefore be
described as a confirmed guarded architecture, not as a confirmed superior
multimodal classifier.

## Prospective design

The candidate was frozen at `2026-08-30T06:02:07.592959+00:00`. The v4 master
protocol and all model rules were fixed before v4 telemetry began at
`2026-08-30T06:13:02+00:00`.

The campaign used ten sequential blocks. Each independently collected and
accepted one database-connection failure, one controlled HTTP 500 failure and
one service stop. The fault duration stayed at 8 seconds while only the
workload interval changed from 1.0 to 0.25 seconds. No block could start until
all earlier blocks passed their acceptance gates.

The locked candidate was evaluated once after all blocks passed. There was no
v4 fitting, threshold selection, feature selection, calibration or model
choice. The source, runtime, specification and model hashes were verified
before deserialisation, and each block recorded the same candidate hash before
and after collection.

## Data and collection quality

| Check | Result | Acceptance |
|---|---:|---:|
| Accepted blocks | 10/10 | 10/10 |
| Total incidents | 30 | 30 |
| Incidents per class | 10 | 10 |
| Maximum realised fault duration | 10.131 s | ≤20 s |
| Maximum metric latency | 400.085 ms | ≤1,000 ms |
| Complete detection pairs | 30/30 | 30/30 |
| False-positive detections | 0 | 0 |
| Recovery failures | 0 | 0 |
| Candidate artifact changed | No | No |
| Workload restored after every block | Yes | Yes |
| Final database health | Healthy | Healthy |
| Final HTTP-500 injection | Disabled | Disabled |

All incident IDs, split groups, campaign run IDs and raw-telemetry hashes were
disjoint from v1 training. Every v4 incident began after candidate freeze.

## Primary frozen result

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Gated robust-fusion v2 | 1.000 | 1.000 | 1.000 | 1.000 |
| Internal logs-only comparator | 1.000 | 1.000 | 1.000 | 1.000 |

For the gated candidate, every class had precision, recall and F1 of 1.000 on
ten incidents.

## OOD routing

The locked gate used robust fusion only when no more than 10% of the 36
engineered metrics fell outside their v1 training ranges.

| Actual class | Records | Fusion | Logs fallback | Mean OOD fraction | OOD range |
|---|---:|---:|---:|---:|---:|
| Database connection failure | 10 | 2 | 8 | 0.161 | 0.083–0.222 |
| HTTP 500 failure | 10 | 3 | 7 | 0.161 | 0.056–0.278 |
| Service stopped | 10 | 0 | 10 | 0.247 | 0.222–0.250 |
| **Total** | **30** | **5** | **25** | **0.190** | **0.056–0.278** |

Fusion coverage was 0.1667 and fallback rate was 0.8333. This is lower than
the post-hoc v3 diagnostic coverage of 0.3333, so the low coverage replicated
rather than disappearing on new data.

## Secondary no-fit comparisons

These comparisons were calculated only after the primary frozen v4 result and
did not affect the candidate or decision threshold.

| Frozen prediction configuration | Macro F1 | Database recall | Interpretation |
|---|---:|---:|---|
| Gated robust-fusion v2 | 1.0000 | 1.000 | Gate plus log fallback |
| Robust fusion branch without routing | 0.8222 | 0.500 | Five database failures mapped to service stop |
| Original v1 logs-plus-metrics model | 0.5556 | 0.000 | All database failures mapped to service stop |
| Original frozen logs-only model | 1.0000 | 1.000 | Stable strong operational signatures |

Removing absolute volume, clipping and robust scaling improved the ungated
fusion branch relative to the original combined model, but did not eliminate
workload sensitivity. The OOD gate prevented those remaining metric-driven
errors from reaching the final output.

## Rule baseline

The operational rule detector matched and correctly classified all 30
incidents with no misses or false positives. Event precision, recall, F1 and
classification accuracy were all 1.000. Mean detection latency was 2.798
seconds, median detection latency 3.277 seconds and mean recovery-detection
latency 1.912 seconds. Four timestamp-boundary adjustments were recorded by
the established evaluator.

## Interpretation and research claim

Supported claim:

> Under a four-times-higher request workload on the same local system and
> three known fault classes, a training-range OOD gate with a logs-only
> fallback prevented the workload-sensitive metric branch from degrading
> retrospective incident classification across 30 new, post-freeze incidents.

Unsupported claims:

- Metrics improved accuracy over logs alone.
- The fusion classifier itself generalised reliably under workload shift.
- The result generalises to new systems, unknown faults or online early
  diagnosis.
- Perfect performance will persist when strong recovery and fault-signature
  logs are absent.

The candidate is useful as a fail-safe design: shifted metrics may contribute
when they remain in range, but they are prevented from overruling stable log
evidence when range shift is detected. Its main limitation is that this safety
behaviour activates on most shifted incidents, making the deployed behaviour
largely logs-only.

## Artifacts and hashes

- Candidate model SHA-256:
  `336ef2a404bcd0db3e783638081a6b541e153f79b3cd7276b56423d5b7d99c85`
- v4 master SHA-256:
  `2d847ac1d30a5279a84665d52a4b8fdea2b7af606c78ac630622906b0c802acb`
- Processed v4 data:
  `data/processed/incidents/local-campaign-v4.jsonl`
- Processed data SHA-256:
  `656e50f714aded0e772c6fb23c2a0100a3bc1d48a14bc8a1ae7653822ed5165e`
- Processed manifest SHA-256:
  `0d1aa64da08efc5e69c51acc13caed83418056f0104fb39bbd1f588fe9765bc1`
- Campaign summary:
  `data/results/local-campaign-v4-summary.json`
- Campaign summary SHA-256:
  `14e6cef7fa15525496580c9f8b75405a4306e069707e279b66be0509aa31901e`
- Frozen model result:
  `data/results/gated-robust-fusion-v2-v4.json`
- Model result SHA-256:
  `f7fe92b88e0922c75ec38f864c5500a97fbacaf6ec33dc8e05ba97d1639df595`
- Figure:
  `data/results/figures/gated-robust-fusion-v2-v4.png`
- Figure SHA-256:
  `3e330a8149dd3314224c58699c29aae531f450486ff752f8f2fcea5f351d1948`

## Next research step

Do not tune this frozen candidate on v4 and continue calling the result
independent. Two defensible paths remain:

1. retain v2 as the confirmed safety model and move the thesis evaluation to
   harder early-window, weak-signature and public-data tasks; or
2. use v4 only as development evidence for a new candidate designed to improve
   fusion coverage, freeze that new candidate, and require a fresh v5
   confirmation campaign.

The first path is lower risk for the current dissertation because the current
evidence already supports a clear, bounded result: metric fusion can fail under
load shift, and explicit OOD fallback can contain that failure.
