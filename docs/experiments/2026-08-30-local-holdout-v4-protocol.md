# Local holdout v4 confirmation protocol

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

Date registered: 2026-08-30

## Purpose

v4 is the first dataset eligible to confirm the frozen
`gated-robust-fusion-v2` candidate. The model, engineered features, classifier,
OOD definition and strict 0.10 routing threshold were frozen before any v4
telemetry was collected.

## Fixed collection design

- Same local order platform and three known fault classes.
- Ten sequential blocks, each containing one database-connection failure, one
  controlled HTTP 500 failure and one service stop.
- Thirty incidents in total, ten per class.
- Fault duration fixed at 8 seconds and cooldown fixed at 8 seconds.
- Workload interval changed from the one-second baseline to 0.25 seconds.
- Telemetry pre-window 10 seconds, post-window 10 seconds, one-second probe
  interval and 0.8-second request timeout.
- Each block receives a unique seed, run ID, incident IDs, split groups and raw
  telemetry files.

The master protocol is
`experiment/configs/local-campaign-v4-blocks.json`. Prepared block
configurations are immutable and include the master and orchestrator hashes.

## Acceptance gates

Every block must independently pass before the next block can run:

- exactly three completed incidents and one of each fault class;
- maximum realised fault duration no greater than 20 seconds;
- maximum metric latency no greater than 1,000 ms;
- complete recovery and detection pairs with no false-positive detections;
- workload image unchanged and interval restored to one second;
- post-run database health and disabled HTTP-500 injection;
- candidate model hash verified before and after collection and unchanged.

Final aggregation rechecks balance, durations, metric latency, raw provenance,
model-source/runtime/specification hashes, training/test disjointness and that
all telemetry began after the candidate freeze time.

## Locked analysis

The frozen candidate is evaluated exactly once after all ten blocks pass.
There is no v4 fit, threshold selection, feature selection, calibration or
model choice. Mandatory outputs are accuracy, Macro F1, per-class F1, fusion
coverage, fallback rate, OOD-fraction distribution and the logs-only
comparator. The rule detector remains an operational baseline.

If any model or routing rule is changed after v4 labels are inspected, the
changed model becomes a new candidate and requires another independent
confirmation campaign.
