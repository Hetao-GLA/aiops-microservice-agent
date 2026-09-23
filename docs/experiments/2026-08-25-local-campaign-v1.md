# Frozen local fault campaign v1 - 25 August 2026

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

## Purpose

Produce the first balanced, incident-level local dataset for the traditional
machine-learning baseline. This campaign supersedes the three-case telemetry
smoke test: its configuration was fixed before execution, all three controlled
fault classes were interleaved in a seeded order, and Ground Truth, rule
detections and raw telemetry were kept as separate evidence channels.

## Frozen configuration

| Setting | Value |
|---|---:|
| Campaign | `local-campaign-v1` |
| Configuration SHA-256 | `13a252d62b27c1cb2e79d2278aefe8a8cfec0b87edf50a91d364b0726ac92496` |
| Random seed | `20260825` |
| Repetitions per fault | 20 |
| Total incidents | 60 |
| Fault duration | 8 s |
| Cooldown | 8 s |
| Telemetry pre / post window | 10 s / 10 s |
| Probe interval / request timeout | 1 s / 0.8 s |

The run started at `2026-08-25T09:43:56Z` and ended at
`2026-08-25T10:22:47Z`, a wall-clock duration of approximately 38 minutes 51
seconds. The telemetry run identifier is
`mixed-20260825T094356Z-c2ba451f`.

## Dataset result

| Fault label | Incidents |
|---|---:|
| `database_connection_failure` | 20 |
| `http_500_failure` | 20 |
| `service_stopped` | 20 |
| **Total** | **60** |

Incident-schema conversion retained 19,729 Docker-log records and 4,965 probe
records. The union of metric-summary features contains 45 fields. The output is
one complete incident per JSONL row; its SHA-256 is
`dd45cf4378e01caa5f9d3b698bba07870dad9f8d7306a1bedbe56ddf6d75ace6`.

All six conversion checks passed:

- unique incident identifiers;
- unique split groups;
- complete Ground Truth start/end pairs;
- telemetry counts matching the raw metadata;
- no collector errors;
- complete rule-detection start/recovery pairs.

## Frozen rule-baseline result

Evaluation used only the run-specific detection archive, which contains 60
failure detections and 60 recovery detections. Historical events in the shared
detector file were excluded.

| Fault | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Database connection failure | 20 | 1.000 | 1.000 | 1.000 |
| HTTP 500 failure | 20 | 1.000 | 1.000 | 1.000 |
| Service stopped | 20 | 1.000 | 1.000 | 1.000 |
| **Overall / macro** | **60** | **1.000** | **1.000** | **1.000** |

There were 60 matched and correctly classified incidents, zero missed
incidents and zero false-positive detections. Mean detection latency was 3.057
seconds (median 3.167 seconds), and mean recovery-detection latency was 2.300
seconds. Matching used the predeclared early/late tolerances of 5 and 15
seconds; six matches required the evaluator's 0.5-second timing-boundary
adjustment. One service-stop detection preceded its Ground Truth timestamp by
0.677 seconds because detector polling and the injection-side Ground Truth
write are asynchronous. It remained within the predeclared early tolerance and
is retained as a negative latency rather than silently changed.

## Interpretation and limitations

The perfect rule score is a valid result for this deliberately closed,
controlled three-fault environment. It is not evidence that the detector will
generalise to unseen services, fault variants or operational noise: its rules
were designed around these known failure signatures. It also creates a ceiling
for the current hypothesis that a learned classifier will outperform rules on
macro F1. A learned method can match, but cannot exceed, 1.000 on this exact
test condition.

The campaign supplies enough balanced observations for a first traditional-ML
experiment, but 60 incidents are not strong support for training a deep model.
The next comparison should therefore use an incident-grouped, stratified
cross-validation pipeline for TF-IDF plus logistic regression. Logs-only and
logs-plus-metrics variants must use identical folds, and all vocabulary,
scaling and feature selection must be fitted inside each training fold. Public
RCAEval labels remain a separate root-cause-localisation task and must not be
silently merged with these local classes.

## Reproduction

The completed v1 artifacts are immutable. To reproduce the method, copy the
configuration, assign a new campaign identifier and unused output paths, then
validate that new configuration without causing faults:

```powershell
.\.venv\Scripts\python.exe -m experiment.campaign_runner `
    --config experiment\configs\local-campaign-v2.json `
    --dry-run
```

Remove `--dry-run` only after the plan and preflight output have been reviewed.
The runner intentionally refuses to overwrite the completed v1 artifacts.

The principal machine-readable results are:

- `data/results/local-campaign-v1-summary.json`;
- `data/results/evaluation-local-campaign-v1.json`;
- `data/processed/incidents/local-campaign-v1-manifest.json`;
- `data/processed/incidents/local-campaign-v1.jsonl`.
