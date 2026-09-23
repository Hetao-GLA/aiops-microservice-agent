# Documentation and evidence index

[Project overview](../README.md) · [Run the demo](GETTING_STARTED.md) ·
[Data and reproduction](REPRODUCIBILITY.md)

## How to read this archive

The dated reports below record the research sequence, including failed
candidates, rejected runs and limitations. Their results have not been rerun
as part of publishing this repository. Historical "next steps" are not the
current installation instructions: start with the demo guide above.

Artifact paths and hashes identify the original research files. Raw telemetry,
processed incidents, models, generated JSON results and figures are not
included in this source-only checkout. Read the reproduction guide before
executing archived commands. Dissertation drafts, supervisor materials and
submission files are deliberately not published here.

## Recommended reading

1. [Local ML baseline](experiments/2026-08-28-local-ml-baseline-v1.md): the initial 60-incident evaluation.
2. [Workload-shifted v3 confirmation](experiments/2026-08-30-local-holdout-v3.md) and [contribution audit](experiments/2026-08-30-frozen-contribution-audit-v1-to-v3.md): why naive fusion failed.
3. [Gated v4 confirmation](experiments/2026-08-30-local-holdout-v4-confirmation.md): selective fallback and its limitations.
4. [Public shift audit](experiments/2026-08-30-rcaeval-re2-ob-shift-audit-v1.md): the previously missing README link.
5. [Failed Sock Shop confirmation](experiments/2026-08-30-rcaeval-re2-ss-confirmation-v1.md), [candidate v2 development](experiments/2026-08-30-rcaeval-service-delta-v2-development.md), and [locked Train Ticket confirmation](experiments/2026-08-30-rcaeval-re2-tt-service-delta-v2-confirmation.md): the public candidate's development and confirmation.

## Local platform and initial campaigns

- [First database-failure baseline](experiments/2026-08-10-database-failure-baseline.md)
- [Database-failure batch](experiments/2026-08-10-database-failure-batch.md)
- [Service-stop batch](experiments/2026-08-10-service-stop-batch.md)
- [HTTP-500 batch](experiments/2026-08-10-http-500-batch.md)
- [Mixed faults and normal operation](experiments/2026-08-10-mixed-and-normal-baseline.md)
- [Telemetry smoke experiment](experiments/2026-08-24-local-telemetry-smoke.md)
- [Accepted 60-incident v1 campaign](experiments/2026-08-25-local-campaign-v1.md)
- [Traditional ML baseline](experiments/2026-08-28-local-ml-baseline-v1.md)

## Local shift, gating and diagnostic analyses

- [Initial holdout protocol](experiments/2026-08-28-holdout-v2-protocol.md)
- [v2-r1 result, not accepted as confirmation](experiments/2026-08-29-local-holdout-v2-r1.md)
- [Accepted v3 confirmation](experiments/2026-08-30-local-holdout-v3.md)
- [Frozen v1-to-v3 contribution audit](experiments/2026-08-30-frozen-contribution-audit-v1-to-v3.md)
- [Robust-fusion v2 specification](experiments/2026-08-30-robust-fusion-v2-spec.md)
- [Gated candidate development](experiments/2026-08-30-gated-robust-fusion-v2-development.md)
- [v4 protocol](experiments/2026-08-30-local-holdout-v4-protocol.md) and [v4 confirmation](experiments/2026-08-30-local-holdout-v4-confirmation.md)
- [Early-diagnosis protocol](experiments/2026-08-30-early-diagnosis-v4-stress-protocol.md) and [results](experiments/2026-08-30-early-diagnosis-v4-stress-test.md)
- [Log-evidence ablation protocol](experiments/2026-08-30-log-evidence-ablation-v4-1s-protocol.md) and [results](experiments/2026-08-30-log-evidence-ablation-v4-1s.md)

## Public service-localisation studies

The local task predicts one of three fault classes. These public studies
instead rank known root-cause services. They are separate evaluations.
OB means Online Boutique, SS means Sock Shop and TT means Train Ticket.

- [RE2-OB baseline protocol](experiments/2026-08-30-rcaeval-re2-ob-public-baseline-protocol.md) and [results](experiments/2026-08-30-rcaeval-re2-ob-public-baseline-v1.md)
- [RE2-OB shift-audit protocol](experiments/2026-08-30-rcaeval-re2-ob-shift-audit-protocol.md) and [results](experiments/2026-08-30-rcaeval-re2-ob-shift-audit-v1.md)
- [RE2-OB delta-candidate protocol](experiments/2026-08-30-rcaeval-re2-ob-delta-candidates-protocol.md) and [results](experiments/2026-08-30-rcaeval-re2-ob-delta-candidates-v1.md)
- [RE2-SS confirmation protocol](experiments/2026-08-30-rcaeval-re2-ss-confirmation-protocol.md) and [failed v1 confirmation](experiments/2026-08-30-rcaeval-re2-ss-confirmation-v1.md)
- [Service-delta v2 development protocol](experiments/2026-08-30-rcaeval-service-delta-v2-development-protocol.md) and [results](experiments/2026-08-30-rcaeval-service-delta-v2-development.md)
- [RE2-TT locked confirmation protocol](experiments/2026-08-30-rcaeval-re2-tt-service-delta-v2-confirmation-protocol.md) and [results](experiments/2026-08-30-rcaeval-re2-tt-service-delta-v2-confirmation.md)

## Data contracts and Agent design

- [Public dataset compatibility audit](data/2026-08-20-public-dataset-compatibility.md)
- [Incident schema v1](data/incident-schema-v1.md)
- [Local telemetry collection](data/local-telemetry-collection.md)
- [Historical Agent MVP design](agent/2026-08-30-agent-mvp.md)

For current UI labels and a controlled recovery demonstration, use
[Getting started](GETTING_STARTED.md), not the dated Agent walkthrough.
