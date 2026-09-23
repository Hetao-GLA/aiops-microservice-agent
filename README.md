# AI-assisted Microservice Operations

This repository contains the experimental platform for the project:

> **An AI Agent-assisted Monitoring and Self-healing System for Microservice Operations**

The project investigates whether an AI-assisted approach using operational logs and metrics can identify microservice failures and their root causes more effectively than a traditional rule-based monitoring approach.

## Current status

The controlled Docker baseline is complete for three fault classes: database
connection failure, service stoppage and application HTTP 500. The frozen rule
baseline has been exercised in isolated and mixed batches, and the public-data
compatibility audit is complete.

The first production local-data campaign and traditional-ML baseline are
complete. The frozen campaign generated 60 incident-schema records (20 per
fault class), containing 19,729 Docker-log rows, 4,965 probe rows and 45 metric
features. All collection and conversion quality checks passed.

After excluding rule-detector and fault-control leakage, both the logs-only
TF-IDF logistic-regression model and the logs-plus-metrics model correctly
classified all 60 out-of-fold incidents in shared five-fold grouped
cross-validation. Both achieved Macro F1 1.000, so adding metrics produced no
measured F1 improvement in this ceiling-limited condition. The next phase is
an independent robustness campaign and external holdout test, not deep-model
training on these 60 records.

The first frozen-model holdout attempt is complete but not accepted as the
confirmatory test. Under a joint workload/duration shift, logs-only retained
Macro F1 1.000 while logs plus metrics fell to 0.556 and mapped all database
failures to service stoppage. Three incidents were affected by host
suspension/scheduling, so these scores are exploratory. The collector now
retries transient Windows directory locks, and the campaign runner can enforce
actual-duration and metric-latency limits before publishing a dataset. See the
[v2-r1 report](docs/experiments/2026-08-29-local-holdout-v2-r1.md).

The corrected workload-only v3 confirmation is now complete. Thirty balanced
incidents were collected as ten independently accepted three-incident blocks;
all duration, latency, recovery, detector-pair and provenance gates passed.
With the v1 pipelines kept frozen, logs-only retained Macro F1 1.000 while
logs plus metrics scored 0.556 and mapped every database failure to service
stoppage. This confirms workload-sensitive covariate shift in the metric branch
rather than a v2 timing artifact. See the
[v3 report](docs/experiments/2026-08-30-local-holdout-v3.md).

A no-fit
[frozen contribution audit](docs/experiments/2026-08-30-frozen-contribution-audit-v1-to-v3.md)
reconstructs the database-versus-service-stop margin exactly. The log branch
favours the correct database class on all ten v3 database incidents, but the
metric branch overwhelms it. Absolute pre-window request-count features shift
about 72 within-class standard deviations and are the largest wrong-class
contributors.

The gated robust-fusion v2 candidate is implemented, tested, frozen and has
now completed its independent v4 confirmation. It removes absolute
workload-volume metrics, derives request rates, clips and robustly scales
metrics using training-only statistics, and falls back to the unchanged
logs-only model when more than 10% of engineered metrics are outside their
training ranges. On 30 new post-freeze v4 incidents it achieved Accuracy and
Macro F1 1.000, but used fusion for only 5 incidents and the logs-only fallback
for 25. This confirms that the gate contains metric-shift failures; it does not
show that metrics improve over logs alone. See the
[v4 confirmation report](docs/experiments/2026-08-30-local-holdout-v4-confirmation.md)
and the earlier
[development report](docs/experiments/2026-08-30-gated-robust-fusion-v2-development.md).

A no-fit early-diagnosis stress test then rebuilt all 30 v4 incidents at
+1, +3, +5 and +8 seconds after injection, excluding recovery evidence. The
logs-only and gated models retained Macro F1 1.000 at every cutoff, including
+1 second, while the ungated fusion branch rose from 0.556 at +1 second to
0.933 at +8 seconds. This is early classification with a known event boundary,
not one-second incident detection. The gate caught every fusion error but used
the logs fallback for most incidents. See the
[early-diagnosis report](docs/experiments/2026-08-30-early-diagnosis-v4-stress-test.md).

A fixed +1-second evidence ablation found that the early text result depends on
combined application and workload context. All sources retained Macro F1 1.000
even after uniform direct-signature masking; application-only logs scored
0.822, workload-only logs 0.556 and masked workload-only logs 0.167. The
ungated fusion branch stayed at 0.556 for every log view, confirming dominant
early metric error. See the
[log-evidence ablation report](docs/experiments/2026-08-30-log-evidence-ablation-v4-1s.md).

Incident schema v1 keeps one complete incident per JSONL row, enforces grouped
train/test splitting and keeps RCAEval labels separate from the local
three-class label space. The six-case RCAEval pilot has now been expanded to
the complete 90-case RE2-OB public benchmark. In repetition-held-out service
localisation, logs-only scored Macro F1 0.532, metrics-only 0.785 and logs plus
metrics 0.775. In fault-type-held-out evaluation the corresponding scores were
0.537, 0.856 and 0.867. The public result therefore supports metrics but does
not show a consistent advantage for simple fusion. See the
[public baseline report](docs/experiments/2026-08-30-rcaeval-re2-ob-public-baseline-v1.md).

A frozen-prediction shift audit explains the two main public weaknesses.
Repetition 1 has a metric range-breach fraction of 0.051 versus 0.018 for the
other repetitions, with the shift concentrated in absolute pre-window latency
levels; it does not contain more unseen log templates. The unseen `delay`
fault has lower numeric drift but a much larger log-source-mixture distance and
only 0.667 metric-signal hit@1, consistent with latency propagation across
services. The next development candidate will therefore remove absolute metric
levels and test service-grouped deltas before considering deep learning. See
the [shift audit](docs/experiments/2026-08-30-rcaeval-re2-ob-shift-audit-v1.md).

The resulting delta-candidate development experiment is complete. Delta-only
logistic regression raises repetition-held-out Macro F1 from 0.785 to 0.934
but falls below the fixed delay gate. A training-median/IQR service-grouped
delta score achieves Macro F1 0.922 in both repetition- and fault-type-held-out
evaluation, with Top-3 accuracy 1.000, and passes all four pre-registered
promotion requirements. It is selected for confirmation on another public
system; the RE2-OB result remains development evidence. See the
[delta candidate report](docs/experiments/2026-08-30-rcaeval-re2-ob-delta-candidates-v1.md).

The selected robust service-delta candidate has now undergone frozen
cross-system confirmation on 90 independent RCAEval RE2-SS cases. Confirmation
failed: the candidate scored Macro F1 0.778/0.758 in repetition/fault holdouts,
below the full-metric reference at 0.849/0.841 and below the locked absolute
requirements. Its Top-3 accuracy remained 0.978/1.000, but the fixed score cap
saturated on 89 of 90 cases; every candidate error occurred in a zero-margin
tie. The RE2-OB score therefore remains development evidence, and candidate v2
must fix score aggregation and tie handling before testing on a third system.
See the
[RE2-SS confirmation report](docs/experiments/2026-08-30-rcaeval-re2-ss-confirmation-v1.md).

Service-delta v2 replaces the saturated maximum score with an uncapped
empirical-tail score aggregated over each service's two strongest delta
features. On the OB/SS development systems it achieved Macro F1
0.956/0.945 and 0.924/0.945 across repetition/fault holdouts, passed every
locked promotion check and produced no exact numeric ties. The fixed soft-vote
candidate also passed, but the label-free empirical-tail scorer was selected
by the pre-registered order. See the
[v2 development report](docs/experiments/2026-08-30-rcaeval-service-delta-v2-development.md).

The selected v2 scorer then passed its single untouched RE2-TT confirmation.
Its repetition- and fault-type-held-out Macro F1 values were 0.956 and 0.945,
versus 0.798 and 0.741 for the fixed full-metric logistic reference. All eight
confirmation checks passed; Top-3 accuracy was 0.989 in both designs, the
largest predicted-class fraction was 0.233 and there were no exact numeric
ties. This supports algorithmic generalisation across Online Boutique, Sock
Shop and Train Ticket under the stated RE2 service-localisation setup, not
pre-trained-model transfer or production causality. See the
[RE2-TT confirmation report](docs/experiments/2026-08-30-rcaeval-re2-tt-service-delta-v2-confirmation.md).

A host-side, human-approved operations Agent MVP is now implemented for the
three controlled local faults. It combines read-only evidence tools, validated
local Runbooks, a deterministic or optional OpenAI Responses API explanation
planner, an explicit approval gate, allow-listed recovery and post-action
health verification. Agent incident state and an append-only audit trail are
persisted under `data/agent/`. The implementation is unit-tested, but no
manual-versus-Agent MTTR or operator study has yet been run. See the
[Agent design and demonstration guide](docs/agent/2026-08-30-agent-mvp.md).

## Documentation

- [Project definition](docs/project-definition.md)
- [First database failure baseline](docs/experiments/2026-08-10-database-failure-baseline.md)
- [Three-trial database failure batch](docs/experiments/2026-08-10-database-failure-batch.md)
- [Three-trial service-stop batch](docs/experiments/2026-08-10-service-stop-batch.md)
- [Three-trial HTTP 500 batch](docs/experiments/2026-08-10-http-500-batch.md)
- [Mixed-fault and normal-operation baseline](docs/experiments/2026-08-10-mixed-and-normal-baseline.md)
- [Public dataset compatibility audit](docs/data/2026-08-20-public-dataset-compatibility.md)
- [Incident schema v1](docs/data/incident-schema-v1.md)
- [Local telemetry collection](docs/data/local-telemetry-collection.md)
- [Local telemetry smoke experiment](docs/experiments/2026-08-24-local-telemetry-smoke.md)
- [Frozen 60-incident local campaign](docs/experiments/2026-08-25-local-campaign-v1.md)
- [Traditional-ML baseline](docs/experiments/2026-08-28-local-ml-baseline-v1.md)
- [Gated robust-fusion v2 development](docs/experiments/2026-08-30-gated-robust-fusion-v2-development.md)
- [Gated robust-fusion v2 v4 confirmation](docs/experiments/2026-08-30-local-holdout-v4-confirmation.md)
- [Early-diagnosis v4 stress test](docs/experiments/2026-08-30-early-diagnosis-v4-stress-test.md)
- [+1-second log-evidence ablation](docs/experiments/2026-08-30-log-evidence-ablation-v4-1s.md)
- [RCAEval RE2-OB public baseline](docs/experiments/2026-08-30-rcaeval-re2-ob-public-baseline-v1.md)
- [RCAEval RE2-OB shift audit](docs/experiments/2026-08-30-rcaeval-re2-ob-shift-audit-v1.md)
- [RCAEval RE2-OB delta candidates](docs/experiments/2026-08-30-rcaeval-re2-ob-delta-candidates-v1.md)
- [RCAEval RE2-SS confirmation](docs/experiments/2026-08-30-rcaeval-re2-ss-confirmation-v1.md)
- [RCAEval service-delta v2 development](docs/experiments/2026-08-30-rcaeval-service-delta-v2-development.md)
- [RCAEval RE2-TT service-delta v2 confirmation](docs/experiments/2026-08-30-rcaeval-re2-tt-service-delta-v2-confirmation.md)
- [Human-approved operations Agent MVP](docs/agent/2026-08-30-agent-mvp.md)
- [Thesis positioning and final evidence map](docs/thesis/2026-08-30-thesis-positioning-and-evidence-map.md)
- [Thesis method, results and discussion draft](docs/thesis/2026-08-30-method-results-discussion-draft.md)
- [Dissertation completion checklist](docs/thesis/2026-08-30-next-writing-checklist.md)
- [Stage progress report](docs/reports/2026-08-20-stage-progress-report.docx)
- [Stage progress report (English)](docs/reports/2026-08-20-stage-progress-report-en.docx)

## Public dataset compatibility audit

Download only the RCAEval and LO2v2 metadata indexes:

```powershell
.\.venv\Scripts\python.exe -m experiment.data_import.download_public_indexes
```

Install the isolated data-analysis dependencies and generate the audit:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe -m experiment.data_import.public_index_audit
```

Download the approved six-case RCAEval pilot (logs and metrics only):

```powershell
.\.venv\Scripts\python.exe -m experiment.data_import.download_rcaeval_pilot
```

Download the locked complete RE2-OB subset (90 cases, logs and metrics only):

```powershell
.\.venv\Scripts\python.exe -m experiment.data_import.download_rcaeval_subset
```

External indexes and telemetry remain under `data/external/` and are excluded
from Git. Public fault labels are kept separate from the local three-class
ground truth unless a compatibility audit establishes semantic equivalence.

## Incident-level data engineering

Convert the approved RCAEval pilot into one JSONL row per complete incident:

```powershell
.\.venv\Scripts\python.exe -m experiment.data_engineering.build_rcaeval_incidents
```

Convert and evaluate the complete RE2-OB public subset:

```powershell
.\.venv\Scripts\python.exe -m experiment.data_engineering.build_rcaeval_incidents `
    --root data\external\rcaeval\re2-ob `
    --output data\processed\incidents\rcaeval-re2-ob.jsonl `
    --manifest data\processed\incidents\rcaeval-re2-ob-manifest.json

.\.venv\Scripts\python.exe -m experiment.rcaeval_public_baseline
```

The generated JSONL file contains ordered log text, pre/post metric summaries,
source labels, root-cause service, time-window metadata and source-file hashes.
The companion manifest verifies unique split groups and confirms that public
labels have not been mapped into the local three-class task. Generated files
are written under `data/processed/incidents/` and excluded from Git.

Convert telemetry-enabled local runs into the same top-level incident schema:

```powershell
.\.venv\Scripts\python.exe -m experiment.data_engineering.build_local_incidents `
    --run-id <approved-run-id>
```

Only approved production runs should be selected. Local records retain the
three-class target, while RCAEval records remain in the separate public
root-cause-localisation task.

## Frozen local data campaign

The completed v1 artifacts are protected from accidental overwrite. To run a
new campaign, first copy the configuration, choose a new campaign ID and use
new output paths. Then validate it without injecting faults:

```powershell
.\.venv\Scripts\python.exe -m experiment.campaign_runner `
    --config experiment\configs\local-campaign-v2.json `
    --dry-run
```

See the [campaign report](docs/experiments/2026-08-25-local-campaign-v1.md) for
the frozen settings, quality checks, rule-baseline results and interpretation.

## Traditional machine-learning baseline

Install the data-analysis dependencies and run a protected reproduction with
new output paths:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe -m experiment.ml_baseline `
    --output data\results\local-ml-baseline-reproduction.json `
    --figure data\results\figures\local-ml-baseline-reproduction.png
```

The pipeline removes rule-detector output and control-plane fault labels before
feature extraction, fits all preprocessing inside each training fold, and uses
identical incident-level folds for the logs-only and logs-plus-metrics models.
See the [ML experiment report](docs/experiments/2026-08-28-local-ml-baseline-v1.md)
for the confusion matrices, feature interpretation and limitations.

## Frozen gated robust-fusion candidate

The v2 candidate bundle has already been frozen. Do not rerun
`develop-freeze` against its existing output directory; the command refuses to
overwrite it. The v4 confirmation has been completed. For auditability, a
future new holdout evaluation keeps this command form:

```powershell
.\.venv\Scripts\python.exe -m experiment.robust_fusion_v2 evaluate `
    --input data\processed\incidents\<new-campaign>.jsonl `
    --output data\results\gated-robust-fusion-v2-<new-campaign>.json
```

The evaluator verifies the frozen source, runtime, specification and model
hashes; rejects overlapping or pre-freeze telemetry; and never fits or tunes on
the confirmation set. The completed v4 result is documented in the
[confirmation report](docs/experiments/2026-08-30-local-holdout-v4-confirmation.md);
the development result and v3 diagnostic remain in the
[robust-fusion report](docs/experiments/2026-08-30-gated-robust-fusion-v2-development.md).

## Phase 1 quick start

Prerequisite: Docker Desktop must be installed and running.

```powershell
docker compose up --build
```

After startup:

- Service information: <http://localhost:8000/>
- Interactive API documentation: <http://localhost:8000/docs>
- Liveness check: <http://localhost:8000/health/live>
- Database health: <http://localhost:8000/health/database>

Start the human-approved operations Agent in a second PowerShell window:

```powershell
.\.venv\Scripts\python.exe -m ops_agent
```

Then open the Agent console at <http://127.0.0.1:8100/> or its interactive API
documentation at <http://127.0.0.1:8100/docs>. The default deterministic
planner needs no external API. Optional OpenAI planner setup and the complete
three-fault demonstration are documented in the
[Agent guide](docs/agent/2026-08-30-agent-mvp.md).

Create a sample order:

```powershell
$body = @{
    customer_id = "customer-001"
    item = "sample-item"
    quantity = 1
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri http://localhost:8000/orders `
    -ContentType "application/json" `
    -Body $body
```

Stop the platform with:

```powershell
docker compose down
```

The `workload` container submits one simulated order per second after the API
becomes healthy. Follow its output with:

```powershell
docker compose logs -f workload
```

Inject a 30-second database disconnection from a second PowerShell window:

```powershell
.\.venv\Scripts\python.exe -m experiment.fault_injector `
    database_disconnect `
    --duration 30
```

Ground-truth start and end events are appended to
`data/ground_truth/incidents.jsonl` independently of the application logs.

The `detector` container polls the database health endpoint and writes the
rule-based baseline output to `data/detections/detections.jsonl`. This output
can later be matched against the independent ground truth to calculate
detection latency and classification accuracy.

Evaluate all recorded incidents:

```powershell
.\.venv\Scripts\python.exe -m experiment.evaluate
```

Run three repeatable database-failure trials after starting the Compose stack:

```powershell
.\.venv\Scripts\python.exe -m experiment.batch_runner `
    --trials 3 `
    --duration 10 `
    --cooldown 5
```

Batch and mixed runners capture per-incident probes and bounded Docker logs by
default under `data/raw/local/`. They also archive only detector events appended
during the current run under `data/detections/<run-id>.jsonl`; use that archive
for evaluation so historical detections cannot be counted as new false
positives. See [Local telemetry collection](docs/data/local-telemetry-collection.md)
for the file contract and window controls.

Run the service/container-stop fault instead:

```powershell
.\.venv\Scripts\python.exe -m experiment.batch_runner `
    --fault service_stop `
    --trials 3 `
    --duration 8 `
    --cooldown 4 `
    --ground-truth data\ground_truth\service-stop-batch.jsonl `
    --output data\results\service-stop-batch.json
```

Run the controlled HTTP 500 fault:

```powershell
.\.venv\Scripts\python.exe -m experiment.batch_runner `
    --fault http_500 `
    --trials 3 `
    --duration 8 `
    --cooldown 8 `
    --ground-truth data\ground_truth\http-500-batch.jsonl `
    --output data\results\http-500-batch.json
```

Run a balanced, reproducibly shuffled mixed batch containing two repetitions
of every fault class:

```powershell
.\.venv\Scripts\python.exe -m experiment.mixed_runner `
    --repetitions-per-fault 2 `
    --seed 20260810 `
    --duration 8 `
    --cooldown 8
```

Run a 30-second no-fault observation period:

```powershell
.\.venv\Scripts\python.exe -m experiment.normal_observation `
    --duration 30
```
