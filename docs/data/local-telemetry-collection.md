# Local incident telemetry collection

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

The batch and mixed-fault runners now capture a bounded, incident-specific
telemetry package by default. The package is written under
`data/raw/local/<run-id>/<incident-id>/` and remains excluded from Git.

## Per-incident files

| File | Content |
|---|---|
| `metadata.json` | Incident ID, expected fault, collection boundaries, sampling settings, row counts and collection errors. |
| `probes.jsonl` | Host-side samples of liveness, database health and five-second order metrics. |
| `docker-logs.jsonl` | Time-bounded logs from the API, workload, rule detector and PostgreSQL containers. |

Probe samples include their UTC timestamp, endpoint, HTTP status, response
payload, availability and request latency. An unavailable service is retained
as an observation with an error type rather than dropped.

Docker logs are requested with explicit `--since` and `--until` boundaries.
Each JSONL record retains the service, container, stream and original Docker
timestamp. A missing container or failed log export fails the telemetry step
and is recorded in metadata.

## Detection-run isolation

The long-running detector appends to `data/detections/detections.jsonl`. Reusing
that file directly would mix historical detections into a new evaluation.
Each runner therefore records the file byte offset after the system health
check and archives only records appended during that run:

`data/detections/<run-id>.jsonl`

Evaluation must use this run-specific archive, not the shared append-only
source file.

## Collection windows

Command-line defaults are:

- 10 seconds before fault injection;
- 10 seconds after recovery;
- one probe cycle per second;
- a request timeout automatically set to 80% of the probe interval, bounded to
  0.2-2.0 seconds;
- all four Compose containers included in the Docker-log export.

The pre/post windows are part of the experiment configuration and must remain
fixed across a comparison. The values can be changed for a pilot:

```powershell
.\.venv\Scripts\python.exe -m experiment.batch_runner `
    --fault http_500 `
    --trials 1 `
    --duration 3 `
    --telemetry-pre-seconds 2 `
    --telemetry-post-seconds 4 `
    --telemetry-interval-seconds 0.5
```

Use `--no-telemetry` only for a pipeline test that will not contribute to the
machine-learning dataset. Use `--no-detection-archive` only when detector
output is intentionally out of scope.

## Data-quality gates

Before converting a local incident into schema v1, verify:

1. Ground Truth contains both `fault_started` and `fault_ended` for the ID.
2. `metadata.json` has no collector or Docker-capture error.
3. Probe samples cover the configured pre- and post-fault boundaries.
4. The run-specific detection archive contains complete start/end pairs.
5. The incident directory and detection archive belong to the same `run_id`.
6. Dataset splitting later operates on the complete incident ID.

## Convert approved local runs

The local converter creates one schema-v1 JSONL row per incident:

```powershell
.\.venv\Scripts\python.exe -m experiment.data_engineering.build_local_incidents `
    --run-id <approved-run-id> `
    --run-id <another-approved-run-id>
```

Its manifest reports label counts, total log/probe records, feature count,
source hashes and all quality-gate outcomes. Debugging or superseded runs should
remain under `data/raw/local/` but must not be selected for a production
dataset.
