# Incident schema v1

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

The machine-learning data layer uses one JSONL row per complete incident. A
row is the indivisible unit for training, validation and testing. Log lines or
metric samples from the same incident must never be split across different
dataset partitions.

## Core identity and labels

| Field | Meaning |
|---|---|
| `schema_version` | Schema version; currently `1`. |
| `incident_id` | Globally unique incident identifier, including the source prefix. |
| `source` | Data origin, for example `local` or `rcaeval`. |
| `dataset` | Dataset or experiment-batch name. |
| `system_name` | System from which the telemetry was collected. |
| `task` | `local_fault_classification` or `public_root_cause_localisation`. |
| `root_cause_service` | Service identified by the dataset's own Ground Truth. |
| `original_fault_label` | Unmodified source label. |
| `local_fault_label` | Local three-class label, or `null` for public incidents. |
| `split_group` | Grouping key that must remain wholly within one data split. |

The local label space is limited to:

- `database_connection_failure`;
- `service_stopped`;
- `http_500_failure`.

RCAEval labels are not mapped into this label space. RCAEval remains a
separate root-cause-localisation task unless a later, explicitly documented
mapping experiment is approved.

## Time window and telemetry

| Field | Meaning |
|---|---|
| `injected_at` | Ground-truth injection time in timezone-aware ISO 8601 UTC. |
| `window_start`, `window_end` | Inclusive telemetry window around injection. |
| `modalities` | Telemetry present in the record, initially logs and metrics. |
| `log_text` | Ordered log messages in `[service] message` format for text models. |
| `log_record_count` | Number of source log rows included in the window. |
| `log_services` | Services represented in the log window. |
| `metric_row_count` | Number of metric time steps in the complete window. |
| `metric_pre_rows`, `metric_post_rows` | Metric rows before and from injection onward. |
| `metric_features` | Per-metric pre-mean, post-mean and post-minus-pre mean delta. |
| `provenance` | Case name, source-file hashes and window parameters. |

The RCAEval pilot uses 60 seconds before injection and 120 seconds after
injection. This is an importer-validation choice, not a claim about the ideal
diagnostic window. Later experiments must compare window sizes on training
data only and keep the selected value fixed for the final test set.

## Leakage controls

1. `incident_id` and `split_group` must be unique in the generated pilot.
2. Dataset splitting must operate on `split_group`, never on log or metric rows.
3. Public records keep `local_fault_label = null`.
4. File hashes and window parameters are stored for reproducibility.
5. Any preprocessing vocabulary, scaler or feature selection must be fitted on
   the training split only.

## Build the RCAEval pilot

After installing `requirements-data.txt` and downloading the approved pilot:

```powershell
.\.venv\Scripts\python.exe -m experiment.data_engineering.build_rcaeval_incidents
```

The command writes the processed JSONL dataset and a leakage-check manifest to
`data/processed/incidents/`. Processed telemetry remains excluded from Git.

## Build local incidents

After running telemetry-enabled fault batches, convert raw incident packages
plus their independent Ground Truth and run-specific detection archives:

```powershell
.\.venv\Scripts\python.exe -m experiment.data_engineering.build_local_incidents
```

Use repeated `--run-id` arguments when only approved runs should enter a
dataset. The converter rejects incomplete Ground Truth, failed recovery,
collector errors, Docker-log errors, mismatched row counts, telemetry windows
that do not contain the complete fault, and incomplete detector start/end
pairs.

Local and RCAEval records have identical top-level fields, but their tasks and
labels remain separate:

- local records use `task = local_fault_classification` and one of the three
  local labels;
- RCAEval records use `task = public_root_cause_localisation` and keep
  `local_fault_label = null`.
