# Public dataset compatibility audit

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

Audit generated from locally downloaded metadata indexes on
2026-08-20T20:36:06.317746+00:00.

## Project target

The local experiment classifies three operational failures:

1. `database_connection_failure`;
2. `service_stopped`;
3. `http_500_failure`.

The public datasets must not be merged into this label space unless the fault
mechanism and ground-truth meaning are equivalent.

## RCAEval

- Index size: **735 failure cases** across nine dataset/system
  combinations.
- Cases with logs: **359**.
- Cases with traces: **240**.
- Labels describe root-cause service and public fault type, not our local
  three-class labels.
- `RE2-OB` contains **90 cases**, including
  **15 socket cases** across five services.
- `RE3-OB` contains **30 code-level cases**.

Decision: **accept for a separate root-cause localisation benchmark**. Do not
directly relabel `socket` as database failure. A socket case may be used only
as a domain-transfer proxy, with results reported separately. RE3 cases may
support log/stack-trace representation experiments, but their F1-F5 meanings
must be confirmed case by case before any mapping.

Recommended first telemetry sample:

- three `RE2-OB` socket cases from one service, to validate the importer;
- three `RE3-OB cartservice_f1` cases, because the public tutorial documents
  the injected incorrect-parameter fault and its stack trace;
- retain the public root-cause labels unchanged.

Source: <https://github.com/phamquiluan/RCAEval>

## LO2v2

- Runs: **115**.
- Tests: **6210**, consisting of **115** `correct`
  tests and **6095** deliberately invalid API tests.
- Response-code distribution encoded in test names: HTTP 400: 3450, HTTP 401: 575, HTTP 404: 2070.
- Explicit HTTP 500 test in the index: **False**.
- Indexed volume: **2,574,594,145 log lines** and
  **910,792 trace lines**.

Decision: **do not use LO2v2 to train the local three-fault classifier**. Its
error tests primarily represent expected API validation/authentication
responses (400/401/404), not a stopped service, lost database connection, or
internal HTTP 500 failure. It can later provide an external robustness test:
the detector should distinguish expected client errors from operational
incidents. Downloading its full telemetry is deferred because the index alone
shows a task mismatch and a very large data volume.

Source: <https://zenodo.org/records/18937117>

## AIOps Challenge 2020

The official repository describes business metrics, infrastructure metrics,
trace data, and a fault table containing time, type, and location. It does not
describe application log text as a primary modality.

Decision: **defer**. It is a possible external metric/trace root-cause
benchmark, but not appropriate for the planned TF-IDF log classifier. Its
non-commercial research/teaching licence condition must be recorded if it is
used.

Source: <https://github.com/NetManAIOps/AIOps-Challenge-2020-Data>

## Final integration decision

1. Keep locally generated incidents as the only training and test source for
   the three operational fault classes.
2. Use RCAEval as a separate public root-cause benchmark and, optionally, a
   transfer-learning source.
3. Keep LO2v2 as a later robustness dataset rather than a fault-label source.
4. Split all datasets by complete incident/case, never by individual log line.
5. Report local and public results separately to expose domain shift.
