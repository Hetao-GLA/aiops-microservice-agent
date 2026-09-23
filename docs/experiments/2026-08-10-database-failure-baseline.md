# Database Failure Baseline Experiment

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

## Experiment metadata

- Date: 10 August 2026
- Environment: Docker Compose on Windows with Docker Desktop
- Workload: one simulated order request per second
- Fault: PostgreSQL container stopped for 10 seconds
- Detector: `database-health-rule-v1`
- Ground-truth incident: `INC-7C1A2CDE4045`
- Detection incident: `DET-A58B39F65C61`

## Expected result

The independent fault injector should record a database connection failure,
the API should become unable to store orders, the rule detector should identify
the database failure, and normal traffic should resume after PostgreSQL is
started again.

## Observed timeline

| Event | UTC timestamp | Result |
|---|---:|---|
| Ground-truth fault started | 14:23:55.618 | PostgreSQL container stopped |
| Rule detection started | 14:23:59.419 | Database connection failure detected |
| Ground-truth fault ended | 14:24:06.495 | PostgreSQL container started |
| Rule detection ended | 14:24:08.828 | Database health returned HTTP 200 |

## Results

- Predicted fault: `database_connection_failure`
- Expected fault: `database_connection_failure`
- Classification result: correct
- Detection latency: **3.801 seconds**
- Initial detector evidence: HTTP `ReadTimeout`
- Subsequent evidence during the incident: HTTP `503 Service Unavailable`
- Recovery detection: successful
- Post-recovery state: database, API, workload generator, and detector healthy
- Post-recovery workload: successful HTTP 201 order responses resumed

## Interpretation

The experiment demonstrates the first complete evidence chain:

```text
workload
-> controlled database fault
-> independent ground truth
-> observable service degradation
-> rule-based detection
-> database restoration
-> recovery detection
-> resumed workload
```

The 3.801-second detection latency was dominated by the detector's four-second
HTTP timeout on its first failed probe. This timeout is therefore an explicit
baseline configuration and should remain fixed during comparable experiments,
or be tuned in a separate sensitivity experiment.

## Limitations

- This is a single trial and cannot support statistical conclusions.
- Only the database connection failure class was tested.
- Detection and ground-truth incidents are not yet correlated automatically.
- The detector classifies the failed component but cannot distinguish every
  possible underlying reason for database unavailability.
- Logs and metrics have not yet been combined in the detector.

## Next experiment work

1. Automate matching of detection events to ground-truth incidents.
2. Repeat the database failure under a fixed workload.
3. Calculate detection latency, recovery latency, false positives, and misses.
4. Add the service/container-stop fault class.
5. Preserve the rule configuration as the research baseline.
