# Independent local holdout v2: prespecified protocol

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

## Research question / 研究问题

Can the two v1 classifiers maintain incident-level classification performance
on a later local campaign with a shorter workload interval and longer faults,
without any fitting, feature selection or hyperparameter tuning on the test data?

第二轮检验“固定模型在新运行条件下是否仍然有效”。这不是重新在新数据上做交叉验证，
也不是跨系统测试。先保存基于第一轮全部 60 条数据训练的完整流水线，再采集独立测试集。

## Locked design

| Setting | Training campaign v1 | Independent test campaign v2 |
|---|---|---|
| System | Local order platform | Same platform and fault mechanisms |
| Incidents | 60; 20 per class | 30; 10 per class |
| Balanced schedule seed | 20260825 | 20260828 |
| Fault duration | 8 seconds | 12 seconds |
| Workload interval | 1 second | 0.25 seconds |
| Pre/post capture | 10/10 seconds | Unchanged |
| Probe interval / timeout | 1 / 0.8 seconds | Unchanged |
| Between-incident cooldown | 8 seconds | Unchanged |
| Models | TF-IDF + logistic regression; same + scaled metrics | Frozen v1 models, prediction only |

The workload interval is a sleep **after each completed request**. It is not an
exact throughput target; slower responses and timeouts lower achieved throughput.
Only the workload container environment changes temporarily; its Docker image
must remain identical. The base Compose file is not edited. Allow a 10-second
warm-up after changing the interval.

Workload and fault duration change together. Consequently, results cannot isolate
the effect of either factor. Use separate load-only and duration-only campaigns
if single-factor attribution is subsequently needed.

## Freeze and contamination controls

- Fit both complete pipelines only on v1 using the existing hyperparameters and
  seed `20260825`. Preserve the original v1 CV result; the full-data training fit
  is a separate artifact, not a replacement for its out-of-fold scores.
- Save both pipelines, including TF-IDF vocabulary/IDF, metric vectorisation,
  scaler and classifier, in `data/models/local-v1-frozen/models.joblib`.
- Record training input hash, IDs, groups, campaign IDs, raw telemetry hashes,
  time ranges, metric schema, code hashes and dependency versions in the manifest.
- Verify model bytes, code and versions before prediction. Reject reused incident
  IDs, split groups, campaign IDs or raw telemetry, nonlocal tasks, changed metric
  schemas and telemetry captured before model freezing.
- Keep removal of rule-detector rows, fault-toggle messages and injected
  `fault_type` fields identical to v1. Do not change these controls after viewing
  test results.
- Register the actual model/config/code hashes before fault injection in
  `data/results/local-campaign-v2-protocol.json`. Preserve failed attempts rather
  than deleting or overwriting them.
- Only load locally produced joblib files. A checksum is not a security guarantee
  for untrusted pickle-based artifacts.

## Prespecified outputs and interpretation

Primary outcome: test Macro F1 for each model and their paired difference on the
same 30 incidents. Secondary outcomes: accuracy, per-class precision/recall/F1,
confusion matrices and individual predictions. Probabilities are uncalibrated
descriptive outputs, not reliability guarantees.

The existing online rule detector is also evaluated against independent ground
truth on this campaign. Its detection/recovery latencies are **not** directly
comparable with these retrospective classifiers: their complete incident windows
include post-onset and recovery evidence.

All three known fault types have explicit operational signatures. A high score
on 30 incidents would support only this limited same-system condition shift;
it would not establish unknown-fault detection, cross-system root-cause analysis,
production readiness or autonomous recovery. No threshold or hypothesis will be
rewritten to turn a null result into an improvement claim.

## Execution

Run each command as a complete line in PowerShell from the project directory:

```powershell
# Replace this placeholder with the path to your local clone.
Set-Location -LiteralPath 'C:\path\to\aiops-microservice-agent'
.\.venv\Scripts\python.exe -m experiment.frozen_holdout freeze
.\.venv\Scripts\python.exe -m experiment.holdout_campaign --dry-run
.\.venv\Scripts\python.exe -m experiment.holdout_campaign
.\.venv\Scripts\python.exe -m experiment.frozen_holdout evaluate
```

The collector temporarily stops the experimental API/database and enables
controlled HTTP 500 responses. Do not run it against production services or
concurrently with another fault-injection demonstration. Each injector attempts
recovery in `finally`; the wrapper also restores the baseline workload on success
or Python exceptions. Hard process termination or Docker shutdown may bypass
cleanup, so always check final health and the runtime protocol.

The collector and evaluator refuse to overwrite existing outputs. A completed
campaign should be reviewed, not rerun with the same filenames. Do not use
`campaign_runner` directly for v2: the holdout wrapper is what applies and records
the workload condition and its restoration.

## Next decision

If both models remain at ceiling, investigate prespecified log-quality/early-window
ablations next. If performance drops, inspect errors and design the next experiment
without tuning on v2 and presenting it as an untouched test set. Either outcome
is more informative than adding a deep model to the same 60 training incidents.
