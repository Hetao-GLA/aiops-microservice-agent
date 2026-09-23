# Early-diagnosis stress test on v4 incidents

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

Date: 2026-08-30

## Decision

The frozen logs-only branch and the gated candidate classified all 30 v4
incidents correctly with only one second of post-injection telemetry. Their
Macro F1 remained 1.000 at the preregistered +1, +3, +5 and +8 second cutoffs.

This is evidence that class-specific log signatures appear almost immediately
in the controlled local faults. It is not evidence of end-to-end one-second
detection: the analysis is given a known injection boundary and performs
classification on a pre-segmented incident window.

The ungated robust-fusion branch was strongly window-sensitive. Its Macro F1
rose from 0.5556 at +1 second to 0.9327 at +8 seconds, then fell to 0.8222 in
the complete retrospective window. The OOD gate caught every ungated error at
all windows, but achieved this safety with low fusion coverage and many
unnecessary log fallbacks.

## Locked post-hoc design

The analysis protocol was written after v4 confirmation but before any
early-window prediction. The same 30 accepted v4 incidents were reconstructed
at +1, +3, +5 and +8 seconds after fault injection.

- The full ten-second pre-fault baseline was retained.
- Raw Docker logs and probes after the cutoff were excluded.
- Every realised v4 fault lasted more than eight seconds, so all early variants
  excluded recovery evidence.
- Raw metadata, probe and Docker-log hashes were verified against v4
  provenance.
- The original 45-feature metric schema was preserved.
- A missing early post-fault numeric probe field used the preregistered 0.0
  failure sentinel.
- The candidate, logs-only branch and ungated fusion branch were never fitted
  or tuned.

Because the same observed v4 incidents are reused, this is a descriptive
post-hoc stress curve and not another independent confirmation.

## Telemetry retained

| Cutoff | Incidents | Log rows | Probe rows | Post-injection probe rows |
|---|---:|---:|---:|---:|
| +1 s | 30 | 8,073 | 1,023 | 60 |
| +3 s | 30 | 8,633 | 1,168 | 205 |
| +5 s | 30 | 9,268 | 1,313 | 350 |
| +8 s | 30 | 10,172 | 1,530 | 567 |

The large log counts include the common ten-second pre-fault baseline under a
four-times-higher request workload. The model does not need to infer the start
of an incident; the cutoff is supplied by the experimental Ground Truth.

## Classification results

| Window | Gated Macro F1 | Logs Macro F1 | Ungated Macro F1 | Ungated database F1 | Ungated HTTP 500 F1 | Ungated service-stop F1 |
|---|---:|---:|---:|---:|---:|---:|
| +1 s | 1.0000 | 1.0000 | 0.5556 | 0.0000 | 1.0000 | 0.6667 |
| +3 s | 1.0000 | 1.0000 | 0.6238 | 0.1818 | 1.0000 | 0.6897 |
| +5 s | 1.0000 | 1.0000 | 0.7341 | 0.4615 | 1.0000 | 0.7407 |
| +8 s | 1.0000 | 1.0000 | 0.9327 | 0.8889 | 1.0000 | 0.9091 |
| Complete | 1.0000 | 1.0000 | 0.8222 | 0.6667 | 1.0000 | 0.8000 |

The ungated branch first exceeded the preregistered descriptive Macro-F1
milestones of 0.8 and 0.9 at +8 seconds. It never reached 1.0. The gated and
logs-only models already met all three milestones at +1 second.

The decline from +8 seconds to the complete window shows that adding recovery
and post-fault metric samples does not monotonically improve this fusion model.
Metric aggregation over a longer window can move the representation away from
the fault-period state.

## Gate behaviour

| Window | Fusion coverage | Fallback rate | Mean OOD fraction | Ungated errors | Errors caught by gate | Ungated-correct cases also sent to fallback |
|---|---:|---:|---:|---:|---:|---:|
| +1 s | 0.1667 | 0.8333 | 0.4120 | 10 | 10 | 15 |
| +3 s | 0.2333 | 0.7667 | 0.3287 | 9 | 9 | 14 |
| +5 s | 0.1333 | 0.8667 | 0.3287 | 7 | 7 | 19 |
| +8 s | 0.0000 | 1.0000 | 0.3824 | 2 | 2 | 28 |
| Complete | 0.1667 | 0.8333 | 0.1898 | 5 | 5 | 20 |

The gate had perfect error capture on these variants: no ungated error was
allowed through the fusion route. Its specificity was poor. At +8 seconds it
sent all 30 incidents to logs even though the ungated branch was correct on 28.
The current rule is therefore best understood as a conservative safety guard,
not an efficient fusion-selection mechanism.

## Research interpretation

Supported findings:

1. Complete recovery evidence is not necessary to classify these three local
   faults once an incident window has already been identified.
2. Direct operational log signatures appear within the first post-injection
   second in this controlled environment.
3. Metric fusion is time-window sensitive, especially for database failures.
4. Training-range OOD routing contains all observed fusion errors but usually
   disables fusion, confirming the low-coverage limitation seen in v4.

Claims not supported:

- incident detection within one second;
- online performance under an unknown event start;
- robustness after removing explicit database, HTTP-status and connection
  failure phrases;
- cross-system, unknown-fault or public-data generalisation.

The immediate perfect log result means the next experiment should not add a
deeper classifier. It should reduce the strength of the available log evidence
or change the task. A source/signature ablation can establish which messages
carry the early result, while RCAEval should remain a separate public
root-cause-localisation benchmark.

## Artifacts

- Locked specification:
  `experiment/configs/early-diagnosis-v4-stress-spec.json`
- Result:
  `data/results/early-diagnosis-v4-stress-test.json`
- Result SHA-256:
  `fab44ed037dfb689034529140f5cd11809f64e6bfaddafa86c29c8125a2e1ced`
- Figure:
  `data/results/figures/early-diagnosis-v4-stress-test.png`
- Figure SHA-256:
  `d0afff6285aaef176977f256681656c66d1a0884cd2b3f91da4b4c933f7f4d3b`
- +1 s dataset SHA-256:
  `566f0c544b8887d4aa42db2c1c1be8708d560253f2efbac091843c50b22c2c21`
- +3 s dataset SHA-256:
  `6b8958c3c3af60af30882216eb4f60391e489641fd145b21554cf08e912628dd`
- +5 s dataset SHA-256:
  `febed2ed0b06a863d91eadaf6155330d0255f5ec600cc44f733de93417b7c98d`
- +8 s dataset SHA-256:
  `1f40fba3ef93a98e4685fec0a41a41c45e045dda357845cda383a9b5e43aca71`

## Next step

Run a separately locked evidence-ablation analysis on the +1-second variants:

- application-service logs only;
- workload-client logs only;
- logs with direct fault/status phrases uniformly masked;
- metrics-only and gated configurations as secondary comparisons.

If perfect performance survives uniform signature masking, the experiment can
move to a prospective weak-signature campaign. If it does not, the thesis can
state clearly that current early performance depends on explicit operational
messages rather than deeper temporal reasoning.
