# RCAEval RE2-TT service-delta v2 confirmation

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

Date: 2026-08-30  
Decision: **three-system confirmation passed**.

## Confirmatory setup

The protocol, candidate, reference and gate were locked before downloading or
evaluating RE2-TT telemetry. The empirical-tail Top-2 scorer was the only
eligible candidate. The full pre/post/delta logistic regression was the fixed
reference. Both used shared repetition-held-out and fault-type-held-out folds.

RE2-TT contains 90 balanced Train Ticket incidents: five root-cause services,
six fault types and three repetitions for every pair. All 90 have metrics, but
one has no log data. Because both algorithms are metrics-only, the acquisition
was fixed at `inject_time.txt` plus `metrics.parquet` for all 90 cases. No log
was fabricated and no case was dropped.

Data validation passed for 90 cases, 180 files and 84,661,051 bytes. Every file
hash was verified, the processed JSONL contains 90 unique incidents, and the
public service labels remain separate from the local three-fault task.

## Main results

| Evaluation | Full-metric LR | Empirical-tail Top-2 | Difference |
|---|---:|---:|---:|
| Repetition held out | 0.798 | **0.956** | **+0.158** |
| Fault type held out | 0.741 | **0.945** | **+0.204** |

Values are out-of-fold Macro F1. Candidate Top-1 accuracy was 0.956 and 0.944;
Top-3 accuracy was 0.989 in both designs.

### Fold stability

| Held-out group | Full-metric LR | Empirical-tail Top-2 |
|---|---:|---:|
| Repetition 1 | 0.538 | **1.000** |
| Repetition 2 | 0.935 | 0.933 |
| Repetition 3 | 0.901 | **0.935** |
| CPU | 0.505 | **1.000** |
| Delay | 0.931 | 0.931 |
| Disk | 1.000 | 1.000 |
| Loss | 0.594 | **0.737** |
| Memory | 0.213 | **1.000** |
| Socket | 1.000 | 1.000 |

The candidate made four errors under repetition holdout and five under fault
holdout. Errors were distributed across service pairs rather than collapsing
onto one label. Its largest predicted-class fraction was 0.233 and its exact
numeric tie fraction was 0.000.

## Locked gate

| Check | Required | Observed | Pass |
|---|---:|---:|:---:|
| Mean Macro F1 difference | >= 0.000 | +0.181 | yes |
| Worst design difference | >= -0.050 | +0.158 | yes |
| Minimum design Macro F1 | >= 0.750 | 0.945 | yes |
| Minimum repetition-fold Macro F1 | >= 0.550 | 0.933 | yes |
| Delay difference | >= -0.020 | 0.000 | yes |
| Minimum Top-3 accuracy | >= 0.950 | 0.989 | yes |
| Maximum predicted-class fraction | <= 0.400 | 0.233 | yes |
| Maximum exact numeric tie fraction | <= 0.050 | 0.000 | yes |

All eight checks passed in the single locked run. Together with the strong OB
and SS development results, this supplies independent third-system evidence
that the empirical-tail service-localisation rule generalises across Online
Boutique, Sock Shop and Train Ticket under the stated RE2 setup.

The fixed logistic reference emitted convergence warnings in three folds after
reaching its locked 2,000-iteration limit. The run completed and was not
repeated or retuned. This does not affect the candidate's computation, which
does not use logistic optimisation, but it is a limitation when interpreting
the size of the candidate-reference difference.

## Reproducibility

- Protocol:
  `docs/experiments/2026-08-30-rcaeval-re2-tt-service-delta-v2-confirmation-protocol.md`
- Locked specification:
  `experiment/configs/rcaeval-re2-tt-service-delta-v2-confirmation-spec.json`
  (`1cae996e81bc078476e836b04323ef1170c13e3496ffeb358a6a3f1f6fd4a516`)
- Incident JSONL:
  `data/processed/incidents/rcaeval-re2-tt-metrics.jsonl`
  (`f5dbb8a3d69ecb2f869ecf650db8d8f7704eafd4aee630f7956def92fb1a7a7d`)
- Result:
  `data/results/rcaeval-re2-tt-service-delta-v2-confirmation.json`
  (`6a7fcef78d2aca40a2669b67155625b6935a1f32dd4ca987b5876635d6bd62f2`)
- Figure:
  `data/results/figures/rcaeval-re2-tt-service-delta-v2-confirmation.png`
  (`30fe4fbd342ba4ea66243a835df567cca5804a39b8fb9afc82318e78d075fc61`)

## Claim boundary and next step

This confirms an algorithmic scoring rule, not transfer of one pre-trained
model, causal root-cause identification, or production performance. The next
step should be thesis integration and a fixed explanatory ablation of why the
Top-2 empirical tail works; RE2-TT must remain frozen rather than becoming a
new tuning set.
