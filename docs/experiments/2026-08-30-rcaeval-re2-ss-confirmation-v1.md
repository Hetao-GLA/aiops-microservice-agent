# RCAEval RE2-SS robust-delta confirmation v1

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

## Decision

**Cross-system confirmation failed.**

The robust service-grouped delta score selected on Online Boutique did not
reproduce its Macro F1 advantage on the independent Sock Shop system. The
pre-registered gate was applied unchanged; six of nine requirements failed.

The failure is informative rather than inconclusive. The candidate retains
high Top-3 accuracy, but its hard score cap and maximum aggregation create
many Top-1 ties on RE2-SS. Every candidate error occurs in a zero-margin tie.

## Protocol and acquisition integrity

The protocol was locked before RE2-SS telemetry was downloaded. Its SHA-256
is `5907b68a338829f453b55f32e7b1b2a6904d6fc33ffc4c884d72e41180c5c859`.

The confirmation verified the frozen candidate and reference implementation
hashes, the RE2-OB development result, the public index, all 270 downloaded
files and the processed JSONL before prediction.

The independent dataset contains:

- 90 Sock Shop incidents;
- five services with 18 incidents each;
- six faults with 15 incidents each;
- three repetitions of every service/fault pair;
- 101,086,392 downloaded bytes;
- 179,350,101 bytes in the unified incident JSONL;
- no trace modality;
- no mapping to the local three-fault labels.

Key acquisition hashes:

- download manifest:
  `9b4ad00f1d0a5d5d0d71d2ad529c6bae0d973af17cd2bef746929954e20ba90b`;
- processed manifest:
  `fac5044e6b1aa4cb41e0dfbec9481709587019ef8bbb19285b57ecc522348bc5`;
- incident JSONL:
  `c85ce8a488d99d4e2e85e2433eb98e69b3e8624842a117a9421699e327b38eda`.

## Aggregate confirmation results

| Evaluation | Full-metric LR reference | Robust service delta | Difference |
| --- | ---: | ---: | ---: |
| Repetition-held-out Macro F1 | **0.849414** | 0.777857 | -0.071557 |
| Fault-type-held-out Macro F1 | **0.841287** | 0.758406 | -0.082881 |
| Repetition-held-out Top-3 | 0.955556 | **0.977778** | +0.022222 |
| Fault-type-held-out Top-3 | 0.955556 | **1.000000** | +0.044444 |

The candidate makes 20 repetition-held-out and 21 fault-type-held-out errors,
versus 14 and 14 for the reference.

## Fold results

### Repetition held out

| Repetition | Reference Macro F1 | Candidate Macro F1 |
| ---: | ---: | ---: |
| 1 | 0.550380 | **0.569802** |
| 2 | **1.000000** | 0.935065 |
| 3 | **0.966434** | 0.831795 |

Both algorithms find repetition 1 difficult, but the candidate also loses
accuracy on repetitions 2 and 3. Its worst repetition Macro F1 is 0.569802,
below the pre-registered 0.65 requirement.

### Fault type held out

| Fault | Reference Macro F1 | Candidate Macro F1 |
| --- | ---: | ---: |
| CPU | 0.564286 | **0.931429** |
| Delay | **1.000000** | 0.554286 |
| Disk | 0.720000 | **0.813333** |
| Loss | 0.704762 | **0.793333** |
| Memory | **1.000000** | 0.793333 |
| Socket | **1.000000** | 0.673333 |

The candidate improves CPU, disk and loss, but the large delay, memory and
socket regressions dominate the aggregate. Delay passes the candidate's
absolute minimum of 0.55 by a narrow margin, yet is 0.445714 below the
reference and therefore fails the comparative requirement.

## Confirmation gate

| Requirement | Minimum | Observed | Pass |
| --- | ---: | ---: | --- |
| Candidate repetition Macro F1 | 0.800000 | 0.777857 | No |
| Candidate fault Macro F1 | 0.800000 | 0.758406 | No |
| Worst repetition-fold Macro F1 | 0.650000 | 0.569802 | No |
| Candidate delay Macro F1 | 0.550000 | 0.554286 | Yes |
| Candidate repetition Top-3 | 0.950000 | 0.977778 | Yes |
| Candidate fault Top-3 | 0.950000 | 1.000000 | Yes |
| Candidate minus reference, repetition | 0.000000 | -0.071557 | No |
| Candidate minus reference, fault | 0.000000 | -0.082881 | No |
| Candidate minus reference, delay | 0.000000 | -0.445714 | No |

The confirmation decision is therefore
`cross_system_confirmation_failed`.

## Failure mechanism

RE2-SS has eight delta features for each of the five candidate services, so
the failure is not caused by unequal feature counts. It is caused by score
saturation and tie resolution:

- 89 of 90 cases reach the fixed maximum score of 20;
- repetition-held-out evaluation has 26 zero-margin Top-1 ties;
- fault-type-held-out evaluation has 31 zero-margin Top-1 ties;
- all 20 repetition-held-out errors and all 21 fault-held-out errors occur in
  those zero-margin cases;
- the lexicographic tie rule contributes to `carts` being predicted 29 times
  although it has only 18 true cases;
- the true service is still in the Top-3 for 88 of 90 repetition-held-out cases
  and all 90 fault-held-out cases.

The candidate therefore preserves useful candidate-set information but fails
to resolve first place reliably in a different metric distribution. The
combination of maximum aggregation, hard capping and lexicographic tie-breaking
is not cross-system robust.

## Research conclusion

1. The RE2-OB candidate score of 0.922 must remain development evidence and
   must not be presented as a generally validated improvement.
2. The full-metric logistic reference remains the stronger cross-system
   service classifier in this experiment.
3. A future delta scorer must replace saturated maximum scores with a method
   that preserves magnitude/rank information and handles multiple abnormal
   services explicitly.
4. Candidate v2 should be developed as a new version, with new thresholds,
   and confirmed on a third untouched system rather than retested as if this
   RE2-SS result had not occurred.
5. Deep learning is still not the immediate remedy. The observed defect is a
   deterministic scoring and tie-calibration problem.

## Limitations

- This confirms algorithm behaviour across two public systems, not deployment
  transfer of a single pre-trained model.
- Both methods know the five candidate service names in each training fold.
- No traces are used.
- The service-classification scores are not official RCAEval metric-level
  Avg@k scores.

## Reproduction and artifacts

```powershell
.\.venv\Scripts\python.exe -m experiment.rcaeval_re2ss_confirmation
```

- Result:
  `data/results/rcaeval-re2-ss-robust-delta-confirmation-v1.json`, SHA-256
  `7bd84c7cbde7e54a06f47a37decb1ab9211b70a36a59391cf4c21d9b5a967ecc`.
- Figure:
  `data/results/figures/rcaeval-re2-ss-robust-delta-confirmation-v1.png`,
  SHA-256
  `e03c0a7b176e70079d6894b198610e0f1a72f19c4156bfff9534c5a415c02ed4`.
- Protocol:
  `docs/experiments/2026-08-30-rcaeval-re2-ss-confirmation-protocol.md`.
