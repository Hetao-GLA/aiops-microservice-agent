# RCAEval RE2-OB delta candidate development v1

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

## Outcome

Removing absolute `pre_mean` and `post_mean` metric levels substantially
improves repetition stability. The pre-registered promotion gate selected
`robust_service_delta_score` for confirmation on a different public system.

This is a development result on the same RE2-OB data that motivated the
candidates. It is not independent confirmation.

## Locked design

The candidate protocol and thresholds were frozen before generating candidate
predictions. The specification SHA-256 is
`f35e8c045b9c8a3a210365fe83c7f66b24382ea51545ec879b8bddc8a1fd6b20`.

The experiment reused the frozen metrics-only predictions as its reference and
evaluated three candidates on the identical folds:

1. delta-only logistic regression;
2. logs plus delta-only logistic regression;
3. a robust service-grouped delta score using training median/IQR and no fitted
   target weights.

Only 86 `__mean_delta` features were available to the candidates. All 172
absolute pre/post features were excluded. There was no hyperparameter search.

## Aggregate results

| Model | Repetition-held-out Macro F1 | Fault-type-held-out Macro F1 |
| --- | ---: | ---: |
| Frozen metrics-only reference | 0.784922 | 0.855535 |
| Delta-only LR | **0.933849** | 0.855712 |
| Logs + delta LR | **0.933849** | 0.888307 |
| Robust service delta | 0.922000 | **0.922000** |

Relative to the frozen reference, repetition-held-out Macro F1 changes by
+0.148927 for both logistic-regression candidates and +0.137078 for the robust
service score. Fault-type-held-out changes are +0.000177, +0.032772 and
+0.066465 respectively.

The robust candidate has Top-1 accuracy 0.922222 and Top-3 accuracy 1.000 in
both designs. It makes seven out-of-fold errors in each design, compared with
20 repetition-held-out and 13 fault-held-out errors for the frozen reference.

## Focus results

| Model | Repetition-1 Macro F1 | Held-out-delay Macro F1 |
| --- | ---: | ---: |
| Frozen metrics-only reference | 0.525556 | 0.457143 |
| Delta-only LR | **0.933100** | 0.423810 |
| Logs + delta LR | **0.933100** | 0.538571 |
| Robust service delta | 0.897862 | **0.664762** |

The delta-only LR confirms that absolute levels caused most repetition-1
instability, but it fails the pre-registered delay requirement. Adding logs
does not change any repetition-held-out prediction, although it improves
unseen-fault performance, particularly delay.

The robust service score sacrifices a small amount of repetition performance
relative to delta LR but is much stronger for unseen delay. Its fault-held-out
errors comprise four delay, two loss and one socket incident; it is perfect on
the held-out CPU, disk and memory folds.

## Promotion decision

The fixed requirements were:

- repetition-held-out Macro F1 at least 0.804922;
- repetition-1 Macro F1 at least 0.650000;
- fault-type-held-out Macro F1 at least 0.835535;
- delay Macro F1 at least 0.457143.

| Candidate | Repetition overall | Repetition 1 | Fault overall | Delay | Pass |
| --- | ---: | ---: | ---: | ---: | --- |
| Delta-only LR | 0.933849 | 0.933100 | 0.855712 | 0.423810 | No |
| Robust service delta | 0.922000 | 0.897862 | 0.922000 | 0.664762 | **Yes** |
| Logs + delta LR | 0.933849 | 0.933100 | 0.888307 | 0.538571 | Yes |

The selection order favoured delta-only LR first, robust service delta second
and logs plus delta third. Delta-only LR failed the delay threshold, so the
robust service delta score is selected.

The choice also has a methodological advantage: it fits only training-fold
medians and interquartile ranges and does not learn target-specific
coefficients. Its service structure matches the root-cause-localisation task
directly.

## Interpretation

1. Absolute operating levels were harmful under public repetition/session
   shift, just as they were in the earlier local workload-shift experiment.
2. Mean changes retain enough information for strong service localisation.
3. Text features help some unseen faults but add no value in the
   repetition-held-out design.
4. A service-structured robust score is more consistent than flat logistic
   regression when delay effects propagate across services.
5. Deep learning remains unnecessary at this stage: the interpretable
   candidate reaches Macro F1 0.922 and directly addresses the observed shift.

## Next step

Freeze the robust scoring implementation and confirm it without modification
on `RE2-SS` or another RCAEval system. The confirmation must compare it with
the same fixed full-metric logistic reference, report repetition and fault
holdouts separately, and retain a dedicated delay result.

## Limitations

- All candidates were designed after observing RE2-OB, so these scores are
  optimistic development evidence.
- The robust method assumes service-qualified metric names.
- It selects among five known service classes and does not discover an unseen
  service.
- The task remains service-level localisation without traces and is not
  directly comparable to official metric-level RCAEval Avg@k results.

## Reproduction and artifacts

```powershell
.\.venv\Scripts\python.exe -m experiment.rcaeval_delta_candidates
```

- Result: `data/results/rcaeval-re2-ob-delta-candidates-v1.json`, SHA-256
  `652b1ea2fe7fd39d1dc1d1f466d2fd97c90626e3034633f253f272f90c017ab8`.
- Figure: `data/results/figures/rcaeval-re2-ob-delta-candidates-v1.png`,
  SHA-256
  `5395950bd97f85ef8acf2edcdf99eb490e1d81e7e7040719e3979500138f8a8d`.
- Protocol:
  `docs/experiments/2026-08-30-rcaeval-re2-ob-delta-candidates-protocol.md`.
