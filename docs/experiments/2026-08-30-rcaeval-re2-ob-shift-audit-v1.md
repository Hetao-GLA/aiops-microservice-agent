# RCAEval RE2-OB frozen-prediction shift audit v1

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

## Outcome

The audit separates the two public-baseline weaknesses:

1. Repetition 1 is primarily associated with **metric baseline/session
   shift**, especially in pre-injection latency levels.
2. The unseen `delay` fault is primarily associated with **root-cause signal
   ambiguity and propagation**, not unusually large out-of-range metric values
   or unseen log templates.

The audit reused the frozen public-baseline predictions. It did not refit a
classifier, select a threshold or change a hyperparameter.

## Frozen inputs and integrity

The audit specification was locked before computing feature-level statistics.
Its SHA-256 is
`c36fa0419b04bf0ec4425e8ca3eb800fb407f3c26c61eac285b7fc418f6898a6`.

The audit verified these inputs before running:

- incident JSONL SHA-256:
  `d234d0682fcdc19594d4172904a7edb79842c93dd012bf82eaf427beb952d8c5`;
- frozen public result SHA-256:
  `9c7f8095e2d3cacb0c86d32f70e4375014d31d5338d516c1598e145147e7fb4d`;
- public baseline specification SHA-256:
  `574751ca0c032965865ebcce6618e7098237b8836e4d87baa4108a12953a3439`.

The analysis covered all 90 incidents, 258 metric features and approximately
1.89 million windowed log rows.

## Repetition-1 diagnosis

| Diagnostic | Repetition 1 | Repetitions 2/3 | Difference |
| --- | ---: | ---: | ---: |
| Metric range-breach fraction | 0.051421 | 0.017507 | +0.033914 |
| Mean absolute training z | 0.755607 | 0.496256 | +0.259351 |
| Mean absolute robust z | 1.689098 | 0.999139 | +0.689959 |
| Unseen log-line fraction | 0.001175 | 0.001499 | -0.000324 |
| Unique unseen-template fraction | 0.013851 | 0.015690 | -0.001839 |
| Log-source-proportion L1 distance | 0.032900 | 0.051914 | -0.019014 |

Repetition 1 has approximately 2.94 times the metric range-breach fraction of
the other held-out repetitions. It does not have more unseen log content or a
more unusual service-source mixture by the fixed measures.

The metric shift is concentrated in absolute pre-injection levels:

| Feature block | Range breach | Mean absolute z |
| --- | ---: | ---: |
| `pre_mean` | 0.087985 | 1.123535 |
| `post_mean` | 0.031008 | 0.623058 |
| `mean_delta` | 0.035271 | 0.520230 |

The largest shifted features are baseline latency summaries, including
`frontend_latency-50__pre_mean`,
`currencyservice_latency-50__pre_mean`,
`frontend_latency-90__pre_mean` and checkout-service pre-window latency.
This matches the earlier local finding that absolute operating levels are
fragile under workload/session change.

The frozen metrics-only errors are also descriptively associated with greater
metric shift. Correct predictions have mean range-breach fraction 0.021096 and
mean absolute z 0.513860; wrong predictions have 0.055814 and 0.823668. For the
frozen fusion model, the corresponding correct/wrong pairs are
0.021346/0.053341 and 0.515293/0.804207. These are associations, not causal
effect estimates.

## Delay diagnosis

| Diagnostic | Held-out delay | Other held-out faults | Difference |
| --- | ---: | ---: | ---: |
| Metric range-breach fraction | 0.012403 | 0.030905 | -0.018502 |
| Mean absolute training z | 0.478067 | 0.625160 | -0.147093 |
| Mean absolute robust z | 0.789366 | 1.213693 | -0.424327 |
| Unseen log-line fraction | 0.000900 | 0.001442 | -0.000542 |
| Log-source-proportion L1 distance | 0.097823 | 0.035799 | +0.062024 |

Delay is not the most out-of-range fault. Its metric and log novelty measures
are lower than the other fault folds. Its log-source mixture, however, is much
farther from the training mean.

The training-normalised, service-grouped `mean_delta` diagnostic ranks the
true service first for only 10 of 15 delay incidents (hit@1 0.666667), although
the true service is within the top three for all 15. The largest delay shifts
are often payment- and shipping-service latency features. This is consistent
with delay effects propagating through calls and making the injected service
less separable from downstream services.

For comparison, the same unsupervised metric-signal check has hit@1 0.933 for
CPU and 1.000 for disk, memory and socket. Its overall hit@1 is 0.900 under
repetition-held-out folds and 0.889 under fault-type-held-out folds; hit@3 is
0.989 and 1.000 respectively. These values are diagnostic rankings, not a new
trained baseline result.

## Log diagnosis

Line-weighted unseen-template fractions remain below 0.004 in every fold. Most
listed unseen templates are low-frequency recommendation product-ID list
variants rather than clear fault signatures. The simple template-normalisation
audit therefore does not explain the large logs-only error rate. A future text
analysis should focus on discriminative signal strength and service attribution
rather than vocabulary novelty alone.

## Decision for the next model iteration

The next public-data model experiment should remain conventional and
interpretable:

1. test a delta-only metric logistic-regression candidate that excludes
   `pre_mean` and `post_mean` absolute levels;
2. test a service-grouped `mean_delta` score with training-only robust
   normalisation;
3. retain the existing metrics-only model as the fixed reference;
4. treat development on RE2-OB as exploratory and require confirmation on a
   new public system or suite before claiming improvement;
5. report delay separately because aggregate accuracy hides its propagation
   ambiguity.

The evidence still does not require a deep-learning model. The present failure
modes concern feature stability and causal localisation structure, which a
larger model would not automatically solve.

## Limitations

- The audit is descriptive and based on one frozen public benchmark result.
- Correct/wrong comparisons are not statistically confirmatory or causal.
- Template normalisation may merge distinct messages and does not measure
  semantic novelty.
- Service-qualified metric names are used by the signal-ranking diagnostic;
  this is appropriate for localisation but is not an official RCAEval score.

## Reproduction and artifacts

```powershell
.\.venv\Scripts\python.exe -m experiment.rcaeval_shift_audit
```

- Result: `data/results/rcaeval-re2-ob-shift-audit-v1.json`, SHA-256
  `013ce2ad72a1f0a9d204489393fa0b8eadf68380a3f3758654a90bb3e6dd90ce`.
- Figure: `data/results/figures/rcaeval-re2-ob-shift-audit-v1.png`, SHA-256
  `297452b33b2fbffc2c481823520a725ad14dca98fd109f33cab7202dcffab550`.
- Protocol:
  `docs/experiments/2026-08-30-rcaeval-re2-ob-shift-audit-protocol.md`.
