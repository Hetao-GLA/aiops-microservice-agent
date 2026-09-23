# Frozen-model contribution audit: v1 to v3 - 30 August 2026

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

## Question

Why did the v1 logs-plus-metrics model classify all ten v3 database failures
as service stoppage while the independently frozen logs-only model remained
correct?

This is a post-hoc mechanism audit. It does not fit a model, change a
transformer, select a threshold or tune a hyperparameter.

## Method

The audit loads the locally frozen v1 pipeline and verifies its model, manifest,
source and dependency hashes. It then applies the already fitted TF-IDF,
`DictVectorizer`, `StandardScaler` and logistic-regression classifier to v3.

For every v3 database incident, the service-stop-versus-database decision
margin is reconstructed exactly:

`full margin = intercept + log contribution + metric contribution`

Positive values favour the observed wrong service-stop class; negative values
favour the correct database class. The maximum numerical reconstruction error
is 0.0 at the stored precision.

Metric-shift statistics compare v3 with the original v1 training values. All
feature ranges and scaler values come from the frozen training data. No v3
values are used to refit or normalise anything.

## Branch-level result

| Component | Mean margin | Minimum | Maximum | Direction |
|---|---:|---:|---:|---|
| Intercept | 0.000 | — | — | Neutral |
| Logs | -0.253 | -0.259 | -0.250 | Correct database class |
| Metrics | +7.043 | +5.586 | +8.172 | Wrong service-stop class |
| Full model | +6.791 | +5.327 | +7.919 | Wrong service-stop class |

Without the intercept, the log branch prefers `database_connection_failure`
for all ten database incidents. The metric branch prefers `service_stopped`
for all ten. The full classifier therefore follows the metric branch on every
case.

The key conclusion is that useful log evidence is present, but its mean margin
of -0.253 is overwhelmed by a metric margin approximately 28 times larger in
the opposite direction.

## Distribution shift

Of the 45 metric features, 33 have at least one v3 value outside their global
v1 training range. The strongest database-class shifts include:

| Metric | v1 DB mean | v3 DB mean | Class-range violations | Standardised shift |
|---|---:|---:|---:|---:|
| Successful requests, pre mean | 4.94 | 19.34 | 10/10 | +72.22 SD |
| Total requests, pre mean | 4.94 | 19.34 | 10/10 | +72.22 SD |
| Total requests, post mean | 2.79 | 8.31 | 10/10 | +43.27 SD |
| Total requests, mean delta | -2.15 | -11.03 | 10/10 | -40.60 SD |
| Successful requests, mean delta | -2.73 | -12.01 | 10/10 | -39.37 SD |
| Successful requests, post mean | 2.20 | 7.32 | 10/10 | +34.63 SD |

The 0.25-second workload interval produces approximately four times as many
requests as the one-second v1 condition. Absolute count features therefore
encode workload intensity as well as failure behaviour.

## Largest changes in wrong-class margin

| Metric | v3 minus v1 margin contribution |
|---|---:|
| Successful requests, pre mean | +6.05 |
| Total requests, pre mean | +6.05 |
| Successful requests, mean delta | -4.79 |
| Successful requests, post mean | +2.96 |
| Order-metrics pre latency | +1.24 |
| Database-health pre latency | +0.64 |
| Liveness pre latency | +0.47 |

Some delta features push back toward the correct database class, but they do
not cancel the two duplicated absolute pre-count contributions and the
post-count contribution. The combined metric margin remains strongly positive.

## Interpretation

The failure mechanism is not missing log evidence. It is an extrapolation
failure caused primarily by load-dependent absolute-count features, sparse
scaling without centring or clipping, and unconstrained linear addition of the
metric and log logits.

The two pre-count features are also near duplicates. Their equal positive
contributions effectively count the same workload signal twice. Logistic
regression probabilities become close to 1.0 because the decision margin is
large, even though the class is wrong. Probability magnitude is therefore not
an out-of-distribution detector.

This decomposition is exact for the frozen linear model, but it is not a causal
intervention on the production system. It identifies how the model computed
its answer, not what would happen after retraining with a changed feature set.

## Constraints for the next model

The next model specification should be written before fitting and should:

1. remove or normalise absolute request-volume features so that failure
   evidence is separated from workload intensity;
2. avoid duplicate count signals such as successful and total requests when
   they are equivalent in the healthy pre-window;
3. include a training-range or robust-distance gate for metric inputs;
4. fall back to logs-only or abstain when metrics are outside the supported
   distribution;
5. cap branch influence or use a gated fusion design so one shifted modality
   cannot overwhelm consistent evidence from another;
6. report abstention/coverage as well as classification F1;
7. treat v3 as consumed during model design and use a new, prespecified v4
   dataset for confirmation.

Clipping the v3 values, removing features and then reporting improved v3
accuracy would only be post-hoc model development. It can be used as a
diagnostic pilot, but not as the final confirmatory result.

## Artifacts

- Audit implementation:
  `experiment/frozen_contribution_audit.py`
- Machine-readable audit:
  `data/results/local-frozen-contribution-audit-v1-to-v3.json`
- Figure:
  `data/results/figures/local-frozen-contribution-audit-v1-to-v3.png`
- Frozen model:
  `data/models/local-v1-frozen/models.joblib`
- v3 frozen-model result:
  `data/results/local-frozen-holdout-v3.json`
