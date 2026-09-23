# RCAEval service-delta v2 development result

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

Date: 2026-08-30  
Status: completed development on RE2-OB and RE2-SS; not confirmation evidence.

## Question

Can the v1 robust service-delta scorer's hard-cap saturation and tie problem be
removed without adding a deep model? Two candidates were fixed before this
run: an empirical-tail Top-2 service score and a 0.5/0.5 full/delta logistic
soft vote. Stored full-metric logistic predictions were the reference.

## Results

| System and design | Full-metric LR | Empirical-tail Top-2 | Fixed soft vote |
|---|---:|---:|---:|
| RE2-OB, repetition held out | 0.785 | **0.956** | 0.783 |
| RE2-OB, fault type held out | 0.856 | **0.945** | 0.887 |
| RE2-SS, repetition held out | 0.849 | **0.924** | 0.874 |
| RE2-SS, fault type held out | 0.841 | **0.945** | 0.864 |

Values are out-of-fold Macro F1. Both candidates passed every locked promotion
check. The empirical-tail candidate was selected by the pre-registered order
because it passed first and fits no target coefficients.

Its observed gate values were:

- mean Macro F1 difference from reference: +0.109;
- worst system/design difference: +0.074;
- minimum system/design Macro F1: 0.924;
- minimum repetition-fold Macro F1: 0.810;
- mean held-out-delay difference from reference: +0.237;
- minimum Top-3 accuracy: 1.000;
- maximum predicted-class fraction: 0.256;
- maximum exact numeric tie fraction: 0.000.

The v1 failure mechanism is therefore resolved on both development systems:
there is no hard score cap, no exact numeric tie, and no dominant predicted
class. This result promoted only the empirical-tail candidate to an untouched
third-system test; it did not itself establish confirmation.

## Reproducibility

- Protocol:
  `docs/experiments/2026-08-30-rcaeval-service-delta-v2-development-protocol.md`
- Locked specification:
  `experiment/configs/rcaeval-service-delta-v2-development-spec.json`
  (`8b879ff7c4da4c06ea0836d58bf40af06e3f15de2a259ba8ba1db5fa538e3752`)
- Result:
  `data/results/rcaeval-service-delta-v2-development.json`
  (`3e3c8356ab9c70ebe0d16a876a4e2d406c716387c478649a5a98ab0ae389aeed`)
- Figure:
  `data/results/figures/rcaeval-service-delta-v2-development.png`
  (`9761f9e0feaf6f46aa639959ae0dfdaace1d02a532a75a41c40f8e53ff93ae0d`)

## Interpretation limit

OB and SS both informed candidate v2, so neither is an independent test. The
score assumes service-qualified metric names and a known candidate-service
set. It uses no traces and is not directly comparable with the official
RCAEval metric-level Avg@k task.
