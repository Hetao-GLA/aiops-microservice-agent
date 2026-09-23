# RCAEval RE2-OB public baseline v1

> Archived research document. Referenced datasets, models, JSON results and
> figures are not included in this source-only GitHub checkout. Read
> [data and reproduction requirements](../REPRODUCIBILITY.md) before running
> historical commands. Dated "next steps" describe the original research stage.

## Outcome

The project now has a complete independent public-data experiment rather than
only a six-case importer pilot. All 90 RCAEval `RE2-OB` cases were downloaded,
converted to the common incident schema and evaluated under the protocol that
was locked before the full subset was downloaded.

The main result is that the public task is metric-led, unlike the ceiling-level
local classifier. In repetition-held-out evaluation, metrics-only achieved
Macro F1 **0.784922**, logs plus metrics achieved **0.775312**, and logs-only
achieved **0.532444**. Fusion therefore did not improve the primary result over
metrics alone.

In the separate fault-type-held-out evaluation, logs plus metrics achieved
Macro F1 **0.866647**, metrics-only **0.855535**, and logs-only **0.536543**.
The small fusion gain in this second design does not establish a general fusion
advantage because the direction reverses in the primary design.

## Protocol and data integrity

The protocol was frozen in
`experiment/configs/rcaeval-re2-ob-public-baseline-spec.json` before the full
telemetry download. Its SHA-256 is
`574751ca0c032965865ebcce6618e7098237b8836e4d87baa4108a12953a3439`.

The selected data contain:

- 90 Online Boutique incidents;
- five root-cause services with 18 incidents each;
- six fault types with 15 incidents each;
- exactly three repetitions of every service/fault pair;
- 1,888,891 log rows inside the fixed incident windows;
- 258 metric features in the union of all pre/post summaries;
- logs and metrics only, with no traces.

The 270 downloaded telemetry files occupy 43,202,139 bytes. Their manifest is
`data/external/rcaeval/re2-ob/subset-manifest.json`, SHA-256
`f24974a1c60104e7b0bea8c483a3cf8e4c6de8b4026b8e84795ccc0fc8c2ae59`.
There were no leftover partial downloads.

The common-schema JSONL contains 90 unique incident IDs and occupies
106,260,301 bytes. Its SHA-256 is
`d234d0682fcdc19594d4172904a7edb79842c93dd012bf82eaf427beb952d8c5`.
All public labels remain separate from the local three-fault label space.

## Fixed models

Three conventional models were evaluated with no hyperparameter search:

1. logs-only TF-IDF unigram/bigram features plus logistic regression;
2. metrics-only pre mean, post mean and mean-delta features plus logistic
   regression;
3. the union of the same log and metric feature blocks plus logistic
   regression.

Every vectoriser and scaler was fitted only on the training portion of its
fold. Case names, metadata fault labels and target fields were excluded from
model input. Telemetry source identity, such as a container name in a log or a
service-qualified metric name, was retained because location is an intended
root-cause-analysis signal.

## Results

| Evaluation design | Model | Top-1 accuracy | Macro F1 | Top-3 accuracy |
| --- | --- | ---: | ---: | ---: |
| Repetition held out | Logs only | 0.522222 | 0.532444 | 0.866667 |
| Repetition held out | Metrics only | **0.777778** | **0.784922** | **0.955556** |
| Repetition held out | Logs + metrics | 0.766667 | 0.775312 | 0.955556 |
| Fault type held out | Logs only | 0.544444 | 0.536543 | 0.855556 |
| Fault type held out | Metrics only | 0.855556 | 0.855535 | 0.988889 |
| Fault type held out | Logs + metrics | **0.866667** | **0.866647** | **0.988889** |

For repetition-held-out evaluation, fusion minus logs-only Macro F1 is
**+0.242868**, while fusion minus metrics-only is **-0.009610**. For
fault-type-held-out evaluation, the corresponding differences are
**+0.330104** and **+0.011112**.

## Fold diagnostics

The primary evaluation exposes strong repetition variation:

| Held-out repetition | Logs Macro F1 | Metrics Macro F1 | Fusion Macro F1 |
| ---: | ---: | ---: | ---: |
| 1 | 0.292315 | 0.525556 | 0.525556 |
| 2 | 0.592381 | 0.863170 | 0.830070 |
| 3 | 0.635470 | 0.966434 | 0.966434 |

The first repetition is consistently harder for all modalities. Simple volume
checks do not explain this: the three repetitions have similar downloaded
sizes, incident-window log counts and metric-row counts. The result should be
treated as acquisition/session shift that requires further feature-level
audit, not averaged away.

The fault-type-held-out fusion folds were:

| Held-out fault | Top-1 accuracy | Macro F1 |
| --- | ---: | ---: |
| CPU | 0.866667 | 0.864762 |
| Delay | 0.600000 | 0.538571 |
| Disk | 0.933333 | 0.931429 |
| Loss | 0.866667 | 0.862857 |
| Memory | 1.000000 | 1.000000 |
| Socket | 0.933333 | 0.931429 |

Delay is the clear unseen-fault weakness. The apparently higher aggregate
fault-held-out score does not mean that design is universally easier or
harder: it withholds a fault mechanism but allows all repetitions to appear on
both sides, while the primary design withholds a complete repetition and
therefore exposes session shift.

## Interpretation

1. The independent public benchmark breaks the local logs-only ceiling:
   logs-only Macro F1 is about 0.53 rather than 1.00.
2. Metrics carry most of the useful public root-cause signal. This supports
   retaining the metric branch, while also confirming that its shift behaviour
   must be controlled.
3. Simple feature concatenation is not consistently better than metrics alone.
   The measured fusion difference is small and changes sign across designs.
4. Repetition 1 and unseen delay faults are the next concrete failure modes to
   investigate.
5. These results do not yet justify replacing the transparent baselines with a
   deep model. The immediate research value is in explaining repetition shift,
   auditing feature stability and comparing with an official RCA baseline.

## Validity limits

- The experiment predicts one of five root-cause services. RCAEval's official
  methods can rank more granular root-cause indicators, so these scores are not
  directly comparable with official Avg@k tables.
- Only Online Boutique is tested; cross-system transfer is unmeasured.
- Traces are excluded.
- Training and test cases share the same public benchmark environment. This is
  external to the local platform but not a deployment-domain test.
- The fault-type-held-out design shares repetitions between train and test;
  the repetition-held-out design shares fault types. Both results are needed.

RCAEval source: <https://github.com/phamquiluan/RCAEval>

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m experiment.data_import.download_rcaeval_subset

.\.venv\Scripts\python.exe -m experiment.data_engineering.build_rcaeval_incidents `
    --root data\external\rcaeval\re2-ob `
    --output data\processed\incidents\rcaeval-re2-ob.jsonl `
    --manifest data\processed\incidents\rcaeval-re2-ob-manifest.json

.\.venv\Scripts\python.exe -m experiment.rcaeval_public_baseline
```

The protected evaluator refuses to overwrite its result or figure unless
`--overwrite` is explicitly supplied.

## Artifacts

- Result: `data/results/rcaeval-re2-ob-public-baseline-v1.json`, SHA-256
  `9c7f8095e2d3cacb0c86d32f70e4375014d31d5338d516c1598e145147e7fb4d`.
- Figure: `data/results/figures/rcaeval-re2-ob-public-baseline-v1.png`,
  SHA-256
  `9e4c428ef27889c51b889c4e1670a9d1401783f21592db0bc880b0982b578188`.
- Protocol:
  `docs/experiments/2026-08-30-rcaeval-re2-ob-public-baseline-protocol.md`.
