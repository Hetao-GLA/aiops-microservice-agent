# Data, reproduction and verification boundaries

[Project overview](../README.md) · [Documentation index](README.md) ·
[Local demonstration](GETTING_STARTED.md)

## What is available in this GitHub checkout?

| Material | Included? | Consequence |
| --- | --- | --- |
| Application, Agent, experiment code and tests | Yes | The local deterministic demo can be set up without a research dataset |
| Experiment specifications and campaign configurations | Yes | Selection rules and evaluation settings can be inspected |
| Selected research reports and protocols | Yes | The reported results, failures and claim boundaries can be read |
| Raw public and local telemetry | No | Data must be acquired or generated before conversion/evaluation |
| Processed incident JSONL and provenance manifests | No | These must be restored or regenerated; they are not supplied by a clone |
| Frozen model bundles and generated results/figures | No | Historical frozen-model verification requires additional artifacts |
| Dissertation, recordings and submission package | No | These are separate academic deliverables, not runtime dependencies |

The earlier academic submission ZIP and this GitHub source checkout are not
the same artifact. Do not assume files included in a submission archive are
also present here. There is currently no downloadable, versioned archive of
all historical research artifacts attached to this repository.

## Three distinct activities

1. **Run the software demonstration.** Follow the getting-started guide. The
   Agent uses current local health evidence, deterministic rules and runbooks;
   it does not load the trained research classifiers.
2. **Run a new experiment.** Generate/acquire data, record its provenance and
   versions, and use new output paths. This tests the method in your environment;
   it does not by itself reproduce the original numerical result.
3. **Verify the historical frozen result.** Restore the exact original inputs,
   model/development artifacts and compatible environment, and pass all recorded
   hashes and provenance checks. Source code alone is insufficient.

Commands in dated reports describe the original research workspace. A path
such as `data/results/example.json` or a SHA-256 in a report is an artifact
identifier, not a promise that the file is distributed in Git.

## Public data source and task

The public study uses selected cases from the
[RCAEval benchmark](https://github.com/phamquiluan/RCAEval), distributed through
[the RCAEval dataset repository](https://huggingface.co/datasets/phamquiluan/RCAEval).
Consult the upstream dataset documentation and terms before downloading,
redistributing or reusing its data. Dataset licensing does not automatically
license this project's own code.

The three 90-case subsets are RE2-OB (Online Boutique), RE2-SS (Sock Shop) and
RE2-TT (Train Ticket). OB/SS acquisition uses logs and metrics; the fixed TT
confirmation uses metrics and injection timestamps only. Public targets are
root-cause services, not the local three fault labels. Traces are not inputs to
these experiments.

The metadata-only compatibility audit also considered LO2v2 and AIOps Challenge
2020. They were not added to the principal 390-case evaluation; see the
[historical compatibility decision](data/2026-08-20-public-dataset-compatibility.md).

## Example: acquire and evaluate a new RE2-OB baseline

Use a fresh research checkout, not an existing frozen evidence directory.
The following commands download data and write derived outputs. They were
checked against the CLI source; publishing these docs did not redownload
telemetry or rerun the experiment. Run each command only after the previous one
succeeds. Network access, disk space and the upstream files are required.

First install the analysis dependencies and fetch the metadata index if absent:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
New-Item -ItemType Directory -Force -Path data/external/rcaeval | Out-Null
if (-not (Test-Path -LiteralPath 'data/external/rcaeval/cases.parquet')) {
    Invoke-WebRequest -Uri 'https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/main/cases.parquet?download=true' -OutFile 'data/external/rcaeval/cases.parquet'
}
```

The project's alternative `experiment.data_import.download_public_indexes`
command fetches both RCAEval and LO2v2 metadata; LO2v2 is unnecessary for this
OB-only example.

Check the case selection without downloading the case telemetry:

```powershell
.\.venv\Scripts\python.exe -m experiment.data_import.download_rcaeval_subset --spec experiment/configs/rcaeval-re2-ob-public-baseline-spec.json --root data/external/rcaeval/re2-ob --dry-run
```

The locked selection should contain 90 cases. If the index no longer matches
the recorded selection or hash, investigate the upstream version rather than
editing the historical specification to make a confirmation pass.

Download the selected telemetry, then build incident records:

```powershell
.\.venv\Scripts\python.exe -m experiment.data_import.download_rcaeval_subset --spec experiment/configs/rcaeval-re2-ob-public-baseline-spec.json --root data/external/rcaeval/re2-ob
.\.venv\Scripts\python.exe -m experiment.data_engineering.build_rcaeval_incidents --root data/external/rcaeval/re2-ob --output data/processed/incidents/rcaeval-re2-ob-reproduction.jsonl --manifest data/processed/incidents/rcaeval-re2-ob-reproduction-manifest.json
.\.venv\Scripts\python.exe -m experiment.rcaeval_public_baseline --input data/processed/incidents/rcaeval-re2-ob-reproduction.jsonl --output data/results/reproduction/rcaeval-re2-ob-baseline.json --figure data/results/reproduction/rcaeval-re2-ob-baseline.png
```

The converter writes its output paths; choose unused paths if running again.
The evaluator refuses existing results unless explicitly told to overwrite.
Prefer a new run directory instead of using `--overwrite`. The upstream
download URLs use `main`, so identical data bytes cannot be assumed indefinitely.
Compare manifests and SHA-256 values before describing a run as an exact
reproduction.

## Why the SS/TT confirmation commands need more than telemetry

The locked SS and TT evaluators check an evidence chain, not just a feature
matrix. They require:

- The matching locked specification, metadata index and source-code hashes.
- Frozen development-result JSON with the expected hash and selected candidate.
- Every raw telemetry file listed in the acquisition manifest.
- Processed incident JSONL and its conversion manifest.

Both `validate_acquisition()` implementations hash the raw files before
evaluation. **Processed JSONL alone is not sufficient.** Raw public data can be
downloaded with the subset importer using the SS/TT specs, but downloading it
does not restore the omitted historical development results.

The code is available in
[SS confirmation](../experiment/rcaeval_re2ss_confirmation.py) and
[TT confirmation](../experiment/rcaeval_re2tt_confirmation.py).
Do not disable these checks, edit expected hashes or present a fresh tuning run
as the original locked confirmation. Reproducing the exact result requires the
original evidence chain, or a separately documented new experimental protocol.

## Local campaigns and frozen models

Local incidents were collected using the Docker testbed. The 60-incident v1
campaign, 30-incident v3 holdout and 30-incident v4 confirmation are not included
as raw or processed data here. A new campaign intentionally injects faults;
review its configuration and run it only in an isolated local demo environment.

The frozen local evaluators verify Python, scikit-learn, NumPy, SciPy and joblib
versions against the bundle manifest, along with source, feature, model and
provenance checks. The broad ranges in `requirements-data.txt` are not a frozen
environment lock. Byte-level source checks can also be affected by line-ending
conversion. Restore the original bundle/environment for historical verification;
do not remove checks to accommodate a newly installed environment.

Model bundles use joblib/pickle: never load a bundle from an untrusted source.
Hash verification can detect changes but cannot make an untrusted pickle safe.

## Verification and publication checks

The submission-era verification recorded 129 passing automated tests and a
successful Agent startup from an extracted submission copy using the existing
Python environment. Those checks were not a clean-machine dependency install,
a new Docker-image build or a complete live recovery study. They do not imply
that the missing experimental artifacts are available in a fresh Git clone.

For this documentation update, local links are checked against the files
eligible for publication, not merely against every file on the author's disk.
The code, frozen experiment configurations and reported numerical results are
not changed. No confirmation experiment or fault injection is run as part of
documentation cleanup.

Before committing further material, review `git diff --cached --name-only`.
Do not force-add ignored data, keys, `.env` files, thesis drafts or submission
archives. A limited content scan reduces accidental disclosure risk; it is not
a guarantee that arbitrary future files are safe to publish.
