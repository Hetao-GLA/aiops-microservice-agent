# AIOps Microservice Agent

A research prototype for microservice fault diagnosis, evidence inspection and
**human-approved recovery**. It combines a small Docker-based operations demo
with separate machine-learning experiments.

The research asks a practical question: **when logs and metrics disagree under
workload shift, how can a diagnosis avoid being made worse by unreliable
metrics?** The local experiments investigate selective fallback to log evidence;
the public experiments investigate root-cause service localisation.

**Scope:** this is a local research demonstrator, not a production-ready or
fully autonomous self-healing system. The default Agent uses deterministic
diagnosis and planning; the experimental ML models are not automatically loaded
by the live console.

## Start here

- [Install and run the local demo](docs/GETTING_STARTED.md)
- [Browse the experiment reports and protocols](docs/README.md)
- [Understand data requirements and reproduction limits](docs/REPRODUCIBILITY.md)

This GitHub checkout contains source code, tests, configurations and selected
research documentation. It does **not** contain the dissertation, submission
archive, recordings, API keys, raw datasets, generated results or frozen model
bundles. A fresh clone therefore has no historical Agent incidents or trained
models. See the reproduction guide before running evaluation commands.

## What the software does

- Runs an order-service API, PostgreSQL database, workload generator and
  rule-based detector using Docker Compose.
- Supports three controlled local faults: database disconnection, service stop
  and application HTTP 500.
- Presents health evidence, detections and proposed recovery in a
  browser-based Agent console, with bounded log access available through
  read-only tools.
- Requires explicit approval before executing a recovery action from a local
  allow-list, then checks service health and records an audit trail.
- Provides offline pipelines for TF-IDF/logistic-regression baselines,
  OOD-gated fusion, and RCAEval service-localisation experiments.

The optional language-model planner helps explain evidence and use read-only
tools. It does not replace the trusted diagnosis, approval gate or action
allow-list. The default deterministic mode requires no external API key.

## Architecture and technology

| Component | Implementation | Responsibility |
| --- | --- | --- |
| Demo application | [app/](app/), FastAPI, SQLAlchemy, PostgreSQL | Orders, health checks, metrics and controlled faults |
| Experiment platform | [experiment/](experiment/), Python, Docker Compose | Workloads, fault injection, telemetry, data conversion and evaluation |
| Operations Agent | [ops_agent/](ops_agent/), FastAPI, local runbooks, HTML/JavaScript | Evidence collection, approved recovery and audit records |
| Offline models | scikit-learn, pandas, NumPy/SciPy | TF-IDF logistic regression, guarded fusion and metric-based service ranking |
| Verification | [tests/](tests/), pytest | Application, Agent, data and evaluation checks |

There are two distinct workflows:

```text
Local demo:   evidence -> diagnosis -> recovery proposal -> human approval
                                                      -> action -> health check

Research:    incident data -> grouped evaluation -> frozen candidate
                                               -> independent confirmation
```

## Quick start (Windows PowerShell)

Install Python 3.12 and start Docker Desktop with Linux containers. Clone the
repository into your chosen working directory. If you already cloned it, open
PowerShell in the repository root and skip the first two commands.

```powershell
git clone https://github.com/Hetao-GLA/aiops-microservice-agent.git
Set-Location aiops-microservice-agent
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Before starting Compose, change the local `api.ports` mapping in
`docker-compose.yml` from `"8000:8000"` to `"127.0.0.1:8000:8000"` to restrict
the demo API to this machine. Then start the testbed and Agent:

```powershell
docker compose up -d --build
.\.venv\Scripts\python.exe -m ops_agent
```

Open the Agent console at **http://127.0.0.1:8100/**. The application's API
documentation is at **http://127.0.0.1:8000/docs**. Keep the Agent terminal open;
use another terminal for health checks or a controlled demonstration.

If an existing Compose stack already uses port 8000 or the fixed `aiops-*`
container names, do not start a second copy. Follow the
[detailed setup and demo guide](docs/GETTING_STARTED.md).

**Security:** use a trusted local machine only. The prototype has development
database credentials and management/fault-control endpoints without production
authentication. Compose publishes port 8000 on host interfaces by default.
Do not expose the services through a public server, tunnel or port-forwarding
rule. The approval gate is an application workflow, not an authentication
boundary.

## Research findings

These are the recorded experiment results, not measurements produced by simply
cloning this repository. Local fault classification and public root-cause
service localisation are **different tasks** and must not be pooled into one
accuracy claim.

| Evaluation | Recorded finding | Evidence |
| --- | --- | --- |
| Local baseline: 60 incidents, three balanced fault classes | Logs-only and logs-plus-metrics both reached Macro F1 1.000 in grouped cross-validation | [Baseline report](docs/experiments/2026-08-28-local-ml-baseline-v1.md) |
| Workload-shifted v3 holdout: 30 incidents | Logs-only remained at 1.000; naive fusion fell to 0.556 | [v3 confirmation](docs/experiments/2026-08-30-local-holdout-v3.md) |
| Post-freeze v4 confirmation: 30 new incidents | OOD-gated fusion reached 1.000; ungated robust fusion reached 0.822; the gate used logs-only for 25 of 30 incidents | [v4 confirmation](docs/experiments/2026-08-30-local-holdout-v4-confirmation.md) |
| Train Ticket: 90 cases, repetition/fault-type holdouts | Empirical-tail Top-2 scoring reached 0.956 / 0.945 versus 0.798 / 0.741 for the fixed full-metric logistic reference | [Locked public confirmation](docs/experiments/2026-08-30-rcaeval-re2-tt-service-delta-v2-confirmation.md) |

The principal evidence covers **390 accepted unique incident/case records**:
60 local baseline incidents, 30 v3 incidents, 30 v4 incidents, and 90 cases from
each of Online Boutique, Sock Shop and Train Ticket. Early-window variants and
ablations reuse incidents and are not counted again. Incidents are the
evaluation units, not individual log lines; within-incident observations are
not independent samples.

Online Boutique and Sock Shop informed public-candidate development. Train
Ticket was reserved for one locked confirmation. The historical failure of the
first candidate on Sock Shop is retained in the
[documentation index](docs/README.md), alongside the improved candidate.

### What these results do not establish

- The local gate limited a known harmful fusion failure; it did not show that
  metrics outperform logs alone, and fallback was frequent.
- Public confirmation supports the scoring algorithm under the stated RE2
  service-localisation setup, not transfer of a single pretrained model,
  unknown-service handling or causal identification in production.
- The fixed Train Ticket logistic reference emitted convergence warnings in
  three folds, which limits interpretation of the improvement margin.
- No operator study or manual-versus-Agent recovery-time evaluation has been
  completed. Software tests do not establish operational effectiveness.

## Tests

The full suite also needs the development and data-analysis dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-data.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The archived submission verification recorded 129 passing tests. This is a
historical software-verification result, not an experimental sample count or a
claim that a fresh dependency installation has been tested on every platform.
The Agent tests include mocked recovery tools; passing them does not demonstrate
a live Docker recovery. See [verification boundaries](docs/REPRODUCIBILITY.md).

## Public data and attribution

Public service-localisation experiments use the
[RCAEval benchmark](https://github.com/phamquiluan/RCAEval) and its
[dataset distribution](https://huggingface.co/datasets/phamquiluan/RCAEval).
The project uses selected RE2 cases from Online Boutique, Sock Shop and Train
Ticket; it does not redistribute the full datasets or claim an official
RCAEval leaderboard result.

The [data guide](docs/REPRODUCIBILITY.md) explains acquisition scripts, missing
artifacts, provenance checks and the separation from the local fault labels.
Third-party data and dependencies retain their own licensing requirements.
No project-wide open-source licence has been selected for this repository yet.
