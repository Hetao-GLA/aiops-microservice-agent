# Getting started: local source-only demonstration

This guide runs the order-service testbed and the human-approved operations
Agent from a fresh checkout. It does not reproduce the dissertation's frozen
experiments or load its trained models.

The default console diagnoses three controlled fault types using live rule
probes. Its deterministic planner explains the selected local Runbook; it is
not an LLM and does not load the research TF-IDF/logistic-regression or
OOD-gated models. The optional model-backed planner is a separate explanation
and read-only-tool adapter. Neither mode permits arbitrary recovery commands.

## 1. Prerequisites and safety boundary

- Windows with PowerShell, Git and Python 3.12 available as `py -3.12`.
- Docker Desktop running with Linux containers and the Compose CLI available.
- Internet access for the initial Git clone, package installation and container
  image downloads. The deterministic demonstration itself needs no external
  model API or API key.
- Ports 8000 and 8100 free, and no other copy of this testbed running. Compose
  uses fixed container names such as `aiops-database`; separate checkout
  directories do not isolate those containers.

Use a disposable local development environment, not a production service.
The testbed includes unauthenticated fault-control endpoints, a demonstration
database password (`postgres`), and synthetic order generation. The Agent API
also has no user authentication: entering an operator name records a decision
but does not authenticate that person. Its approval gate and action allow-list
are workflow safeguards, not an Internet-facing security boundary.

The Agent entry point binds to `127.0.0.1:8100`. The checked-in Compose file's
`8000:8000` mapping can expose the business API on host network interfaces.
Before starting a local demo, change the `api.ports` entry in your local
`docker-compose.yml` from `"8000:8000"` to `"127.0.0.1:8000:8000"`.
Do not publish either service through a public tunnel or router port forwarding.

## 2. Clone and install the local Python dependencies

Run these commands from the directory in which you want to keep the checkout:

```powershell
git clone https://github.com/Hetao-GLA/aiops-microservice-agent.git
Set-Location -LiteralPath '.\aiops-microservice-agent'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the repository is private, GitHub authorization is required to clone it.
If you already have the checkout, open PowerShell in its root instead of
cloning again. All remaining commands assume the repository root as the
working directory. Run each command only after the previous one succeeds.
Calling the virtual environment's interpreter directly avoids PowerShell
activation-policy changes.

`requirements.txt` supplies the deterministic Agent and testbed dependencies.
It uses version ranges, not the exact historical frozen-evaluation environment.
Installing it does not establish equivalence with the dissertation's model
environment. Installing packages on the host also does not install them in
Docker: the Dockerfile installs its own dependencies during the build.

## 3. Start the Docker testbed

Open Docker Desktop, wait for its engine to be ready, and check:

```powershell
docker version
docker compose version
```

`docker version` must show a working **Server**, not only a Client. If it
cannot connect to the Docker Desktop Linux engine, resolve that before
continuing. Starting a Python virtual environment does not start Docker.

After applying the loopback port mapping described above, run:

```powershell
docker compose -f .\docker-compose.yml up -d --build
docker compose -f .\docker-compose.yml ps
```

The four Compose services are `api`, `database`, `workload` and `detector`.
The workload writes synthetic orders to PostgreSQL; the detector writes local
JSONL records under `data/detections/`. Leave time for the database and API
health checks to pass, then run:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health/live' -TimeoutSec 5
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health/database' -TimeoutSec 5
```

Both should return `status: healthy`. If they do not, inspect the services
before continuing:

```powershell
docker compose -f .\docker-compose.yml logs --tail 60 api database
```

Review logs locally; redact credentials, personal identifiers and unrelated
system information before sharing them.

## 4. Start the host-side Agent

In a second PowerShell window opened at the same repository root, run:

```powershell
$env:OPS_AGENT_PLANNER = 'deterministic'
$env:OPS_AGENT_PROJECT_ROOT = (Get-Location).Path
$env:OPS_AGENT_COMPOSE_FILE = 'docker-compose.yml'
$env:OPS_AGENT_TARGET_BASE_URL = 'http://127.0.0.1:8000'
$env:OPS_AGENT_DATA_DIR = 'data/agent'
$env:OPS_AGENT_DETECTION_PATH = 'data/detections/detections.jsonl'
.\.venv\Scripts\python.exe -m ops_agent
```

Keep this window open. The host-side Agent needs access to the same Docker
engine and Compose project to read logs and execute the approved fixed start
commands. It is not a Compose service in this repository.

The Agent reads environment variables; copying `.env.example` to `.env` does
not automatically load those values into the host Python process. The explicit
PowerShell assignments above avoid that ambiguity.

Open the console at <http://127.0.0.1:8100/>. The interactive API documentation
is at <http://127.0.0.1:8100/docs>. In another terminal, you can check:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8100/agent/health' -TimeoutSec 5
Invoke-RestMethod -Uri 'http://127.0.0.1:8100/agent/runbooks' -TimeoutSec 5
```

The health response should name the `deterministic` planner and report
`human_approval_required: true` and `arbitrary_command_execution: false`.
The Runbooks endpoint should list three entries.

A fresh clone has no historical Agent incidents, trained model bundles, raw
telemetry or processed experiment datasets. An empty incident list is expected.
New incident snapshots and audit events are created under `data/agent/` as you
use the console. Do not publish these runtime records without review.

If all services are healthy, **Diagnose current fault** should report that no
active supported fault was found. That is expected; the system should not
propose a recovery just to populate the interface.

## 5. Optional controlled database-failure demonstration

This section deliberately interrupts the local demo database. Orders will
temporarily fail until the database is started again. Do not run it against an
existing environment used by other people or another experiment. Confirm both
health checks pass first and keep the manual restoration command below ready.

In the first terminal, from this checkout's root:

```powershell
docker compose -f .\docker-compose.yml stop database
```

This command does **not** set a timer or automatically restore the database.
It is a manual software demonstration, not a new registered research incident
or a replacement for the experiment collection scripts.

In the Agent console:

1. Select **Diagnose current fault**.
2. Use **View** to inspect the diagnosis, health evidence and Runbook. For this
   controlled setup, expect `database_connection_failure` and `start_database`.
   If the evidence or proposed action is different, stop and investigate.
3. Select **Approve**, enter an operator label and a decision note in the
   prompts. This records approval; it does not yet execute recovery. **Reject**
   records rejection without executing an action.
4. Select **Execute recovery** and confirm the browser prompt. This starts only
   the allow-listed `database` Compose service, then checks application
   liveness and database health.
5. Inspect the final status and verification evidence. `Resolved` means those
   implemented checks passed; command completion alone is not treated as
   recovery success. **Refresh data** reloads the displayed records.

If you reject the proposal, close the Agent, or encounter a failure, manually
restore the intentionally stopped service from the repository root:

```powershell
docker compose -f .\docker-compose.yml start database
```

Allow PostgreSQL to become ready, then repeat the two business health checks
from section 3. A manual restoration is not an Agent-executed success. If the
incident is specifically in `verification_failed` state, **Retry verification**
checks health again without repeating the recovery action. Do not delete the
database volume to resolve a failed demonstration.

The other implemented controlled-fault mappings are:

| Fault label | Policy-locked recovery |
| --- | --- |
| `service_stopped` | Start the existing `api` Compose service |
| `http_500_failure` | Disable the application's controlled HTTP-500 flag |

These labels reflect a restricted testbed, not proof of arbitrary production
root-cause diagnosis. A failed health probe can have causes beyond a stopped
container. Always review the evidence and current fault before approving.

## 6. Optional dependencies and tests

The optional model-backed planner requires `requirements-agent.txt`, provider
credentials and a model configuration. It may send evidence to an external
service and incur charges. It does not replace the trusted fault label or the
locked recovery policy. No such service is needed for this guide; never commit
API keys. The deterministic demonstration is not evidence that an LLM improves
diagnosis or recovery time.

To install the additional dependencies used by the automated test suite:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-data.txt
.\.venv\Scripts\python.exe -m pytest
```

The data requirements matter: several tests import pandas, NumPy, joblib and
scikit-learn through the experiment modules. Test success is software
verification, not a new run of the published experiments or a complete Docker
fault-recovery acceptance test. Historical test counts are not a guarantee for
every future resolution of the dependency ranges.

## 7. Stop the demonstration

Restore any deliberately stopped service first. Press **Ctrl+C** in the
Agent's terminal. Then, from the repository root:

```powershell
docker compose -f .\docker-compose.yml down
```

This removes the demo containers and Compose network but retains the named
PostgreSQL volume. Do not add `-v` unless you intentionally want to delete that
database data. Host-side files under `data/` also remain.

For the source layout and research boundaries, return to the
[project README](../README.md).
