"""Run a preregistered holdout with temporary workload changes and restoration."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

import httpx

from experiment.campaign_runner import CampaignConfig, campaign_plan, load_config, preflight, run_campaign
from experiment.frozen_holdout import verify_bundle, write_json
from experiment.ml_baseline import sha256_file


def _command(arguments: list[str]) -> str:
    completed = subprocess.run(arguments, capture_output=True, text=True, check=True, timeout=60)
    return completed.stdout.strip()


def workload_state() -> dict[str, Any]:
    container_id = _command(["docker", "compose", "-f", "docker-compose.yml", "ps", "-q", "workload"])
    if not container_id or "\n" in container_id:
        raise RuntimeError("Expected exactly one running workload container")
    env = json.loads(_command(["docker", "inspect", "--format", "{{json .Config.Env}}", container_id]))
    values = dict(item.split("=", 1) for item in env if "=" in item)
    return {
        "container_id": container_id,
        "image_id": _command(["docker", "inspect", "--format", "{{.Image}}", container_id]),
        "interval_seconds": float(values["WORKLOAD_INTERVAL_SECONDS"]),
    }


def set_workload(override: Path | None) -> None:
    command = ["docker", "compose", "-f", "docker-compose.yml"]
    if override is not None:
        command += ["-f", str(override)]
    _command(command + ["up", "-d", "--no-deps", "--no-build", "--pull", "never", "workload"])


def _healthy_and_no_fault(base_url: str) -> None:
    health = httpx.get(base_url + "/health/database", timeout=5)
    health.raise_for_status()
    fault = httpx.get(base_url + "/internal/faults/http-500", timeout=5)
    fault.raise_for_status()
    if fault.json().get("enabled") is not False:
        raise RuntimeError("HTTP-500 fault must be disabled before/after the campaign")


def holdout_plan(config: CampaignConfig) -> dict[str, Any]:
    settings = config.payload["holdout"]
    bundle = Path(settings["frozen_bundle"])
    manifest = verify_bundle(bundle)
    checks = preflight(config)
    protocol_path = Path(settings["protocol_output"])
    if protocol_path.exists():
        raise FileExistsError(f"Protocol already exists; refusing to overwrite: {protocol_path}")
    override = Path(settings["workload_override"])
    if not override.is_file():
        raise FileNotFoundError(override)
    warmup = float(settings["warmup_seconds"])
    baseline = float(settings["baseline_interval_seconds"])
    experimental = float(settings["experimental_interval_seconds"])
    if not (0 <= warmup <= 60 and baseline > 0 and experimental > 0):
        raise ValueError("Invalid workload/warmup settings")
    code_paths = [
        Path(__file__),
        Path("docker-compose.yml"),
        override,
        Path("experiment/campaign_runner.py"),
        Path("experiment/mixed_runner.py"),
        Path("experiment/local_telemetry.py"),
        Path("experiment/workload.py"),
    ]
    code_paths.extend(
        Path(value) for value in settings.get("additional_code_paths", [])
    )
    if len(set(code_paths)) != len(code_paths):
        raise ValueError("Holdout code paths must be unique")
    missing_code = [str(path) for path in code_paths if not path.is_file()]
    if missing_code:
        raise FileNotFoundError(f"Holdout code paths are missing: {missing_code}")
    return {
        "schema_version": 1,
        "registered_at": datetime.now(UTC).isoformat(),
        "campaign": campaign_plan(config), "preflight": checks,
        "frozen_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "frozen_model_sha256": manifest["model_sha256"],
        "frozen_at": manifest["created_at"],
        "training_dataset_sha256": manifest["training"]["sha256"],
        "workload": settings,
        "code_and_config_sha256": {
            str(path): sha256_file(path) for path in code_paths
        },
        "interpretation": settings.get(
            "interpretation",
            "Joint condition shift; same-system closed-set retrospective "
            "classification, no tuning on this campaign.",
        ),
    }


def run_holdout_campaign(config: CampaignConfig) -> dict[str, Any]:
    protocol = holdout_plan(config)
    settings = config.payload["holdout"]
    path = Path(settings["protocol_output"])
    before = workload_state()
    if before["interval_seconds"] != float(settings["baseline_interval_seconds"]):
        raise RuntimeError("Workload is not at the expected baseline; refusing to change it")
    _healthy_and_no_fault(str(config.payload["base_url"]))
    protocol.update({"status": "registered", "runtime_before": before})
    write_json(path, protocol)
    try:
        set_workload(Path(settings["workload_override"]))
        during = workload_state()
        protocol["runtime_during"] = during
        if during["interval_seconds"] != float(settings["experimental_interval_seconds"]):
            raise RuntimeError("Experimental workload interval was not applied")
        if during["image_id"] != before["image_id"]:
            raise RuntimeError("Workload image changed; only the interval may change")
        time.sleep(float(settings["warmup_seconds"]))
        protocol["status"] = "collecting"
        write_json(path, protocol)
        summary = run_campaign(config)
        protocol["campaign_summary"] = summary
        protocol["status"] = "collected"
    except BaseException as exc:
        protocol["status"] = "failed"
        protocol["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            set_workload(None)
            after = workload_state()
            protocol["runtime_after"] = after
            protocol["workload_restored"] = (
                after["interval_seconds"] == before["interval_seconds"]
                and after["image_id"] == before["image_id"]
            )
            if not protocol["workload_restored"]:
                raise RuntimeError("Original workload settings were not restored")
            _healthy_and_no_fault(str(config.payload["base_url"]))
            protocol["post_run_health_check_passed"] = True
        except BaseException as exc:
            protocol["restoration_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            protocol["finished_at"] = datetime.now(UTC).isoformat()
            write_json(path, protocol)
    print(json.dumps({"protocol": str(path), "status": protocol["status"],
                      "workload_restored": protocol["workload_restored"]}, indent=2), flush=True)
    return protocol


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiment/configs/local-campaign-v2.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(holdout_plan(config), ensure_ascii=False, indent=2))
    else:
        run_holdout_campaign(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
