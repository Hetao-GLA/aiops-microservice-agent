from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from experiment.data_import.download_public_indexes import download_file, sha256_file


REPOSITORY = "https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/main"
DEFAULT_SPEC = Path(
    "experiment/configs/rcaeval-re2-ob-public-baseline-spec.json"
)


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional data environment
        raise RuntimeError(
            "RCAEval subset download requires requirements-data.txt"
        ) from exc
    return pd


def remote_url(case: str, filename: str) -> str:
    return f"{REPOSITORY}/{case}/{filename}?download=true"


def load_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported RCAEval subset spec version")
    return payload


def select_cases(index_path: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    expected_hash = str(spec["source"]["index_sha256"])
    actual_hash = sha256_file(index_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"RCAEval index hash mismatch: expected={expected_hash}, actual={actual_hash}"
        )

    pd = _pandas()
    frame = pd.read_parquet(index_path)
    required = {
        "case",
        "dataset",
        "system_name",
        "root_cause_service",
        "fault",
        "repetition",
        "has_logs",
    }
    if spec["selection"].get("require_metrics", False):
        required.add("n_metrics")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"RCAEval index is missing columns: {sorted(missing)}")

    selection = spec["selection"]
    subset = frame[
        frame["dataset"].eq(selection["dataset"])
        & frame["system_name"].eq(selection["system_name"])
    ].copy()
    if selection["require_logs"]:
        subset = subset[subset["has_logs"].eq(True)]  # noqa: E712
    if selection.get("require_metrics", False):
        subset = subset[subset["n_metrics"].gt(0)]
    subset = subset.sort_values("case", kind="stable")

    expected_services = sorted(selection["expected_root_cause_services"])
    expected_faults = sorted(selection["expected_faults"])
    actual_services = sorted(subset["root_cause_service"].astype(str).unique())
    actual_faults = sorted(subset["fault"].astype(str).unique())
    if len(subset) != int(selection["expected_cases"]):
        raise ValueError(f"Unexpected selected case count: {len(subset)}")
    if actual_services != expected_services:
        raise ValueError(f"Unexpected root-cause services: {actual_services}")
    if actual_faults != expected_faults:
        raise ValueError(f"Unexpected fault types: {actual_faults}")

    expected_repetitions = int(
        selection["expected_repetitions_per_service_fault_pair"]
    )
    pair_counts = Counter(
        zip(
            subset["root_cause_service"].astype(str),
            subset["fault"].astype(str),
        )
    )
    expected_pairs = {
        (service, fault)
        for service in expected_services
        for fault in expected_faults
    }
    if set(pair_counts) != expected_pairs or set(pair_counts.values()) != {
        expected_repetitions
    }:
        raise ValueError("RCAEval service/fault balance does not match the locked spec")

    return [row.to_dict() for _, row in subset.iterrows()]


def _reuse_cached_file(candidates: Iterable[Path], destination: Path) -> bool:
    if destination.exists():
        return True
    for candidate in candidates:
        if candidate.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
            return True
    return False


def download_subset(
    cases: list[dict[str, Any]],
    *,
    filenames: list[str],
    root: Path,
    cache_roots: Iterable[Path] = (),
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for metadata in cases:
        case = str(metadata["case"])
        files: list[dict[str, Any]] = []
        for filename in filenames:
            destination = root / case / filename
            reused = False
            if not overwrite:
                reused = _reuse_cached_file(
                    (cache_root / case / filename for cache_root in cache_roots),
                    destination,
                )
            if not destination.exists() or overwrite:
                print(f"Downloading {case}/{filename}")
                download_file(
                    remote_url(case, filename), destination, overwrite=overwrite
                )
            elif reused:
                print(f"Reused cached {case}/{filename}")
            files.append(
                {
                    "name": filename,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
        entries.append(
            {
                "case": case,
                "root_cause_service": str(metadata["root_cause_service"]),
                "fault": str(metadata["fault"]),
                "repetition": int(metadata["repetition"]),
                "files": files,
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a locked RCAEval public subset."
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--index", type=Path, default=Path("data/external/rcaeval/cases.parquet")
    )
    parser.add_argument(
        "--root", type=Path, default=Path("data/external/rcaeval/re2-ob")
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        action="append",
        default=[Path("data/external/rcaeval/pilot")],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    cases = select_cases(args.index, spec)
    filenames = [str(value) for value in spec["telemetry"]["files"]]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "cases": len(cases),
                    "files_per_case": filenames,
                    "root_cause_services": dict(
                        sorted(Counter(str(row["root_cause_service"]) for row in cases).items())
                    ),
                    "faults": dict(
                        sorted(Counter(str(row["fault"]) for row in cases).items())
                    ),
                },
                indent=2,
            )
        )
        return

    entries = download_subset(
        cases,
        filenames=filenames,
        root=args.root,
        cache_roots=args.cache_root,
        overwrite=args.overwrite,
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
        "source": spec["source"]["repository"],
        "spec": str(args.spec),
        "spec_sha256": sha256_file(args.spec),
        "index_sha256": sha256_file(args.index),
        "includes_traces": False,
        "cases": entries,
        "totals": {
            "cases": len(entries),
            "files": sum(len(entry["files"]) for entry in entries),
            "bytes": sum(
                int(file["bytes"])
                for entry in entries
                for file in entry["files"]
            ),
        },
    }
    manifest_path = args.root / "subset-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(entries)}-case manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
