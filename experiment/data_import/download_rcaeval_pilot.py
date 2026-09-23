from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment.data_import.download_public_indexes import download_file, sha256_file


REPOSITORY = "https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/main"

PILOT_CASES: tuple[str, ...] = (
    "re2ob_checkoutservice_socket_1",
    "re2ob_checkoutservice_socket_2",
    "re2ob_checkoutservice_socket_3",
    "re3ob_cartservice_f1_1",
    "re3ob_cartservice_f1_2",
    "re3ob_cartservice_f1_3",
)

CORE_FILES: tuple[str, ...] = (
    "inject_time.txt",
    "logs.parquet",
    "metrics.parquet",
)


def remote_url(case: str, filename: str) -> str:
    return f"{REPOSITORY}/{case}/{filename}?download=true"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the six-case RCAEval logs-and-metrics pilot."
    )
    parser.add_argument(
        "--root", type=Path, default=Path("data/external/rcaeval/pilot")
    )
    parser.add_argument("--include-traces", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    filenames = (*CORE_FILES, "traces.parquet") if args.include_traces else CORE_FILES
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source": "https://huggingface.co/datasets/phamquiluan/RCAEval",
        "selection_reason": (
            "Three RE2-OB socket cases and three documented RE3-OB "
            "cartservice F1 code-level cases for importer validation only."
        ),
        "includes_traces": args.include_traces,
        "cases": [],
    }

    case_entries: list[dict[str, object]] = []
    for case in PILOT_CASES:
        files: list[dict[str, object]] = []
        for filename in filenames:
            destination = args.root / case / filename
            print(f"Downloading {case}/{filename}")
            download_file(
                remote_url(case, filename), destination, overwrite=args.overwrite
            )
            files.append(
                {
                    "name": filename,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
        case_entries.append({"case": case, "files": files})

    manifest["cases"] = case_entries
    manifest_path = args.root / "pilot-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote pilot manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
