from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


DEFAULT_ROOT = Path("data/external")

DATASETS: tuple[dict[str, str], ...] = (
    {
        "name": "rcaeval",
        "url": (
            "https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/"
            "main/cases.parquet?download=true"
        ),
        "relative_path": "rcaeval/cases.parquet",
        "source": "https://github.com/phamquiluan/RCAEval",
        "license": "MIT",
    },
    {
        "name": "lo2v2",
        "url": (
            "https://zenodo.org/records/18937117/files/"
            "LO2v2_index.json?download=1"
        ),
        "relative_path": "lo2v2/LO2v2_index.json",
        "source": "https://zenodo.org/records/18937117",
        "license": "CC-BY-4.0",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, *, overwrite: bool = False) -> None:
    if destination.exists() and not overwrite:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "microservice-ops-research/1.0"})

    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_manifest(root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        path = root / dataset["relative_path"]
        entries.append(
            {
                **dataset,
                "downloaded": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256_file(path) if path.exists() else None,
            }
        )
    return {"schema_version": 1, "datasets": entries}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download only the RCAEval and LO2v2 metadata indexes."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for dataset in DATASETS:
        destination = args.root / dataset["relative_path"]
        print(f"Downloading {dataset['name']} index -> {destination}")
        download_file(dataset["url"], destination, overwrite=args.overwrite)

    manifest = build_manifest(args.root)
    manifest_path = args.root / "download-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
