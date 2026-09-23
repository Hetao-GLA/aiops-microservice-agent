import hashlib
from pathlib import Path

from experiment.data_import.download_public_indexes import DATASETS, build_manifest


def test_public_index_destinations_are_unique() -> None:
    destinations = [item["relative_path"] for item in DATASETS]

    assert len(destinations) == len(set(destinations))
    assert all(not Path(path).is_absolute() for path in destinations)


def test_build_manifest_records_existing_files(tmp_path: Path) -> None:
    for index, dataset in enumerate(DATASETS):
        path = tmp_path / dataset["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"index-{index}".encode())

    manifest = build_manifest(tmp_path)

    assert manifest["schema_version"] == 1
    assert all(item["downloaded"] for item in manifest["datasets"])
    first_bytes = b"index-0"
    assert manifest["datasets"][0]["sha256"] == hashlib.sha256(first_bytes).hexdigest()
