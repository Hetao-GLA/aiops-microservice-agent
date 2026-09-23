from pathlib import Path
import pytest
from experiment.local_telemetry import _rename_directory_with_retry


def test_transient_lock_retries(tmp_path, monkeypatch):
    source = tmp_path / ".trial"
    target = tmp_path / "incident"
    source.mkdir()
    (source / "evidence.txt").write_text("keep")
    original = Path.rename
    calls = []
    sleeps = []

    def locked(path, destination):
        calls.append(path)
        if len(calls) < 3:
            error = PermissionError("locked")
            error.winerror = 32
            raise error
        return original(path, destination)

    monkeypatch.setattr(Path, "rename", locked)
    _rename_directory_with_retry(source, target, sleep=sleeps.append)
    assert sleeps == [0.1, 0.2]
    assert (target / "evidence.txt").read_text() == "keep"


@pytest.mark.parametrize("winerror,retries", [(5, 3), (None, 0)])
def test_failure_preserves_evidence(tmp_path, monkeypatch, winerror, retries):
    source = tmp_path / ".trial"
    source.mkdir()
    target = tmp_path / "incident"
    sleeps = []

    def denied(path, destination):
        error = PermissionError("denied")
        error.winerror = winerror
        raise error

    monkeypatch.setattr(Path, "rename", denied)
    with pytest.raises(PermissionError):
        _rename_directory_with_retry(source, target, attempts=4, sleep=sleeps.append)
    assert len(sleeps) == retries
    assert source.exists()
    assert not target.exists()


def test_existing_target_is_not_replaced(tmp_path):
    source = tmp_path / ".trial"
    target = tmp_path / "incident"
    source.mkdir()
    target.mkdir()
    with pytest.raises(FileExistsError):
        _rename_directory_with_retry(source, target)
    assert source.exists() and target.exists()
