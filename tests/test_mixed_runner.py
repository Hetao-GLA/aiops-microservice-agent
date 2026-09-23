from collections import Counter

from experiment.mixed_runner import SUPPORTED_FAULTS, build_balanced_schedule


def test_mixed_schedule_is_balanced_and_reproducible() -> None:
    first = build_balanced_schedule(repetitions_per_fault=2, seed=42)
    second = build_balanced_schedule(repetitions_per_fault=2, seed=42)

    assert first == second
    assert Counter(first) == {fault: 2 for fault in SUPPORTED_FAULTS}
    assert len(first) == 6

